"""Synthetic end-to-end smoke test -- no GPU or nnU-Net weights required.

Builds a fake CBCT + DentalSegmentator-style labelmap where the LOWER arch is
dominant, with lower-tooth crowns sticking up above the mandibular crest. Checks:
  - only the dominant (mandible) arch is output; maxilla/upper teeth dropped
  - tooth roots keep the part at/below the crest; crowns above it are removed
  - outputs are geometry-aligned with the input

Run:  python tests/test_synthetic.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbct_pipeline.pipeline import run_pipeline  # noqa: E402

CONFIG = {
    "arches": {
        "lower": {"name": "Mandible", "bone_labels": [2], "teeth_labels": [4],
                  "crown_direction": "superior"},
        "upper": {"name": "Maxilla & Upper bone", "bone_labels": [1],
                  "teeth_labels": [3], "crown_direction": "inferior"},
    },
    "dominance": {"metric": "bone"},
    "root": {"bone_close_mm": 1.0, "smooth_iter": 1},
    "output": {"teeth_name": "teeth", "root_name": "travecular"},
}

CREST_Z = 25   # top of mandible bone along the S (axis-2, +index=superior) axis


def build_synthetic(tmp: Path):
    shape = (60, 60, 44)            # axes (X, Y, Z); Z is superior-inferior
    spacing = (0.3, 0.3, 0.3)
    affine = np.diag([*spacing, 1.0])   # aff2axcodes -> ('R','A','S'); axis 2 = S

    lm = np.zeros(shape, dtype=np.uint8)
    # Lower arch (dominant): big mandible block, z in [10, CREST_Z]
    lm[10:50, 10:50, 10:CREST_Z + 1] = 2
    # Lower teeth: root + crown AND a tip that overshoots BELOW the bone base
    # (z 5..9 is below the mandible bottom at z=10 -> must be clipped away).
    lm[20:40, 20:40, 5:36] = 4
    # Upper arch (smaller -> not dominant): maxilla + upper teeth, high z
    lm[15:30, 15:30, 30:38] = 1
    lm[18:27, 18:27, 26:31] = 3

    img = np.full(shape, 300, np.float32)
    img[lm == 2] = 2000
    img[lm == 4] = 2600
    img_path = tmp / "0102871823.nii.gz"
    lm_path = tmp / "seg.nii.gz"
    nib.save(nib.Nifti1Image(img, affine), str(img_path))
    nib.save(nib.Nifti1Image(lm, affine), str(lm_path))
    return img_path, lm_path, affine


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        img_path, lm_path, affine = build_synthetic(tmp)
        out_dir = tmp / "out"

        written = run_pipeline(img_path, out_dir, CONFIG, labelmap_path=lm_path)

        # Only the dominant arch should be produced; bone uses the arch name.
        assert set(written) == {"bone", "teeth", "root"}, set(written)
        assert written["bone"].name == "0102871823_Mandible.nii.gz", written["bone"].name
        assert written["teeth"].name == "0102871823_teeth.nii.gz"
        assert written["root"].name == "0102871823_travecular.nii.gz"
        assert not list(out_dir.glob("*Maxilla*")), "upper arch was not discarded"

        for key, p in sorted(written.items()):
            m = nib.load(str(p))
            assert np.allclose(m.affine, affine), "affine not preserved"
            n = int(np.asanyarray(m.dataobj).sum())
            assert n > 0, f"{p.name} empty"
            print(f"  OK  {p.name:34s} voxels={n}")

        teeth = np.asanyarray(
            nib.load(str(written["teeth"])).dataobj).astype(bool)
        root = np.asanyarray(
            nib.load(str(written["root"])).dataobj).astype(bool)
        # Root must be a strict subset of full teeth (crowns removed).
        assert (root & teeth).sum() == root.sum(), "root not a subset of teeth"
        assert root.sum() < teeth.sum(), "no crown was removed"
        # Full teeth keep crown + the overshoot tip (z 5..35); root must not.
        ztz = np.where(teeth.any(axis=(0, 1)))[0]
        assert ztz.max() > CREST_Z, "full teeth missing the crown"
        assert ztz.min() < 10, "full teeth missing the below-bone tip"
        # Root must sit within the bone band [base=10, crest=25]: no crown above,
        # no overshoot below the bone base.
        zsup = np.where(root.any(axis=(0, 1)))[0]
        assert zsup.max() <= CREST_Z, f"crown not removed: root reaches z={zsup.max()}"
        assert zsup.min() >= 10, f"root overshoots below bone base: z={zsup.min()}"
        print(f"  OK  full teeth z[{ztz.min()},{ztz.max()}] (crown+tip); "
              f"root z[{zsup.min()},{zsup.max()}] bounded to bone band [10,{CREST_Z}]")

    print("\nAll synthetic checks passed.")
    test_no_float_above_bone()
    return 0


def test_no_float_above_bone():
    """Root must track the LOCAL bone level, not a tall neighbouring septum, so
    the cervical neck above the bone is not kept (no 'red above blue')."""
    sp = (0.3, 0.3, 0.3)
    aff = np.diag([*sp, 1.0])
    sh = (60, 60, 40)
    bone = np.zeros(sh, bool)
    bone[10:22, 10:50, 10:31] = True     # tall left wall (to z=30)
    bone[38:50, 10:50, 10:31] = True     # tall right wall (to z=30)
    bone[22:38, 10:50, 10:21] = True     # low bone near the tooth (to z=20)
    tooth = np.zeros(sh, bool)
    tooth[26:34, 24:36, 15:40] = True    # neck/crown up to z=40
    from cbct_pipeline.roots import extract_root
    r = extract_root(tooth, bone, aff, "superior", spacing=sp,
                     bone_close_mm=1.0, smooth_iter=0)
    zs = np.where(r.any(axis=(0, 1)))[0]
    assert zs.size and zs.max() <= 22, \
        f"root floats above local bone to z={zs.max()} (tall septum at z=30)"
    print(f"  OK  enclosure: root stops at local bone z<= {zs.max()} "
          "(not the tall septum at z=30)")


if __name__ == "__main__":
    raise SystemExit(main())
