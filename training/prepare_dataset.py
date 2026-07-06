#!/usr/bin/env python3
"""Build an nnU-Net v2 raw dataset for the tooth-ROOT model (design A).

Each training case becomes a multi-channel input + a root label:
  channel 0: CBCT image
  channel 1: tooth mask   (from DentalSegmentator: upper+lower teeth)
  channel 2: bone mask    (from DentalSegmentator: maxilla+mandible)
  label    : your root ground truth (binary; 0=bg, 1=root)

Giving the model the tooth and bone masks makes the task well-posed: it learns
"which part of THIS tooth is root", with the bone crest as context. That's what
"reuse DentalSegmentator" means here.

The tooth/bone channels can be supplied precomputed (--tooth/--bone dirs) or
generated on the fly from DentalSegmentator (--dentalseg-config). Cases are
matched across folders by the leading patient id in the filename.

Output layout (nnU-Net v2):
  <out>/Dataset<ID>_<NAME>/
    imagesTr/<case>_0000.nii.gz  # CBCT
    imagesTr/<case>_0001.nii.gz  # tooth   (omitted if channels excludes it)
    imagesTr/<case>_0002.nii.gz  # bone
    labelsTr/<case>.nii.gz       # root
    dataset.json

Usage
-----
  # labels + precomputed tooth/bone masks already on disk
  python training/prepare_dataset.py \
      --images data/img --roots data/root_gt \
      --tooth data/tooth --bone data/bone \
      --out /path/nnUNet_raw --dataset-id 201 --dataset-name ToothRoot

  # generate tooth/bone from DentalSegmentator instead
  python training/prepare_dataset.py \
      --images data/img --roots data/root_gt \
      --dentalseg-config config.dentalsegmentator.yaml \
      --out /path/nnUNet_raw --dataset-id 201 --dataset-name ToothRoot

  # simplest: CBCT-only input (no DentalSegmentator channels)
  python training/prepare_dataset.py --images data/img --roots data/root_gt \
      --channels image --out /path/nnUNet_raw --dataset-id 201 --dataset-name ToothRoot
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbct_pipeline import labels as L                     # noqa: E402
from cbct_pipeline.pipeline import derive_patient_id      # noqa: E402

log = logging.getLogger("prepare_dataset")
SUFFIXES = (".nii", ".nii.gz")
CHANNEL_INDEX = {"image": 0, "tooth": 1, "bone": 2}


def index_folder(folder: Path) -> dict:
    """{patient_id: path} for every .nii/.nii.gz in folder."""
    out = {}
    if folder is None:
        return out
    for p in sorted(Path(folder).iterdir()):
        if p.name.endswith(SUFFIXES):
            out.setdefault(derive_patient_id(p, None), p)
    return out


def _strip_ext(name: str) -> str:
    for suf in (".nii.gz", ".nii"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _pid_and_suffix(path: Path):
    """Split '0102330056_teeth.nii.gz' -> ('0102330056', '_teeth')."""
    stem = _strip_ext(path.name)
    m = re.match(r"([A-Za-z0-9]+)(.*)$", stem)
    if not m:
        return None, stem
    return m.group(1), m.group(2)


def index_by_suffix(folder: Path, wanted) -> dict:
    """{patient_id: path} for files whose suffix (text after the id) is in
    `wanted` (a list of exact suffix strings, e.g. ['', '_teeth'])."""
    out = {}
    if folder is None:
        return out
    wanted = list(wanted)
    for p in sorted(Path(folder).iterdir()):
        if not p.name.endswith(SUFFIXES):
            continue
        pid, suf = _pid_and_suffix(p)
        if pid is None:
            continue
        if suf in wanted:
            out.setdefault(pid, p)
    return out


def load(path: Path):
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj), img.affine, img.header


def same_grid(a_shape, a_aff, b_shape, b_aff) -> bool:
    return a_shape == b_shape and np.allclose(a_aff, b_aff, atol=1e-3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    help="single folder holding image + masks per patient, named "
                         "<id>[_suffix].nii.gz (uses the --*-suffix options below)")
    ap.add_argument("--images", type=Path, help="folder of CBCT images (if not using --dir)")
    ap.add_argument("--roots", type=Path, help="root ground-truth masks (if not using --dir)")
    ap.add_argument("--tooth", type=Path, help="precomputed tooth masks (optional)")
    ap.add_argument("--bone", type=Path, help="precomputed bone masks (optional)")
    ap.add_argument("--image-suffix", default="", help="[--dir] suffix for the CBCT (default: none)")
    ap.add_argument("--root-suffix", default="_travecular", help="[--dir] suffix for the root label")
    ap.add_argument("--tooth-suffix", default="_teeth", help="[--dir] suffix for the tooth mask")
    ap.add_argument("--bone-suffix", default="_Mandible,_Maxilla & Upper Skull",
                    help="[--dir] comma list of possible bone-mask suffixes; first match per case")
    ap.add_argument("--dentalseg-config", type=Path,
                    help="generate tooth/bone from DentalSegmentator using this config")
    ap.add_argument("--channels", default="image,tooth,bone",
                    help="comma list from image,tooth,bone (default all three)")
    ap.add_argument("--out", required=True, type=Path, help="nnUNet_raw root")
    ap.add_argument("--dataset-id", required=True, type=int)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--mask-channel-noNorm", action="store_true",
                    help="name tooth/bone channels 'noNorm' so nnU-Net skips z-scoring them")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    for c in channels:
        if c not in CHANNEL_INDEX:
            raise SystemExit(f"Unknown channel {c!r} (use image,tooth,bone)")
    if "image" not in channels:
        raise SystemExit("channels must include 'image'")
    need_masks = ("tooth" in channels) or ("bone" in channels)

    # Resolve where each input comes from: single folder (--dir, by suffix) or
    # explicit folders (--images/--roots/--tooth/--bone).
    dir_files = {}
    if args.dir:
        if not Path(args.dir).is_dir():
            raise SystemExit(f"--dir folder not found: {args.dir}")
        bone_suffixes = [s.strip() for s in args.bone_suffix.split(",")]
        # group every file by patient id -> {suffix: path}
        for p in sorted(Path(args.dir).iterdir()):
            if not p.name.endswith(SUFFIXES):
                continue
            pid, suf = _pid_and_suffix(p)
            if pid is not None:
                dir_files.setdefault(pid, {})[suf] = p

        imgs, roots, tooth_dir, bone_dir = {}, {}, {}, {}
        role_suffixes = {args.image_suffix, args.root_suffix, args.tooth_suffix}
        for pid, sufs in dir_files.items():
            if args.image_suffix in sufs:
                imgs[pid] = sufs[args.image_suffix]
            if args.root_suffix in sufs:
                roots[pid] = sufs[args.root_suffix]
            if args.tooth_suffix in sufs:
                tooth_dir[pid] = sufs[args.tooth_suffix]
            # bone: try the explicit suffix list, else take the leftover file
            bpath = next((sufs[b] for b in bone_suffixes if b in sufs), None)
            if bpath is None:
                leftover = [pp for ss, pp in sufs.items() if ss not in role_suffixes]
                if len(leftover) == 1:
                    bpath = leftover[0]
                elif len(leftover) > 1:  # prefer an obvious bone name
                    pref = [pp for pp in leftover if any(
                        k in pp.name for k in ("andible", "axilla", "kull", "one"))]
                    bpath = (pref or leftover)[0]
            if bpath is not None:
                bone_dir[pid] = bpath
    else:
        for name, folder in (("--images", args.images), ("--roots", args.roots)):
            if folder is None:
                raise SystemExit(f"{name} is required (or use --dir for a single folder)")
            if not Path(folder).is_dir():
                raise SystemExit(
                    f"{name} folder not found: {folder}\n"
                    "Point it at a real folder containing your .nii/.nii.gz files.")
        imgs = index_folder(args.images)
        roots = index_folder(args.roots)
        tooth_dir = index_folder(args.tooth)
        bone_dir = index_folder(args.bone)

    masks_on_disk = bool(tooth_dir) and bool(bone_dir)

    cases = sorted(set(imgs) & set(roots))
    if not cases:
        raise SystemExit("No patient ids shared between the image and root files. "
                         "Check the suffixes / folders.")
    log.info("Matched %d case(s) with both image and root label", len(cases))

    predictor_cfg = None
    if need_masks and args.dentalseg_config:
        import yaml
        predictor_cfg = yaml.safe_load(open(args.dentalseg_config))
    if need_masks and not masks_on_disk and predictor_cfg is None:
        raise SystemExit(
            "tooth/bone channels requested but none found on disk. Either provide "
            "them (--dir with _teeth/bone files, or --tooth/--bone), pass "
            "--dentalseg-config to generate them, or use --channels image for a "
            "CBCT-only model (no tooth/bone needed).")

    ds = args.out / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    imagesTr = ds / "imagesTr"
    labelsTr = ds / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    def masks_for(pid, img_path, ref_shape, ref_aff):
        """Return (tooth_bool, bone_bool) aligned to the image grid.
        Prefer masks on disk; fall back to DentalSegmentator if configured."""
        if pid in tooth_dir and pid in bone_dir:
            t, ta, _ = load(tooth_dir[pid])
            b, ba, _ = load(bone_dir[pid])
            if not same_grid(t.shape, ta, ref_shape, ref_aff) or \
               not same_grid(b.shape, ba, ref_shape, ref_aff):
                raise ValueError(f"{pid}: tooth/bone mask grid != image grid")
            return t > 0, b > 0
        if predictor_cfg is None:
            present = sorted(dir_files.get(pid, {}).keys())
            hint = f" (found suffixes: {present})" if present else ""
            raise FileNotFoundError(
                f"{pid}: need both tooth and bone{hint}. "
                "Pass --dentalseg-config to auto-generate, or --channels image.")
        # generate with DentalSegmentator
        import tempfile
        from cbct_pipeline.segmentation import segment
        with tempfile.TemporaryDirectory() as tmp:
            lm_path = segment(img_path, Path(tmp), predictor_cfg["nnunet"])
            lm, la, _ = load(lm_path)
        if not same_grid(lm.shape, la, ref_shape, ref_aff):
            raise ValueError(f"{pid}: DentalSegmentator output grid != image grid")
        tl = predictor_cfg.get("arches", {})
        # default DentalSegmentator ids: teeth 3,4 ; bone 1,2
        tooth = L.mask_from_labels(lm, [3, 4])
        bone = L.mask_from_labels(lm, [1, 2])
        return tooth, bone

    written = 0
    skipped = []
    for pid in cases:
        img_path = imgs[pid]
        try:
            image, aff, hdr = load(img_path)
            root, ra, _ = load(roots[pid])
            if not same_grid(root.shape, ra, image.shape, aff):
                log.warning("%s: root grid != image grid -- skipping", pid)
                skipped.append((pid, "root grid mismatch"))
                continue
            chans = {"image": image.astype(np.float32)}
            if need_masks:
                tooth, bone = masks_for(pid, img_path, image.shape, aff)
                chans["tooth"] = tooth.astype(np.float32)
                chans["bone"] = bone.astype(np.float32)

            case = f"{args.dataset_name}_{pid}"
            for c in channels:
                nib.save(nib.Nifti1Image(chans[c], aff, hdr),
                         str(imagesTr / f"{case}_{CHANNEL_INDEX[c]:04d}.nii.gz"))
            rootbin = (root > 0).astype(np.uint8)
            lbl = nib.Nifti1Image(rootbin, aff, hdr)
            lbl.header.set_data_dtype(np.uint8)
            nib.save(lbl, str(labelsTr / f"{case}.nii.gz"))
            written += 1
            log.info("[%d] %s", written, case)
        except Exception as exc:
            log.error("%s: failed (%s)", pid, exc)
            skipped.append((pid, str(exc)))

    if written == 0:
        raise SystemExit("No cases written.")

    # dataset.json
    def cname(c):
        if c in ("tooth", "bone") and args.mask_channel_noNorm:
            return "noNorm"
        return c
    channel_names = {str(CHANNEL_INDEX[c]): cname(c) for c in channels}
    dataset_json = {
        "channel_names": channel_names,
        "labels": {"background": 0, "root": 1},
        "numTraining": written,
        "file_ending": ".nii.gz",
    }
    with open(ds / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    log.info("Wrote %d cases to %s", written, ds)
    if skipped:
        miss = [pid for pid, why in skipped if "no tooth/bone" in why]
        log.warning("Skipped %d of %d matched case(s).", len(skipped), len(cases))
        if miss:
            log.warning("%d skipped for missing tooth/bone mask. To include them, "
                        "either add --dentalseg-config to auto-generate the masks, "
                        "or use --channels image for a CBCT-only model.", len(miss))
    print("\nNext steps (set nnU-Net env vars first: nnUNet_raw / nnUNet_preprocessed / nnUNet_results):")
    print(f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity")
    print(f"  nnUNetv2_train {args.dataset_id} 3d_fullres 0   # repeat folds 1..4, or use 'all'")
    print(f"  nnUNetv2_predict -i IN -o OUT -d {args.dataset_id} -c 3d_fullres -f all")
    print(f"\ndataset.json channel_names: {channel_names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
