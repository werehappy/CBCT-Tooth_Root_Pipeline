#!/usr/bin/env python3
"""Deeper alignment diagnosis: distinguishes a real content mismatch from a mere
orientation/flip difference, and checks that the root label lines up with the CBCT
(which is what matters for a CBCT-only model).

For each patient it loads <id>.nii.gz (image), <id>_teeth.nii.gz, and
<id>_travecular.nii.gz, and reports:
  * root-inside-teeth overlap in RAW ARRAY order (what the previous check did)
  * root-inside-teeth overlap after reorienting both to canonical (RAS)
  * whether image and root share the same affine/orientation

Usage (from the training folder):  python diagnose_alignment.py --dir data
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np


def pid_of(name: str) -> str:
    for s in (".nii.gz", ".nii"):
        if name.endswith(s):
            name = name[: -len(s)]
            break
    m = re.match(r"([A-Za-z0-9]+)", name)
    return m.group(1) if m else name


def frac_inside(a_bool: np.ndarray, b_bool: np.ndarray) -> float:
    n = int(a_bool.sum())
    return (int((a_bool & b_bool).sum()) / n) if n else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--image-suffix", default="")
    ap.add_argument("--root-suffix", default="_travecular")
    ap.add_argument("--tooth-suffix", default="_teeth")
    a = ap.parse_args()

    arr_ov, can_ov = [], []
    img_root_affine_ok = 0
    img_root_affine_bad = []
    orients = Counter()

    roots = sorted(a.dir.glob(f"*{a.root_suffix}.nii.gz"))
    checked = 0
    for rp in roots:
        pid = pid_of(rp.name)
        tp = a.dir / f"{pid}{a.tooth_suffix}.nii.gz"
        ip = a.dir / f"{pid}{a.image_suffix}.nii.gz"
        if not tp.exists():
            continue
        checked += 1
        rn = nib.load(str(rp))
        tn = nib.load(str(tp))
        r = np.asanyarray(rn.dataobj) > 0
        t = np.asanyarray(tn.dataobj) > 0

        # raw array-order overlap
        if r.shape == t.shape:
            arr_ov.append(frac_inside(r, t))

        # canonical (RAS) overlap
        rc = np.asanyarray(nib.as_closest_canonical(rn).dataobj) > 0
        tc = np.asanyarray(nib.as_closest_canonical(tn).dataobj) > 0
        if rc.shape == tc.shape:
            can_ov.append(frac_inside(rc, tc))

        orients[nib.aff2axcodes(rn.affine)] += 1
        orients[nib.aff2axcodes(tn.affine)] += 1

        # image vs root affine (matters for CBCT-only training)
        if ip.exists():
            inn = nib.load(str(ip))
            same = (np.asanyarray(inn.dataobj).shape == r.shape and
                    np.allclose(inn.affine, rn.affine, atol=1e-2))
            if same:
                img_root_affine_ok += 1
            else:
                img_root_affine_bad.append(pid)

    def m(x):
        x = [v for v in x if v == v]  # drop nan
        return statistics.mean(x) if x else float("nan")

    print(f"cases checked : {checked}")
    print(f"root-inside-teeth  RAW ARRAY order : {m(arr_ov):.1%}")
    print(f"root-inside-teeth  CANONICAL (RAS) : {m(can_ov):.1%}")
    print(f"orientations seen among teeth/root : "
          f"{dict((''.join(k), v) for k, v in orients.items())}")
    print(f"image==root affine : {img_root_affine_ok} ok, "
          f"{len(img_root_affine_bad)} mismatched")
    if img_root_affine_bad:
        print("  image/root affine mismatch: " + ", ".join(img_root_affine_bad[:10]) +
              (" ..." if len(img_root_affine_bad) > 10 else ""))

    print("\nDiagnosis:")
    print("  * RAW low but CANONICAL high  -> just an ORIENTATION/FLIP difference; "
          "the data corresponds. Reorient masks (or rebuild with reorientation).")
    print("  * RAW low AND CANONICAL low   -> real CONTENT mismatch (wrong/other-arch "
          "teeth). Use CBCT-only, provided image==root affine is OK.")
    print("  * image==root all ok          -> CBCT-only training is safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
