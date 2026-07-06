#!/usr/bin/env python3
"""Sanity-check that each root label sits INSIDE its tooth mask (design A premise).

For every <id>_travecular file it finds the matching <id>_teeth file and reports
what fraction of root voxels fall inside the tooth mask. Should be ~100%. Low or
variable values mean the tooth and root files are misaligned (different arches,
different grids, or mixed pipeline versions) and the 3-channel model would be
learning a broken task.

Usage (from the training folder):   python check_alignment.py --dir data
      (from the project root):      python training\\check_alignment.py --dir training\\data
"""

from __future__ import annotations

import argparse
import re
import statistics
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--root-suffix", default="_travecular")
    ap.add_argument("--tooth-suffix", default="_teeth")
    a = ap.parse_args()
    if not a.dir.is_dir():
        raise SystemExit(f"folder not found: {a.dir}")

    rows, mism = [], []
    for rp in sorted(a.dir.glob(f"*{a.root_suffix}.nii.gz")):
        pid = pid_of(rp.name)
        tp = a.dir / f"{pid}{a.tooth_suffix}.nii.gz"
        if not tp.exists():
            continue
        r = np.asanyarray(nib.load(str(rp)).dataobj) > 0
        t = np.asanyarray(nib.load(str(tp)).dataobj) > 0
        if r.shape != t.shape:
            mism.append(pid)
            continue
        rv = int(r.sum())
        if rv == 0:
            continue
        rows.append((pid, int((r & t).sum()) / rv))

    if not rows and not mism:
        raise SystemExit("No <id>_travecular + <id>_teeth pairs found. Check --dir "
                         "and the suffixes.")

    vals = [v for _, v in rows]
    print(f"cases checked : {len(vals)}")
    if vals:
        print(f"mean root-inside-teeth : {statistics.mean(vals):.1%}")
        print(f"min  root-inside-teeth : {min(vals):.1%}")
        worst = sorted(rows, key=lambda x: x[1])[:10]
        print("worst 10 cases:")
        for pid, v in worst:
            print(f"  {pid}: {v:.1%}")
    if mism:
        print(f"\nSHAPE MISMATCH (tooth grid != root grid) for {len(mism)} case(s):")
        print("  " + ", ".join(mism))

    print("\nInterpretation: ~100% = aligned, good. Low/variable = tooth and root "
          "don't correspond; fix the pairs or train CBCT-only (--channels image).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
