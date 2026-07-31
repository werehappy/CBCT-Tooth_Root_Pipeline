#!/usr/bin/env python3
"""Evaluate a CBCT tooth-root segmentation against ground truth.

This goes beyond nnU-Net's own summary.json (which reports only Dice/IoU/FP/FN).
For each case it computes, in three groups:

  Overlap      Dice, IoU, precision, recall (sensitivity)
  Volume       predicted/reference voxel ratio, FP:FN ratio  (over/under-seg)
  Geometry     HD95 and ASSD in millimetres  (boundary agreement)
  Cut          the crown/root CUT-HEIGHT error in millimetres  <-- the key one

Why the cut metric matters: a root is mostly a compact blob, so a high Dice can
coexist with a crown/root cut that sits 1-2 mm too high or too low. The model's
whole job is to put that cut in the right place, and Dice is nearly blind to it.
This script measures it directly: per (x,y) column it finds the crown-side edge
of the root in the prediction and in the reference, and reports the signed
difference in mm (positive = prediction extends further toward the crown, i.e.
over-segments upward -- the failure mode seen on the over-filling models).

Inputs are two folders of masks (prediction vs ground truth), matched by file
name. Everything is reoriented to canonical (RAS+) first so the superior-inferior
axis is well defined, and spacing is read from each file's affine so distances
are true millimetres.

Usage
-----
  python evaluate_root.py --pred PRED_DIR --gt GT_DIR --out eval_out

  # crown points inferior (upper arch)         -> which edge is "the cut"
  python evaluate_root.py --pred P --gt G --crown inferior

  # files differ by a suffix (e.g. pred "<id>.nii.gz", gt "<id>_root.nii.gz")
  python evaluate_root.py --pred P --gt G --gt-suffix _root

Outputs (in --out): per_case.csv, summary.json, evaluation_report.png
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
from scipy import ndimage

log = logging.getLogger("evaluate_root")
SUFFIXES = (".nii.gz", ".nii")


# --------------------------------------------------------------------------- #
# file matching
# --------------------------------------------------------------------------- #
def strip_ext(name: str) -> str:
    for s in SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return name


def key_of(path: Path, suffix: str) -> str:
    """Matching key = filename stem with an optional trailing suffix removed."""
    stem = strip_ext(path.name)
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def index_folder(folder: Path, suffix: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sorted(Path(folder).iterdir()):
        if p.name.endswith(SUFFIXES):
            out.setdefault(key_of(p, suffix), p)
    return out


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_canonical(path: Path, label: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bool mask in canonical RAS+, per-axis spacing in mm)."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    arr = np.asanyarray(img.dataobj)
    mask = (arr == label) if label is not None else (arr > 0)
    if not mask.any():                       # fall back to any-foreground
        mask = arr > 0
    spacing = nib.affines.voxel_sizes(img.affine)
    return mask.astype(bool), np.asarray(spacing, dtype=float)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    n_pred, n_ref = tp + fp, tp + fn
    dice = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else math.nan
    iou = (tp / (tp + fp + fn)) if (tp + fp + fn) else math.nan
    prec = (tp / n_pred) if n_pred else math.nan
    rec = (tp / n_ref) if n_ref else math.nan
    return {
        "dice": dice, "iou": iou, "precision": prec, "recall": rec,
        "tp": tp, "fp": fp, "fn": fn, "n_pred": n_pred, "n_ref": n_ref,
        "vol_ratio_pred_over_ref": (n_pred / n_ref) if n_ref else math.nan,
        "fp_over_fn": (fp / fn) if fn else math.nan,
    }


def _surface(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels of a binary mask."""
    if not mask.any():
        return np.zeros_like(mask)
    return mask ^ ndimage.binary_erosion(mask, iterations=1)


def surface_metrics(pred: np.ndarray, gt: np.ndarray,
                    spacing: np.ndarray) -> Dict[str, float]:
    """Symmetric surface distances (mm): HD95 and ASSD."""
    ps, gs = _surface(pred), _surface(gt)
    if not ps.any() or not gs.any():
        return {"hd95_mm": math.nan, "assd_mm": math.nan}
    dt_to_gt = ndimage.distance_transform_edt(~gs, sampling=spacing)
    dt_to_pred = ndimage.distance_transform_edt(~ps, sampling=spacing)
    d_pg = dt_to_gt[ps]      # pred surface -> gt surface
    d_gp = dt_to_pred[gs]    # gt surface -> pred surface
    both = np.concatenate([d_pg, d_gp])
    return {"hd95_mm": float(np.percentile(both, 95)),
            "assd_mm": float(both.mean())}


def _edge_along_z(mask: np.ndarray, side: str) -> np.ndarray:
    """Per-column z index of the crown-side edge (nan where the column is empty).

    Canonical RAS+: axis 2 is S/I with increasing index = superior.
    side='superior' -> highest z (max);  side='inferior' -> lowest z (min)."""
    z = np.arange(mask.shape[2])
    if side == "superior":
        e = np.where(mask, z[None, None, :], -1).max(axis=2).astype(float)
        e[e < 0] = np.nan
    else:
        big = mask.shape[2]
        e = np.where(mask, z[None, None, :], big).min(axis=2).astype(float)
        e[e >= big] = np.nan
    return e


def cut_metrics(pred: np.ndarray, gt: np.ndarray, spacing: np.ndarray,
                crown: str) -> Dict[str, float]:
    """Crown/root cut-height error in mm, measured on the crown-side edge.

    Positive signed error = prediction's cut is further toward the crown than the
    reference's (root extends too far up = over-segmentation toward the crown)."""
    sz = float(spacing[2])
    ep = _edge_along_z(pred, crown)
    eg = _edge_along_z(gt, crown)
    both = ~np.isnan(ep) & ~np.isnan(eg)          # columns present in both
    if not both.any():
        return {"cut_signed_mm": math.nan, "cut_abs_mm": math.nan,
                "cut_p95_abs_mm": math.nan, "col_coverage": 0.0}
    sign = 1.0 if crown == "superior" else -1.0    # so + always means "toward crown"
    err = sign * (ep[both] - eg[both]) * sz
    gt_cols = int(np.count_nonzero(~np.isnan(eg)))
    covered = int(np.count_nonzero(both))
    return {
        "cut_signed_mm": float(np.mean(err)),
        "cut_abs_mm": float(np.mean(np.abs(err))),
        "cut_p95_abs_mm": float(np.percentile(np.abs(err), 95)),
        "col_coverage": (covered / gt_cols) if gt_cols else 0.0,
    }


# --------------------------------------------------------------------------- #
# aggregation + reporting
# --------------------------------------------------------------------------- #
AGG_KEYS = ["dice", "iou", "precision", "recall", "vol_ratio_pred_over_ref",
            "fp_over_fn", "hd95_mm", "assd_mm", "cut_signed_mm", "cut_abs_mm",
            "cut_p95_abs_mm", "col_coverage"]


def summarize(rows: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for k in AGG_KEYS:
        vals = np.array([r[k] for r in rows if r.get(k) == r.get(k)], dtype=float)
        if vals.size == 0:
            out[k] = {"n": 0}
            continue
        out[k] = {
            "n": int(vals.size),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "iqr": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))],
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out


def make_plot(rows: List[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dice = [r["dice"] for r in rows if r["dice"] == r["dice"]]
    hd95 = [r["hd95_mm"] for r in rows if r["hd95_mm"] == r["hd95_mm"]]
    cut = [r["cut_signed_mm"] for r in rows if r["cut_signed_mm"] == r["cut_signed_mm"]]
    vr = [r["vol_ratio_pred_over_ref"] for r in rows]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist(dice, bins=15, color="#2C6E8F", edgecolor="white")
    ax[0, 0].axvline(np.median(dice), color="crimson", ls="--",
                     label=f"median {np.median(dice):.3f}")
    ax[0, 0].set_title("Dice per case"); ax[0, 0].set_xlabel("Dice"); ax[0, 0].legend()

    ax[0, 1].hist(hd95, bins=15, color="#8A5A12", edgecolor="white")
    ax[0, 1].set_title("HD95 (surface, mm)"); ax[0, 1].set_xlabel("mm")

    ax[1, 0].hist(cut, bins=15, color="#B9770E", edgecolor="white")
    ax[1, 0].axvline(0, color="black", lw=1)
    ax[1, 0].set_title("Crown-cut error (signed, mm)\n+ = over toward crown")
    ax[1, 0].set_xlabel("mm")

    ax[1, 1].scatter(vr, dice, s=18, color="#23495C", alpha=0.8)
    ax[1, 1].axvline(1.0, color="black", lw=1, ls=":")
    ax[1, 1].set_title("Volume ratio vs Dice")
    ax[1, 1].set_xlabel("pred / ref volume"); ax[1, 1].set_ylabel("Dice")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def fmt(s: dict, key: str, unit: str = "") -> str:
    d = s.get(key, {})
    if not d.get("n"):
        return f"{key}: n/a"
    return (f"{key:>24}: median {d['median']:.3f}{unit}  "
            f"mean {d['mean']:.3f}  IQR [{d['iqr'][0]:.3f}, {d['iqr'][1]:.3f}]  "
            f"min {d['min']:.3f}  max {d['max']:.3f}  (n={d['n']})")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, type=Path, help="folder of prediction masks")
    ap.add_argument("--gt", required=True, type=Path, help="folder of ground-truth masks")
    ap.add_argument("--out", type=Path, default=Path("eval_out"), help="output folder")
    ap.add_argument("--label", type=int, default=1, help="foreground label id (default 1)")
    ap.add_argument("--crown", choices=["superior", "inferior"], default="superior",
                    help="crown direction: superior for lower arch, inferior for upper")
    ap.add_argument("--pred-suffix", default="", help="suffix to strip from pred names")
    ap.add_argument("--gt-suffix", default="", help="suffix to strip from gt names")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for name, d in (("--pred", a.pred), ("--gt", a.gt)):
        if not Path(d).is_dir():
            raise SystemExit(f"{name} folder not found: {d}")

    preds = index_folder(a.pred, a.pred_suffix)
    gts = index_folder(a.gt, a.gt_suffix)
    keys = sorted(set(preds) & set(gts))
    if not keys:
        raise SystemExit(
            "No matching cases. Prediction keys and GT keys don't intersect.\n"
            f"  example pred keys: {list(preds)[:3]}\n"
            f"  example gt   keys: {list(gts)[:3]}\n"
            "Use --pred-suffix / --gt-suffix if the names differ by a suffix.")
    log.info("Matched %d case(s); crown side = %s", len(keys), a.crown)

    rows: List[dict] = []
    skipped: List[Tuple[str, str]] = []
    for k in keys:
        try:
            pred, sp_p = load_canonical(preds[k], a.label)
            gt, sp_g = load_canonical(gts[k], a.label)
            if pred.shape != gt.shape:
                skipped.append((k, f"shape {pred.shape} != {gt.shape}"))
                continue
            if not np.allclose(sp_p, sp_g, atol=1e-3):
                log.warning("%s: pred/gt spacing differ (%s vs %s); using GT spacing",
                            k, np.round(sp_p, 3), np.round(sp_g, 3))
            spacing = sp_g
            row = {"case": k}
            row.update(overlap_metrics(pred, gt))
            row.update(surface_metrics(pred, gt, spacing))
            row.update(cut_metrics(pred, gt, spacing, a.crown))
            rows.append(row)
            log.info("[%d/%d] %s  Dice=%.3f  HD95=%.2fmm  cut=%.2fmm",
                     len(rows), len(keys), k, row["dice"], row["hd95_mm"],
                     row["cut_signed_mm"])
        except Exception as exc:                       # keep going on per-case failure
            skipped.append((k, str(exc)))
            log.error("%s: failed (%s)", k, exc)

    if not rows:
        raise SystemExit("No cases evaluated successfully.")

    a.out.mkdir(parents=True, exist_ok=True)

    # per-case CSV
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = a.out / "per_case.csv"
    df.to_csv(csv_path, index=False)

    # summary JSON
    summary = summarize(rows)
    summary["_meta"] = {"n_cases": len(rows), "crown": a.crown,
                        "skipped": [{"case": c, "why": w} for c, w in skipped]}
    with open(a.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if not a.no_plot:
        try:
            make_plot(rows, a.out / "evaluation_report.png")
        except Exception as exc:
            log.warning("plot skipped: %s", exc)

    # console summary
    print("\n" + "=" * 78)
    print(f"ROOT EVALUATION  —  {len(rows)} case(s), crown side = {a.crown}")
    print("=" * 78)
    print("Overlap")
    print("  " + fmt(summary, "dice"))
    print("  " + fmt(summary, "iou"))
    print("  " + fmt(summary, "precision"))
    print("  " + fmt(summary, "recall"))
    print("Volume / over-segmentation")
    print("  " + fmt(summary, "vol_ratio_pred_over_ref"))
    print("  " + fmt(summary, "fp_over_fn"))
    print("Boundary (mm)")
    print("  " + fmt(summary, "hd95_mm", " mm"))
    print("  " + fmt(summary, "assd_mm", " mm"))
    print("Crown/root cut (mm)  [the decision-relevant metric]")
    print("  " + fmt(summary, "cut_signed_mm", " mm"))
    print("  " + fmt(summary, "cut_abs_mm", " mm"))
    print("  " + fmt(summary, "cut_p95_abs_mm", " mm"))
    print("  " + fmt(summary, "col_coverage"))
    if skipped:
        print(f"\nSkipped {len(skipped)} case(s): "
              + ", ".join(f"{c} ({w})" for c, w in skipped[:6])
              + (" ..." if len(skipped) > 6 else ""))
    print(f"\nWrote: {csv_path}")
    print(f"       {a.out / 'summary.json'}")
    if not a.no_plot:
        print(f"       {a.out / 'evaluation_report.png'}")
    print("\nReading the cut metric: cut_signed_mm > 0 means the predicted root")
    print("reaches further toward the crown than ground truth (over-segmentation);")
    print("< 0 means it stops short. cut_abs_mm is the typical placement error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
