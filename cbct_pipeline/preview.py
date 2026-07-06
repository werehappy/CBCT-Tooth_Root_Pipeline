"""Slice-overlay QC previews: render a 2D slice of the CBCT with the bone, teeth,
and root masks colour-coded on top. Used by the app and available standalone."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt        # noqa: E402
import nibabel as nib                  # noqa: E402
import numpy as np                     # noqa: E402

log = logging.getLogger(__name__)

# Plane -> axis index in RAS-canonical space (x=R/L, y=A/P, z=S/I)
_PLANE_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}

# Overlay colours (RGB 0-1)
COLORS = {
    "bone":  (0.30, 0.55, 1.00),   # blue
    "teeth": (1.00, 0.85, 0.20),   # yellow
    "root":  (1.00, 0.25, 0.25),   # red
}


def _canon(path) -> np.ndarray:
    return np.asanyarray(nib.as_closest_canonical(nib.load(str(path))).dataobj)


def _to_display(arr2d: np.ndarray, plane: str) -> np.ndarray:
    """Orient a 2D slice so superior is up (for sagittal/coronal)."""
    if plane == "axial":
        return np.flipud(arr2d.T)
    return np.flipud(arr2d.T)  # rows become the S-I axis -> superior on top


def make_overlay(
    image_path,
    mask_paths: Dict[str, Path],
    out_png: Path,
    plane: str = "sagittal",
    slice_index: Optional[int] = None,
    alpha: float = 0.45,
    title: Optional[str] = None,
) -> Path:
    """Render image + colour-coded masks to a PNG. mask_paths keys among
    {'bone','teeth','root'}. Slice auto-picked to show the most root (else teeth)."""
    vol = _canon(image_path).astype(float)
    masks = {k: _canon(p).astype(bool) for k, p in mask_paths.items()
             if Path(p).exists()}
    axis = _PLANE_AXIS[plane]

    # Pick the slice that best shows the cut: prefer root, then teeth, then any.
    pick = next((k for k in ("root", "teeth", "bone") if k in masks), None)
    if slice_index is None and pick is not None:
        other = tuple(a for a in range(3) if a != axis)
        counts = masks[pick].sum(axis=other)
        slice_index = int(np.argmax(counts)) if counts.any() else vol.shape[axis] // 2
    if slice_index is None:
        slice_index = vol.shape[axis] // 2

    def take(a):
        sl = [slice(None)] * 3
        sl[axis] = slice_index
        return a[tuple(sl)]

    base = _to_display(take(vol), plane)
    # Window the grayscale for contrast (robust percentiles)
    lo, hi = np.percentile(base, [2, 98])
    base_n = np.clip((base - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.stack([base_n] * 3, axis=-1)

    for key in ("bone", "teeth", "root"):
        if key not in masks:
            continue
        m = _to_display(take(masks[key]), plane)
        color = np.array(COLORS[key])
        rgb[m] = (1 - alpha) * rgb[m] + alpha * color

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(title or f"{plane} slice {slice_index}", fontsize=11)
    handles = [mpatches.Patch(color=COLORS[k], label=lbl)
               for k, lbl in (("bone", "bone"), ("teeth", "teeth"),
                              ("root", "root (travecular)")) if k in masks]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.7)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    log.info("Wrote preview %s (%s slice %d)", out_png.name, plane, slice_index)
    return out_png
