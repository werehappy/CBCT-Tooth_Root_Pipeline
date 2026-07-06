"""Cortical vs trabecular bone separation.

Two strategies, both operating strictly inside the segmented bone mask:

  morphological : erode the dense cortical rind by a physical thickness (mm).
                  The interior is trabecular. Essentially intensity-independent,
                  so it degrades gracefully across mixed scanners -- this is the
                  default and the recommended one for varied CBCT input.

  intensity     : per-scan Otsu threshold computed ONLY on voxels inside the
                  bone mask. Relative, so it tolerates the gray-value drift that
                  makes fixed HU-style thresholds unreliable on CBCT -- but still
                  more scanner-sensitive than the morphological method.

Both return (cortical_mask, trabecular_mask).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)


def _mm_to_voxels(thickness_mm: float, spacing: Tuple[float, float, float]) -> int:
    """Number of erosion iterations approximating a physical thickness."""
    return max(1, int(round(thickness_mm / min(spacing))))


def trabecular_morphological(
    bone_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    cortical_thickness_mm: float = 1.5,
    fill_holes: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Geometry-based split. Scanner-independent."""
    mask = bone_mask.astype(bool)
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    iters = _mm_to_voxels(cortical_thickness_mm, spacing)
    interior = ndimage.binary_erosion(mask, iterations=iters)
    cortical = mask & ~interior
    trabecular = interior & bone_mask.astype(bool)  # keep within original bone
    log.info("Morphological split: cortical=%d trabecular=%d (erosion iters=%d, %.2fmm)",
             int(cortical.sum()), int(trabecular.sum()), iters, cortical_thickness_mm)
    return cortical, trabecular


def trabecular_intensity(
    volume: np.ndarray,
    bone_mask: np.ndarray,
    percentile: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-scan adaptive threshold computed inside the bone mask only.

    If `percentile` is given (0-100), use that percentile of in-bone intensities
    as the cortical/trabecular cut instead of Otsu. Useful when Otsu is unstable
    on a particular scan.
    """
    bone = bone_mask.astype(bool)
    vals = volume[bone]
    if vals.size == 0:
        log.warning("Empty bone mask -- returning empty cortical/trabecular masks")
        empty = np.zeros_like(bone)
        return empty, empty

    if percentile is not None:
        thr = float(np.percentile(vals, percentile))
        method = f"p{percentile:g}"
    else:
        # Lazy import so scikit-image is only required when this path is used.
        from skimage.filters import threshold_otsu
        thr = float(threshold_otsu(vals))
        method = "otsu"

    cortical = bone & (volume >= thr)    # dense -> cortical
    trabecular = bone & (volume < thr)   # porous/darker -> trabecular
    log.info("Intensity split (%s, thr=%.2f): cortical=%d trabecular=%d",
             method, thr, int(cortical.sum()), int(trabecular.sum()))
    return cortical, trabecular


def separate(method: str, *, volume, bone_mask, spacing, params: dict
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch on method name from config."""
    method = method.lower()
    if method == "morphological":
        return trabecular_morphological(
            bone_mask, spacing,
            cortical_thickness_mm=params.get("cortical_thickness_mm", 1.5),
            fill_holes=params.get("fill_holes", True),
        )
    if method == "intensity":
        return trabecular_intensity(
            volume, bone_mask,
            percentile=params.get("percentile"),
        )
    raise ValueError(f"Unknown trabecular method: {method!r} "
                     "(expected 'morphological' or 'intensity')")
