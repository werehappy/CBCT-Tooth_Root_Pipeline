"""Tooth-root extraction by bone enclosure.

A tooth voxel is part of the root if, in its axial (occlusal) slice, the alveolar
bone wraps around it (found by per-slice hole-filling, with a small closing to
bridge the thin periodontal gap). The crown and cervical neck float above the
bone with no bone wrapping them, so they are excluded, and the top of the root
lands where the bone surface is -- a seamless connection, with no root floating
above the bone.

Geometric heuristic, not a learned model. It tracks the *bone* level (cuts at the
alveolar bone, not the anatomical CEJ). Assumes teeth run roughly along the S-I
axis (used only to pick the slice plane).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)


def _si_axis(affine: np.ndarray) -> int:
    """Index of the superior-inferior voxel axis (the tooth long axis)."""
    for ax, c in enumerate(nib.aff2axcodes(affine)):
        if c in ("S", "I"):
            return ax
    raise ValueError(f"No S/I axis in affine (codes={nib.aff2axcodes(affine)})")


def _signed_si_axis(affine: np.ndarray, crown_direction: str) -> Tuple[int, int]:
    """(axis, sign) so that sign*index increases toward crown_direction."""
    for ax, c in enumerate(nib.aff2axcodes(affine)):
        if c in ("S", "I"):
            plus_is = "superior" if c == "S" else "inferior"
            return ax, (1 if plus_is == crown_direction else -1)
    raise ValueError("No S/I axis in affine")


def _fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    nan = np.isnan(a)
    if nan.all():
        return np.zeros_like(a)
    if nan.any():
        _, idx = ndimage.distance_transform_edt(
            nan, return_distances=True, return_indices=True)
        a = a[idx[0], idx[1]]
    return a


def smooth_root_top_to_bone(root_mask: np.ndarray, bone_mask: np.ndarray,
                            affine: np.ndarray, crown_direction: str,
                            spacing: Tuple[float, float, float],
                            sigma_mm: float) -> np.ndarray:
    """Smooth ONLY the root's top (cut) surface and blend it into the bone crest.

    The top of the root is a height field h(x,y) along the tooth long axis (the
    'y'/S-I value of the highest root voxel per column). We build a combined
    surface -- root height over the tooth, bone-crest height everywhere else --
    Gaussian-smooth that height field, and trim the root down to it. Near the rim
    the smoothed height ramps from the root top to the surrounding bone level, so
    the two surfaces meet continuously. The deep root (apex) is untouched.
    """
    root = root_mask.astype(bool)
    bone = bone_mask.astype(bool)
    if sigma_mm <= 0 or not root.any():
        return root

    ax, sign = _signed_si_axis(affine, crown_direction)
    rm = np.moveaxis(root, ax, -1)
    bm = np.moveaxis(bone, ax, -1)
    if sign < 0:
        rm = rm[..., ::-1].copy()
        bm = bm[..., ::-1].copy()
    Z = rm.shape[-1]
    zidx = np.arange(Z)

    rtop = np.where(rm, zidx[None, None, :], -1).max(axis=-1).astype(float)
    rtop[rtop < 0] = np.nan
    btop = np.where(bm, zidx[None, None, :], -1).max(axis=-1).astype(float)
    btop[btop < 0] = np.nan

    have_root = ~np.isnan(rtop)
    surface = np.where(have_root, rtop, btop)   # root height on teeth, bone else
    surface = _fill_nan_nearest(surface)

    plane_sp = min(spacing[i] for i in range(3) if i != ax)
    surface_s = ndimage.gaussian_filter(surface, sigma=sigma_mm / plane_sp)

    keep = zidx[None, None, :] <= surface_s[..., None]   # trim top to smoothed surface
    out = rm & keep
    if sign < 0:
        out = out[..., ::-1]
    return np.moveaxis(out, -1, ax)


def extract_root(
    tooth_mask: np.ndarray,
    bone_mask: np.ndarray,
    affine: np.ndarray,
    crown_direction: str = "superior",   # accepted for API compatibility; unused
    *,
    spacing: Optional[Tuple[float, float, float]] = None,
    bone_close_mm: float = 1.0,
    smooth_iter: int = 1,
    **legacy,                            # tolerate old config keys
) -> np.ndarray:
    """Return the tooth-root mask: the part of the tooth enclosed by the alveolar
    bone (bone wrapping around it in its axial slice), decided slice by slice."""
    tooth = tooth_mask.astype(bool)
    bone = bone_mask.astype(bool)
    if not tooth.any():
        return np.zeros_like(tooth)
    if not bone.any():
        log.warning("No bone for this arch -- keeping nothing")
        return np.zeros_like(tooth)

    ax = _si_axis(affine)
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
    plane_sp = min(s for i, s in enumerate(spacing) if i != ax)
    close_it = max(0, int(round(bone_close_mm / plane_sp)))

    tm = np.moveaxis(tooth, ax, -1)
    bm = np.moveaxis(bone, ax, -1)
    root = np.zeros_like(tm)

    for k in range(tm.shape[-1]):
        T = tm[..., k]
        if not T.any():
            continue
        B = bm[..., k]
        if not B.any():
            continue  # no bone in this slice -> crown / overshoot, exclude
        B2 = ndimage.binary_closing(B, iterations=close_it) if close_it else B
        root[..., k] = T & ndimage.binary_fill_holes(B2)

    root = np.moveaxis(root, -1, ax)

    if smooth_iter and smooth_iter > 0:
        root = ndimage.binary_closing(root, iterations=int(smooth_iter)) & tooth

    log.info("Root extraction (bone-enclosure, close=%d vox, smooth=%d): "
             "tooth=%d -> root=%d (removed=%d)",
             close_it, smooth_iter, int(tooth.sum()), int(root.sum()),
             int(tooth.sum()) - int(root.sum()))
    return root
