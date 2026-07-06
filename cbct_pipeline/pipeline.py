"""End-to-end orchestration for a single CBCT volume.

Flow:
  1. Segment (DentalSegmentator/nnU-Net) or load a provided labelmap.
  2. Build the upper and lower arch groups (bone + teeth) from config.
  3. Pick the dominant arch by voxel count; discard the other entirely.
  4. Extract tooth roots: cut teeth at the alveolar bone boundary.
  5. Write two masks for the winning arch: bone, and tooth roots.

Outputs (lower arch dominant, DentalSegmentator defaults):
    <patient>_mandible_bone.nii.gz
    <patient>_mandible_teeth_root.nii.gz
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional

from . import labels as L
from .roots import extract_root, smooth_root_top_to_bone
from .io_utils import load_volume, save_mask
from .segmentation import segment

log = logging.getLogger(__name__)


def derive_patient_id(input_path: Path, override: Optional[str]) -> str:
    if override:
        return override
    stem = input_path.name
    for suf in (".nii.gz", ".nii"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    m = re.match(r"([A-Za-z0-9]+)", stem)
    return m.group(1) if m else stem


def _score(bone_n: int, teeth_n: int, metric: str) -> int:
    if metric == "bone":
        return bone_n
    if metric == "teeth":
        return teeth_n
    return bone_n + teeth_n


def run_pipeline(input_path: str | Path, output_dir: str | Path, cfg: dict,
                 *, patient_id: Optional[str] = None,
                 labelmap_path: Optional[str | Path] = None) -> Dict[str, Path]:
    """Run segmentation, pick the dominant arch, extract roots, write 2 masks."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pid = derive_patient_id(input_path, patient_id)
    log.info("=== Patient %s ===", pid)

    # root_only: skip DentalSegmentator/teeth/bone entirely and output just the
    # trained model's root (travecular). No DentalSegmentator needed.
    if cfg.get("root_only"):
        rcfg = cfg.get("root", {})
        if str(rcfg.get("method", "")).lower() != "model" or not rcfg.get("model_dir"):
            raise ValueError("root_only requires root.method: model and root.model_dir")
        image = load_volume(input_path, as_int=False)
        with tempfile.TemporaryDirectory() as rtmp:
            root_ncfg = {
                "backend": "python",
                "model_dir": rcfg["model_dir"],
                "folds": str(rcfg.get("folds", "0")),
                "checkpoint": rcfg.get("checkpoint", "checkpoint_final.pth"),
                "device": rcfg.get("device", cfg.get("nnunet", {}).get("device", "cuda")),
                "disable_tta": rcfg.get("disable_tta", False),
            }
            log.info("root_only: running trained root model %s", rcfg["model_dir"])
            rlm = segment(input_path, Path(rtmp), root_ncfg)
            root = load_volume(rlm, as_int=True).data == int(rcfg.get("root_label", 1))
        root_name = cfg.get("output", {}).get("root_name", "travecular")
        p = save_mask(root, image, output_dir / f"{pid}_{root_name}.nii.gz")
        log.info("Done (root_only): wrote %s", p.name)
        return {"root": p}

    arches = cfg.get("arches")
    if not arches:
        raise ValueError("Config must define `arches` (upper/lower) -- see "
                         "config.dentalsegmentator.yaml")

    # 1. Image (geometry + affine for orientation-correct root cut)
    image = load_volume(input_path, as_int=False)

    # 2. Segmentation
    with tempfile.TemporaryDirectory() as tmp:
        if labelmap_path is not None:
            log.info("Using provided labelmap: %s", labelmap_path)
            lm_path = Path(labelmap_path)
        else:
            lm_path = segment(input_path, Path(tmp), cfg["nnunet"])
        labelmap = load_volume(lm_path, as_int=True).data

    # 3. Build arch groups and pick the dominant one
    metric = cfg.get("dominance", {}).get("metric", "bone")
    groups = {}
    scores = {}
    for key, spec in arches.items():
        bone = L.mask_from_labels(labelmap, spec["bone_labels"])
        teeth = L.mask_from_labels(labelmap, spec["teeth_labels"])
        groups[key] = (bone, teeth, spec)
        scores[key] = _score(int(bone.sum()), int(teeth.sum()), metric)
        log.info("Arch '%s': bone=%d teeth=%d score(%s)=%d",
                 key, int(bone.sum()), int(teeth.sum()), metric, scores[key])

    winner = max(scores, key=scores.get)
    bone, teeth, spec = groups[winner]
    archname = spec.get("name", winner)
    log.info("Dominant arch: '%s' (%s) -- discarding the other arch", winner, archname)

    if scores[winner] == 0:
        log.warning("Dominant arch is empty -- check labels / segmentation output")

    # 4. Root: either the geometric heuristic, or a trained nnU-Net root model.
    rcfg = cfg.get("root", {})
    method = str(rcfg.get("method", "heuristic")).lower()
    if method == "model":
        if not rcfg.get("model_dir"):
            raise ValueError("root.method is 'model' but root.model_dir is not set")
        with tempfile.TemporaryDirectory() as rtmp:
            root_ncfg = {
                "backend": "python",
                "model_dir": rcfg["model_dir"],
                "folds": str(rcfg.get("folds", "0")),
                "checkpoint": rcfg.get("checkpoint", "checkpoint_final.pth"),
                "device": rcfg.get("device", cfg.get("nnunet", {}).get("device", "cuda")),
                "disable_tta": rcfg.get("disable_tta", False),
            }
            log.info("Running trained root model: %s", rcfg["model_dir"])
            rlm = segment(input_path, Path(rtmp), root_ncfg)
            root_lm = load_volume(rlm, as_int=True).data
        root = root_lm == int(rcfg.get("root_label", 1))
    else:
        root = extract_root(
            teeth, bone, image.affine,
            crown_direction=spec.get("crown_direction", "superior"),
            spacing=image.spacing,
            bone_close_mm=rcfg.get("bone_close_mm", 1.0),
            smooth_iter=rcfg.get("smooth_iter", 1),
        )
        # Smooth ONLY the root's top (cut) surface and blend into the bone crest.
        root = smooth_root_top_to_bone(
            root, bone, image.affine,
            spec.get("crown_direction", "superior"), image.spacing,
            rcfg.get("top_smooth_mm", rcfg.get("surface_smooth_mm", 1.5)))

    # 5. Write the outputs for the winning arch
    #    bone  -> <patient>_<arch name>.nii.gz   (e.g. _Mandible)
    #    teeth -> <patient>_<teeth_name>.nii.gz  (full teeth, crown + root)
    #    root  -> <patient>_<root_name>.nii.gz   (roots only, crowns cut off)
    ocfg = cfg.get("output", {})
    teeth_name = ocfg.get("teeth_name", "teeth")
    root_name = ocfg.get("root_name", "travecular")
    written: Dict[str, Path] = {
        "bone": save_mask(bone, image, output_dir / f"{pid}_{archname}.nii.gz"),
        "teeth": save_mask(teeth, image, output_dir / f"{pid}_{teeth_name}.nii.gz"),
        "root": save_mask(root, image, output_dir / f"{pid}_{root_name}.nii.gz"),
    }
    log.info("Done: wrote %d files for patient %s", len(written), pid)
    return written
