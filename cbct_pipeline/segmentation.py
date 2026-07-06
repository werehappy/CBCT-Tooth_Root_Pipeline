"""nnU-Net v2 inference for DentalSegmentator (and any nnU-Net model).

Two backends, selected by `nnunet.backend` in config:

  "python" (default, recommended for running in-project)
      Calls nnU-Net's Python predictor API directly. No reliance on the
      nnUNetv2_predict console script being on PATH. Loads the model once.
      You point `nnunet.model_dir` straight at the trained-model folder, e.g.
        .../nnUNet_results/Dataset111_453CT/nnUNetTrainer__nnUNetPlans__3d_fullres

  "cli"
      Shells out to `nnUNetv2_predict`. Uses nnUNet_results + dataset_id like
      the standard command line.

Both copy the input to nnU-Net's expected `CASE_0000.nii.gz` naming, run
inference, and return the path to the produced labelmap.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _parse_folds(folds) -> tuple:
    """'all' -> ('all',);  '0 1 2 3 4' -> (0,1,2,3,4);  0 -> (0,)."""
    toks = str(folds).split()
    return tuple("all" if t == "all" else int(t) for t in toks)


def _stage_input(input_volume: Path, work_dir: Path) -> tuple[Path, Path]:
    in_dir = work_dir / "nnunet_in"
    out_dir = work_dir / "nnunet_out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    # nnU-Net single-channel naming: <case>_0000.nii.gz
    shutil.copy(input_volume, in_dir / "CASE_0000.nii.gz")
    return in_dir, out_dir


def _find_output(out_dir: Path) -> Path:
    out = out_dir / "CASE.nii.gz"
    if out.exists():
        return out
    produced = list(out_dir.glob("*.nii.gz"))
    if not produced:
        raise RuntimeError(f"nnU-Net produced no output in {out_dir}")
    return produced[0]


# --------------------------------------------------------------------------- #
# Python predictor backend
# --------------------------------------------------------------------------- #
def run_nnunet_python(
    input_volume: Path,
    work_dir: Path,
    *,
    model_dir: str,
    folds="all",
    checkpoint: str = "checkpoint_final.pth",
    device: str = "cuda",
    disable_tta: bool = False,
) -> Path:
    """Run inference via the nnUNetv2 Python API. `model_dir` is the folder that
    directly contains plans.json, dataset.json and the fold_* subdirs."""
    # Lazy imports so the rest of the pipeline works without torch/nnunetv2.
    try:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as e:
        raise RuntimeError(
            "nnunetv2 / torch not importable. Install with `pip install nnunetv2` "
            "(and a CUDA-matched torch from https://pytorch.org if using GPU)."
        ) from e

    model_dir = str(model_dir)
    if not Path(model_dir).is_dir():
        raise RuntimeError(
            f"model_dir does not exist: {model_dir}\n"
            "Point nnunet.model_dir at the trained-model folder, e.g. "
            "<nnUNet_results>/Dataset111_453CT/nnUNetTrainer__nnUNetPlans__3d_fullres"
        )

    want_cuda = str(device).lower() == "cuda"
    use_cuda = want_cuda and torch.cuda.is_available()
    if want_cuda and not use_cuda:
        log.warning("CUDA requested but unavailable -- falling back to CPU (slow).")
    dev = torch.device("cuda", 0) if use_cuda else torch.device("cpu")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not disable_tta,
        perform_everything_on_device=use_cuda,
        device=dev,
        verbose=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=_parse_folds(folds),
        checkpoint_name=checkpoint,
    )
    log.info("Loaded model: %s (folds=%s, device=%s)", model_dir, folds, dev)

    in_dir, out_dir = _stage_input(input_volume, work_dir)
    predictor.predict_from_files(
        [[str(in_dir / "CASE_0000.nii.gz")]],
        str(out_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )
    out = _find_output(out_dir)
    log.info("nnU-Net labelmap: %s", out)
    return out


# --------------------------------------------------------------------------- #
# CLI backend
# --------------------------------------------------------------------------- #
def run_nnunet_cli(
    input_volume: Path,
    work_dir: Path,
    *,
    dataset_id: str,
    configuration: str = "3d_fullres",
    folds: str = "all",
    trainer: str | None = None,
    plans: str | None = None,
    nnunet_results: str | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    if shutil.which("nnUNetv2_predict") is None:
        raise RuntimeError(
            "nnUNetv2_predict not found on PATH. Use backend: python, or install "
            "nnunetv2 and activate its environment."
        )
    env = os.environ.copy()
    if nnunet_results:
        env["nnUNet_results"] = str(nnunet_results)
    if "nnUNet_results" not in env:
        raise RuntimeError(
            "nnUNet_results is not set. Set nnunet.results_dir in config or the "
            "nnUNet_results env var."
        )

    in_dir, out_dir = _stage_input(input_volume, work_dir)
    cmd = ["nnUNetv2_predict", "-i", str(in_dir), "-o", str(out_dir),
           "-d", str(dataset_id), "-c", configuration, "-f", *str(folds).split()]
    if trainer:
        cmd += ["-tr", trainer]
    if plans:
        cmd += ["-p", plans]
    if extra_args:
        cmd += list(extra_args)

    log.info("Running: %s", " ".join(cmd))
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        log.error("stdout:\n%s\nstderr:\n%s", res.stdout, res.stderr)
        raise RuntimeError(f"nnUNetv2_predict failed (exit {res.returncode})")
    return _find_output(out_dir)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def segment(input_volume: Path, work_dir: Path, ncfg: dict) -> Path:
    """Run inference using the backend named in config (`nnunet.backend`)."""
    backend = ncfg.get("backend", "python").lower()
    if backend == "python":
        return run_nnunet_python(
            input_volume, work_dir,
            model_dir=ncfg["model_dir"],
            folds=ncfg.get("folds", "all"),
            checkpoint=ncfg.get("checkpoint", "checkpoint_final.pth"),
            device=ncfg.get("device", "cuda"),
            disable_tta=ncfg.get("disable_tta", False),
        )
    if backend == "cli":
        return run_nnunet_cli(
            input_volume, work_dir,
            dataset_id=ncfg["dataset_id"],
            configuration=ncfg.get("configuration", "3d_fullres"),
            folds=str(ncfg.get("folds", "all")),
            trainer=ncfg.get("trainer"),
            plans=ncfg.get("plans"),
            nnunet_results=ncfg.get("results_dir"),
            extra_args=ncfg.get("extra_args"),
        )
    raise ValueError(f"Unknown nnunet.backend: {backend!r} (expected 'python' or 'cli')")
