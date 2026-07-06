#!/usr/bin/env python3
"""Local UI for the CBCT dominant-arch / tooth-root pipeline.

Run:  streamlit run app.py
Then open the URL it prints (default http://localhost:8501).

Add multiple patient .nii/.nii.gz files (upload or point at a folder), set
parameters, run the batch, and review per-patient outputs with a slice overlay.
"""

from __future__ import annotations

import copy
import time
import traceback
from pathlib import Path

import streamlit as st
import yaml

from cbct_pipeline.pipeline import run_pipeline, derive_patient_id
from cbct_pipeline.preview import make_overlay

ROOT = Path(__file__).resolve().parent
PRESETS = {
    "DentalSegmentator (ready weights)": ROOT / "config.dentalsegmentator.yaml",
    "ToothFairy2 (per-tooth)": ROOT / "config.yaml",
}
SUFFIXES = (".nii", ".nii.gz")

st.set_page_config(page_title="CBCT Arch & Root Pipeline", page_icon="🦷",
                   layout="wide")

# ----------------------------- session state ------------------------------- #
ss = st.session_state
ss.setdefault("queue", [])      # list of {"name","patient_id","path"}
ss.setdefault("results", [])    # list of {"patient_id","outputs","image","error"}


def _load_cfg(path: Path) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        st.error(
            f"Could not parse **{path.name}**.\n\n"
            "If you edited a Windows path, use forward slashes "
            "(`C:/Users/...`) or single quotes (`'C:\\Users\\...'`). "
            "Double quotes with backslashes break YAML. Or just leave the path "
            "empty in the file and set it in the sidebar below.\n\n"
            f"Details: {e}")
        st.stop()


def _add_to_queue(path: Path):
    path = Path(path)
    pid = derive_patient_id(path, None)
    if any(q["path"] == str(path) for q in ss.queue):
        return
    ss.queue.append({"name": path.name, "patient_id": pid, "path": str(path)})


def _find_labelmap(folder: Path, pid: str):
    if not folder or not Path(folder).is_dir():
        return None
    for p in sorted(Path(folder).iterdir()):
        if p.name.endswith(SUFFIXES) and pid in p.name:
            return p
    return None


# ------------------------------- sidebar ----------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")

    preset = st.selectbox("Model preset", list(PRESETS.keys()))
    cfg = _load_cfg(PRESETS[preset])
    backend = cfg.get("nnunet", {}).get("backend", "python")

    st.subheader("Model")
    if backend == "python":
        cfg["nnunet"]["model_dir"] = st.text_input(
            "Model folder (model_dir)", value=cfg["nnunet"].get("model_dir", ""),
            help="Folder containing plans.json, dataset.json and fold_* dirs")
        cfg["nnunet"]["device"] = st.selectbox(
            "Device", ["cuda", "cpu"],
            index=0 if cfg["nnunet"].get("device", "cuda") == "cuda" else 1)
        cfg["nnunet"]["disable_tta"] = st.checkbox(
            "Disable TTA (faster)", value=bool(cfg["nnunet"].get("disable_tta", False)))
    else:
        cfg["nnunet"]["results_dir"] = st.text_input(
            "nnUNet_results folder", value=cfg["nnunet"].get("results_dir") or "")
        cfg["nnunet"]["dataset_id"] = st.text_input(
            "Dataset id", value=str(cfg["nnunet"].get("dataset_id", "")))
    cfg["nnunet"]["folds"] = st.text_input(
        "Folds", value=str(cfg["nnunet"].get("folds", "all")),
        help='"all", "0 1 2 3 4", or "0"')

    st.subheader("Arch & root")
    dom = cfg.setdefault("dominance", {})
    dom["metric"] = st.selectbox(
        "Dominant-arch metric", ["bone", "teeth", "total"],
        index=["bone", "teeth", "total"].index(dom.get("metric", "bone")))
    rcfg = cfg.setdefault("root", {})
    rcfg["bone_close_mm"] = st.number_input(
        "Bone close (mm)", value=float(rcfg.get("bone_close_mm", 1.0)),
        min_value=0.0, max_value=5.0, step=0.5,
        help="Bridge thin bone gaps so a near-closed socket ring encloses the root")
    rcfg["smooth_iter"] = st.number_input(
        "Edge smoothing (iter)", value=int(rcfg.get("smooth_iter", 1)),
        min_value=0, max_value=5, step=1)
    rcfg["surface_smooth_mm"] = st.number_input(
        "Root surface smooth (mm)", value=float(rcfg.get("surface_smooth_mm", 0.7)),
        min_value=0.0, max_value=3.0, step=0.1,
        help="Softens the visible cut/seam in 3D renders (0 = off)")

    st.subheader("Inference")
    skip = st.checkbox("Skip inference — use existing labelmaps", value=False,
                       help="For testing without a GPU/model")
    labelmap_dir = ""
    if skip:
        labelmap_dir = st.text_input("Labelmaps folder",
                                     help="Files matched to patients by id in the name")

    out_dir = st.text_input("Output folder", value=str(ROOT / "out"))
    plane = st.selectbox("Preview plane", ["sagittal", "coronal", "axial"])

# ------------------------------- main -------------------------------------- #
st.title("🦷 CBCT Arch & Root Pipeline")
st.caption("Segment → keep the dominant arch → output bone, teeth, and tooth roots "
           "(travecular). Add one or many patients below.")

tab_folder, tab_upload = st.tabs(["📁 Folder on this machine", "⬆️ Upload files"])

with tab_folder:
    folder = st.text_input("Folder containing .nii / .nii.gz files")
    if st.button("Add folder", key="add_folder") and folder:
        f = Path(folder)
        if f.is_dir():
            found = [p for p in sorted(f.iterdir()) if p.name.endswith(SUFFIXES)]
            for p in found:
                _add_to_queue(p)
            st.success(f"Added {len(found)} file(s).")
        else:
            st.error("Not a folder.")

with tab_upload:
    ups = st.file_uploader("Select one or more volumes", type=["nii", "gz"],
                           accept_multiple_files=True)
    if st.button("Add uploaded", key="add_up") and ups:
        updir = Path(out_dir) / "_uploads"
        updir.mkdir(parents=True, exist_ok=True)
        for uf in ups:
            dest = updir / uf.name
            dest.write_bytes(uf.getbuffer())
            _add_to_queue(dest)
        st.success(f"Added {len(ups)} file(s).")

# Queue table
st.subheader(f"Patient queue ({len(ss.queue)})")
if ss.queue:
    st.dataframe(
        [{"Patient": q["patient_id"], "File": q["name"]} for q in ss.queue],
        width="stretch", hide_index=True)
    c1, c2 = st.columns([1, 6])
    if c1.button("Clear queue"):
        ss.queue = []
        st.rerun()
else:
    st.info("No patients yet. Add a folder or upload files above.")

run = st.button("▶️ Run pipeline", type="primary", disabled=not ss.queue)

# ------------------------------- run --------------------------------------- #
if run:
    ss.results = []
    out_root = Path(out_dir)
    progress = st.progress(0.0, text="Starting…")
    n = len(ss.queue)
    for i, q in enumerate(ss.queue, start=1):
        pid, src = q["patient_id"], Path(q["path"])
        progress.progress((i - 1) / n, text=f"[{i}/{n}] {pid} — segmenting…")
        lm = _find_labelmap(Path(labelmap_dir), pid) if skip else None
        rec = {"patient_id": pid, "outputs": {}, "image": str(src), "error": None}
        try:
            if skip and lm is None:
                raise FileNotFoundError(
                    f"No labelmap containing '{pid}' in {labelmap_dir}")
            outs = run_pipeline(src, out_root, copy.deepcopy(cfg),
                                labelmap_path=lm)
            rec["outputs"] = {k: str(v) for k, v in outs.items()}
        except Exception as exc:
            rec["error"] = f"{exc}\n{traceback.format_exc()}"
        ss.results.append(rec)
        progress.progress(i / n, text=f"[{i}/{n}] {pid} — done")
    progress.progress(1.0, text=f"Completed {n} patient(s)")
    time.sleep(0.3)

# ------------------------------- results ----------------------------------- #
if ss.results:
    ok = sum(1 for r in ss.results if not r["error"])
    st.subheader(f"Results — {ok}/{len(ss.results)} succeeded")
    for r in ss.results:
        pid = r["patient_id"]
        if r["error"]:
            with st.expander(f"❌ {pid} — failed", expanded=False):
                st.code(r["error"])
            continue
        with st.expander(f"✅ {pid}", expanded=True):
            left, right = st.columns([3, 4])
            with left:
                st.markdown("**Output files**")
                for key, path in r["outputs"].items():
                    p = Path(path)
                    if p.exists():
                        st.download_button(
                            f"⬇️ {p.name}", data=p.read_bytes(),
                            file_name=p.name, key=f"dl_{pid}_{key}",
                            mime="application/gzip")
            with right:
                try:
                    masks = {k: Path(v) for k, v in r["outputs"].items()
                             if k in ("bone", "teeth", "root")}
                    png = Path(r["outputs"]["bone"]).parent / f"{pid}_preview_{plane}.png"
                    make_overlay(r["image"], masks, png, plane=plane,
                                 title=f"{pid} — {plane}")
                    st.image(str(png), width="stretch")
                except Exception as exc:
                    st.warning(f"Preview unavailable: {exc}")
