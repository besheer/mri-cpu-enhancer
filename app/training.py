from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from app.admin import esc, layout, require_admin
from app.store import audit, connect

router = APIRouter(prefix="/admin/training", tags=["training"])
DATA_ROOT = Path(os.getenv("TRAINING_DATA_ROOT", "/datasets")).resolve()


@router.get("", response_class=HTMLResponse)
def training_page(message: str = "", error: str = "", _: str = Depends(require_admin)):
    with connect() as db:
        datasets = db.execute("SELECT * FROM training_datasets ORDER BY id DESC").fetchall()
        models = db.execute("SELECT * FROM model_registry ORDER BY id DESC").fetchall()
    dataset_rows = "".join(
        f"<tr><td>{d['id']}</td><td><b>{esc(d['name'])}</b></td><td>{esc(d['anatomy'])}</td><td>{esc(d['sequence'])}</td><td>{esc(d['data_mode'])}</td><td class='route'>{esc(d['path'])}</td><td><span class='badge ok'>{esc(d['status'])}</span></td></tr>"
        for d in datasets) or "<tr><td colspan='7' class='empty'>No registered datasets</td></tr>"
    model_rows = "".join(
        f"<tr><td>{m['id']}</td><td><b>{esc(m['name'])}</b></td><td>{esc(m['version'])}</td><td>{esc(m['format'])}</td><td><span class='badge {'ok' if m['active'] else 'warn'}'>{esc(m['status'])}</span></td></tr>"
        for m in models) or "<tr><td colspan='5' class='empty'>No validated models</td></tr>"
    body = f"""<div class='study-toolbar'><a class='btn gray' href='/admin'>Dashboard</a><a class='btn gray' href='/admin/studies'>Studies</a></div>
    <section class='panel'><div class='panel-head'><h2>🧠 RAD4AI & MONAI Training Workspace</h2><span class='badge ok'>RAD4AI-NAFNet-03T v2.0 Active in Production</span></div>
    <div class='notice'>Start with a specific anatomy and sequence. Models are validated and exported to ONNX before production deployment.</div>
    <form method='post' action='/admin/training/datasets' class='grid'>
      <div class='field'><label>Dataset Name</label><input name='name' required></div>
      <div class='field'><label>Data Path under /datasets</label><input name='path' placeholder='/datasets/shoulder-t1-v1' required></div>
      <div class='field'><label>Anatomy</label><input name='anatomy' placeholder='Shoulder' required></div>
      <div class='field'><label>Sequence</label><input name='sequence' placeholder='T1 COR' required></div>
      <div class='field'><label>Training Mode</label><select name='data_mode'><option value='PAIRED'>Paired low/high quality</option><option value='SELF_SUPERVISED'>Self-supervised denoising</option></select></div>
      <div class='field'><label>Notes</label><input name='notes'></div>
      <div class='wide'><button class='btn'>Validate & Register Dataset</button></div>
    </form></section>
    <section class='panel'><h2>Registered Clinical Datasets</h2><div class='table-wrap'><table><tr><th>ID</th><th>Name</th><th>Anatomy</th><th>Sequence</th><th>Mode</th><th>Path</th><th>Status</th></tr>{dataset_rows}</table></div></section>
    <section class='panel'><h2>AI Model Registry</h2><p class='muted'>Production inference runs on CPU using the approved ONNX runtime engine.</p><div class='table-wrap'><table><tr><th>ID</th><th>Model</th><th>Version</th><th>Format</th><th>Status</th></tr>{model_rows}</table></div></section>"""
    return layout("AI & MONAI Training", body, error or message, bool(error))


@router.post("/datasets")
def register_dataset(name: str = Form(...), path: str = Form(...), anatomy: str = Form(...),
                     sequence: str = Form(...), data_mode: str = Form(...), notes: str = Form(""),
                     _: str = Depends(require_admin)):
    if data_mode not in {"PAIRED", "SELF_SUPERVISED"}:
        raise HTTPException(422, "Invalid training mode")
    candidate = Path(path).resolve()
    if candidate != DATA_ROOT and DATA_ROOT not in candidate.parents:
        raise HTTPException(422, "Dataset path must be under /datasets")
    if not candidate.is_dir():
        return RedirectResponse("/admin/training?error=" + quote("Directory not found inside container: " + str(candidate)), 303)
    files = sum(1 for p in candidate.rglob("*") if p.is_file() and p.suffix.lower() in {".dcm", ".nii", ".gz"})
    if files < 10:
        return RedirectResponse("/admin/training?error=" + quote("Dataset needs at least 10 DICOM/NIfTI files"), 303)
    with connect() as db:
        cur = db.execute("INSERT INTO training_datasets(name,anatomy,sequence,data_mode,path,status,notes) VALUES(?,?,?,?,?,'VALIDATED',?)",
                         (name.strip()[:100], anatomy.strip()[:80], sequence.strip()[:80], data_mode, str(candidate), notes.strip()[:1000]))
        dataset_id = cur.lastrowid
    audit("DATASET_REGISTERED", "training_dataset", dataset_id, f"files={files};mode={data_mode}", actor=_)
    return RedirectResponse("/admin/training?message=" + quote(f"Dataset {dataset_id} validated: {files} files"), 303)
