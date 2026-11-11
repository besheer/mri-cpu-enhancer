from __future__ import annotations

import io
import json
import os
import re
import struct
import zlib
from urllib.parse import quote

import httpx
import numpy as np
import pydicom
from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from app.admin import esc, layout, require_admin
from app.profiles import PROFILE_INFO, PROFILES, STRENGTHS, recommendation, validate_selection
from app.store import (audit, connect, enabled_profile_names, enqueue_comparison,
                       enqueue_return, get_center_by_aet, get_comparison_job,
                       list_comparison_jobs, list_processing_tools)


def source_center(api: httpx.Client, series_id: str):
    info = api.get(f"/series/{series_id}")
    info.raise_for_status()
    instances = info.json().get("Instances", [])
    if not instances:
        return None
    origin = api.get(f"/instances/{instances[0]}/metadata/1024")
    return get_center_by_aet(origin.text.strip()) if origin.status_code == 200 else None


router = APIRouter(prefix="/admin", tags=["viewer"])
ORTHANC_ID = re.compile(r"^[0-9a-f]{8}(?:-[0-9a-f]{8}){4}$")


def is_derived_series(description: str) -> bool:
    u = description.upper()
    return u.startswith("ENH") or "[AI-SWIFT]" in u or "[RAD4AI]" in u or "[RAD4AI-PRO]" in u or "[RAD4AI-1.5T]" in u or "[RAD4AI-GE-AW]" in u or "[ENHANCED]" in u


def client() -> httpx.Client:
    return httpx.Client(base_url=os.getenv("ORTHANC_URL", "http://orthanc:8042"),
                        auth=(os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", "")),
                        timeout=60)


def viewer_url(study_uid: str) -> str:
    return f"{os.getenv('OHIF_PUBLIC_URL', '/ohif/viewer')}?StudyInstanceUIDs={quote(study_uid, safe='.')}"


@router.get("/series/{series_id}/thumbnail")
def thumbnail(series_id: str, _: str = Depends(require_admin)):
    if not ORTHANC_ID.fullmatch(series_id):
        raise HTTPException(400, "Invalid series ID")
    try:
        with client() as api:
            response = api.get(f"/series/{series_id}")
            response.raise_for_status()
            instances = response.json().get("Instances", [])
            if not instances:
                raise ValueError("Empty series")
            preview = api.get(f"/instances/{instances[len(instances)//2]}/preview")
            preview.raise_for_status()
        return Response(preview.content, media_type=preview.headers.get("content-type", "image/png"),
                        headers={"Cache-Control": "private, max-age=300"})
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/series/{series_id}/slice/{slice_index}.png")
def series_slice_png(series_id: str, slice_index: int, slab: str = "single", lut: str = "gray", _: str = Depends(require_admin)):
    if not ORTHANC_ID.fullmatch(series_id):
        raise HTTPException(400, "Invalid series ID")
    try:
        with client() as api:
            info = api.get(f"/series/{series_id}")
            info.raise_for_status()
            instances = info.json().get("Instances", [])
            if not instances:
                raise ValueError("Empty series")
            inst_meta = []
            for iid in instances:
                r = api.get(f"/instances/{iid}")
                if r.status_code == 200:
                    num = int(r.json().get("MainDicomTags", {}).get("InstanceNumber", 0) or 0)
                    inst_meta.append((num, iid))
            inst_meta.sort(key=lambda x: x[0])
            idx = max(0, min(slice_index, len(inst_meta) - 1))

            # Fast path if single slice and default grayscale
            if slab == "single" and lut == "gray":
                chosen_id = inst_meta[idx][1]
                preview = api.get(f"/instances/{chosen_id}/preview")
                preview.raise_for_status()
                return Response(preview.content, media_type=preview.headers.get("content-type", "image/png"),
                                headers={"Cache-Control": "private, max-age=120"})

            # GE AW Volume Viewer Slab Projection (MIP-3, MIP-5, MinIP-3, AVG-3) & READY View Color LUTs
            radius = 2 if slab in ("mip5", "avg5") else 1 if slab in ("mip3", "minip3", "avg3") else 0
            start_i = max(0, idx - radius)
            end_i = min(len(inst_meta) - 1, idx + radius)
            arrays = []
            for s_i in range(start_i, end_i + 1):
                raw_dcm = api.get(f"/instances/{inst_meta[s_i][1]}/file")
                raw_dcm.raise_for_status()
                ds = pydicom.dcmread(io.BytesIO(raw_dcm.content))
                arrays.append(ds.pixel_array.astype(np.float32))
            stack = np.stack(arrays, axis=0)
            if slab.startswith("mip"):
                img2d = np.max(stack, axis=0)
            elif slab.startswith("minip"):
                img2d = np.min(stack, axis=0)
            elif slab.startswith("avg"):
                img2d = np.mean(stack, axis=0)
            else:
                img2d = stack[len(stack) // 2]

            p1, p99 = np.percentile(img2d, (1.0, 99.5))
            norm = np.clip((img2d - p1) / max(float(p99 - p1), 1.0), 0.0, 1.0)
            if lut == "ge_gold":
                r_ch = np.clip(norm * 1.35, 0.0, 1.0)
                g_ch = np.clip(norm ** 1.25 * 1.05, 0.0, 1.0)
                b_ch = np.clip(norm ** 2.2 * 0.65, 0.0, 1.0)
                rgb = np.stack([r_ch * 255, g_ch * 255, b_ch * 255], axis=-1).astype(np.uint8)
            elif lut == "hot_iron":
                r_ch = np.clip(norm * 2.0, 0.0, 1.0)
                g_ch = np.clip((norm - 0.35) * 2.0, 0.0, 1.0)
                b_ch = np.clip((norm - 0.70) * 3.3, 0.0, 1.0)
                rgb = np.stack([r_ch * 255, g_ch * 255, b_ch * 255], axis=-1).astype(np.uint8)
            elif lut == "pet_rainbow":
                r_ch = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0) * (norm > 0.05)
                g_ch = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0) * (norm > 0.05)
                b_ch = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0) * (norm > 0.05)
                rgb = np.stack([r_ch * 255, g_ch * 255, b_ch * 255], axis=-1).astype(np.uint8)
            else:
                g8 = (norm * 255.0).astype(np.uint8)
                rgb = np.stack([g8, g8, g8], axis=-1)
        return Response(png_rgb(rgb), media_type="image/png", headers={"Cache-Control": "private, max-age=60"})
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


def png_rgb(array: np.ndarray) -> bytes:
    height, width, channels = array.shape
    if channels != 3 or array.dtype != np.uint8:
        raise ValueError("RGB uint8 image required")
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def profile_form(series_id: str, description: str, instances: int, existing_job=None) -> str:
    recommended, _reason = recommendation(description, instances)
    enabled = enabled_profile_names()
    if "RAD4AI_SPINE_PRO" in enabled and not any(w in description.upper() for w in ("SCANO", "LOCALIZER", "SURVEY")):
        recommended = "RAD4AI_SPINE_PRO"
    elif "AI_SWIFT_DENOISE" in enabled and not any(w in description.upper() for w in ("SCANO", "LOCALIZER", "SURVEY")):
        recommended = "AI_SWIFT_DENOISE"
    options = "".join(
        f"<option value='{key}' {'selected' if key == recommended else ''} title='{esc(PROFILE_INFO[key]['guidance'])}'>{esc(label)}</option>"
        for key, label in PROFILES.items() if key in enabled
    )
    default_strength = "MEDIUM" if recommended in {"AI_SWIFT_DENOISE", "RAD4AI_SPINE_PRO"} else "MILD"
    strengths = "".join(f"<option value='{key}' {'selected' if key == default_strength else ''}>{esc(label)}</option>" for key, label in STRENGTHS.items())
    compare_btn = ""
    if existing_job and existing_job["status"] == "ready" and existing_job["output_series_id"]:
        compare_btn = f"<a class='btn emerald' href='/admin/jobs/{existing_job['id']}/compare'>🔍 Compare Before/After (Job #{existing_job['id']})</a>"
    return (f"<div class='series-choice'><input class='series-check' type='checkbox' "
            f"value='{series_id}' aria-label='Select {esc(description)}'>"
            f"<form method='post' action='/admin/series/{series_id}/ai-swift' class='ai-quick-form'>"
            f"<button class='btn ai-btn' title='Instant RAD4AI Spine & Ortho Pro High-Clarity Enhancement'>⚡ Instant RAD4AI Pro</button></form>"
            f"<form method='post' action='/admin/series/{series_id}/enqueue' class='actions enqueue-form'>"
            f"<select name='profile' style='padding:7px;border-radius:7px'>{options}</select>"
            f"<select name='strength' style='padding:7px;border-radius:7px'>{strengths}</select>"
            "<label class='muted' title='Required for CLAHE or Strong level'><input type='checkbox' name='acknowledged'> Research Ack</label>"
            f"<button class='btn gray'>+ Custom Queue</button></form>"
            f"{compare_btn}</div>")


@router.get("/studies", response_class=HTMLResponse)
def studies(message: str = "", error: str = "", job_status: str = "", profile_filter: str = "", _: str = Depends(require_admin)):
    allowed_statuses = {"", "queued", "processing", "ready", "failed"}
    if job_status not in allowed_statuses:
        job_status = ""
    if profile_filter not in {"", *PROFILES}:
        profile_filter = ""
    all_jobs = list_comparison_jobs(limit=1000)
    jobs = list_comparison_jobs(status=job_status, profile=profile_filter)
    latest = {}
    for job in all_jobs:
        latest.setdefault(job["source_series_id"], job)
    try:
        with client() as api:
            studies_resp = api.get("/studies?expand")
            studies_resp.raise_for_status()
            series_resp = api.get("/series?expand")
            series_resp.raise_for_status()
            series_lookup = {s["ID"]: s for s in series_resp.json()}

            rows = []
            for idx_study, study in enumerate(studies_resp.json()):
                main = study.get("MainDicomTags", {})
                patient = study.get("PatientMainDicomTags", {})
                raw_cells, derived_cells, enhanced_count, thumbnail_series = [], [], 0, ""

                study_series = [series_lookup[sid] for sid in study.get("Series", []) if sid in series_lookup]
                study_series.sort(key=lambda s: int(s.get("MainDicomTags", {}).get("SeriesNumber", 0) or 0))

                for item in study_series:
                    series_id = item["ID"]
                    tags = item.get("MainDicomTags", {})
                    description = str(tags.get("SeriesDescription", "Unnamed Series"))
                    s_num = str(tags.get("SeriesNumber", "—"))
                    inst_count = len(item.get("Instances", []))
                    enhanced = is_derived_series(description)
                    enhanced_count += int(enhanced)
                    if not enhanced and not thumbnail_series:
                        thumbnail_series = series_id
                    job = latest.get(series_id)

                    if enhanced:
                        disp_desc = description.replace("[AI-SWIFT]", "[RAD4AI]")
                        up = description.upper()
                        if "[RAD4AI-RNLM]" in up or "RNLM" in up:
                            badge_type = "🛡️ RAD4AI Rician-VST + NLM"
                        elif "[RAD4AI-3THD]" in up or "3THD" in up:
                            badge_type = "🔬 RAD4AI 3T-HD True Anatomy"
                        elif "[RAD4AI-PRINT]" in up or "PRINT" in up:
                            badge_type = "🖨️ RAD4AI Print-Master Ultra"
                        elif "[RAD4AI-GE-AW]" in up or "GEAW" in up or "GE-AW" in up:
                            badge_type = "🖥️ RAD4AI GE-AW Workstation"
                        elif "[RAD4AI-1.5T]" in up or "-SR-" in up:
                            badge_type = "⚡ RAD4AI 1.5T Super-Resolution"
                        elif "[RAD4AI-PRO]" in up or "-PRO-" in up:
                            badge_type = "🦴 RAD4AI Spine & Ortho Pro"
                        elif any(k in up for k in ("[AI-SWIFT]", "AISWIFT", "[RAD4AI]", "RAD4AI")):
                            badge_type = "✨ RAD4AI NAFNet Denoise"
                        else:
                            badge_type = "✓ Enhanced (Derived)"
                        derived_cells.append(
                            f"<div class='series-item derived-item'>"
                            f"<div><span class='snum-badge derived'>S:{esc(s_num)}</span> <b>{esc(disp_desc)}</b> "
                            f"<span class='muted'>({inst_count} slices)</span></div>"
                            f"<div style='display:flex;align-items:center;gap:8px'>"
                            f"<span class='badge ok'>{badge_type}</span>"
                            f"<form method='post' action='/admin/series/{esc(series_id)}/send-to-ge' style='margin:0' onsubmit=\"return confirm('إرسال هذه السلسلة المحسنة فوراً إلى جهاز GE Workstation (192.168.0.206:4006)؟')\">"
                            f"<button class='btn emerald' type='submit' style='padding:5px 12px;font-size:12px;font-weight:700;display:inline-flex;align-items:center;gap:4px'>📤 Send to GE</button></form>"
                            f"<form method='post' action='/admin/series/{esc(series_id)}/delete' style='margin:0' onsubmit=\"return confirm('مسح هذه السلسلة المحسنة نهائياً؟')\">"
                            f"<button class='btn red' style='padding:5px 10px;font-size:12px' title='Delete Series'>🗑️</button></form>"
                            f"</div></div>"
                        )
                    else:
                        if job and job["status"] in {"queued", "processing"}:
                            action = f"<span class='badge warn'>⏳ Processing: {esc(job['profile'])} ({esc(job['strength'])})</span>"
                        else:
                            action = profile_form(series_id, description, inst_count, job)
                        raw_cells.append(
                            f"<div class='series-item'>"
                            f"<div><span class='snum-badge'>S:{esc(s_num)}</span> <b>{esc(description)}</b> "
                            f"<span class='muted'>({inst_count} raw slices)</span></div>"
                            f"{action}</div>"
                        )

                uid = str(main.get("StudyInstanceUID", ""))
                open_link = f"<a class='btn emerald' target='_blank' rel='noopener' href='{esc(viewer_url(uid))}'>🖥️ Open & Compare in OHIF Viewer</a>" if uid else "Missing UID"
                thumb = f"<img class='thumb' loading='lazy' src='/admin/series/{thumbnail_series}/thumbnail' alt='Study thumbnail'>" if thumbnail_series else "<div class='thumb'></div>"
                open_attr = "open" if idx_study == 0 else ""

                derived_section = ""
                if derived_cells:
                    derived_section = (
                        f"<div style='margin-top:14px'><div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>"
                        f"<h4 style='margin:0;color:#34d399'>✨ AI-Enhanced Series Ready for Comparison ({len(derived_cells)}):</h4>"
                        f"<form method='post' action='/admin/studies/{esc(study.get('ID',''))}/delete-enhanced' style='margin:0' "
                        f"onsubmit=\"return confirm('مسح جميع السلاسل المحسّنة لهذه الحالة فقط مع الإبقاء على الفحص الأصلي؟\n\nClear all enhanced series for this study?')\">"
                        f"<button class='btn warn' style='padding:5px 12px;font-size:12px'>🧹 Clear Enhanced ({len(derived_cells)})</button></form></div>"
                        f"<div class='series-grid'>{''.join(derived_cells)}</div></div>"
                    )

                rows.append(
                    f"<details class='study-card' data-study='{esc(study.get('ID',''))}' {open_attr}>"
                    f"<summary style='display:flex;align-items:center;gap:12px'>"
                    f"<input class='study-check' type='checkbox' value='{esc(study.get('ID',''))}' title='Select Case' style='width:22px;height:22px;accent-color:#ef4444;cursor:pointer' onclick='event.stopPropagation()'>"
                    f"{thumb}<div style='flex:1'><b style='font-size:16px'>{esc(patient.get('PatientName','')) or 'Unknown patient'}</b> — "
                    f"<span style='color:#60a5fa;font-weight:700'>{esc(main.get('StudyDescription',''))}</span><br>"
                    f"<span class='muted'>ID: {esc(patient.get('PatientID',''))} · Date: {esc(main.get('StudyDate',''))} · {len(raw_cells)} original series</span><br>"
                    f"<span class='badge {'ok' if enhanced_count else 'warn'}' style='margin-top:5px'>{enhanced_count} enhanced series ready</span></div>"
                    f"<span class='chevron'></span></summary>"
                    f"<div class='study-body'><div class='study-toolbar'>"
                    f"<button type='button' class='btn gray select-study'>☑️ Select All Series in Case</button>{open_link}"
                    f"<form method='post' action='/admin/studies/{esc(study.get('ID',''))}/delete' style='display:inline-block;margin-left:auto' "
                    f"onsubmit=\"return confirm('⚠️ Are you sure you want to completely DELETE this study? This cannot be undone!\n\nهل أنت متأكد من مسح هذه الحالة بالكامل نهائياً من السيرفر؟')\">"
                    f"<button class='btn red' style='padding:7px 14px'>🗑️ Delete Study (مسح الحالة)</button></form></div>"
                    f"<h4 style='margin:6px 0 8px;color:#93c5fd'>🩻 Original Raw Series:</h4>"
                    f"<div class='series-grid'>{''.join(raw_cells)}</div>"
                    f"{derived_section}</div></details>"
                )
    except httpx.HTTPError as exc:
        return layout("Studies & AI Comparison", "<section class='panel'><h2>Failed to load studies from Orthanc</h2></section>", str(exc), True)

    studies_html = "".join(rows) or "<div class='empty'>No studies available in Orthanc.</div>"
    job_rows = []
    for job in jobs:
        metrics = json.loads(job["metrics_json"]) if job["metrics_json"] else {}
        if metrics:
            sec = metrics.get("seconds", "—")
            nc = metrics.get("noise_change_percent", "—")
            ep = metrics.get("edge_preservation_percent", "—")
            inst = metrics.get("output_instances", metrics.get("instances", "—"))
            metrics_html = (
                f"<div class='kpi-pills'>"
                f"<span class='kpi-pill'>⚡ {esc(sec)}s</span>"
                f"<span class='kpi-pill'>📉 Noise {esc(nc)}%</span>"
                f"<span class='kpi-pill'>🎯 Edge {esc(ep)}%</span>"
                f"<span class='kpi-pill'>🖼️ {esc(inst)} slices</span>"
                f"</div>"
            )
        else:
            metrics_html = "<span class='muted'>—</span>"
        try:
            qc_reasons = " · ".join(json.loads(job["qc_reasons"])) if job["qc_reasons"] else ""
        except (json.JSONDecodeError, TypeError):
            qc_reasons = str(job["qc_reasons"] or "")
        qc_class = "ok" if job["qc_status"] == "PASS" else "bad" if job["qc_status"] == "FAIL" else "warn"
        controls = ""
        if job["status"] == "ready":
            decisions = {"": "Unreviewed", "best": "⭐ Best (Diagnostic)", "acceptable": "✓ Clinically Acceptable", "rejected": "✗ Rejected"}
            options = "".join(f"<option value='{key}' {'selected' if job['decision'] == key else ''}>{label}</option>" for key, label in decisions.items())
            controls = (f"<div class='actions'>"
                        f"<a class='btn ai-btn' href='/admin/jobs/{job['id']}/compare'>🔍 Interactive Compare (Before / After)</a>"
                        f"<a class='btn gray' target='_blank' href='/admin/jobs/{job['id']}/difference.png'>Difference Map</a></div>"
                        f"<form method='post' action='/admin/jobs/{job['id']}/evaluate' class='review-form'>"
                        f"<select name='decision' required>{options}</select>"
                        f"<input name='notes' maxlength='1000' placeholder='Radiologist notes...' value='{esc(job['review_notes'])}'>"
                        "<button class='btn gray'>Save Review</button></form>"
                        + (f"<a class='btn gray' href='/admin/protocols/{job['protocol_study_id']}'>Protocol Study</a>" if job["protocol_study_id"] else
                           f"<form style='margin-top:5px' method='post' action='/admin/jobs/{job['id']}/approve' onsubmit=\"return confirm('Approve result and add to PACS Return Queue?')\"><button class='btn emerald'>📤 Approve & Return to PACS</button></form>"))
        elif job["status"] == "failed":
            controls = f"<form method='post' action='/admin/jobs/{job['id']}/retry'><button class='btn gray'>🔄 Retry Job</button></form>"
        decision_text = {"best": "⭐ Best", "acceptable": "✓ Acceptable", "rejected": "✗ Rejected"}.get(job["decision"], "Unreviewed")
        job_rows.append(f"<tr><td><b>#{job['id']}</b></td><td><b>{esc(job['profile'])}</b><br><span class='badge'>{esc(job['strength'])}</span></td><td><span class='badge {'ok' if job['status']=='ready' else 'bad' if job['status']=='failed' else 'warn'}'>{esc(job['status'])}</span></td><td><span class='badge {qc_class}'>{esc(job['qc_status'])}</span><div class='hint'>{esc(qc_reasons)}</div></td><td>{metrics_html}</td><td><span class='badge {'ok' if job['decision'] in {'best','acceptable'} else 'bad' if job['decision']=='rejected' else 'warn'}'>{decision_text}</span></td><td>{esc(job['error'])}</td><td>{controls}</td></tr>")
    jobs_html = "".join(job_rows) or "<tr><td colspan='8' class='empty'>No jobs matching selected filters.</td></tr>"
    enabled = enabled_profile_names()
    bulk_options = "".join(f"<label class='tool-check'><input type='checkbox' name='bulk-tool' value='{key}' {'checked' if key == 'AI_SWIFT_DENOISE' else ''}> {esc(label)}</label>" for key, label in PROFILES.items() if key in enabled)
    strength_options = "".join(f"<option value='{key}' {'selected' if key == 'MEDIUM' else ''}>{esc(label)}</option>" for key, label in STRENGTHS.items())
    bulk_bar = ("<div id='queue-toast' class='queue-toast' hidden></div><div id='bulk-bar' class='bulk-bar' hidden><b><span id='selected-count'>0</span> series selected</b>"
                f"<div class='bulk-tools'>{bulk_options}</div><select id='bulk-strength'>{strength_options}</select>"
                "<label class='muted'><input id='bulk-ack' type='checkbox'> CLAHE/Strong Research Ack</label>"
                "<button type='button' class='btn ai-btn' id='bulk-enqueue'>⚡ Enhance Selected Now</button>"
                "<button type='button' class='btn gray' id='clear-selection'>Clear Selection</button><span id='bulk-status'></span></div>")
    script = """<script>
const checks=()=>[...document.querySelectorAll('.series-check:checked')];
const bar=document.getElementById('bulk-bar'),count=document.getElementById('selected-count'),statusBox=document.getElementById('bulk-status'),toast=document.getElementById('queue-toast');

// Cases Checkbox & Bulk Delete Handler
const studyChecks = () => [...document.querySelectorAll('.study-check:checked')];
const allCasesCheck = document.getElementById('select-all-cases-check');
const delSelectedCasesBtn = document.getElementById('delete-selected-cases-btn');
const selectedCasesCount = document.getElementById('selected-cases-count');

function syncCasesBar() {
    const n = studyChecks().length;
    if (selectedCasesCount) selectedCasesCount.textContent = n;
    if (delSelectedCasesBtn) delSelectedCasesBtn.style.display = n > 0 ? 'inline-flex' : 'none';
    if (allCasesCheck) {
        const total = document.querySelectorAll('.study-check').length;
        allCasesCheck.checked = total > 0 && n === total;
    }
}

document.querySelectorAll('.study-check').forEach(c => c.addEventListener('change', syncCasesBar));

allCasesCheck?.addEventListener('change', () => {
    const checked = allCasesCheck.checked;
    document.querySelectorAll('.study-check').forEach(c => c.checked = checked);
    syncCasesBar();
});

delSelectedCasesBtn?.addEventListener('click', async () => {
    const selected = studyChecks();
    if (!selected.length) return;
    if (!confirm(`⚠️ هل أنت متأكد من مسح ${selected.length} حالات محددة نهائياً بجميع صورها من السيرفر؟\n\nAre you sure you want to permanently delete ${selected.length} selected cases?`)) return;
    delSelectedCasesBtn.disabled = true;
    delSelectedCasesBtn.textContent = '⏳ Deleting...';
    const data = new FormData();
    selected.forEach(c => data.append('study_ids', c.value));
    try {
        const resp = await fetch('/admin/studies/delete-batch', { method: 'POST', body: data });
        const res = await resp.json();
        if (!resp.ok) throw new Error(res.detail || 'Failed');
        showToast(`Deleted ${res.count} cases successfully`);
        setTimeout(() => location.reload(), 1200);
    } catch (e) {
        showToast('Error: ' + e.message, true);
        delSelectedCasesBtn.disabled = false;
        syncCasesBar();
    }
});

function showToast(message,isError=false){toast.textContent=message;toast.className='queue-toast '+(isError?'error':'ok');toast.hidden=false;setTimeout(()=>toast.hidden=true,4500);}
function syncBar(){const n=checks().length;count.textContent=n;bar.hidden=!n;}
document.querySelectorAll('.series-check').forEach(c=>c.addEventListener('change',syncBar));
document.querySelectorAll('.select-study').forEach(b=>b.addEventListener('click',()=>{const boxes=[...b.closest('.study-card').querySelectorAll('.series-check:not(:disabled)')];const next=boxes.some(c=>!c.checked);boxes.forEach(c=>c.checked=next);syncBar();}));
document.getElementById('clear-selection').addEventListener('click',()=>{checks().forEach(c=>c.checked=false);syncBar();});
document.querySelectorAll('.enqueue-form').forEach(form=>form.addEventListener('submit',async event=>{event.preventDefault();const button=form.querySelector('button');button.disabled=true;try{const response=await fetch(form.action,{method:'POST',body:new FormData(form),headers:{'X-Requested-With':'fetch'}});const data=await response.json();if(!response.ok)throw new Error(data.detail||'Failed to enqueue job');button.textContent='Queued ✓';showToast(`Job #${data.job_id} added to processing queue`);setTimeout(()=>location.reload(),1800);}catch(error){showToast(error.message,true);}finally{button.disabled=false;}}));
document.querySelectorAll('.ai-quick-form').forEach(form=>form.addEventListener('submit',async event=>{event.preventDefault();const button=form.querySelector('button');button.disabled=true;button.textContent='⏳ Enhancing...';try{const response=await fetch(form.action,{method:'POST',headers:{'X-Requested-With':'fetch'}});const data=await response.json();if(!response.ok)throw new Error(data.detail||'Failed to start RAD4AI');button.textContent='✓ Queued!';showToast(`⚡ RAD4AI enhancement started (Job #${data.job_id})...`);setTimeout(()=>location.reload(),2200);}catch(error){showToast(error.message,true);button.textContent='⚡ Instant RAD4AI Enhance';}finally{button.disabled=false;}}));
document.getElementById('bulk-delete')?.addEventListener('click',async()=>{
    const selected=checks();
    if(!selected.length){showToast('Please select at least one series to delete', true); return;}
    if(!confirm(`⚠️ تحذير: هل أنت متأكد من مسح ${selected.length} سلاسل محددة نهائياً من السيرفر؟\\n\\nAre you sure you want to permanently delete ${selected.length} selected series?`))return;
    const button=document.getElementById('bulk-delete'),data=new FormData();
    selected.forEach(c=>data.append('series_ids',c.value));
    button.disabled=true;statusBox.textContent=' Deleting...';
    try{
        const response=await fetch('/admin/series/delete-batch',{method:'POST',body:data,headers:{'X-Requested-With':'fetch'}});
        const result=await response.json();
        if(!response.ok)throw new Error(result.detail||'Failed to delete batch');
        showToast(`Deleted ${result.count} series successfully`);
        setTimeout(()=>location.reload(),1200);
    }catch(error){
        showToast(error.message,true);
    }finally{
        button.disabled=false;
    }
});
document.getElementById('bulk-enqueue').addEventListener('click',async()=>{const selected=checks(),profiles=[...document.querySelectorAll("input[name='bulk-tool']:checked")];if(!selected.length)return;if(!profiles.length){showToast('Select at least one enhancer profile',true);return;}const button=document.getElementById('bulk-enqueue'),data=new FormData();selected.forEach(c=>data.append('series_ids',c.value));profiles.forEach(p=>data.append('profiles',p.value));data.append('strength',document.getElementById('bulk-strength').value);if(document.getElementById('bulk-ack').checked)data.append('acknowledged','on');button.disabled=true;statusBox.textContent=' Enqueuing...';try{const response=await fetch('/admin/series/enqueue-batch',{method:'POST',body:data,headers:{'X-Requested-With':'fetch'}});const result=await response.json();if(!response.ok)throw new Error(result.detail||'Failed to enqueue batch');selected.forEach(c=>{c.checked=false;c.disabled=true;});statusBox.textContent='';showToast(`Added ${result.count} jobs to the processing queue`);syncBar();setTimeout(()=>location.reload(),2000);}catch(error){statusBox.textContent='';showToast(error.message,true);}finally{button.disabled=false;}});
</script>"""
    status_options = "".join(f"<option value='{value}' {'selected' if job_status == value else ''}>{label}</option>" for value, label in [("", "All Statuses"), ("queued", "Queued"), ("processing", "Processing"), ("ready", "Ready"), ("failed", "Failed")])
    profile_options = "<option value=''>All Enhancers</option>" + "".join(f"<option value='{key}' {'selected' if profile_filter == key else ''}>{esc(label)}</option>" for key, label in PROFILES.items())
    filters = f"<form method='get' class='actions queue-filters'><select name='job_status'>{status_options}</select><select name='profile_filter'>{profile_options}</select><button class='btn gray'>Filter</button><a class='btn gray' href='/admin/studies'>Reset</a></form>"
    body = (
        f"<section class='panel'><div class='panel-head'><h2>🩻 MRI Studies & RAD4AI Enhancement</h2>"
        f"<div class='study-toolbar' style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>"
        f"<button class='btn gray' type='button' onclick=\"document.querySelectorAll('.study-card').forEach(x=>x.open=true)\">Expand All</button>"
        f"<button class='btn gray' type='button' onclick=\"document.querySelectorAll('.study-card').forEach(x=>x.open=false)\">Collapse All</button>"
        f"<label style='display:inline-flex;align-items:center;gap:6px;cursor:pointer;font-weight:700;margin-left:10px;padding:6px 12px;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.3);border-radius:8px'>"
        f"<input type='checkbox' id='select-all-cases-check' style='width:18px;height:18px;accent-color:#ef4444'>"
        f"<span>☑️ Select All Cases (تحديد كل الحالات)</span></label>"
        f"<button type='button' class='btn red' id='delete-selected-cases-btn' style='display:none;padding:7px 14px'>🗑️ Delete Selected (<span id='selected-cases-count'>0</span>)</button>"
        f"<form method='post' action='/admin/studies/delete-all' style='margin:0;margin-left:auto' "
        f"onsubmit=\"return confirm('⚠️ تحذير شديد: هل أنت متأكد تماماً من مسح جميع الحالات والفحوصات بالكامل نهائياً من السيرفر؟\\n\\nAre you sure you want to completely DELETE ALL studies? This cannot be undone!')\">"
        f"<button class='btn red' style='padding:7px 14px;background:#991b1b;border-color:#7f1d1d'>⚠️ Delete ALL Cases (مسح كل الحالات)</button></form>"
        f"</div></div>"
        f"<p class='muted'>Original raw series keep their base number (e.g. <b>S:14</b>), while RAD4AI-enhanced series add 100 (e.g. <b>S:114 [RAD4AI]</b>). Click <b>⚡ Instant RAD4AI Enhance</b> for 1-click enhancement, or <b>🔍 Compare Before/After</b> to inspect results.</p>"
        f"{studies_html}</section>"
        f"<section class='panel'><div class='panel-head'><h2>⚡ Processing Queue & Clinical Quality Gate</h2>{filters}</div>"
        f"<div class='table-wrap'><table><tr><th>Job #</th><th>Enhancer & Level</th><th>Status</th><th>Quality Gate (QC)</th><th>Performance & Edge Metrics</th><th>Radiologist Review</th><th>Notes / Error</th><th>Compare & Action</th></tr>{jobs_html}</table></div></section>{bulk_bar}{script}"
    )
    return layout("Studies & RAD4AI Comparison", body, error or message, bool(error))


@router.get("/jobs/{job_id}/compare", response_class=HTMLResponse)
def interactive_compare(job_id: int, message: str = "", error: str = "", _: str = Depends(require_admin)):
    job = get_comparison_job(job_id)
    if job is None or job["status"] != "ready" or not job["output_series_id"]:
        raise HTTPException(404, "Comparison job is not ready or does not exist")
    metrics = json.loads(job["metrics_json"]) if job["metrics_json"] else {}
    total_slices = int(metrics.get("output_instances", metrics.get("instances", 16)) or 16)
    mid_slice = max(0, total_slices // 2)
    src_sid = job["source_series_id"]
    out_sid = job["output_series_id"]

    gemini_data = None
    try:
        if "gemini_audit" in job.keys() and job["gemini_audit"]:
            gemini_data = json.loads(job["gemini_audit"])
    except Exception:
        pass

    gemini_html = ""
    if gemini_data and not gemini_data.get("error"):
        findings_li = "".join(f"<li>{esc(f)}</li>" for f in gemini_data.get("preliminary_findings", []))
        score_val = gemini_data.get("safety_score", 10)
        score_badge = f"<span class='badge ok'>🛡️ Safety Score: {score_val}/10 ({gemini_data.get('verdict','PASS')})</span>"
        model_badge = f"<span class='badge' style='background:#1e1b4b;color:#a5b4fc;border:1px solid #4338ca'>⚡ {esc(gemini_data.get('model_used','Gemini Vision'))}</span>"
        gemini_html = f"""
        <div style='background:linear-gradient(135deg, rgba(30,27,75,.4), rgba(15,23,42,.8));border:1px solid #4338ca;border-radius:12px;padding:14px 18px;margin-bottom:16px;color:#e2e8f0'>
          <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px'>
            <div style='display:flex;align-items:center;gap:10px'>
              <h3 style='margin:0;color:#c7d2fe;font-size:16px'>🤖 Gemini Clinical AI Auditor & Second-Opinion</h3>
              {model_badge}
              {score_badge}
            </div>
            <form method='post' action='/admin/jobs/{job_id}/gemini-audit' style='margin:0'>
              <button class='btn gray' style='font-size:12px;padding:4px 10px'>🔄 Re-Analyze with Gemini</button>
            </form>
          </div>
          <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;font-size:13.5px'>
            <div style='background:rgba(0,0,0,.35);padding:10px 14px;border-radius:8px;border:1px solid rgba(255,255,255,0.06)'>
              <b style='color:#38bdf8'>📋 Preliminary Radiologist Findings ({esc(gemini_data.get('sequence_region','MRI'))}):</b>
              <ul style='margin:6px 0 0 18px;padding:0;color:#cbd5e1'>{findings_li}</ul>
            </div>
            <div style='background:rgba(0,0,0,.35);padding:10px 14px;border-radius:8px;border:1px solid rgba(255,255,255,0.06)'>
              <b style='color:#34d399'>💬 ملخص التقرير الطبي (Clinical Impression):</b>
              <p style='margin:6px 0 0;color:#e2e8f0;line-height:1.6;direction:rtl;text-align:right'>{esc(gemini_data.get('summary_ar',''))}</p>
              <small style='display:block;margin-top:6px;color:#94a3b8'>{esc(gemini_data.get('summary_en',''))}</small>
            </div>
          </div>
        </div>
        """
    else:
        err_msg = f"<small style='color:#ef4444;margin-left:10px'>{esc(gemini_data.get('error'))}</small>" if (gemini_data and gemini_data.get("error")) else ""
        gemini_html = f"""
        <div style='background:#090d1a;border:1px solid #312e81;border-radius:10px;padding:10px 16px;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px'>
          <div style='display:flex;align-items:center;gap:10px'>
            <span style='font-size:20px'>🤖</span>
            <div>
              <b style='color:#c7d2fe;font-size:14px'>Gemini Clinical AI Auditor:</b>
              <span class='muted' style='margin-left:6px'>Run instant Second-Opinion, Diagnostic Safety Audit & Preliminary Findings on this scan.</span>
              {err_msg}
            </div>
          </div>
          <form method='post' action='/admin/jobs/{job_id}/gemini-audit' style='margin:0'>
            <button class='btn ai-btn' type='submit' style='background:linear-gradient(135deg,#4f46e5,#7c3aed);border-color:#6366f1;font-weight:700'>
              🧠 Run Gemini Clinical AI Audit
            </button>
          </form>
        </div>
        """

    study_uid = ""
    src_desc = "Original Raw"
    out_desc = "RAD4AI Enhanced"
    from app.profiles import PROFILES, STRENGTHS
    prof_key = job["profile"] if "profile" in job.keys() else ""
    strength_val = job["strength"] if "strength" in job.keys() else "MEDIUM"
    prof_name = PROFILES.get(prof_key, prof_key.replace("_", " "))
    src_snum = ""
    out_snum = ""
    patient_name = ""
    try:
        with client() as api:
            s_info = api.get(f"/series/{src_sid}").json()
            o_info = api.get(f"/series/{out_sid}").json()
            src_desc = s_info.get("MainDicomTags", {}).get("SeriesDescription", src_desc)
            out_desc = o_info.get("MainDicomTags", {}).get("SeriesDescription", out_desc).replace("[AI-SWIFT]", "[RAD4AI]")
            src_snum = s_info.get("MainDicomTags", {}).get("SeriesNumber", "")
            out_snum = o_info.get("MainDicomTags", {}).get("SeriesNumber", "")
            parent_study = s_info.get("ParentStudy", "")
            if parent_study:
                st_info = api.get(f"/studies/{parent_study}").json()
                study_uid = st_info.get("MainDicomTags", {}).get("StudyInstanceUID", "")
                patient_name = st_info.get("PatientMainDicomTags", {}).get("PatientName", "")
    except Exception:
        pass

    ohif_btn = f"<a class='btn emerald' target='_blank' href='{esc(viewer_url(study_uid))}'>🖥️ Open Full Study in OHIF Viewer</a>" if study_uid else ""

    page = f"""
    <div class='study-toolbar' style='justify-content:space-between'>
      <div class='actions' style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>
        <a class='btn gray' href='/admin/studies'>← Back to Studies</a>
        {ohif_btn}
        <button type='button' class='btn ai-btn' onclick='exportComparisonSheet()'>📸 Download Clinical Comparison Report (PNG)</button>
        <form method='post' action='/admin/jobs/{job_id}/send-to-ge' style='margin:0' onsubmit="return confirm('إرسال هذه السلسلة المحسّنة الآن إلى جهاز GE Workstation (192.168.0.206)؟\n\nSend this enhanced series directly to GE Workstation?')">
          <button class='btn emerald' type='submit' style='display:inline-flex;align-items:center;gap:6px;font-weight:700;padding:8px 16px'>
            📤 Send to GE Workstation (إرسال للجهاز)
          </button>
        </form>
      </div>
      <div class='kpi-pills'>
        <span class='kpi-pill'>Patient: <b>{esc(patient_name)}</b></span>
        <span class='kpi-pill' style='background:rgba(16,185,129,0.18);border:1px solid #10b981;color:#34d399'>🔮 Profile: <b>{esc(prof_name)}</b></span>
        <span class='kpi-pill' style='background:rgba(56,189,248,0.18);border:1px solid #38bdf8;color:#38bdf8'>💪 Strength: <b>{esc(strength_val)}</b></span>
        <span class='kpi-pill'>🧬 Auto-Seq: <b>{esc(metrics.get('sequence_mode','AUTO'))}</b></span>
        <span class='kpi-pill'>🧊 Context: <b>{esc(metrics.get('context_mode','2.5D Multi-Slice'))}</b></span>
        <span class='kpi-pill'>⚡ Time: <b>{esc(metrics.get('seconds','—'))}s</b></span>
        <span class='kpi-pill'>📉 Noise Delta: <b>{esc(metrics.get('noise_change_percent','—'))}%</b></span>
        <span class='kpi-pill'>🎯 Edge Preservation: <b>{esc(metrics.get('edge_preservation_percent','—'))}%</b></span>
        <span class='badge ok'>QC: {esc(job['qc_status'])}</span>
      </div>
    </div>

    <section class='panel' style='padding:18px'>
      <div class='panel-head'>
        <h2>🔍 Interactive Clinical Comparison (Job #{job_id}): <span class='snum-badge'>S:{esc(src_snum)} {esc(src_desc)}</span> vs <span class='snum-badge derived'>S:{esc(out_snum)} {esc(out_desc)}</span></h2>
        <div class='actions'>
          <button type='button' class='btn ai-btn' id='mode-side' onclick='setMode("side")'>🪟 Side-by-Side (2x1)</button>
          <button type='button' class='btn gray' id='mode-wipe' onclick='setMode("wipe")'>↔️ Split Wipe (A/B)</button>
          <button type='button' class='btn gray' id='mode-diff' onclick='setMode("diff")'>🔥 Difference Map</button>
        </div>
      </div>

      <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;background:#070d19;padding:14px;border-radius:12px;border:1px solid #1e293b;margin-bottom:16px;color:#e2e8f0'>
        <div>
          <label style='font-size:13px;font-weight:700;display:flex;justify-content:space-between'>
            <span>🎞️ Slice Scrubber (Use Mouse Wheel):</span>
            <span id='slice-label' style='color:#38bdf8'>{mid_slice + 1} / {total_slices}</span>
          </label>
          <input type='range' id='slice-slider' min='0' max='{max(0, total_slices - 1)}' value='{mid_slice}' style='width:100%;margin-top:6px;accent-color:#3b82f6'>
        </div>
        <div>
          <label style='font-size:13px;font-weight:700;display:flex;justify-content:space-between'>
            <span>☀️ Brightness:</span>
            <span id='br-label'>100%</span>
          </label>
          <input type='range' id='br-slider' min='50' max='180' value='100' style='width:100%;margin-top:6px;accent-color:#3b82f6'>
        </div>
        <div>
          <label style='font-size:13px;font-weight:700;display:flex;justify-content:space-between'>
            <span>🌗 Contrast:</span>
            <span id='ct-label'>105%</span>
          </label>
          <input type='range' id='ct-slider' min='60' max='220' value='105' style='width:100%;margin-top:6px;accent-color:#3b82f6'>
        </div>
        <div>
          <label style='font-size:13px;font-weight:700;display:flex;justify-content:space-between'>
            <span>🔍 Zoom:</span>
            <span id='zm-label'>100%</span>
          </label>
          <input type='range' id='zm-slider' min='100' max='260' value='100' style='width:100%;margin-top:6px;accent-color:#3b82f6'>
        </div>
        <div>
          <label style='font-size:13px;font-weight:700;display:block;margin-bottom:5px'>🧊 GE AW Volume Slab Mode:</label>
          <select id='slab-select' style='width:100%;padding:6px 8px;border-radius:7px;background:#0f172a;color:#38bdf8;border:1px solid #334155;font-weight:700'>
            <option value='single'>Single Slice (2D Native)</option>
            <option value='mip3'>GE Volume MIP (3-Slice Vessel/MRV)</option>
            <option value='mip5'>GE Volume MIP (5-Slice Thick)</option>
            <option value='avg3'>GE Thick-Slice Average (3-Slice Ortho)</option>
            <option value='minip3'>GE Volume MinIP (3-Slice Dark Fluid)</option>
          </select>
        </div>
        <div>
          <label style='font-size:13px;font-weight:700;display:block;margin-bottom:5px'>🎨 GE READY View Color LUT:</label>
          <select id='lut-select' style='width:100%;padding:6px 8px;border-radius:7px;background:#0f172a;color:#34d399;border:1px solid #334155;font-weight:700'>
            <option value='gray'>Clinical Grayscale (Standard)</option>
            <option value='ge_gold'>GE AW Warm Gold (Perfusion/Ortho)</option>
            <option value='hot_iron'>GE Hot Iron (STIR Edema / Thermal)</option>
            <option value='pet_rainbow'>GE READY View Spectrum (Multi-Parametric)</option>
          </select>
        </div>
      </div>

      <div id='roi-bar' style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;background:#0b1528;border:1px solid #1e3a5f;border-radius:10px;padding:9px 14px;margin-bottom:14px;font-size:13px;color:#cbd5e1'>
        <div>🎯 <b>GE AW Interactive ROI & SNR Probe:</b> <span style='color:#94a3b8'>Click anywhere on the MRI image to measure local tissue ROI (24×24 px) Before vs. After.</span></div>
        <div id='roi-readout' style='font-family:Consolas,monospace;color:#38bdf8;font-weight:700'>Click on image to inspect tissue SNR & SD...</div>
      </div>

{gemini_html}
      <!-- Mode 1: Side by Side -->
      <div id='view-side' style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>
        <div style='background:#020617;border:2px solid #334155;border-radius:12px;overflow:hidden;text-align:center;position:relative'>
          <div style='position:absolute;top:10px;left:12px;z-index:10;background:rgba(15,23,42,.85);padding:5px 12px;border-radius:8px;border:1px solid #475569;font-weight:700;font-size:13px;color:#e2e8f0'>
            Original Raw · <span style='color:#93c5fd'>S:{esc(src_snum)}</span>
          </div>
          <div style='height:480px;display:flex;align-items:center;justify-content:center;overflow:hidden' class='wheel-zone'>
            <img id='img-src-side' class='mri-img' src='/admin/series/{src_sid}/slice/{mid_slice}.png' style='max-height:100%;max-width:100%;object-fit:contain;transition:transform .1s'>
          </div>
        </div>
        <div style='background:#020617;border:2px solid #059669;border-radius:12px;overflow:hidden;text-align:center;position:relative;box-shadow:0 0 25px rgba(16,185,129,.15)'>
          <div style='position:absolute;top:10px;left:12px;z-index:10;background:rgba(6,78,59,.92);padding:6px 14px;border-radius:8px;border:1px solid #10b981;font-weight:700;font-size:13px;color:#34d399;box-shadow:0 4px 12px rgba(0,0,0,0.6)'>
            ✨ {esc(prof_name)} [{esc(strength_val)}] · <span>S:{esc(out_snum)}</span>
          </div>
          <div style='height:480px;display:flex;align-items:center;justify-content:center;overflow:hidden' class='wheel-zone'>
            <img id='img-out-side' class='mri-img' src='/admin/series/{out_sid}/slice/{mid_slice}.png' style='max-height:100%;max-width:100%;object-fit:contain;transition:transform .1s'>
          </div>
        </div>
      </div>

      <!-- Mode 2: Split Wipe A/B -->
      <div id='view-wipe' style='display:none;background:#020617;border:2px solid #3b82f6;border-radius:12px;padding:16px;text-align:center;color:#e2e8f0'>
        <div style='margin-bottom:10px'>
          <label style='font-weight:700;font-size:13.5px'>↔️ Drag Split Wipe Slider (Left: {esc(prof_name)} [{esc(strength_val)}] | Right: Original Raw):</label>
          <input type='range' id='wipe-slider' min='0' max='100' value='50' style='width:320px;vertical-align:middle;accent-color:#10b981'>
        </div>
        <div style='position:relative;width:512px;height:512px;margin:0 auto;overflow:hidden;border-radius:10px;border:1px solid #334155' class='wheel-zone'>
          <img id='img-out-wipe' class='mri-img' src='/admin/series/{out_sid}/slice/{mid_slice}.png' style='position:absolute;top:0;left:0;width:100%;height:100%;object-fit:contain'>
          <div id='wipe-clip' style='position:absolute;top:0;left:0;width:100%;height:100%;clip-path:inset(0 0 0 50%)'>
            <img id='img-src-wipe' class='mri-img' src='/admin/series/{src_sid}/slice/{mid_slice}.png' style='width:100%;height:100%;object-fit:contain'>
          </div>
          <div id='wipe-line' style='position:absolute;top:0;bottom:0;left:50%;width:3px;background:#10b981;box-shadow:0 0 10px #10b981;pointer-events:none'></div>
        </div>
      </div>

      <!-- Mode 3: Difference Map -->
      <div id='view-diff' style='display:none;background:#020617;border:2px solid #f59e0b;border-radius:12px;padding:16px;text-align:center'>
        <p class='muted' style='color:#94a3b8'>Isolated Noise Map (Bright regions show high-frequency Rician noise removed by RAD4AI while preserving anatomical edges):</p>
        <img src='/admin/jobs/{job_id}/difference.png' style='height:460px;object-fit:contain;border-radius:10px;border:1px solid #334155'>
      </div>
    </section>

    <script>
      const srcSid = "{src_sid}", outSid = "{out_sid}", total = {total_slices};
      const sliceSlider = document.getElementById('slice-slider');
      const sliceLabel = document.getElementById('slice-label');
      const brSlider = document.getElementById('br-slider');
      const ctSlider = document.getElementById('ct-slider');
      const zmSlider = document.getElementById('zm-slider');
      const wipeSlider = document.getElementById('wipe-slider');
      const slabSelect = document.getElementById('slab-select');
      const lutSelect = document.getElementById('lut-select');

      function updateSlice(idx) {{
        idx = Math.max(0, Math.min(total - 1, parseInt(idx)));
        sliceSlider.value = idx;
        sliceLabel.textContent = (idx + 1) + ' / ' + total;
        const slab = slabSelect.value;
        const lut = lutSelect.value;
        const q = `?slab=${{encodeURIComponent(slab)}}&lut=${{encodeURIComponent(lut)}}`;
        const srcUrl = `/admin/series/${{srcSid}}/slice/${{idx}}.png${{q}}`;
        const outUrl = `/admin/series/${{outSid}}/slice/${{idx}}.png${{q}}`;
        document.getElementById('img-src-side').src = srcUrl;
        document.getElementById('img-out-side').src = outUrl;
        document.getElementById('img-src-wipe').src = srcUrl;
        document.getElementById('img-out-wipe').src = outUrl;
      }}

      sliceSlider.addEventListener('input', e => updateSlice(e.target.value));
      slabSelect.addEventListener('change', () => updateSlice(sliceSlider.value));
      lutSelect.addEventListener('change', () => updateSlice(sliceSlider.value));

      function computeRoiStats(imgEl, normX, normY) {{
        const c = document.createElement('canvas');
        const w = imgEl.naturalWidth || 256, h = imgEl.naturalHeight || 256;
        c.width = w; c.height = h;
        const ctx = c.getContext('2d');
        ctx.drawImage(imgEl, 0, 0);
        const rSize = 20;
        const cx = Math.max(rSize, Math.min(w - rSize, Math.round(normX * w)));
        const cy = Math.max(rSize, Math.min(h - rSize, Math.round(normY * h)));
        const data = ctx.getImageData(cx - rSize/2, cy - rSize/2, rSize, rSize).data;
        let vals = [];
        for (let i = 0; i < data.length; i += 4) {{
          vals.push((data[i] + data[i+1] + data[i+2]) / 3.0);
        }}
        const mean = vals.reduce((a,b) => a+b, 0) / vals.length;
        const variance = vals.reduce((a,b) => a + (b - mean)**2, 0) / vals.length;
        const sd = Math.max(Math.sqrt(variance), 0.4);
        return {{ mean: mean.toFixed(1), sd: sd.toFixed(2), snr: (mean / sd).toFixed(1) }};
      }}

      ['img-src-side', 'img-out-side'].forEach(id => {{
        const el = document.getElementById(id);
        el.style.cursor = 'crosshair';
        el.addEventListener('click', e => {{
          const rect = el.getBoundingClientRect();
          const nx = (e.clientX - rect.left) / rect.width;
          const ny = (e.clientY - rect.top) / rect.height;
          try {{
            const s1 = computeRoiStats(document.getElementById('img-src-side'), nx, ny);
            const s2 = computeRoiStats(document.getElementById('img-out-side'), nx, ny);
            const gain = (((parseFloat(s2.snr) - parseFloat(s1.snr)) / Math.max(parseFloat(s1.snr), 0.1)) * 100).toFixed(1);
            document.getElementById('roi-readout').innerHTML =
              `Original [Mean: ${{s1.mean}}, SD: ${{s1.sd}}, SNR: ${{s1.snr}}]  ➔  <span style='color:#34d399'>RAD4AI [Mean: ${{s2.mean}}, SD: ${{s2.sd}}, SNR: ${{s2.snr}} | SNR Gain: ${{gain > 0 ? '+'+gain : gain}}% ]</span>`;
          }} catch(err) {{}}
        }});
      }});
      document.querySelectorAll('.wheel-zone').forEach(z => z.addEventListener('wheel', e => {{
        e.preventDefault();
        const delta = e.deltaY > 0 ? 1 : -1;
        updateSlice(parseInt(sliceSlider.value) + delta);
      }}, {{passive: false}}));

      function updateFilters() {{
        const br = brSlider.value, ct = ctSlider.value, zm = zmSlider.value / 100.0;
        document.getElementById('br-label').textContent = br + '%';
        document.getElementById('ct-label').textContent = ct + '%';
        document.getElementById('zm-label').textContent = zmSlider.value + '%';
        document.querySelectorAll('.mri-img').forEach(img => {{
          img.style.filter = `brightness(${{br}}%) contrast(${{ct}}%)`;
          img.style.transform = `scale(${{zm}})`;
        }});
      }}
      brSlider.addEventListener('input', updateFilters);
      ctSlider.addEventListener('input', updateFilters);
      zmSlider.addEventListener('input', updateFilters);
      updateFilters();

      wipeSlider.addEventListener('input', e => {{
        const v = e.target.value;
        document.getElementById('wipe-clip').style.clipPath = `inset(0 0 0 ${{v}}%)`;
        document.getElementById('wipe-line').style.left = v + '%';
      }});

      function setMode(m) {{
        document.getElementById('view-side').style.display = m === 'side' ? 'grid' : 'none';
        document.getElementById('view-wipe').style.display = m === 'wipe' ? 'block' : 'none';
        document.getElementById('view-diff').style.display = m === 'diff' ? 'block' : 'none';
        document.getElementById('mode-side').className = m === 'side' ? 'btn ai-btn' : 'btn gray';
        document.getElementById('mode-wipe').className = m === 'wipe' ? 'btn ai-btn' : 'btn gray';
        document.getElementById('mode-diff').className = m === 'diff' ? 'btn ai-btn' : 'btn gray';
      }}

      function exportComparisonSheet() {{
        const imgSrc = document.getElementById('img-src-side');
        const imgOut = document.getElementById('img-out-side');
        const canvas = document.createElement('canvas');
        canvas.width = 1400;
        canvas.height = 820;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#060d1a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = '#0f223d';
        ctx.fillRect(0, 0, canvas.width, 84);
        ctx.fillStyle = '#38bdf8';
        ctx.font = 'bold 24px Segoe UI, Arial';
        ctx.fillText('RAD4AI 0.3T Clinical AI Enhancement Report — Hitachi AIRIS Vento', 28, 38);
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '15px Segoe UI, Arial';
        ctx.fillText(`Patient: {esc(patient_name)}   |   Job #{job_id} ({esc(job['profile'])})   |   Slice: ${{parseInt(sliceSlider.value)+1}}/${{total}}   |   Edge Preservation: {esc(metrics.get('edge_preservation_percent','—'))}%   |   Noise Delta: {esc(metrics.get('noise_change_percent','—'))}%   |   QC: {esc(job['qc_status'])}`, 28, 66);

        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 2;
        ctx.strokeRect(28, 104, 656, 656);
        ctx.strokeStyle = '#10b981';
        ctx.strokeRect(716, 104, 656, 656);

        ctx.filter = `brightness(${{brSlider.value}}%) contrast(${{ctSlider.value}}%)`;
        try {{
          ctx.drawImage(imgSrc, 36, 144, 640, 600);
          ctx.drawImage(imgOut, 724, 144, 640, 600);
        }} catch(e) {{}}
        ctx.filter = 'none';

        ctx.fillStyle = '#93c5fd';
        ctx.font = 'bold 18px Segoe UI, Arial';
        ctx.fillText(`BEFORE: Original Raw (S:{esc(src_snum)} {esc(src_desc)})`, 44, 134);
        ctx.fillStyle = '#34d399';
        ctx.fillText(`AFTER: {esc(prof_name)} [{esc(strength_val)}] (S:{esc(out_snum)} {esc(out_desc)})`, 732, 134);

        ctx.fillStyle = '#64748b';
        ctx.font = '13px Segoe UI, Arial';
        ctx.fillText('Generated by RAD4AI 0.3T AI Appliance v2.0 (2.5D Multi-Slice NAFNet + Auto-Sequence Intelligence)', 28, 795);

        const link = document.createElement('a');
        link.download = `RAD4AI_Job_{job_id}_Slice_${{parseInt(sliceSlider.value)+1}}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
      }}
    </script>
    """
    return layout(f"Interactive Comparison #{job_id}", page, error or message, bool(error))


@router.post("/series/{series_id}/ai-swift")
def quick_ai_swift(series_id: str, _: str = Depends(require_admin)):
    if not ORTHANC_ID.fullmatch(series_id):
        raise HTTPException(400, "Invalid Orthanc series ID")
    try:
        with client() as api:
            response = api.get(f"/series/{series_id}")
            response.raise_for_status()
            item = response.json()
        description = str(item.get("MainDicomTags", {}).get("SeriesDescription", ""))
        if is_derived_series(description):
            raise ValueError("Cannot reprocess an already enhanced series")
        chosen_profile = "RAD4AI_SPINE_PRO" if "RAD4AI_SPINE_PRO" in enabled_profile_names() else "AI_SWIFT_DENOISE"
        validate_selection(chosen_profile, description, len(item.get("Instances", [])), True, "MEDIUM")
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    job_id = enqueue_comparison(series_id, chosen_profile, "MEDIUM")
    return JSONResponse({"ok": True, "job_id": job_id})


@router.post("/series/{series_id}/enqueue")
def enqueue(series_id: str, profile: str = Form(...), strength: str = Form("MILD"), acknowledged: str | None = Form(None), _: str = Depends(require_admin)):
    if not ORTHANC_ID.fullmatch(series_id):
        raise HTTPException(400, "Invalid Orthanc series ID")
    if profile not in PROFILES or profile not in enabled_profile_names():
        raise HTTPException(422, "Invalid profile")
    if profile == "BASE_MILD":
        strength = "MILD"
    try:
        with client() as api:
            response = api.get(f"/series/{series_id}")
            response.raise_for_status()
            item = response.json()
        description = str(item.get("MainDicomTags", {}).get("SeriesDescription", ""))
        if is_derived_series(description):
            raise ValueError("Cannot reprocess an already enhanced series")
        validate_selection(profile, description, len(item.get("Instances", [])), bool(acknowledged), strength)
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    job_id = enqueue_comparison(series_id, profile, strength)
    return JSONResponse({"ok": True, "job_id": job_id})


@router.post("/series/{series_id}/safe-preset")
def safe_preset(series_id: str, _: str = Depends(require_admin)):
    if not ORTHANC_ID.fullmatch(series_id):
        raise HTTPException(400, "Invalid Orthanc series ID")
    try:
        with client() as api:
            response = api.get(f"/series/{series_id}")
            response.raise_for_status()
            item = response.json()
        description = str(item.get("MainDicomTags", {}).get("SeriesDescription", ""))
        if is_derived_series(description):
            raise ValueError("Cannot run enhancement on a derived series")
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    job_id = enqueue_comparison(series_id, "AI_SWIFT_DENOISE", "MEDIUM")
    return JSONResponse({"ok": True, "count": 1, "job_ids": [job_id]})


@router.post("/series/enqueue-batch")
def enqueue_batch(series_ids: list[str] = Form(...), profiles: list[str] = Form(...), strength: str = Form("MILD"), acknowledged: str | None = Form(None), _: str = Depends(require_admin)):
    unique_ids = list(dict.fromkeys(series_ids))
    if not unique_ids or len(unique_ids) > 100:
        raise HTTPException(422, "Select between 1 and 100 series")
    unique_profiles = list(dict.fromkeys(profiles))
    enabled = enabled_profile_names()
    if not unique_profiles or any(profile not in PROFILES or profile not in enabled for profile in unique_profiles):
        raise HTTPException(422, "One of the selected enhancer profiles is unavailable")
    if len(unique_ids) * len(unique_profiles) > 200:
        raise HTTPException(422, "Maximum 200 jobs per batch submission")
    checked = []
    try:
        with client() as api:
            for series_id in unique_ids:
                if not ORTHANC_ID.fullmatch(series_id):
                    raise ValueError("Invalid Series ID")
                response = api.get(f"/series/{series_id}")
                response.raise_for_status()
                item = response.json()
                description = str(item.get("MainDicomTags", {}).get("SeriesDescription", ""))
                if is_derived_series(description):
                    raise ValueError("Cannot reprocess an already enhanced series")
                for profile in unique_profiles:
                    effective_strength = "MILD" if profile == "BASE_MILD" else strength
                    validate_selection(profile, description, len(item.get("Instances", [])), bool(acknowledged), effective_strength)
                checked.append(series_id)
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    job_ids = [enqueue_comparison(series_id, profile, "MILD" if profile == "BASE_MILD" else strength)
               for series_id in checked for profile in unique_profiles]
    return JSONResponse({"ok": True, "count": len(job_ids), "job_ids": job_ids})


@router.get("/tools", response_class=HTMLResponse)
def tools_page(message: str = "", _: str = Depends(require_admin)):
    requirements = {
        "AI_SWIFT_DENOISE": "Active & Approved: 2.5D Multi-Slice RAD4AI NAFNet v2.0 with Auto-Sequence Intelligence.",
        "RAD4AI_SPINE_PRO": "Active & Approved: 2.5D High-Clarity Spine & Ortho Pro mode with Ultra-Smooth marrow/muscle denoise, micro-edge crisping, and anti-burnout coil shading correction.",
        "RAD4AI_SR_15T": "Active & Approved: Sub-pixel edge synthesis + anti-staircasing + 2.5D Rician suppression for a 1.5T high-field matrix look.",
        "RAD4AI_GE_AW": "Active & Approved: GE AW Workstation look with AIR Recon DL denoise, PURE/SCIC coil illumination equalization, and zero-floor background clamping.",
        "BASE_MILD": "Ready for baseline comparison reference.",
        "DENOISE_MEDIUM": "Ready for noisy scans; inspect fine anatomical edges.",
        "EDGE_DENOISE_MILD": "Reduces Rician noise while protecting anatomical edges.",
        "N4_BIAS": "Ready for consistent volumes (8+ slices) with visible coil shading/bias.",
        "N4_DENOISE": "Ready for combined bias field and noise; compare before approval.",
        "N4_EDGE_DENOISE": "Bias correction + edge-preserving denoise; use when coil bias is visible.",
        "CLAHE_VIEW": "Research viewing only; requires acknowledgement before running.",
        "DIPY_DWI": "Pending: requires b-values and b-vectors validation for DWI.",
        "MONAI_MODEL": "Enabled via RAD4AI ONNX NAFNet v2.0 CPU runtime.",
    }
    rows = []
    for tool in list_processing_tools():
        available = "Available" if tool["available"] else "Unavailable"
        state = "Enabled" if tool["enabled"] else "Disabled"
        automatic = "Auto eligible" if tool["auto_eligible"] else "Manual comparison only"
        if tool["profile"] == "BASE_MILD":
            control = "Protected Baseline"
        elif tool["available"]:
            control = f"<form method='post' action='/admin/tools/{esc(tool['profile'])}/toggle'><button class='btn gray'>Toggle Enable/Disable</button></form>"
        else:
            control = "Pending Requirements"
        readiness = requirements.get(tool["profile"], str(tool["notes"]))
        rows.append(f"<tr><td><b>{esc(tool['label'])}</b><br><span class='muted'>{esc(tool['profile'])}</span></td><td><span class='badge {'ok' if tool['available'] else 'bad'}'>{available}</span></td><td><span class='badge {'ok' if tool['enabled'] else 'warn'}'>{state}</span></td><td><span class='badge {'ok' if tool['auto_eligible'] else 'warn'}'>{automatic}</span></td><td>{esc(tool['notes'])}</td><td>{esc(readiness)}</td><td>{control}</td></tr>")
    body = f"<section class='panel'><h2>⚙️ Enhancement Engines & Tools Management</h2><p class='muted'>The <b>RAD4AI Denoise (NAFNet v2.0)</b> engine is active as the primary clinical enhancer.</p><div class='table-wrap'><table><tr><th>Tool / Engine</th><th>Availability</th><th>Status</th><th>Automation Safety</th><th>Guidance</th><th>Readiness / Notes</th><th>Action</th></tr>{''.join(rows)}</table></div></section>"
    return layout("Enhancement Tools", body, message)


@router.post("/tools/{profile}/toggle")
def toggle_tool(profile: str, _: str = Depends(require_admin)):
    if profile == "BASE_MILD":
        raise HTTPException(409, "Baseline cannot be disabled")
    with connect() as db:
        result = db.execute("UPDATE processing_tools SET enabled=1-enabled WHERE profile=? AND available=1", (profile,))
        if result.rowcount != 1:
            raise HTTPException(404, "Tool unavailable")
    return RedirectResponse("/admin/tools?message=" + quote("Tool status updated"), 303)


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: int, _: str = Depends(require_admin)):
    with connect() as db:
        result = db.execute("UPDATE comparison_jobs SET status='queued',error=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='failed'", (job_id,))
        if result.rowcount != 1:
            raise HTTPException(409, "Only failed jobs can be retried")
    return RedirectResponse("/admin/studies?message=" + quote("Job re-queued for processing"), 303)


@router.post("/jobs/{job_id}/evaluate")
def evaluate_job(job_id: int, decision: str = Form(...), notes: str = Form(""), _: str = Depends(require_admin)):
    if decision not in {"best", "acceptable", "rejected"}:
        raise HTTPException(422, "Invalid review decision")
    notes = notes.strip()[:1000]
    with connect() as db:
        job = db.execute("SELECT status,qc_status FROM comparison_jobs WHERE id=?", (job_id,)).fetchone()
        if job is None:
            raise HTTPException(404, "Job not found")
        if job["status"] != "ready":
            raise HTTPException(409, "Cannot evaluate job before completion")
        if job["qc_status"] not in {"PASS", "WARN"} and decision in {"best", "acceptable"}:
            raise HTTPException(409, "Quality Gate did not pass; reprocess or reject result")
        db.execute("UPDATE comparison_jobs SET decision=?,review_notes=?,reviewed_at=CURRENT_TIMESTAMP,approved=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                   (decision, notes, int(decision in {"best", "acceptable"}), job_id))
    return RedirectResponse("/admin/studies?message=" + quote("Radiologist review saved"), 303)


@router.get("/jobs/{job_id}/difference.png")
def difference_map(job_id: int, _: str = Depends(require_admin)):
    job = get_comparison_job(job_id)
    if job is None or job["status"] != "ready" or not job["output_series_id"]:
        raise HTTPException(404, "Difference Map unavailable")
    try:
        datasets = []
        with client() as api:
            for series_id in (job["source_series_id"], job["output_series_id"]):
                info = api.get(f"/series/{series_id}")
                info.raise_for_status()
                series = []
                for instance_id in info.json().get("Instances", []):
                    payload = api.get(f"/instances/{instance_id}/file")
                    payload.raise_for_status()
                    series.append(pydicom.dcmread(io.BytesIO(payload.content)))
                series.sort(key=lambda ds: (int(ds.get("InstanceNumber", 0)), str(ds.SOPInstanceUID)))
                datasets.append(series)
        if not datasets[0] or len(datasets[0]) != len(datasets[1]):
            raise ValueError("Source and output series slice counts do not match")
        index = len(datasets[0]) // 2
        source = datasets[0][index].pixel_array.astype(np.float32)
        output = datasets[1][index].pixel_array.astype(np.float32)
        if source.shape != output.shape:
            raise ValueError("Image dimensions do not match")
        difference = np.abs(output - source)
        scale = max(float(np.percentile(difference, 99)), 1.0)
        intensity = np.clip(difference / scale, 0, 1)
        rgb = np.stack((255 * intensity, 180 * intensity ** 2, np.zeros_like(intensity)), axis=-1).astype(np.uint8)
        return Response(png_rgb(rgb), media_type="image/png", headers={"Cache-Control": "no-store", "X-Research-Only": "true"})
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/jobs/{job_id}/approve")
def approve(job_id: int, _: str = Depends(require_admin)):
    with connect() as db:
        protected = db.execute("SELECT protocol_study_id FROM comparison_jobs WHERE id=?", (job_id,)).fetchone()
        if protected and protected["protocol_study_id"] is not None:
            raise HTTPException(409, "Protocol Study result is comparison-only and cannot enter Return Queue")
        result = db.execute("UPDATE comparison_jobs SET approved=1,updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='ready' AND qc_status IN ('PASS','WARN') AND decision IN ('best','acceptable')", (job_id,))
        if result.rowcount != 1:
            raise HTTPException(409, "Record a radiologist review as Best or Acceptable before approving PACS return")
        job = db.execute("SELECT * FROM comparison_jobs WHERE id=?", (job_id,)).fetchone()
    with client() as api:
        center = source_center(api, job["source_series_id"])
    if center is None or center["return_policy"] == "DISABLED":
        return RedirectResponse("/admin/studies?message=" + quote("Approved; PACS return is disabled for this center"), 303)
    if center["last_echo_status"] != "PASS":
        return RedirectResponse("/admin/studies?error=" + quote("Approved, but not queued: C-ECHO must pass on the Center page first"), 303)
    enqueue_return(job_id, center["id"], job["output_series_id"])
    audit("MANUAL_RETURN_QUEUED", "comparison_job", job_id, f"center={center['name']}", actor=_)
    return RedirectResponse("/admin/studies?message=" + quote("Approved and added to PACS Return Queue"), 303)



@router.post("/jobs/{job_id}/gemini-audit")
def gemini_audit_route(job_id: int, _: str = Depends(require_admin)):
    from app.gemini_auditor import audit_job_with_gemini
    res = audit_job_with_gemini(job_id)
    if "error" in res:
        return RedirectResponse(f"/admin/jobs/{job_id}/compare?error=" + quote(f"Gemini Audit: {res['error']}"), 303)
    return RedirectResponse(f"/admin/jobs/{job_id}/compare?message=" + quote("✅ اكتمل الفحص والتدقيق الطبي بواسطة Gemini AI بنجاح!"), 303)

@router.post("/jobs/{job_id}/send-to-ge")
def send_job_to_ge(job_id: int, _: str = Depends(require_admin)):
    job = get_comparison_job(job_id)
    if not job or not job["output_series_id"]:
        raise HTTPException(404, "Enhanced series not found for this job")
    out_sid = job["output_series_id"]
    try:
        with client() as api:
            api.put("/modalities/darelha", json={"AET": "adw", "Host": "192.168.0.206", "Port": 4006})
            r = api.post("/modalities/darelha/store", json=out_sid, timeout=120)
            r.raise_for_status()
        with connect() as db:
            db.execute("UPDATE comparison_jobs SET approved=1, updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
            db.execute("INSERT OR REPLACE INTO return_jobs(comparison_job_id, center_id, output_series_id, status, error, updated_at) VALUES(?, 8, ?, 'sent', NULL, CURRENT_TIMESTAMP)", (job_id, out_sid))
        audit("MANUAL_GE_SENT", "comparison_job", job_id, "destination=adw@192.168.0.206:4006", actor=_)
        return RedirectResponse(f"/admin/jobs/{job_id}/compare?message=" + quote("✅ تم إرسال السلسلة المحسّنة بنجاح إلى جهاز GE Workstation (192.168.0.206:4006 / adw)!"), 303)
    except Exception as exc:
        return RedirectResponse(f"/admin/jobs/{job_id}/compare?error=" + quote(f"فشل الإرسال إلى جهاز GE: {exc}"), 303)


@router.post("/series/{series_id}/send-to-ge")
def send_series_to_ge(series_id: str, _: str = Depends(require_admin)):
    if not ORTHANC_ID.match(series_id):
        raise HTTPException(400, "Invalid series ID")
    try:
        with client() as api:
            api.put("/modalities/darelha", json={"AET": "adw", "Host": "192.168.0.206", "Port": 4006})
            r = api.post("/modalities/darelha/store", json=series_id, timeout=120)
            r.raise_for_status()
        audit("MANUAL_SERIES_GE_SENT", "series", 0, f"series={series_id}&destination=adw@192.168.0.206:4006", actor=_)
        return RedirectResponse("/admin/studies?message=" + quote("✅ تم إرسال السلسلة المحسّنة بنجاح إلى جهاز GE Workstation (192.168.0.206:4006 / adw)!"), 303)
    except Exception as exc:
        return RedirectResponse("/admin/studies?error=" + quote(f"فشل الإرسال إلى جهاز GE: {exc}"), 303)

@router.post("/studies/{study_id}/delete")
def delete_study(study_id: str, _: str = Depends(require_admin)):
    if not ORTHANC_ID.match(study_id):
        raise HTTPException(400, "Invalid Orthanc study ID")
    try:
        series_ids = []
        with client() as api:
            study_resp = api.get(f"/studies/{study_id}")
            if study_resp.status_code == 200:
                series_ids = study_resp.json().get("Series", [])
                api.delete(f"/studies/{study_id}").raise_for_status()
        if series_ids:
            with connect() as db:
                for sid in series_ids:
                    job_ids = [r[0] for r in db.execute("SELECT id FROM comparison_jobs WHERE source_series_id=? OR output_series_id=?", (sid, sid)).fetchall()]
                    for jid in job_ids:
                        db.execute("DELETE FROM return_jobs WHERE comparison_job_id=?", (jid,))
                    db.execute("DELETE FROM comparison_jobs WHERE source_series_id=? OR output_series_id=?", (sid, sid))
                    db.execute("DELETE FROM inbound_series WHERE series_id=?", (sid,))
                    db.execute("DELETE FROM processed_series WHERE series_id=?", (sid,))
        audit("STUDY_DELETED", "study", study_id, f"series={len(series_ids)}", actor=_)
        return RedirectResponse("/admin/studies?message=" + quote("Study and all associated series deleted successfully"), 303)
    except Exception as exc:
        return RedirectResponse("/admin/studies?error=" + quote(f"Failed to delete study: {exc}"), 303)


@router.post("/studies/{study_id}/delete-enhanced")
def delete_enhanced_series(study_id: str, _: str = Depends(require_admin)):
    if not ORTHANC_ID.match(study_id):
        raise HTTPException(400, "Invalid Orthanc study ID")
    try:
        deleted_count = 0
        with client() as api:
            study_resp = api.get(f"/studies/{study_id}")
            if study_resp.status_code != 200:
                raise HTTPException(404, "Study not found")
            for sid in study_resp.json().get("Series", []):
                s_resp = api.get(f"/series/{sid}")
                if s_resp.status_code == 200:
                    desc = str(s_resp.json().get("MainDicomTags", {}).get("SeriesDescription", ""))
                    if is_derived_series(desc):
                        api.delete(f"/series/{sid}")
                        deleted_count += 1
                        with connect() as db:
                            job_ids = [r[0] for r in db.execute("SELECT id FROM comparison_jobs WHERE output_series_id=?", (sid,)).fetchall()]
                            for jid in job_ids:
                                db.execute("DELETE FROM return_jobs WHERE comparison_job_id=?", (jid,))
                            db.execute("DELETE FROM comparison_jobs WHERE output_series_id=?", (sid,))
                            db.execute("DELETE FROM processed_series WHERE series_id=?", (sid,))
        audit("ENHANCED_SERIES_DELETED", "study", study_id, f"count={deleted_count}", actor=_)
        return RedirectResponse("/admin/studies?message=" + quote(f"Deleted {deleted_count} enhanced series"), 303)
    except Exception as exc:
        return RedirectResponse("/admin/studies?error=" + quote(f"Failed to delete enhanced series: {exc}"), 303)


@router.post("/series/{series_id}/delete")
def delete_series(series_id: str, _: str = Depends(require_admin)):
    if not ORTHANC_ID.match(series_id):
        raise HTTPException(400, "Invalid Orthanc series ID")
    try:
        with client() as api:
            api.delete(f"/series/{series_id}").raise_for_status()
        with connect() as db:
            job_ids = [r[0] for r in db.execute("SELECT id FROM comparison_jobs WHERE source_series_id=? OR output_series_id=?", (series_id, series_id)).fetchall()]
            for jid in job_ids:
                db.execute("DELETE FROM return_jobs WHERE comparison_job_id=?", (jid,))
            db.execute("DELETE FROM comparison_jobs WHERE source_series_id=? OR output_series_id=?", (series_id, series_id))
            db.execute("DELETE FROM inbound_series WHERE series_id=?", (series_id,))
            db.execute("DELETE FROM processed_series WHERE series_id=?", (series_id,))
        audit("SERIES_DELETED", "series", series_id, actor=_)
        return RedirectResponse("/admin/studies?message=" + quote("Series deleted successfully"), 303)
    except Exception as exc:
        return RedirectResponse("/admin/studies?error=" + quote(f"Failed to delete series: {exc}"), 303)


@router.post("/series/delete-batch")
async def delete_series_batch(request: Request, _: str = Depends(require_admin)):
    form = await request.form()
    series_ids = form.getlist("series_ids")
    if not series_ids:
        raise HTTPException(400, "No series selected")
    deleted_count = 0
    with client() as api:
        for sid in series_ids:
            if ORTHANC_ID.match(sid):
                try:
                    r = api.delete(f"/series/{sid}")
                    if r.status_code == 200:
                        deleted_count += 1
                except Exception:
                    pass
    with connect() as db:
        for sid in series_ids:
            job_ids = [r[0] for r in db.execute("SELECT id FROM comparison_jobs WHERE source_series_id=? OR output_series_id=?", (sid, sid)).fetchall()]
            for jid in job_ids:
                db.execute("DELETE FROM return_jobs WHERE comparison_job_id=?", (jid,))
            db.execute("DELETE FROM comparison_jobs WHERE source_series_id=? OR output_series_id=?", (sid, sid))
            db.execute("DELETE FROM inbound_series WHERE series_id=?", (sid,))
            db.execute("DELETE FROM processed_series WHERE series_id=?", (sid,))
    audit("BATCH_SERIES_DELETED", "series_batch", str(deleted_count), actor=_)
    return JSONResponse({"count": deleted_count})


@router.post("/jobs/{job_id}/delete")
def delete_job(job_id: int, _: str = Depends(require_admin)):
    try:
        with connect() as db:
            db.execute("DELETE FROM return_jobs WHERE comparison_job_id=?", (job_id,))
            db.execute("DELETE FROM comparison_jobs WHERE id=?", (job_id,))
        audit("JOB_DELETED", "comparison_job", str(job_id), actor=_)
        return RedirectResponse("/admin/studies?message=" + quote(f"Job #{job_id} deleted successfully"), 303)
    except Exception as exc:
        return RedirectResponse("/admin/studies?error=" + quote(f"Failed to delete job: {exc}"), 303)

@router.post("/studies/delete-batch")
async def delete_studies_batch(request: Request, _: str = Depends(require_admin)):
    form = await request.form()
    study_ids = form.getlist("study_ids")
    if not study_ids:
        raise HTTPException(400, "No studies selected")
    deleted_count = 0
    all_series_ids = []
    with client() as api:
        for stid in study_ids:
            if ORTHANC_ID.match(stid):
                try:
                    s_resp = api.get(f"/studies/{stid}")
                    if s_resp.status_code == 200:
                        all_series_ids.extend(s_resp.json().get("Series", []))
                    r = api.delete(f"/studies/{stid}")
                    if r.status_code == 200:
                        deleted_count += 1
                except Exception:
                    pass
    if all_series_ids:
        with connect() as db:
            for sid in all_series_ids:
                job_ids = [r[0] for r in db.execute("SELECT id FROM comparison_jobs WHERE source_series_id=? OR output_series_id=?", (sid, sid)).fetchall()]
                for jid in job_ids:
                    db.execute("DELETE FROM return_jobs WHERE comparison_job_id=?", (jid,))
                db.execute("DELETE FROM comparison_jobs WHERE source_series_id=? OR output_series_id=?", (sid, sid))
                db.execute("DELETE FROM inbound_series WHERE series_id=?", (sid,))
                db.execute("DELETE FROM processed_series WHERE series_id=?", (sid,))
    audit("BATCH_STUDIES_DELETED", "studies_batch", str(deleted_count), actor=_)
    return JSONResponse({"count": deleted_count})


@router.post("/studies/delete-all")
def delete_all_studies(_: str = Depends(require_admin)):
    deleted_count = 0
    all_series_ids = []
    with client() as api:
        resp = api.get("/studies")
        if resp.status_code == 200:
            for stid in resp.json():
                try:
                    s_resp = api.get(f"/studies/{stid}")
                    if s_resp.status_code == 200:
                        all_series_ids.extend(s_resp.json().get("Series", []))
                    api.delete(f"/studies/{stid}")
                    deleted_count += 1
                except Exception:
                    pass
    with connect() as db:
        db.execute("DELETE FROM return_jobs")
        db.execute("DELETE FROM comparison_jobs")
        db.execute("DELETE FROM inbound_series")
        db.execute("DELETE FROM processed_series")
    audit("ALL_STUDIES_DELETED", "studies", str(deleted_count), actor=_)
    return RedirectResponse("/admin/studies?message=" + quote(f"All {deleted_count} studies deleted successfully"), 303)

