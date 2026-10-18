from __future__ import annotations

import csv
import io
import json
import os
import re
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from app.admin import esc, layout, require_admin
from app.profiles import PROFILE_INFO, PROFILES, STRENGTHS, validate_selection
from app.store import audit, connect, enabled_profile_names, list_centers

router = APIRouter(prefix="/admin/protocols", tags=["protocol-study"])
ORTHANC_ID = re.compile(r"^[0-9a-f]{8}(?:-[0-9a-f]{8}){4}$")
DECISIONS = {"ACCEPT": "Accept / مقبول", "REVIEW": "Review / مراجعة", "REJECT": "Reject / مرفوض"}


def calculate_reduction(standard_seconds: float, fast_seconds: float) -> float:
    if standard_seconds <= 0 or fast_seconds <= 0 or fast_seconds >= standard_seconds:
        raise ValueError("Fast time must be positive and shorter than Standard time")
    return round((standard_seconds - fast_seconds) * 100.0 / standard_seconds, 2)


def orthanc_client() -> httpx.Client:
    return httpx.Client(
        base_url=os.getenv("ORTHANC_URL", "http://orthanc:8042"),
        auth=(os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", "")),
        timeout=60,
    )


def ohif_url(uid: str) -> str:
    base = os.getenv("OHIF_PUBLIC_URL", "/ohif/viewer")
    return f"{base}?StudyInstanceUIDs={quote(uid, safe='.') }"


def get_protocol(protocol_id: int):
    with connect() as db:
        return db.execute(
            "SELECT p.*,c.name AS center_name FROM protocol_studies p "
            "LEFT JOIN centers c ON c.id=p.center_id WHERE p.id=?", (protocol_id,)
        ).fetchone()


def series_catalog() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    with orthanc_client() as api:
        response = api.get("/studies?expand")
        response.raise_for_status()
        for study in response.json():
            study_tags = study.get("MainDicomTags", {})
            patient_tags = study.get("PatientMainDicomTags", {})
            uid = str(study_tags.get("StudyInstanceUID", ""))
            for series_id in study.get("Series", []):
                item_response = api.get(f"/series/{series_id}")
                if item_response.status_code != 200:
                    continue
                item = item_response.json()
                tags = item.get("MainDicomTags", {})
                description = str(tags.get("SeriesDescription", "Unnamed series"))
                if description.upper().startswith("ENHANC"):
                    continue
                result.append({
                    "id": series_id,
                    "study_id": study.get("ID", ""),
                    "study_uid": uid,
                    "patient_id": str(patient_tags.get("PatientID", "")),
                    "patient_name": str(patient_tags.get("PatientName", "")),
                    "study_date": str(study_tags.get("StudyDate", "")),
                    "study_description": str(study_tags.get("StudyDescription", "")),
                    "description": description,
                    "instances": len(item.get("Instances", [])),
                })
    return result


def checked_series(api: httpx.Client, series_id: str) -> dict[str, object]:
    if not ORTHANC_ID.fullmatch(series_id):
        raise ValueError("Invalid Orthanc Series ID")
    response = api.get(f"/series/{series_id}")
    response.raise_for_status()
    item = response.json()
    description = str(item.get("MainDicomTags", {}).get("SeriesDescription", ""))
    if description.upper().startswith("ENHANC"):
        raise ValueError("Protocol inputs must be original, non-ENHANC series")
    study = api.get(f"/studies/{item['ParentStudy']}")
    study.raise_for_status()
    study_item = study.json()
    return {
        "series": item,
        "study_id": item["ParentStudy"],
        "study_uid": str(study_item.get("MainDicomTags", {}).get("StudyInstanceUID", "")),
        "patient_id": str(study_item.get("PatientMainDicomTags", {}).get("PatientID", "")),
        "description": description,
    }


def rating_options(selected: int = 3) -> str:
    return "".join(f"<option value='{value}' {'selected' if value == selected else ''}>{value}</option>" for value in range(1, 6))


@router.get("", response_class=HTMLResponse)
def protocols(message: str = "", error: str = "", _: str = Depends(require_admin)):
    try:
        catalog = series_catalog()
        options = "".join(
            f"<option value='{esc(s['id'])}'>{esc(s['patient_name'] or s['patient_id'] or 'Unknown')} · "
            f"{esc(s['study_date'])} · {esc(s['description'])} ({s['instances']} images)</option>"
            for s in catalog
        )
    except httpx.HTTPError as exc:
        options = ""
        error = error or f"Orthanc unavailable: {exc}"
    centers = "<option value=''>Unassigned / غير محدد</option>" + "".join(
        f"<option value='{c['id']}'>{esc(c['name'])}</option>" for c in list_centers()
    )
    with connect() as db:
        rows = db.execute(
            "SELECT p.*,c.name AS center_name,(SELECT COUNT(*) FROM protocol_reviews r WHERE r.protocol_study_id=p.id) AS reviews "
            "FROM protocol_studies p LEFT JOIN centers c ON c.id=p.center_id "
            "WHERE p.archived=0 ORDER BY p.updated_at DESC,p.id DESC"
        ).fetchall()
    cards = []
    for row in rows:
        state_class = "ok" if row["status"] == "ACCEPT" else "bad" if row["status"] == "REJECT" else "warn"
        cards.append(
            f"<details class='study-card' id='protocol-{row['id']}'><summary><div><b>#{row['id']} · {esc(row['title'])}</b>"
            f"<br><span class='muted'>{esc(row['center_name'] or 'Unassigned')} · {esc(row['anatomy'])} · {esc(row['sequence'])}</span></div>"
            f"<div><span class='badge {state_class}'>{esc(row['status'])}</span> "
            f"<span class='badge'>{row['reduction_percent']:.1f}% faster</span></div><span class='chevron'></span></summary>"
            f"<div class='study-body'><div class='cards'><div class='card'><span class='muted'>Standard</span><div class='number'>{row['standard_seconds']:.0f}s</div></div>"
            f"<div class='card'><span class='muted'>Fast</span><div class='number'>{row['fast_seconds']:.0f}s</div></div>"
            f"<div class='card'><span class='muted'>Reduction</span><div class='number'>{row['reduction_percent']:.1f}%</div></div>"
            f"<div class='card'><span class='muted'>Reviews</span><div class='number'>{row['reviews']}</div></div></div>"
            f"<p class='route'>STANDARD: {esc(row['standard_series_id'])}<br>FAST: {esc(row['fast_series_id'])}</p>"
            f"<div class='actions'><a class='btn' href='/admin/protocols/{row['id']}'>Open study</a>"
            f"<a class='btn gray' target='_blank' rel='noopener' href='{esc(ohif_url(row['study_instance_uid']))}'>Compare in OHIF</a>"
            f"<form method='post' action='/admin/protocols/{row['id']}/archive' onsubmit=\"return confirm('Archive protocol study?')\"><button class='btn red'>Archive</button></form></div></div></details>"
        )
    create = f"""<section class='panel'><div class='panel-head'><h2>New Protocol Acceleration Study</h2><span class='badge warn'>No Auto Return</span></div>
    <p class='muted'>اربط سلسلتين أصليتين من نفس الفحص: Standard كاملة وFast مسرعة. يمنع النظام السلاسل المشتقة والإرجاع التلقائي.</p>
    <form method='post' action='/admin/protocols' class='grid'>
      <div class='field'><label>Study title *</label><input name='title' maxlength='150' required placeholder='Knee PD Fast 25% Pilot'></div>
      <div class='field'><label>Center</label><select name='center_id'>{centers}</select></div>
      <div class='field'><label>Anatomy *</label><input name='anatomy' maxlength='80' required placeholder='Knee'></div>
      <div class='field'><label>Sequence *</label><input name='sequence' maxlength='100' required placeholder='PD COR FS'></div>
      <div class='field'><label>Scanner model</label><input name='scanner_model' maxlength='120' placeholder='Hitachi AIRIS Elite 0.3T'></div>
      <div class='field'><label>Coil</label><input name='coil' maxlength='120'></div>
      <div class='field wide'><label>Standard Series *</label><select name='standard_series_id' required><option value=''>Select original series…</option>{options}</select></div>
      <div class='field wide'><label>Fast Series *</label><select name='fast_series_id' required><option value=''>Select accelerated series…</option>{options}</select></div>
      <div class='field'><label>Standard time (seconds) *</label><input name='standard_seconds' type='number' min='1' max='86400' step='0.1' required></div>
      <div class='field'><label>Fast time (seconds) *</label><input name='fast_seconds' type='number' min='1' max='86400' step='0.1' required></div>
      <div class='field wide'><label>Protocol changes</label><textarea name='protocol_notes' maxlength='2000' placeholder='NEX 2→1; keep all other parameters unchanged'></textarea></div>
      <div class='wide'><button class='btn'>Create study</button></div>
    </form></section>"""
    body = ("<div class='actions' style='margin-bottom:14px'><a class='btn gray' href='/admin'>Dashboard</a>"
            "<a class='btn gray' href='/admin/protocols/export.csv'>Export CSV</a></div>" + create +
            "<section class='panel'><div class='panel-head'><h2>Protocol Studies</h2><span class='badge'>Research registry</span></div>" +
            ("".join(cards) if cards else "<div class='empty'>No protocol studies yet.</div>") + "</section>")
    return layout("Protocol Acceleration Study", body, error or message, bool(error))


@router.post("")
def create_protocol(
    title: str = Form(...), center_id: str = Form(""), anatomy: str = Form(...), sequence: str = Form(...),
    scanner_model: str = Form(""), coil: str = Form(""), standard_series_id: str = Form(...),
    fast_series_id: str = Form(...), standard_seconds: float = Form(...), fast_seconds: float = Form(...),
    protocol_notes: str = Form(""), actor: str = Depends(require_admin),
):
    title, anatomy, sequence = title.strip()[:150], anatomy.strip()[:80], sequence.strip()[:100]
    if not title or not anatomy or not sequence:
        raise HTTPException(422, "Title, anatomy and sequence are required")
    if standard_series_id == fast_series_id:
        raise HTTPException(422, "Standard and Fast series must be different")
    try:
        reduction = calculate_reduction(standard_seconds, fast_seconds)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        with orthanc_client() as api:
            standard = checked_series(api, standard_series_id)
            fast = checked_series(api, fast_series_id)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if standard["study_id"] != fast["study_id"] or standard["patient_id"] != fast["patient_id"]:
        raise HTTPException(422, "Standard and Fast series must belong to the same Orthanc study/patient")
    try:
        selected_center = int(center_id) if center_id else None
    except ValueError as exc:
        raise HTTPException(422, "Invalid center") from exc
    with connect() as db:
        if selected_center and db.execute("SELECT 1 FROM centers WHERE id=?", (selected_center,)).fetchone() is None:
            raise HTTPException(422, "Center not found")
        cursor = db.execute(
            "INSERT INTO protocol_studies(title,center_id,anatomy,sequence,scanner_model,coil,standard_series_id,fast_series_id,study_instance_uid,standard_seconds,fast_seconds,reduction_percent,protocol_notes,created_by) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (title, selected_center, anatomy, sequence, scanner_model.strip()[:120], coil.strip()[:120],
             standard_series_id, fast_series_id, standard["study_uid"], standard_seconds, fast_seconds,
             reduction, protocol_notes.strip()[:2000], actor),
        )
        protocol_id = int(cursor.lastrowid)
    audit("PROTOCOL_STUDY_CREATED", "protocol_study", protocol_id, f"reduction={reduction}%; standard={standard_series_id}; fast={fast_series_id}", actor)
    return RedirectResponse(f"/admin/protocols/{protocol_id}?message=" + quote("Protocol Study created; Auto Return is blocked"), 303)


@router.get("/export.csv")
def export_csv(_: str = Depends(require_admin)):
    with connect() as db:
        rows = db.execute(
            "SELECT p.*,c.name AS center_name,(SELECT COUNT(*) FROM protocol_reviews r WHERE r.protocol_study_id=p.id) AS review_count "
            "FROM protocol_studies p LEFT JOIN centers c ON c.id=p.center_id WHERE p.archived=0 ORDER BY p.id"
        ).fetchall()
    output = io.StringIO(newline="")
    fields = ["id", "title", "center_name", "anatomy", "sequence", "scanner_model", "coil", "standard_series_id", "fast_series_id", "standard_seconds", "fast_seconds", "reduction_percent", "status", "review_count", "created_at"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        data = dict(row)
        writer.writerow({key: data.get(key, "") for key in fields})
    return Response(output.getvalue(), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": "attachment; filename=protocol-studies.csv", "X-Research-Only": "true"})


@router.get("/{protocol_id}", response_class=HTMLResponse)
def protocol_detail(protocol_id: int, message: str = "", error: str = "", _: str = Depends(require_admin)):
    study = get_protocol(protocol_id)
    if study is None or study["archived"]:
        raise HTTPException(404, "Protocol Study not found")
    with connect() as db:
        jobs = db.execute("SELECT * FROM comparison_jobs WHERE protocol_study_id=? AND archived=0 ORDER BY id DESC", (protocol_id,)).fetchall()
        reviews = db.execute("SELECT * FROM protocol_reviews WHERE protocol_study_id=? ORDER BY id DESC", (protocol_id,)).fetchall()
    enabled = enabled_profile_names()
    profiles = "".join(f"<option value='{key}'>{esc(label)}</option>" for key, label in PROFILES.items() if key in enabled)
    strengths = "".join(f"<option value='{key}' {'selected' if key == 'MILD' else ''}>{esc(label)}</option>" for key, label in STRENGTHS.items())
    job_rows = []
    for job in jobs:
        metrics = ""
        if job["metrics_json"]:
            try:
                metrics = " · ".join(f"{esc(k)}={esc(v)}" for k, v in json.loads(job["metrics_json"]).items())
            except (TypeError, json.JSONDecodeError):
                metrics = esc(job["metrics_json"])
        output = esc(job["output_series_id"] or "Pending")
        job_rows.append(f"<tr><td>{job['id']}</td><td>{esc(job['profile'])}/{esc(job['strength'])}</td><td><span class='badge'>{esc(job['status'])}</span><br><span class='badge'>{esc(job['qc_status'])}</span></td><td class='route'>{output}</td><td>{metrics}</td><td>{esc(job['error'])}</td></tr>")
    review_rows = "".join(
        f"<tr><td>{esc(r['created_at'])}</td><td>{esc(r['reviewer'])}</td><td>{r['image_quality']}</td><td>{r['diagnostic_confidence']}</td><td>{r['noise']}</td><td>{r['sharpness']}</td><td>{r['artifacts']}</td><td><span class='badge'>{esc(r['decision'])}</span></td><td>{esc(r['notes'])}</td></tr>"
        for r in reviews
    ) or "<tr><td colspan='9' class='empty'>No medical reviews yet.</td></tr>"
    options = rating_options()
    decision_options = "".join(f"<option value='{key}'>{label}</option>" for key, label in DECISIONS.items())
    body = f"""<div class='actions' style='margin-bottom:14px'><a class='btn gray' href='/admin/protocols'>Protocol Studies</a><a class='btn' target='_blank' rel='noopener' href='{esc(ohif_url(study['study_instance_uid']))}'>Compare Standard/Fast/Enhanced in OHIF</a></div>
    <section class='panel'><div class='panel-head'><h2>#{study['id']} · {esc(study['title'])}</h2><span class='badge warn'>Comparison only · Return blocked</span></div>
    <div class='cards'><div class='card'><span class='muted'>Standard</span><div class='number'>{study['standard_seconds']:.0f}s</div></div><div class='card'><span class='muted'>Fast</span><div class='number'>{study['fast_seconds']:.0f}s</div></div><div class='card'><span class='muted'>Reduction</span><div class='number'>{study['reduction_percent']:.1f}%</div></div><div class='card'><span class='muted'>Decision</span><div class='number' style='font-size:18px'>{esc(study['status'])}</div></div></div>
    <p><b>{esc(study['anatomy'])} · {esc(study['sequence'])}</b><br><span class='muted'>{esc(study['scanner_model'])} · {esc(study['coil'])}</span></p><p>{esc(study['protocol_notes'])}</p>
    <p class='route'>STANDARD: {esc(study['standard_series_id'])}<br>FAST: {esc(study['fast_series_id'])}</p></section>
    <section class='panel'><h2>Enhance Fast Series</h2><p class='muted'>النتائج تبقى داخل دراسة المقارنة. قاعدة البيانات تمنع إضافتها إلى Return Queue.</p>
    <form method='post' action='/admin/protocols/{protocol_id}/enhance' class='actions'><select name='profile'>{profiles}</select><select name='strength'>{strengths}</select><label class='muted'><input type='checkbox' name='acknowledged'> Research acknowledgement</label><button class='btn'>Add comparison job</button></form>
    <div class='table-wrap'><table><tr><th>Job</th><th>Profile</th><th>Status/QC</th><th>Output</th><th>Metrics</th><th>Error</th></tr>{''.join(job_rows) or '<tr><td colspan="6" class="empty">No enhancement jobs.</td></tr>'}</table></div></section>
    <section class='panel'><h2>Medical Review / تقييم طبي</h2><p class='muted'>قيّم المقارنة مع تثبيت Window/Level والـslice نفسه. الدرجات من 1 (ضعيف) إلى 5 (ممتاز)، بينما Artifacts من 1 (قليلة) إلى 5 (شديدة).</p>
    <form method='post' action='/admin/protocols/{protocol_id}/review' class='grid'>
      <div class='field'><label>Image quality</label><select name='image_quality'>{options}</select></div><div class='field'><label>Diagnostic confidence</label><select name='diagnostic_confidence'>{options}</select></div>
      <div class='field'><label>Noise control</label><select name='noise'>{options}</select></div><div class='field'><label>Sharpness/detail</label><select name='sharpness'>{options}</select></div>
      <div class='field'><label>Artifacts severity</label><select name='artifacts'>{options}</select></div><div class='field'><label>Decision</label><select name='decision'>{decision_options}</select></div>
      <div class='field wide'><label>Clinical notes</label><textarea name='notes' maxlength='2000' required></textarea></div><div class='wide'><button class='btn'>Save immutable review</button></div>
    </form><div class='table-wrap'><table><tr><th>Date</th><th>Reviewer</th><th>Quality</th><th>Confidence</th><th>Noise</th><th>Sharpness</th><th>Artifacts</th><th>Decision</th><th>Notes</th></tr>{review_rows}</table></div></section>"""
    return layout("Protocol Study", body, error or message, bool(error))


@router.post("/{protocol_id}/enhance")
def enhance_fast(protocol_id: int, profile: str = Form(...), strength: str = Form("MILD"), acknowledged: str | None = Form(None), actor: str = Depends(require_admin)):
    study = get_protocol(protocol_id)
    if study is None or study["archived"]:
        raise HTTPException(404, "Protocol Study not found")
    if profile not in enabled_profile_names() or profile not in PROFILE_INFO:
        raise HTTPException(422, "Enhancement profile is unavailable")
    if profile == "BASE_MILD":
        strength = "MILD"
    try:
        with orthanc_client() as api:
            fast = checked_series(api, study["fast_series_id"])
        validate_selection(profile, fast["description"], len(fast["series"].get("Instances", [])), bool(acknowledged), strength)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    with connect() as db:
        existing = db.execute(
            "SELECT id FROM comparison_jobs WHERE protocol_study_id=? AND profile=? AND strength=? AND status IN ('queued','processing','ready') ORDER BY id DESC LIMIT 1",
            (protocol_id, profile, strength),
        ).fetchone()
        if existing:
            job_id = int(existing["id"])
        else:
            cursor = db.execute(
                "INSERT INTO comparison_jobs(source_series_id,profile,strength,protocol_study_id) VALUES(?,?,?,?)",
                (study["fast_series_id"], profile, strength, protocol_id),
            )
            job_id = int(cursor.lastrowid)
    audit("PROTOCOL_ENHANCEMENT_QUEUED", "protocol_study", protocol_id, f"job={job_id}; profile={profile}; strength={strength}; return=blocked", actor)
    return RedirectResponse(f"/admin/protocols/{protocol_id}?message=" + quote(f"Comparison Job {job_id} queued; Return is blocked"), 303)


@router.post("/{protocol_id}/review")
def review_protocol(protocol_id: int, image_quality: int = Form(...), diagnostic_confidence: int = Form(...), noise: int = Form(...), sharpness: int = Form(...), artifacts: int = Form(...), decision: str = Form(...), notes: str = Form(...), actor: str = Depends(require_admin)):
    if any(value not in range(1, 6) for value in (image_quality, diagnostic_confidence, noise, sharpness, artifacts)):
        raise HTTPException(422, "All ratings must be from 1 to 5")
    if decision not in DECISIONS:
        raise HTTPException(422, "Invalid review decision")
    if not notes.strip():
        raise HTTPException(422, "Clinical notes are required")
    study = get_protocol(protocol_id)
    if study is None or study["archived"]:
        raise HTTPException(404, "Protocol Study not found")
    with connect() as db:
        db.execute(
            "INSERT INTO protocol_reviews(protocol_study_id,reviewer,image_quality,diagnostic_confidence,noise,sharpness,artifacts,decision,notes) VALUES(?,?,?,?,?,?,?,?,?)",
            (protocol_id, actor, image_quality, diagnostic_confidence, noise, sharpness, artifacts, decision, notes.strip()[:2000]),
        )
        db.execute("UPDATE protocol_studies SET status=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (decision, protocol_id))
    audit("PROTOCOL_MEDICAL_REVIEW", "protocol_study", protocol_id, f"decision={decision}; quality={image_quality}; confidence={diagnostic_confidence}", actor)
    return RedirectResponse(f"/admin/protocols/{protocol_id}?message=" + quote("Medical review saved; no images were returned"), 303)


@router.post("/{protocol_id}/archive")
def archive_protocol(protocol_id: int, actor: str = Depends(require_admin)):
    with connect() as db:
        result = db.execute("UPDATE protocol_studies SET archived=1,updated_at=CURRENT_TIMESTAMP WHERE id=?", (protocol_id,))
        if result.rowcount != 1:
            raise HTTPException(404, "Protocol Study not found")
        db.execute("UPDATE comparison_jobs SET archived=1 WHERE protocol_study_id=?", (protocol_id,))
    audit("PROTOCOL_STUDY_ARCHIVED", "protocol_study", protocol_id, actor=actor)
    return RedirectResponse("/admin/protocols?message=" + quote("Protocol Study archived"), 303)
