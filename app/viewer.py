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


def png_rgb(array: np.ndarray) -> bytes:
    height, width, channels = array.shape
    if channels != 3 or array.dtype != np.uint8:
        raise ValueError("RGB uint8 image required")
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def profile_form(series_id: str, description: str, instances: int) -> str:
    recommended, reason = recommendation(description, instances)
    enabled = enabled_profile_names()
    options = "".join(
        f"<option value='{key}' {'selected' if key == recommended else ''} title='{esc(PROFILE_INFO[key]['guidance'])}'>{esc(label)}</option>"
        for key, label in PROFILES.items() if key in enabled
    )
    strengths = "".join(f"<option value='{key}' {'selected' if key == 'MILD' else ''}>{esc(label)}</option>" for key, label in STRENGTHS.items())
    return (f"<div class='series-choice'><input class='series-check' type='checkbox' "
            f"value='{series_id}' aria-label='اختيار {esc(description)}'>"
            f"<form method='post' action='/admin/series/{series_id}/enqueue' class='actions enqueue-form'>"
            f"<select name='profile' style='padding:8px;border:1px solid #60728b;border-radius:7px'>{options}</select>"
            f"<select name='strength' style='padding:8px;border:1px solid #60728b;border-radius:7px'>{strengths}</select>"
            "<label class='muted' title='مطلوب عند CLAHE أو المستوى القوي'><input type='checkbox' name='acknowledged'> إقرار بحثي</label>"
            f"<button class='btn gray'>Add to Queue</button></form>"
            f"<form method='post' action='/admin/series/{series_id}/safe-preset' class='preset-form'><button class='btn'>مقارنة محافظة تلقائية</button></form>"
            f"</div><div class='hint'>التوصية: {esc(PROFILES[recommended])} — {esc(reason)}</div>")


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
            response = api.get("/studies?expand")
            response.raise_for_status()
            rows = []
            for study in response.json():
                main = study.get("MainDicomTags", {})
                patient = study.get("PatientMainDicomTags", {})
                cells, enhanced_count, thumbnail_series = [], 0, ""
                for series_id in study.get("Series", []):
                    series_response = api.get(f"/series/{series_id}")
                    if series_response.status_code != 200:
                        continue
                    item = series_response.json()
                    tags = item.get("MainDicomTags", {})
                    description = str(tags.get("SeriesDescription", "بدون اسم"))
                    enhanced = description.upper().startswith("ENHANC")
                    enhanced_count += int(enhanced)
                    if not enhanced and not thumbnail_series:
                        thumbnail_series = series_id
                    job = latest.get(series_id)
                    if enhanced:
                        action = "<span class='badge ok'>Derived</span>"
                    elif job and job["status"] in {"queued", "processing"}:
                        action = f"<span class='badge warn'>{esc(job['profile'])}/{esc(job['strength'])}: {esc(job['status'])}</span>"
                    else:
                        action = profile_form(series_id, description, len(item.get("Instances", [])))
                    cells.append(f"<div style='padding:8px 0;border-bottom:1px solid #eee'><b>{esc(description)}</b> "
                                 f"<span class='muted'>({len(item.get('Instances', []))} images)</span>{action}</div>")
                uid = str(main.get("StudyInstanceUID", ""))
                open_link = f"<a class='btn' target='_blank' rel='noopener' href='{esc(viewer_url(uid))}'>Open / Compare in OHIF</a>" if uid else "UID غير موجود"
                thumb = f"<img class='thumb' loading='lazy' src='/admin/series/{thumbnail_series}/thumbnail' alt='Study thumbnail'>" if thumbnail_series else "<div class='thumb'></div>"
                rows.append(f"<details class='study-card' data-study='{esc(study.get('ID',''))}'><summary>{thumb}<div><b>{esc(patient.get('PatientName','')) or 'Unknown patient'}</b><br><span class='muted'>{esc(patient.get('PatientID',''))} · {esc(main.get('StudyDescription',''))} · {esc(main.get('StudyDate',''))}</span><br><span class='badge {'ok' if enhanced_count else 'warn'}'>{enhanced_count} Enhanced</span></div><span class='chevron'></span></summary><div class='study-body'><div class='study-toolbar'><button type='button' class='btn gray select-study'>تحديد الأصل / Select originals</button>{open_link}</div>{''.join(cells)}</div></details>")
    except httpx.HTTPError as exc:
        return layout("الحالات والمقارنة", "<section class='panel'><h2>تعذر تحميل الحالات</h2></section>", str(exc), True)

    studies_html = "".join(rows) or "<div class='empty'>لا توجد دراسات / No studies.</div>"
    job_rows = []
    for job in jobs:
        metrics = json.loads(job["metrics_json"]) if job["metrics_json"] else {}
        metrics_text = " · ".join(f"{esc(k)}={esc(v)}" for k, v in metrics.items())
        try:
            qc_reasons = " · ".join(json.loads(job["qc_reasons"])) if job["qc_reasons"] else ""
        except (json.JSONDecodeError, TypeError):
            qc_reasons = str(job["qc_reasons"] or "")
        qc_class = "ok" if job["qc_status"] == "PASS" else "bad" if job["qc_status"] == "FAIL" else "warn"
        controls = ""
        if job["status"] == "ready":
            decisions = {"": "لم يُراجع", "best": "الأفضل", "acceptable": "مقبول", "rejected": "مرفوض"}
            options = "".join(f"<option value='{key}' {'selected' if job['decision'] == key else ''}>{label}</option>" for key, label in decisions.items())
            controls = (f"<a class='btn gray' target='_blank' href='/admin/jobs/{job['id']}/difference.png'>Difference Map</a>"
                        f"<form method='post' action='/admin/jobs/{job['id']}/evaluate' class='review-form'>"
                        f"<select name='decision' required>{options}</select>"
                        f"<input name='notes' maxlength='1000' placeholder='ملاحظة الطبيب' value='{esc(job['review_notes'])}'>"
                        "<button class='btn'>حفظ التقييم</button></form>"
                        + (f"<a class='btn gray' href='/admin/protocols/{job['protocol_study_id']}'>Protocol Study · Return blocked</a>" if job["protocol_study_id"] else
                           f"<form method='post' action='/admin/jobs/{job['id']}/approve' onsubmit=\"return confirm('اعتماد النتيجة وإضافتها لطابور الإرجاع حسب سياسة المركز؟')\"><button class='btn'>Approve & Return</button></form>"))
        elif job["status"] == "failed":
            controls = f"<form method='post' action='/admin/jobs/{job['id']}/retry'><button class='btn gray'>Retry</button></form>"
        decision_text = {"best": "الأفضل", "acceptable": "مقبول", "rejected": "مرفوض"}.get(job["decision"], "لم يُراجع")
        job_rows.append(f"<tr><td>{job['id']}</td><td>{esc(job['profile'])}<br><span class='muted'>{esc(job['strength'])}</span></td><td><span class='badge {'ok' if job['status']=='ready' else 'bad' if job['status']=='failed' else 'warn'}'>{esc(job['status'])}</span></td><td><span class='badge {qc_class}'>{esc(job['qc_status'])}</span><div class='hint'>{esc(qc_reasons)}</div></td><td>{esc(metrics_text)}</td><td><span class='badge {'ok' if job['decision'] in {'best','acceptable'} else 'bad' if job['decision']=='rejected' else 'warn'}'>{decision_text}</span></td><td>{esc(job['error'])}</td><td>{controls}</td></tr>")
    jobs_html = "".join(job_rows) or "<tr><td colspan='8' class='empty'>لا توجد مهام مطابقة للفلاتر.</td></tr>"
    enabled = enabled_profile_names()
    bulk_options = "".join(f"<label class='tool-check'><input type='checkbox' name='bulk-tool' value='{key}' {'checked' if key == 'EDGE_DENOISE_MILD' else ''}> {esc(label)}</label>" for key, label in PROFILES.items() if key in enabled)
    strength_options = "".join(f"<option value='{key}' {'selected' if key == 'MILD' else ''}>{esc(label)}</option>" for key, label in STRENGTHS.items())
    bulk_bar = ("<div id='queue-toast' class='queue-toast' hidden></div><div id='bulk-bar' class='bulk-bar' hidden><b><span id='selected-count'>0</span> سلاسل محددة</b>"
                f"<div class='bulk-tools'>{bulk_options}</div><select id='bulk-strength'>{strength_options}</select>"
                "<label class='muted'><input id='bulk-ack' type='checkbox'> إقرار CLAHE/Strong بحثي</label>"
                "<button type='button' class='btn' id='bulk-enqueue'>إضافة المحدد للـQueue</button>"
                "<button type='button' class='btn gray' id='clear-selection'>إلغاء التحديد</button><span id='bulk-status'></span></div>")
    script = """<script>
const checks=()=>[...document.querySelectorAll('.series-check:checked')];
const bar=document.getElementById('bulk-bar'),count=document.getElementById('selected-count'),statusBox=document.getElementById('bulk-status'),toast=document.getElementById('queue-toast');
function showToast(message,isError=false){toast.textContent=message;toast.className='queue-toast '+(isError?'error':'ok');toast.hidden=false;setTimeout(()=>toast.hidden=true,5000);}
function syncBar(){const n=checks().length;count.textContent=n;bar.hidden=!n;}
document.querySelectorAll('.series-check').forEach(c=>c.addEventListener('change',syncBar));
document.querySelectorAll('.select-study').forEach(button=>button.addEventListener('click',()=>{const items=[...button.closest('.study-card').querySelectorAll('.series-check:not(:disabled)')];const select=!items.every(c=>c.checked);items.forEach(c=>c.checked=select);syncBar();}));
document.getElementById('clear-selection').addEventListener('click',()=>{document.querySelectorAll('.series-check').forEach(c=>c.checked=false);syncBar();});
document.querySelectorAll('.enqueue-form').forEach(form=>form.addEventListener('submit',async event=>{event.preventDefault();const button=form.querySelector('button');button.disabled=true;try{const response=await fetch(form.action,{method:'POST',body:new FormData(form),headers:{'X-Requested-With':'fetch'}});const data=await response.json();if(!response.ok)throw new Error(data.detail||'تعذر إضافة المهمة');button.textContent='تمت الإضافة ✓';form.closest('.series-choice').querySelector('.series-check').disabled=true;showToast(`تمت إضافة Job ${data.job_id} إلى Queue`);}catch(error){showToast(error.message,true);}finally{button.disabled=false;}}));
document.querySelectorAll('.preset-form').forEach(form=>form.addEventListener('submit',async event=>{event.preventDefault();const button=form.querySelector('button');button.disabled=true;try{const response=await fetch(form.action,{method:'POST',headers:{'X-Requested-With':'fetch'}});const data=await response.json();if(!response.ok)throw new Error(data.detail||'تعذر إضافة المقارنة');button.textContent='تمت الإضافة ✓';showToast(`تمت إضافة ${data.count} نتائج للمقارنة`);}catch(error){showToast(error.message,true);}finally{button.disabled=false;}}));
document.getElementById('bulk-enqueue').addEventListener('click',async()=>{const selected=checks(),profiles=[...document.querySelectorAll("input[name='bulk-tool']:checked")];if(!selected.length)return;if(!profiles.length){showToast('اختر محسّنًا واحدًا على الأقل',true);return;}const button=document.getElementById('bulk-enqueue'),data=new FormData();selected.forEach(c=>data.append('series_ids',c.value));profiles.forEach(p=>data.append('profiles',p.value));data.append('strength',document.getElementById('bulk-strength').value);if(document.getElementById('bulk-ack').checked)data.append('acknowledged','on');button.disabled=true;statusBox.textContent=' جارٍ الإضافة...';try{const response=await fetch('/admin/series/enqueue-batch',{method:'POST',body:data,headers:{'X-Requested-With':'fetch'}});const result=await response.json();if(!response.ok)throw new Error(result.detail||'تعذر إضافة المهام');selected.forEach(c=>{c.checked=false;c.disabled=true;});statusBox.textContent='';showToast(`تمت إضافة ${result.count} مهمة إلى Queue`);syncBar();}catch(error){statusBox.textContent='';showToast(error.message,true);}finally{button.disabled=false;}});
document.querySelectorAll('.study-card').forEach(card=>{const key='mri-open-'+card.dataset.study;card.open=sessionStorage.getItem(key)==='1';card.addEventListener('toggle',()=>sessionStorage.setItem(key,card.open?'1':'0'));});
</script>"""
    status_options = "".join(f"<option value='{value}' {'selected' if job_status == value else ''}>{label}</option>" for value, label in [("", "كل الحالات"), ("queued", "Queued"), ("processing", "Processing"), ("ready", "Ready"), ("failed", "Failed")])
    profile_options = "<option value=''>كل المحسنات</option>" + "".join(f"<option value='{key}' {'selected' if profile_filter == key else ''}>{esc(label)}</option>" for key, label in PROFILES.items())
    filters = f"<form method='get' class='actions queue-filters'><select name='job_status'>{status_options}</select><select name='profile_filter'>{profile_options}</select><button class='btn gray'>تصفية</button><a class='btn gray' href='/admin/studies'>مسح</a></form>"
    body = f"<div style='margin-bottom:14px'><a class='btn gray' href='/admin'>العودة للوحة التحكم</a></div><section class='panel'><div class='panel-head'><h2>الحالات واختيار المحسّن</h2><span class='badge warn'>Manual review by default</span></div><p class='muted'>كل دراسة مجمعة في بطاقة. افتح البطاقة ثم حدد السلاسل؛ الإضافة تتم دون إعادة تحميل الصفحة.</p><div class='study-toolbar'><button class='btn gray' type='button' onclick=\"document.querySelectorAll('.study-card').forEach(x=>x.open=true)\">توسيع الكل / Expand all</button><button class='btn gray' type='button' onclick=\"document.querySelectorAll('.study-card').forEach(x=>x.open=false)\">طي الكل / Collapse all</button></div>{studies_html}</section><section class='panel'><div class='panel-head'><h2>Processing Queue & Quality Gate</h2>{filters}</div><div class='table-wrap'><table><tr><th>Job</th><th>Profile</th><th>Status</th><th>Quality Gate</th><th>Metrics</th><th>تقييم الطبيب</th><th>Error</th><th>الإجراء</th></tr>{jobs_html}</table></div></section>{bulk_bar}{script}"
    body = body.replace("</a></div>", "</a> <a class='btn gray' href='/admin/tools'>إدارة أدوات التحسين</a></div>", 1)
    return layout("الحالات والمقارنة", body, error or message, bool(error))


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
        validate_selection(profile, description, len(item.get("Instances", [])), bool(acknowledged), strength)
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    job_id = enqueue_comparison(series_id, profile, strength)
    return JSONResponse({"ok": True, "job_id": job_id})


@router.post("/series/{series_id}/safe-preset")
def safe_preset(series_id: str, _: str = Depends(require_admin)):
    if not ORTHANC_ID.fullmatch(series_id):
        raise HTTPException(400, "Invalid Orthanc series ID")
    enabled = enabled_profile_names()
    plan = [("BASE_MILD", "MILD"), ("EDGE_DENOISE_MILD", "MILD"),
            ("EDGE_DENOISE_MILD", "MEDIUM")]
    if any(profile not in enabled for profile, _strength in plan):
        raise HTTPException(422, "فعّل Baseline وEdge Denoise لتشغيل المقارنة المحافظة")
    try:
        with client() as api:
            response = api.get(f"/series/{series_id}")
            response.raise_for_status()
            item = response.json()
        description = str(item.get("MainDicomTags", {}).get("SeriesDescription", ""))
        if description.upper().startswith("ENHANC"):
            raise ValueError("لا يمكن تشغيل Preset على نتيجة مشتقة")
        for profile, strength in plan:
            validate_selection(profile, description, len(item.get("Instances", [])), False, strength)
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    job_ids = [enqueue_comparison(series_id, profile, strength) for profile, strength in plan]
    return JSONResponse({"ok": True, "count": len(job_ids), "job_ids": job_ids})


@router.post("/series/enqueue-batch")
def enqueue_batch(series_ids: list[str] = Form(...), profiles: list[str] = Form(...), strength: str = Form("MILD"), acknowledged: str | None = Form(None), _: str = Depends(require_admin)):
    unique_ids = list(dict.fromkeys(series_ids))
    if not unique_ids or len(unique_ids) > 100:
        raise HTTPException(422, "اختر من 1 إلى 100 سلسلة")
    unique_profiles = list(dict.fromkeys(profiles))
    enabled = enabled_profile_names()
    if not unique_profiles or any(profile not in PROFILES or profile not in enabled for profile in unique_profiles):
        raise HTTPException(422, "أحد المحسّنات المحددة غير متاح")
    if len(unique_ids) * len(unique_profiles) > 200:
        raise HTTPException(422, "الحد الأقصى 200 مهمة في الإضافة الواحدة")
    checked = []
    try:
        with client() as api:
            for series_id in unique_ids:
                if not ORTHANC_ID.fullmatch(series_id):
                    raise ValueError("معرف Series غير صحيح")
                response = api.get(f"/series/{series_id}")
                response.raise_for_status()
                item = response.json()
                description = str(item.get("MainDicomTags", {}).get("SeriesDescription", ""))
                if description.upper().startswith("ENHANC"):
                    raise ValueError("لا يمكن إعادة معالجة سلسلة ENHANC مشتقة")
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
        "BASE_MILD": "جاهز للاستخدام كمرجع للمقارنة.",
        "DENOISE_MEDIUM": "جاهز للصور ذات الضوضاء الواضحة؛ راجع الحواف الدقيقة.",
        "EDGE_DENOISE_MILD": "موصى به كبداية: يقلل الضوضاء مع حماية أفضل للحواف.",
        "N4_BIAS": "جاهز للـVolume المتناسق (8 شرائح أو أكثر) عند وجود Bias مرئي.",
        "N4_DENOISE": "جاهز عند اجتماع Bias وضوضاء؛ يحتاج مقارنة قبل الاعتماد.",
        "N4_EDGE_DENOISE": "Bias وضوضاء مع حماية الحواف؛ استخدمه فقط عند وجود Bias مرئي.",
        "CLAHE_VIEW": "جاهز للعرض البحثي فقط ويحتاج إقرارًا عند التشغيل.",
        "DIPY_DWI": "ناقص: قراءة b-values وb-vectors والتحقق من DWI قبل إضافة المحرك.",
        "MONAI_MODEL": "ناقص: ملف نموذج 0.3T موثّق، manifest، preprocessing واختبار قبول.",
    }
    rows = []
    for tool in list_processing_tools():
        available = "متاح" if tool["available"] else "غير جاهز"
        state = "مفعّل" if tool["enabled"] else "متوقف"
        automatic = "Auto eligible" if tool["auto_eligible"] else "Manual comparison only"
        if tool["profile"] == "BASE_MILD":
            control = "محمي كمرجع"
        elif tool["available"]:
            control = f"<form method='post' action='/admin/tools/{esc(tool['profile'])}/toggle'><button class='btn gray'>تشغيل/إيقاف</button></form>"
        else:
            control = "ينتظر المتطلبات"
        readiness = requirements.get(tool["profile"], str(tool["notes"]))
        rows.append(f"<tr><td><b>{esc(tool['label'])}</b><br><span class='muted'>{esc(tool['profile'])}</span></td><td><span class='badge {'ok' if tool['available'] else 'bad'}'>{available}</span></td><td><span class='badge {'ok' if tool['enabled'] else 'warn'}'>{state}</span></td><td><span class='badge {'ok' if tool['auto_eligible'] else 'warn'}'>{automatic}</span></td><td>{esc(tool['notes'])}</td><td>{esc(readiness)}</td><td>{control}</td></tr>")
    body = f"<div style='margin-bottom:14px'><a class='btn gray' href='/admin/studies'>العودة للحالات</a></div><section class='panel'><h2>إدارة أدوات التحسين</h2><p class='muted'>تعطيل الأداة يخفيها من الاختيارات الجديدة. حالة «غير جاهز» تمنع تشغيل أداة ناقصة أو غير موثقة. Manual comparison only تعني أنها لا تدخل Auto Processing أو Auto Return.</p><div class='table-wrap'><table><tr><th>الأداة</th><th>التوفر</th><th>الحالة</th><th>Automation Safety</th><th>الإرشاد</th><th>الجاهزية/المطلوب</th><th>التحكم</th></tr>{''.join(rows)}</table></div></section>"
    return layout("إدارة أدوات التحسين", body, message)


@router.post("/tools/{profile}/toggle")
def toggle_tool(profile: str, _: str = Depends(require_admin)):
    if profile == "BASE_MILD":
        raise HTTPException(409, "Baseline cannot be disabled")
    with connect() as db:
        result = db.execute("UPDATE processing_tools SET enabled=1-enabled WHERE profile=? AND available=1", (profile,))
        if result.rowcount != 1:
            raise HTTPException(404, "Tool unavailable")
    return RedirectResponse("/admin/tools?message=" + quote("تم تحديث حالة الأداة"), 303)


@router.post("/jobs/{job_id}/retry")
def retry(job_id: int, _: str = Depends(require_admin)):
    with connect() as db:
        db.execute("UPDATE comparison_jobs SET status='queued',error=NULL,approved=0,qc_status='PENDING',qc_reasons='',decision='',review_notes='',reviewed_at=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
    return RedirectResponse("/admin/studies?message=" + quote("Job returned to queue"), 303)


@router.post("/jobs/{job_id}/evaluate")
def evaluate_job(job_id: int, decision: str = Form(...), notes: str = Form(""), _: str = Depends(require_admin)):
    if decision not in {"best", "acceptable", "rejected"}:
        raise HTTPException(422, "قرار التقييم غير صحيح")
    notes = notes.strip()[:1000]
    with connect() as db:
        job = db.execute("SELECT status,qc_status FROM comparison_jobs WHERE id=?", (job_id,)).fetchone()
        if job is None:
            raise HTTPException(404, "Job غير موجود")
        if job["status"] != "ready":
            raise HTTPException(409, "لا يمكن تقييم النتيجة قبل اكتمالها")
        if job["qc_status"] not in {"PASS", "WARN"} and decision in {"best", "acceptable"}:
            raise HTTPException(409, "Quality Gate غير ناجح أو لم يُحسب؛ أعد المعالجة أو ارفض النتيجة")
        db.execute("UPDATE comparison_jobs SET decision=?,review_notes=?,reviewed_at=CURRENT_TIMESTAMP,approved=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                   (decision, notes, int(decision in {"best", "acceptable"}), job_id))
    return RedirectResponse("/admin/studies?message=" + quote("تم حفظ تقييم الطبيب؛ لم يتم إرسال النتيجة"), 303)


@router.get("/jobs/{job_id}/difference.png")
def difference_map(job_id: int, _: str = Depends(require_admin)):
    job = get_comparison_job(job_id)
    if job is None or job["status"] != "ready" or not job["output_series_id"]:
        raise HTTPException(404, "Difference Map غير متاحة")
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
            raise ValueError("السلسلتان غير متطابقتين")
        index = len(datasets[0]) // 2
        source = datasets[0][index].pixel_array.astype(np.float32)
        output = datasets[1][index].pixel_array.astype(np.float32)
        if source.shape != output.shape:
            raise ValueError("أبعاد الصور غير متطابقة")
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
            raise HTTPException(409, "سجل تقييم الطبيب كأفضل أو مقبول قبل اعتماد الإرجاع")
        job = db.execute("SELECT * FROM comparison_jobs WHERE id=?", (job_id,)).fetchone()
    with client() as api:
        center = source_center(api, job["source_series_id"])
    if center is None or center["return_policy"] == "DISABLED":
        return RedirectResponse("/admin/studies?message=" + quote("تم الاعتماد؛ الإرجاع غير مفعّل لهذا المركز"), 303)
    if center["last_echo_status"] != "PASS":
        return RedirectResponse("/admin/studies?error=" + quote("تم الاعتماد لكن لم يُرسل: يجب نجاح C-ECHO من صفحة المركز أولًا"), 303)
    enqueue_return(job_id, center["id"], job["output_series_id"])
    audit("MANUAL_RETURN_QUEUED", "comparison_job", job_id, f"center={center['name']}", actor=_)
    return RedirectResponse("/admin/studies?message=" + quote("تم الاعتماد وإضافة النتيجة إلى Return Queue"), 303)
