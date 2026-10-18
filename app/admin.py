from __future__ import annotations

import html
import ipaddress
import os
import re
import secrets
import sqlite3
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.store import (connect, dashboard_counts, get_center, get_pending_sender,
                       auto_eligible_profile_names, enabled_profile_names, list_centers, list_inbound_series, list_pending_senders,
                       list_queue, list_return_jobs, pipeline_counts, slugify)
from app.profiles import PROFILES as ENHANCE_PROFILES, STRENGTHS
from app.system_health import snapshot

router = APIRouter(prefix="/admin", tags=["admin"])
security = HTTPBasic()
AE_RE = re.compile(r"^[A-Z0-9_]{1,16}$")
PROFILES = {"BASE_ONLY": "BASE فقط — متاح", "N4_ONLY": "N4 فقط — قريبًا", "BASE_AND_N4": "BASE + N4 — قريبًا"}
NETWORK_MODES = {"UNKNOWN": "غير محدد / Unknown", "VPN": "VPN — موصى به", "PUBLIC_STATIC": "Public Static IP", "NAT_FORWARD": "Behind NAT + Port Forward", "AGENT_PULL": "Agent Pull", "NO_RETURN": "No Return"}
RETURN_POLICIES = {"MANUAL": "مراجعة يدوية / Manual", "AUTO_PASS": "تلقائي عند PASS / Auto if PASS", "DISABLED": "إيقاف الإرجاع / Disabled"}


def require_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    valid = secrets.compare_digest(credentials.username, os.getenv("ADMIN_USERNAME", "mriadmin"))
    valid &= secrets.compare_digest(credentials.password, os.getenv("ADMIN_PASSWORD", ""))
    if not valid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"})
    return credentials.username


def esc(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def valid_ip(value: str, label: str, required: bool = True) -> str:
    value = value.strip()
    if not value and not required:
        return ""
    try:
        return str(ipaddress.ip_address(value))
    except ValueError as exc:
        raise ValueError(f"{label}: عنوان IP غير صحيح") from exc


def valid_aet(value: str, label: str) -> str:
    value = value.strip().upper()
    if not AE_RE.fullmatch(value):
        raise ValueError(f"{label}: استخدم 1–16 حرفًا من A-Z أو أرقام أو _ فقط")
    return value


def clean(data: dict[str, object]) -> dict[str, object]:
    name = str(data["name"]).strip()
    if not name:
        raise ValueError("اسم المركز مطلوب")
    port = int(data["return_port"])
    if not 1 <= port <= 65535:
        raise ValueError("منفذ الإرجاع يجب أن يكون بين 1 و65535")
    profile = str(data.get("processing_profile", "BASE_ONLY"))
    if profile not in PROFILES:
        raise ValueError("Processing Profile غير صحيح")
    # Do not silently advertise an unavailable processing engine.
    if profile != "BASE_ONLY":
        raise ValueError("محرك N4 لم يُفعّل بعد؛ اختر BASE فقط في هذه النسخة")
    strength = str(data.get("field_strength", "0.3")).strip()
    network_mode = str(data.get("network_mode", "UNKNOWN"))
    return_policy = str(data.get("return_policy", "MANUAL"))
    if network_mode not in NETWORK_MODES or return_policy not in RETURN_POLICIES:
        raise ValueError("إعداد الشبكة أو سياسة الإرجاع غير صحيح")
    if network_mode in {"UNKNOWN", "AGENT_PULL", "NO_RETURN"} and return_policy == "AUTO_PASS":
        raise ValueError("الإرجاع التلقائي يحتاج VPN أو Public IP/Port Forward مختبرًا")
    attempts = int(data.get("max_return_attempts", 3))
    if not 1 <= attempts <= 10:
        raise ValueError("عدد محاولات الإرجاع يجب أن يكون بين 1 و10")
    auto_profile = str(data.get("auto_profile", "EDGE_DENOISE_MILD"))
    auto_strength = str(data.get("auto_strength", "MILD"))
    if auto_profile not in ENHANCE_PROFILES or auto_strength not in STRENGTHS:
        raise ValueError("Branch enhancement profile غير صحيح")
    auto_enabled = bool(data.get("auto_enhance_enabled"))
    if auto_enabled and auto_profile not in enabled_profile_names():
        raise ValueError("المحسن المختار غير مفعّل في صفحة Tools")
    if auto_enabled and (auto_profile not in auto_eligible_profile_names() or auto_strength == "STRONG"):
        raise ValueError("هذا المحسن/المستوى للمقارنة اليدوية فقط ولا يسمح به في التحسين التلقائي")
    return {
        "name": name, "slug": slugify(name),
        "source_ip": valid_ip(str(data["source_ip"]), "IP الاتصال/VPN"),
        "public_ip": valid_ip(str(data.get("public_ip", "")), "Public IP", False),
        "public_ip_static": int(bool(data.get("public_ip_static"))),
        "source_aet": valid_aet(str(data["source_aet"]), "Calling AE"),
        "return_ip": valid_ip(str(data["return_ip"]), "Return IP"),
        "return_aet": valid_aet(str(data["return_aet"]), "Return AE"),
        "return_port": port,
        "pacs_internal_ip": valid_ip(str(data.get("pacs_internal_ip", "")), "PACS Internal IP", False),
        "device_manufacturer": str(data.get("device_manufacturer", "")).strip()[:100],
        "device_model": str(data.get("device_model", "")).strip()[:100],
        "field_strength": strength[:10], "anatomy": str(data.get("anatomy", "")).strip()[:250],
        "sequences": str(data.get("sequences", "")).strip()[:500],
        "notes": str(data.get("notes", "")).strip()[:1000], "processing_profile": profile,
        "network_mode": network_mode, "return_policy": return_policy,
        "vpn_cidr": str(data.get("vpn_cidr", "")).strip()[:100],
        "max_return_attempts": attempts,
        "auto_enhance_enabled": int(auto_enabled),
        "auto_profile": auto_profile, "auto_strength": auto_strength,
    }


STYLE = """
:root{--nav:#0b172a;--blue:#1769e0;--bg:#f3f6fa;--card:#fff;--text:#172033;--muted:#667085;--green:#087443;--red:#b42318;--amber:#b54708}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Tahoma,Arial,sans-serif}.top{background:var(--nav);color:#fff;padding:18px 4%;display:flex;justify-content:space-between}.top h1{font-size:21px;margin:0}.top small{color:#bdc8d9}.wrap{width:min(1400px,94%);margin:24px auto}.notice{padding:12px 16px;border-radius:9px;margin-bottom:15px;background:#e9f2ff;color:#174ea6}.error{background:#feeceb;color:var(--red)}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,.panel{background:#fff;border:1px solid #e1e7ef;border-radius:12px;box-shadow:0 2px 8px #1822300c}.card{padding:17px}.number{font-size:28px;font-weight:bold}.muted{color:var(--muted);font-size:13px}.panel{margin-top:18px;padding:19px}.panel-head{display:flex;justify-content:space-between;align-items:center;gap:12px}.panel h2{margin:0 0 14px;font-size:19px}.btn{display:inline-block;border:0;border-radius:7px;padding:9px 13px;background:var(--blue);color:#fff;text-decoration:none;cursor:pointer;font-weight:bold}.btn:disabled{opacity:.55;cursor:not-allowed}.gray{background:#e9edf3;color:#27364d}.red{background:#fee4e2;color:var(--red)}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:11px 9px;text-align:right;border-bottom:1px solid #e7ebf0;vertical-align:top}th{background:#f7f9fc}.badge{display:inline-block;border-radius:99px;padding:4px 9px;font-size:12px;background:#e9edf3}.ok{background:#e3f8ec;color:var(--green)}.bad{background:#fee4e2;color:var(--red)}.warn{background:#fff0d7;color:var(--amber)}.actions{display:flex;gap:5px;flex-wrap:wrap}.actions form{margin:0}.route{direction:ltr;text-align:right;font:12px Consolas,monospace}.empty{text-align:center;color:var(--muted);padding:25px}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}.field label{display:block;font-size:13px;font-weight:bold;margin-bottom:6px}.field input,.field select,.field textarea{width:100%;padding:10px;border:1px solid #cfd7e3;border-radius:7px}.field textarea{min-height:80px}.wide{grid-column:1/-1}.check{display:flex;align-items:center;gap:8px}.check input{width:auto}.hint{font-size:12px;color:var(--muted);margin-top:5px}.series-choice{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.series-check{width:19px;height:19px;accent-color:var(--blue);cursor:pointer}.select-study{margin:0 0 8px}.bulk-bar{position:sticky;bottom:12px;z-index:20;margin:16px auto 0;padding:13px 16px;display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap;background:#fff;border:2px solid var(--blue);border-radius:12px;box-shadow:0 8px 28px #17203333}.bulk-bar[hidden]{display:none}.bulk-bar select{padding:9px;border:1px solid #ccd5e0;border-radius:7px}.footer{color:var(--muted);font-size:12px;padding:22px;text-align:center}@media(max-width:800px){.cards,.grid{grid-template-columns:1fr}.table-wrap{overflow:auto}.bulk-bar{bottom:5px}}
.queue-toast{position:fixed;top:18px;left:50%;transform:translateX(-50%);z-index:40;padding:12px 18px;border-radius:9px;box-shadow:0 6px 25px #17203333;font-weight:bold}.queue-toast[hidden]{display:none}.bulk-tools{display:flex;gap:8px;flex-wrap:wrap;max-width:720px}.tool-check{padding:7px 9px;background:#f3f6fa;border-radius:7px;font-size:12px;white-space:nowrap}.tool-check input{accent-color:var(--blue)}.review-form{display:grid;gap:5px;min-width:210px;margin-top:6px}.review-form select,.review-form input,.queue-filters select{padding:8px;border:1px solid #ccd5e0;border-radius:7px}.queue-filters{align-items:center}
.health-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.meter{height:10px;background:#d8e0eb;border-radius:99px;overflow:hidden;margin-top:9px}.meter>span{display:block;height:100%;background:var(--blue);width:0;transition:width .4s}.chart{width:100%;height:150px;background:#0b1421;border-radius:9px}.pipeline{display:grid;grid-template-columns:repeat(7,1fr);gap:8px}.pipeline .card{text-align:center;padding:12px}.health-up{color:var(--green)}.health-down{color:var(--red)}@media(max-width:900px){.health-grid,.pipeline{grid-template-columns:repeat(2,1fr)}}
.top-actions{display:flex;gap:8px;align-items:center}.icon-btn{border:1px solid #64748b;background:#14243b;color:#fff;border-radius:8px;padding:7px 10px;cursor:pointer}.study-card{margin:12px 0;border:1px solid #dce4ee;border-radius:12px;background:var(--card);overflow:hidden}.study-card summary{display:grid;grid-template-columns:110px 1fr auto;gap:16px;align-items:center;padding:14px;cursor:pointer;list-style:none}.study-card summary::-webkit-details-marker{display:none}.study-card[open] summary{border-bottom:1px solid #dce4ee}.study-body{padding:14px}.thumb{width:104px;height:84px;object-fit:contain;background:#05070a;border-radius:8px}.chevron:before{content:'＋'}.study-card[open] .chevron:before{content:'−'}.series-row{padding:11px;border-bottom:1px solid #e7ebf0}.study-toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
html[data-theme='dark']{--nav:#050b14;--bg:#0d1522;--card:#141f2e;--text:#f1f5f9;--muted:#c2ccda;--blue:#438af3;color-scheme:dark}html[data-theme='dark'] body,html[data-theme='dark'] .panel,html[data-theme='dark'] .card,html[data-theme='dark'] .study-card,html[data-theme='dark'] summary,html[data-theme='dark'] td{color:var(--text)}html[data-theme='dark'] .panel,html[data-theme='dark'] .card,html[data-theme='dark'] .bulk-bar,html[data-theme='dark'] .study-card{background:var(--card);border-color:#34445a}html[data-theme='dark'] th,html[data-theme='dark'] .gray,html[data-theme='dark'] .tool-check{background:#203047;color:#f8fafc}html[data-theme='dark'] th{color:#fff;border-color:#53657b}html[data-theme='dark'] td,html[data-theme='dark'] .series-row,html[data-theme='dark'] .study-card[open] summary{border-color:#34445a}html[data-theme='dark'] .muted,html[data-theme='dark'] .hint,html[data-theme='dark'] .footer,html[data-theme='dark'] small{color:#c2ccda}html[data-theme='dark'] a:not(.btn){color:#78b3ff}html[data-theme='dark'] input,html[data-theme='dark'] select,html[data-theme='dark'] textarea{background:#0b1421;color:#fff;border-color:#60728b}html[data-theme='dark'] input::placeholder,html[data-theme='dark'] textarea::placeholder{color:#a8b5c7}html[data-theme='dark'] option{background:#101b2a;color:#fff}html[data-theme='dark'] .badge{background:#e7edf5;color:#172033}html[data-theme='dark'] .badge.ok,html[data-theme='dark'] .ok{background:#c9f2da;color:#075b36}html[data-theme='dark'] .badge.warn,html[data-theme='dark'] .warn{background:#ffedc7;color:#8a3500}html[data-theme='dark'] .badge.bad,html[data-theme='dark'] .bad{background:#ffd7d4;color:#8d1610}html[data-theme='dark'] .notice{background:#183457;color:#dbeafe}html[data-theme='dark'] .notice.error,html[data-theme='dark'] .error{background:#4c1d1d;color:#fee2e2}html[data-theme='dark'] .btn{color:#fff}html[data-theme='dark'] .btn.gray{color:#f8fafc}html[data-theme='dark'] .btn.red{background:#5d2020;color:#fecaca}
"""


def layout(title: str, body: str, message: str = "", error: bool = False) -> str:
    alert = f"<div class='notice {'error' if error else ''}'>{esc(message)}</div>" if message else ""
    shell = """<script>
const root=document.documentElement;
const savedTheme=localStorage.getItem('mri-theme')||'system';
const effective=savedTheme==='system'?(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light'):savedTheme;
root.dataset.theme=effective;
function toggleTheme(){const next=root.dataset.theme==='dark'?'light':'dark';root.dataset.theme=next;localStorage.setItem('mri-theme',next)}
function toggleLang(){const next=(localStorage.getItem('mri-lang')||'en')==='ar'?'en':'ar';localStorage.setItem('mri-lang',next);applyLang(next)}
function applyLang(lang){root.lang=lang;root.dir=lang==='ar'?'rtl':'ltr';document.querySelectorAll('[data-ar][data-en]').forEach(x=>x.textContent=x.dataset[lang]);document.getElementById('lang-button').textContent=lang==='ar'?'English':'العربية'}
document.addEventListener('DOMContentLoaded',()=>{applyLang(localStorage.getItem('mri-lang')||'en');const f=document.querySelector('.footer');if(f)f.textContent='MRI Enhancer v1.5.0 — Research only · Not for diagnosis'});
</script>"""
    return f"<!doctype html><html lang='ar' dir='rtl'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{esc(title)}</title><style>{STYLE}</style>{shell}</head><body><header class='top'><div><h1>MRI 0.3T Control Center</h1><small data-ar='المراكز · التوجيه · الطابور · العارض' data-en='Centers · Routing · Queue · Viewer'>المراكز · التوجيه · الطابور · العارض</small></div><div class='top-actions'><button id='lang-button' class='icon-btn' onclick='toggleLang()'>English</button><button class='icon-btn' onclick='toggleTheme()'>☀ / ☾</button><span class='badge warn'>RESEARCH ONLY</span></div></header><main class='wrap'>{alert}{body}<div class='footer'>MRI Enhancer v1.1.0 — Research only · Not for diagnosis</div></main></body></html>"


def center_form(center=None) -> str:
    c = dict(center) if center else {}
    editing = bool(c.get("id") and c.get("name"))
    action = f"/admin/centers/{c['id']}/edit" if editing else "/admin/centers"
    def val(k: str, default: str = "") -> str: return esc(c.get(k, default))
    checked = "checked" if not center or c.get("public_ip_static") else ""
    options = "".join(f"<option value='{key}' {'selected' if c.get('processing_profile','BASE_ONLY')==key else ''}>{label}</option>" for key, label in PROFILES.items())
    network_options = "".join(f"<option value='{key}' {'selected' if c.get('network_mode','UNKNOWN')==key else ''}>{label}</option>" for key,label in NETWORK_MODES.items())
    return_options = "".join(f"<option value='{key}' {'selected' if c.get('return_policy','MANUAL')==key else ''}>{label}</option>" for key,label in RETURN_POLICIES.items())
    auto_allowed = auto_eligible_profile_names()
    auto_profiles = "".join(
        f"<option value='{key}' {'selected' if c.get('auto_profile','EDGE_DENOISE_MILD')==key else ''}>{esc(label)}</option>"
        for key, label in ENHANCE_PROFILES.items() if key in auto_allowed
    )
    auto_strengths = "".join(f"<option value='{key}' {'selected' if c.get('auto_strength','MILD')==key else ''}>{esc(label)}</option>" for key,label in STRENGTHS.items())
    auto_checked = "checked" if c.get("auto_enhance_enabled") else ""
    return f"""<form method='post' action='{action}' class='grid'>
    <div class='field'><label>اسم المركز *</label><input name='name' value='{val('name')}' required></div><div class='field'><label>IP المسموح/VPN *</label><input name='source_ip' value='{val('source_ip')}' required><div class='hint'>العنوان الذي يراه السيرفر، ويُستخدم في ACL.</div></div>
    <div class='field'><label>Public IP</label><input name='public_ip' value='{val('public_ip')}'></div><div class='field check'><input id='static' type='checkbox' name='public_ip_static' {checked}><label for='static'>Public IP ثابت</label></div>
    <div class='field'><label>Calling AE للجهاز/PACS *</label><input name='source_aet' value='{val('source_aet')}' maxlength='16' pattern='[A-Za-z0-9_]{{1,16}}' required></div><div class='field'><label>Called AE للسيرفر</label><input value='{esc(os.getenv('ORTHANC_CALLED_AET','MRI03T'))}' disabled></div>
    <div class='field'><label>Return IP *</label><input name='return_ip' value='{val('return_ip')}' required></div><div class='field'><label>Return AE *</label><input name='return_aet' value='{val('return_aet')}' maxlength='16' pattern='[A-Za-z0-9_]{{1,16}}' required></div>
    <div class='field'><label>Return DICOM Port *</label><input name='return_port' type='number' min='1' max='65535' value='{val('return_port','104')}' required></div><div class='field'><label>PACS Internal IP</label><input name='pacs_internal_ip' value='{val('pacs_internal_ip')}'></div>
    <div class='field'><label>Network / NAT Mode *</label><select name='network_mode'>{network_options}</select><div class='hint'>VPN هو الخيار الموصى به. لا يعني نجاح الاستقبال أن الإرجاع العكسي متاح.</div></div>
    <div class='field'><label>Return Policy *</label><select name='return_policy'>{return_options}</select><div class='hint'>Manual افتراضيًا. Auto if PASS يعمل فقط بعد نجاح Quality Gate.</div></div>
    <div class='field'><label>VPN CIDR</label><input name='vpn_cidr' value='{val('vpn_cidr')}' placeholder='10.20.0.0/24'></div>
    <div class='field'><label>Maximum Return Attempts</label><input name='max_return_attempts' type='number' min='1' max='10' value='{val('max_return_attempts','3')}'></div>
    <div class='field check'><input id='auto-enhance' type='checkbox' name='auto_enhance_enabled' {auto_checked}><label for='auto-enhance'>Auto Enhance incoming series / تحسين الوارد تلقائيًا</label></div>
    <div class='field'><label>Branch Enhancement Profile</label><select name='auto_profile'>{auto_profiles}</select><div class='hint'>يطبق على السلاسل الجديدة المستقرة من Calling AE لهذا الفرع.</div></div>
    <div class='field'><label>Enhancement Level</label><select name='auto_strength'>{auto_strengths}</select></div>
    <div class='field'><label>Processing Profile لكل AE *</label><select name='processing_profile'>{options}</select><div class='hint'>N4 ظاهر للتخطيط فقط ولن يقبل الحفظ قبل تركيب المحرك.</div></div><div class='field'><label>قوة المجال</label><input name='field_strength' value='{val('field_strength','0.3')}'></div>
    <div class='field'><label>الشركة المصنعة</label><input name='device_manufacturer' value='{val('device_manufacturer')}'></div><div class='field'><label>موديل الجهاز</label><input name='device_model' value='{val('device_model')}'></div>
    <div class='field'><label>مناطق التصوير</label><input name='anatomy' value='{val('anatomy')}' placeholder='Brain, Spine, Knee'></div><div class='field'><label>Sequences</label><input name='sequences' value='{val('sequences')}' placeholder='T1, T2, FLAIR, DWI'></div>
    <div class='field wide'><label>ملاحظات</label><textarea name='notes'>{val('notes')}</textarea></div><div class='wide'><button class='btn'>حفظ</button> <a class='btn gray' href='/admin'>إلغاء</a></div></form>"""


@router.get("/api/status", response_class=JSONResponse)
def live_status(_: str = Depends(require_admin)):
    health = snapshot()
    counts = pipeline_counts()
    with connect() as db:
        active = [dict(row) for row in db.execute(
            "SELECT id,source_series_id,profile,strength,status,total_instances,processed_instances,error "
            "FROM comparison_jobs WHERE status IN ('queued','processing') ORDER BY id LIMIT 20"
        )]
    for job in active:
        total = max(int(job["total_instances"] or 0), 1)
        job["progress_percent"] = round(100 * int(job["processed_instances"] or 0) / total) if job["status"] == "processing" else 0
    return {"system": health, "pipeline": counts, "active_jobs": active}


@router.get("", response_class=HTMLResponse)
def home(message: str = "", error: str = "", _: str = Depends(require_admin)) -> str:
    n = dashboard_counts(); running = any(c["auto_enhance_enabled"] for c in list_centers())
    centers = "".join(f"<tr><td><b>{esc(c['name'])}</b><br><span class='muted'>{esc(c['device_manufacturer'])} {esc(c['device_model'])} · {esc(c['field_strength'])}T</span></td><td>{esc(c['source_ip'])}<br><b>{esc(c['source_aet'])}</b></td><td><span class='badge'>{esc(c['processing_profile'])}</span><br><span class='muted'>{esc(c['network_mode'])} · {esc(c['return_policy'])}</span></td><td class='route'>{esc(c['return_ip'])}:{c['return_port']} ← {esc(c['return_aet'])}<br><span class='badge {'ok' if c['last_echo_status']=='PASS' else 'bad' if c['last_echo_status']=='FAIL' else 'warn'}'>C-ECHO {esc(c['last_echo_status'])}</span></td><td><span class='badge {'ok' if c['enabled'] else 'bad'}'>{'مفعّل' if c['enabled'] else 'متوقف'}</span></td><td><div class='actions'><a class='btn gray' href='/admin/centers/{c['id']}/edit'>تعديل</a><form method='post' action='/admin/centers/{c['id']}/echo'><button class='btn gray'>Test C-ECHO</button></form><form method='post' action='/admin/centers/{c['id']}/toggle'><button class='btn gray'>تشغيل/إيقاف</button></form><form method='post' action='/admin/centers/{c['id']}/delete' onsubmit=\"return confirm('حذف المركز؟')\"><button class='btn red'>حذف</button></form></div></td></tr>" for c in list_centers()) or "<tr><td colspan='6' class='empty'>لا توجد مراكز.</td></tr>"
    queue = "".join(f"<tr><td class='route'>{esc(q['series_id'])}</td><td>{esc(q['center_name'] or 'غير معروف')}</td><td><span class='badge {'ok' if q['status']=='sent' else 'bad' if q['status']=='failed' else 'warn'}'>{esc(q['status'])}</span></td><td>{esc(q['updated_at'])}</td><td>{esc(q['error'])}</td><td><form method='post' action='/admin/queue/{quote(str(q['series_id']),safe='')}/retry'><button class='btn gray'>إعادة المحاولة</button></form></td></tr>" for q in list_queue()) or "<tr><td colspan='6' class='empty'>الطابور فارغ.</td></tr>"
    pending = "".join(f"<tr><td>{esc(p['source_ip'])}</td><td><b>{esc(p['source_aet'])}</b></td><td>{esc(p['called_aet'])}</td><td>{p['attempts']}</td><td>{esc(p['last_seen'])}</td><td><div class='actions'><a class='btn' href='/admin/pending/{p['id']}/register'>Register</a><form method='post' action='/admin/pending/{p['id']}/reject'><button class='btn red'>رفض</button></form></div></td></tr>" for p in list_pending_senders()) or "<tr><td colspan='6' class='empty'>لا توجد محاولات تسجيل جديدة.</td></tr>"
    returns = "".join(f"<tr><td>{r['id']}</td><td>{esc(r['center_name'])}</td><td><span class='badge {'ok' if r['status']=='sent' else 'bad' if r['status']=='failed' else 'warn'}'>{esc(r['status'])}</span></td><td>{r['attempts']}</td><td class='route'>{esc(r['return_aet'])}@{esc(r['return_ip'])}:{r['return_port']}</td><td>{esc(r['error'])}</td><td><form method='post' action='/admin/returns/{r['id']}/retry'><button class='btn gray'>Retry</button></form></td></tr>" for r in list_return_jobs()) or "<tr><td colspan='7' class='empty'>لا توجد عمليات إرجاع.</td></tr>"
    inbound = "".join(f"<tr><td class='route'>{esc(i['series_id'])}</td><td>{esc(i['center_name'] or i['source_aet'] or 'Unmatched')}</td><td>{esc(i['description'])}</td><td>{i['instances']}</td><td><span class='badge {'ok' if i['status']=='READY' else 'bad' if i['status']=='FAILED' else 'warn'}'>{esc(i['status'])}</span></td><td>{esc(i['updated_at'])}</td><td>{esc(i['error'])}</td></tr>" for i in list_inbound_series()) or "<tr><td colspan='7' class='empty'>لا توجد سلاسل مستلمة.</td></tr>"
    p = pipeline_counts()
    health_panel = f"""<section class='panel'><div class='panel-head'><h2>System Health & Live Progress</h2><span id='orthanc-state' class='badge warn'>Checking…</span></div><p class='muted'>Runtime/container measurements — ليست قياسات جهاز MRI.</p><div class='health-grid'><div class='card'><b>CPU</b><div class='number' id='cpu-value'>--%</div><div class='meter'><span id='cpu-meter'></span></div></div><div class='card'><b>RAM</b><div class='number' id='ram-value'>--%</div><div class='meter'><span id='ram-meter'></span></div><span id='ram-detail' class='muted'></span></div><div class='card'><b>Disk</b><div class='number' id='disk-value'>--%</div><div class='meter'><span id='disk-meter'></span></div><span id='disk-detail' class='muted'></span></div><div class='card'><b>Enhancer RAM</b><div class='number' id='process-ram'>-- MB</div></div></div><canvas id='health-chart' class='chart' width='1100' height='150' style='margin-top:14px'></canvas><h3>Pipeline / مسار الحالات</h3><div class='pipeline'>{''.join(f"<div class='card'><span class='muted'>{label}</span><div class='number' id='pipe-{key}'>{p[key]}</div></div>" for key,label in [('received','Received'),('queued','Queued'),('processing','Processing'),('ready','Ready'),('return_queued','Return Queue'),('sent','Sent'),('failed','Failed')])}</div><div id='active-progress' style='margin-top:14px'></div></section>"""
    body = f"<div class='cards'><div class='card'><span class='muted'>المراكز</span><div class='number'>{n['centers']}</div></div><div class='card'><span class='muted'>Pending Senders</span><div class='number'>{n['pending']}</div></div><div class='card'><span class='muted'>قيد المعالجة</span><div class='number'>{n['processing']}</div></div><div class='card'><span class='muted'>فشلت</span><div class='number'>{n['failed']}</div></div></div><section class='panel'><div class='panel-head'><h2>Register New Sender</h2><span class='badge warn'>غير مسموح قبل اعتمادك</span></div><div class='table-wrap'><table><tr><th>Source IP</th><th>Calling AE</th><th>Called AE</th><th>المحاولات</th><th>آخر محاولة</th><th>القرار</th></tr>{pending}</table></div></section><section class='panel'><div class='panel-head'><h2>المراكز وRouting Rules</h2><a class='btn' href='/admin/centers/new'>+ إضافة مركز</a></div><div class='table-wrap'><table><tr><th>المركز والجهاز</th><th>Source IP / AE</th><th>Profile</th><th>Return Route</th><th>الحالة</th><th>إجراءات</th></tr>{centers}</table></div></section><section class='panel'><div class='panel-head'><h2>Queue المتابعة</h2><span class='badge {'ok' if running else 'warn'}'>{'Auto Processing يعمل' if running else 'Auto Processing متوقف'}</span></div><div class='table-wrap'><table><tr><th>Series ID</th><th>المركز</th><th>الحالة</th><th>التحديث</th><th>الخطأ</th><th>إجراء</th></tr>{queue}</table></div></section><section class='panel'><h2>Return Queue / طابور الإرجاع</h2><div class='table-wrap'><table><tr><th>ID</th><th>Center</th><th>Status</th><th>Attempts</th><th>Destination</th><th>Error</th><th>Action</th></tr>{returns}</table></div></section><section class='panel'><h2>مسار التسجيل</h2><p class='route'>UNKNOWN TEST C-STORE → PENDING + REJECT → ADMIN REGISTER → RESEND → QUEUE</p><a href='/health'>System Health JSON</a></section>"
    body = "<div style='margin-bottom:14px' class='actions'><a class='btn' href='/admin/studies'>الحالات والمقارنة في OHIF</a><a class='btn gray' href='/admin/protocols'>Protocol Study</a><a class='btn gray' href='/admin/synthetic'>Synthetic MRI Lab</a><a class='btn gray' href='/admin/queues'>Queues</a><a class='btn gray' href='/admin/services'>Services</a><a class='btn gray' href='/admin/training'>MONAI Training</a><a class='btn gray' href='/admin/tools'>Tools</a></div>" + health_panel + body
    body += f"<section class='panel'><h2>Receive Queue / طابور الاستقبال</h2><div class='table-wrap'><table><tr><th>Series</th><th>Branch</th><th>Description</th><th>Images</th><th>Status</th><th>Updated</th><th>Error</th></tr>{inbound}</table></div></section>" + """<script>
const hs=[];function meter(id,v){document.getElementById(id).style.width=Math.min(100,v)+'%'}function chart(){const c=document.getElementById('health-chart'),g=c.getContext('2d'),w=c.width,h=c.height;g.clearRect(0,0,w,h);g.strokeStyle='#34445a';for(let i=1;i<4;i++){g.beginPath();g.moveTo(0,h*i/4);g.lineTo(w,h*i/4);g.stroke()}[['cpu','#4f9cff'],['ram','#45c486']].forEach(([k,color])=>{g.strokeStyle=color;g.lineWidth=2;g.beginPath();hs.forEach((s,i)=>{const x=i*w/Math.max(hs.length-1,1),y=h-s[k]*h/100;i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()})}async function live(){try{const d=await(await fetch('/admin/api/status',{cache:'no-store'})).json(),s=d.system;document.getElementById('cpu-value').textContent=s.cpu_percent+'%';document.getElementById('ram-value').textContent=s.ram_percent+'%';document.getElementById('disk-value').textContent=s.disk_percent+'%';document.getElementById('ram-detail').textContent=s.ram_used_gb+' / '+s.ram_total_gb+' GB';document.getElementById('disk-detail').textContent=s.disk_free_gb+' GB free';document.getElementById('process-ram').textContent=s.process_ram_mb+' MB';meter('cpu-meter',s.cpu_percent);meter('ram-meter',s.ram_percent);meter('disk-meter',s.disk_percent);const o=document.getElementById('orthanc-state');o.textContent='Orthanc '+s.orthanc;o.className='badge '+(s.orthanc==='UP'?'ok':'bad');Object.entries(d.pipeline).forEach(([k,v])=>{const e=document.getElementById('pipe-'+k);if(e)e.textContent=v});document.getElementById('active-progress').innerHTML=d.active_jobs.map(j=>`<div class='series-row'><b>Job ${j.id} · ${j.profile}/${j.strength}</b> <span class='badge warn'>${j.status} ${j.progress_percent}%</span><div class='meter'><span style='width:${j.progress_percent}%'></span></div></div>`).join('')||'<span class="muted">No active enhancement jobs</span>';hs.push({cpu:s.cpu_percent,ram:s.ram_percent});if(hs.length>60)hs.shift();chart()}catch(e){const o=document.getElementById('orthanc-state');o.textContent='Health unavailable';o.className='badge bad'}}live();setInterval(live,5000);
</script>"""
    return layout("لوحة تحكم MRI", body, error or message, bool(error))


@router.get("/centers/new", response_class=HTMLResponse)
def new(_: str = Depends(require_admin)): return layout("إضافة مركز", "<section class='panel'><h2>إضافة مركز وقاعدة توجيه</h2>" + center_form() + "</section>")


@router.get("/pending/{sender_id}/register", response_class=HTMLResponse)
def register_page(sender_id: int, _: str = Depends(require_admin)):
    sender = get_pending_sender(sender_id)
    if sender is None: raise HTTPException(404, "Pending sender not found")
    prefill = {"source_ip": sender["source_ip"], "source_aet": sender["source_aet"],
               "public_ip": sender["source_ip"], "return_ip": sender["source_ip"],
               "return_aet": sender["source_aet"], "return_port": 104,
               "processing_profile": "BASE_ONLY", "field_strength": "0.3"}
    return layout("Register Sender", "<section class='panel'><h2>تسجيل المرسل الجديد</h2><div class='notice'>تم ملء IP وAE تلقائيًا. راجع بيانات الإرجاع وأكمل اسم المركز.</div>" + center_form(prefill) + "</section>")


@router.post("/pending/{sender_id}/reject")
def reject_sender(sender_id: int, _: str = Depends(require_admin)):
    with connect() as db: db.execute("UPDATE pending_senders SET status='rejected' WHERE id=?", (sender_id,))
    return RedirectResponse("/admin?message="+quote("تم رفض المرسل"),303)


@router.get("/centers/{center_id}/edit", response_class=HTMLResponse)
def edit_page(center_id: int, _: str = Depends(require_admin)):
    center = get_center(center_id)
    if center is None: raise HTTPException(404, "Center not found")
    return layout("تعديل مركز", "<section class='panel'><h2>تعديل المركز وقاعدة التوجيه</h2>" + center_form(center) + "</section>")


def values(name, source_ip, public_ip, public_ip_static, source_aet, return_ip, return_aet, return_port, pacs_internal_ip, device_manufacturer, device_model, field_strength, anatomy, sequences, notes, processing_profile, network_mode, return_policy, vpn_cidr, max_return_attempts, auto_enhance_enabled, auto_profile, auto_strength):
    return clean(locals())


@router.post("/centers")
def add(name: str=Form(...), source_ip: str=Form(...), public_ip: str=Form(""), public_ip_static: str|None=Form(None), source_aet: str=Form(...), return_ip: str=Form(...), return_aet: str=Form(...), return_port: int=Form(...), pacs_internal_ip: str=Form(""), device_manufacturer: str=Form(""), device_model: str=Form(""), field_strength: str=Form("0.3"), anatomy: str=Form(""), sequences: str=Form(""), notes: str=Form(""), processing_profile: str=Form("BASE_ONLY"), network_mode: str=Form("UNKNOWN"), return_policy: str=Form("MANUAL"), vpn_cidr: str=Form(""), max_return_attempts: int=Form(3), auto_enhance_enabled: str|None=Form(None), auto_profile: str=Form("EDGE_DENOISE_MILD"), auto_strength: str=Form("MILD"), _: str=Depends(require_admin)):
    try:
        d = values(name,source_ip,public_ip,public_ip_static,source_aet,return_ip,return_aet,return_port,pacs_internal_ip,device_manufacturer,device_model,field_strength,anatomy,sequences,notes,processing_profile,network_mode,return_policy,vpn_cidr,max_return_attempts,auto_enhance_enabled,auto_profile,auto_strength)
        with connect() as db: db.execute(f"INSERT INTO centers({','.join(d)}) VALUES({','.join('?' for _ in d)})", tuple(d.values()))
        with connect() as db: db.execute("UPDATE pending_senders SET status='approved' WHERE source_ip=? AND source_aet=?", (d["source_ip"], d["source_aet"]))
    except (ValueError, sqlite3.IntegrityError) as exc: return RedirectResponse("/admin?error="+quote(str(exc)),303)
    return RedirectResponse("/admin?message="+quote("تمت إضافة المركز وقاعدة التوجيه"),303)


@router.post("/centers/{center_id}/edit")
def edit(center_id: int, name: str=Form(...), source_ip: str=Form(...), public_ip: str=Form(""), public_ip_static: str|None=Form(None), source_aet: str=Form(...), return_ip: str=Form(...), return_aet: str=Form(...), return_port: int=Form(...), pacs_internal_ip: str=Form(""), device_manufacturer: str=Form(""), device_model: str=Form(""), field_strength: str=Form("0.3"), anatomy: str=Form(""), sequences: str=Form(""), notes: str=Form(""), processing_profile: str=Form("BASE_ONLY"), network_mode: str=Form("UNKNOWN"), return_policy: str=Form("MANUAL"), vpn_cidr: str=Form(""), max_return_attempts: int=Form(3), auto_enhance_enabled: str|None=Form(None), auto_profile: str=Form("EDGE_DENOISE_MILD"), auto_strength: str=Form("MILD"), _: str=Depends(require_admin)):
    try:
        d=values(name,source_ip,public_ip,public_ip_static,source_aet,return_ip,return_aet,return_port,pacs_internal_ip,device_manufacturer,device_model,field_strength,anatomy,sequences,notes,processing_profile,network_mode,return_policy,vpn_cidr,max_return_attempts,auto_enhance_enabled,auto_profile,auto_strength)
        with connect() as db: db.execute("UPDATE centers SET "+",".join(f"{k}=?" for k in d)+" WHERE id=?",(*d.values(),center_id))
    except (ValueError, sqlite3.IntegrityError) as exc: return RedirectResponse("/admin?error="+quote(str(exc)),303)
    return RedirectResponse("/admin?message="+quote("تم حفظ التعديلات"),303)


@router.post("/centers/{center_id}/toggle")
def toggle(center_id:int,_:str=Depends(require_admin)):
    with connect() as db: db.execute("UPDATE centers SET enabled=1-enabled WHERE id=?",(center_id,))
    return RedirectResponse("/admin",303)


@router.post("/centers/{center_id}/delete")
def delete(center_id:int,_:str=Depends(require_admin)):
    with connect() as db: db.execute("DELETE FROM centers WHERE id=?",(center_id,))
    return RedirectResponse("/admin",303)


@router.post("/queue/{series_id}/retry")
def retry(series_id:str,_:str=Depends(require_admin)):
    with connect() as db: db.execute("DELETE FROM processed_series WHERE series_id=?",(series_id,))
    return RedirectResponse("/admin?message="+quote("تمت إعادة الحالة إلى قائمة الانتظار"),303)


@router.post("/returns/{return_id}/retry")
def retry_return(return_id: int, _: str = Depends(require_admin)):
    with connect() as db:
        result = db.execute("UPDATE return_jobs SET status='queued',error=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='failed'", (return_id,))
        if result.rowcount != 1:
            raise HTTPException(409, "Only failed return jobs can be retried")
    return RedirectResponse("/admin?message=" + quote("تمت إعادة الإرسال إلى الطابور"), 303)


@router.post("/centers/{center_id}/echo")
def echo_center(center_id: int, _: str = Depends(require_admin)):
    import httpx
    center = get_center(center_id)
    if center is None:
        raise HTTPException(404, "Center not found")
    try:
        base = os.getenv("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
        auth = (os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", ""))
        with httpx.Client(auth=auth, timeout=20) as api:
            api.put(f"{base}/modalities/{center['slug']}", json={"AET": center["return_aet"], "Host": center["return_ip"], "Port": center["return_port"]}).raise_for_status()
            api.post(f"{base}/modalities/{center['slug']}/echo").raise_for_status()
        state, message = "PASS", "C-ECHO succeeded"
    except Exception as exc:
        state, message = "FAIL", str(exc)[:500]
    with connect() as db:
        db.execute("UPDATE centers SET last_echo_status=?,last_echo_at=CURRENT_TIMESTAMP WHERE id=?", (state, center_id))
    return RedirectResponse("/admin?" + ("message=" if state == "PASS" else "error=") + quote(message), 303)
