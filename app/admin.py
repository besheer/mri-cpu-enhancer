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
AE_RE = re.compile(r"^[A-Za-z0-9_]{1,16}$")
PROFILES = {"BASE_ONLY": "BASE Only — Available", "N4_ONLY": "N4 Only — Coming Soon", "BASE_AND_N4": "BASE + N4 — Coming Soon"}
NETWORK_MODES = {"UNKNOWN": "Unknown", "VPN": "VPN — Recommended", "PUBLIC_STATIC": "Public Static IP", "NAT_FORWARD": "Behind NAT + Port Forward", "AGENT_PULL": "Agent Pull", "NO_RETURN": "No Return"}
RETURN_POLICIES = {"MANUAL": "Manual Review", "AUTO_PASS": "Auto Return if QC PASS", "DISABLED": "Return Disabled"}


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
        raise ValueError(f"{label}: Invalid IP address") from exc


def valid_aet(value: str, label: str) -> str:
    value = value.strip()
    if not AE_RE.fullmatch(value):
        raise ValueError(f"{label}: Use 1–16 characters (A-Z, 0-9, or _ only)")
    return value


def clean(data: dict[str, object]) -> dict[str, object]:
    name = str(data["name"]).strip()
    if not name:
        raise ValueError("Center name is required")
    port = int(data["return_port"])
    if not 1 <= port <= 65535:
        raise ValueError("Return port must be between 1 and 65535")
    profile = str(data.get("processing_profile", "BASE_ONLY"))
    if profile not in PROFILES:
        raise ValueError("Invalid Processing Profile")
    # Do not silently advertise an unavailable processing engine.
    if profile != "BASE_ONLY":
        raise ValueError("N4 engine is not enabled for branch default; select BASE_ONLY")
    strength = str(data.get("field_strength", "0.3")).strip()
    network_mode = str(data.get("network_mode", "UNKNOWN"))
    return_policy = str(data.get("return_policy", "MANUAL"))
    if network_mode not in NETWORK_MODES or return_policy not in RETURN_POLICIES:
        raise ValueError("Invalid network mode or return policy")
    if network_mode in {"UNKNOWN", "AGENT_PULL", "NO_RETURN"} and return_policy == "AUTO_PASS":
        raise ValueError("Auto Return requires a verified VPN or Public IP/Port Forward")
    attempts = int(data.get("max_return_attempts", 3))
    if not 1 <= attempts <= 10:
        raise ValueError("Max return attempts must be between 1 and 10")
    auto_profile = str(data.get("auto_profile", "EDGE_DENOISE_MILD"))
    auto_strength = str(data.get("auto_strength", "MILD"))
    if auto_profile not in ENHANCE_PROFILES or auto_strength not in STRENGTHS:
        raise ValueError("Invalid branch enhancement profile")
    auto_enabled = bool(data.get("auto_enhance_enabled"))
    if auto_enabled and auto_profile not in enabled_profile_names():
        raise ValueError("Selected enhancer is not enabled on the Tools page")
    if auto_enabled and (auto_profile not in auto_eligible_profile_names() ):
        raise ValueError("This enhancer/strength is for manual comparison only and is not allowed for Auto-Enhance")
    return {
        "name": name, "slug": slugify(name),
        "source_ip": valid_ip(str(data["source_ip"]), "Source/VPN IP"),
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
:root{
  --nav:#07101e;--blue:#2563eb;--blue-hover:#1d4ed8;--cyan:#06b6d4;
  --bg:#f1f5f9;--card:#ffffff;--text:#0f172a;--muted:#64748b;
  --green:#059669;--red:#dc2626;--amber:#d97706;--border:#e2e8f0;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:'Segoe UI',Tahoma,Arial,sans-serif;line-height:1.5}
.top{background:linear-gradient(135deg,#060d1a 0%,#0f223d 100%);color:#fff;padding:14px 3%;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1e3a5f;box-shadow:0 4px 20px rgba(0,0,0,.25);position:sticky;top:0;z-index:50;flex-wrap:wrap;gap:12px}
.brand{display:flex;align-items:center;gap:12px}
.brand-logo{width:40px;height:40px;border-radius:10px;background:linear-gradient(135deg,#3b82f6,#06b6d4);display:flex;align-items:center;justify-content:center;font-weight:900;font-size:18px;color:#fff;box-shadow:0 0 15px rgba(59,130,246,.5)}
.top h1{font-size:19px;margin:0;letter-spacing:.3px;display:flex;align-items:center;gap:8px}
.top small{color:#94a3b8;font-size:12px;display:block}
.main-nav{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.nav-link{color:#cbd5e1;text-decoration:none;padding:8px 13px;border-radius:8px;font-size:13px;font-weight:600;transition:all .2s;border:1px solid transparent;display:inline-flex;align-items:center;gap:6px}
.nav-link:hover{background:rgba(255,255,255,.08);color:#fff}
.nav-link.active{background:rgba(59,130,246,.22);color:#60a5fa;border-color:rgba(59,130,246,.45)}
.nav-link.ohif-btn{background:linear-gradient(135deg,#059669,#10b981);color:#fff;border:none;box-shadow:0 2px 10px rgba(16,185,129,.3)}
.nav-link.ohif-btn:hover{filter:brightness(1.1)}
.wrap{width:min(1440px,95%);margin:22px auto}
.notice{padding:13px 18px;border-radius:10px;margin-bottom:16px;background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe;font-weight:600}
.error{background:#fef2f2;color:var(--red);border-color:#fecaca}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}
.card,.panel{background:var(--card);border:1px solid var(--border);border-radius:14px;box-shadow:0 4px 14px rgba(15,23,42,.04)}
.card{padding:18px;transition:transform .15s}
.card:hover{transform:translateY(-2px)}
.number{font-size:28px;font-weight:800;margin:4px 0}
.muted{color:var(--muted);font-size:13px}
.panel{margin-top:18px;padding:20px}
.panel-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.panel h2{margin:0;font-size:18px;display:flex;align-items:center;gap:8px}
.btn{display:inline-flex;align-items:center;gap:6px;border:0;border-radius:8px;padding:8px 14px;background:var(--blue);color:#fff;text-decoration:none;cursor:pointer;font-weight:700;font-size:13px;transition:all .15s}
.btn:hover{background:var(--blue-hover)}
.btn:disabled{opacity:.55;cursor:not-allowed}
.btn.ai-btn{background:linear-gradient(135deg,#2563eb,#06b6d4);color:#fff;box-shadow:0 2px 10px rgba(37,99,235,.3)}
.btn.ai-btn:hover{filter:brightness(1.08)}
.btn.emerald{background:linear-gradient(135deg,#059669,#10b981);color:#fff}
.gray{background:#e2e8f0;color:#1e293b}
.gray:hover{background:#cbd5e1}
.red{background:#fee2e2;color:var(--red)}
.btn.red{background:#dc2626;color:#ffffff;border:1px solid #b91c1c}
.btn.red:hover{background:#b91c1c}
.btn.warn{background:#d97706;color:#ffffff;border:1px solid #b45309}
.btn.warn:hover{background:#b45309}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:12px 10px;text-align:left;border-bottom:1px solid var(--border);vertical-align:middle}
html[dir='rtl'] th,html[dir='rtl'] td{text-align:right}
th{background:#f8fafc;font-weight:700;color:#475569}
.badge{display:inline-flex;align-items:center;gap:4px;border-radius:99px;padding:4px 10px;font-size:12px;font-weight:700;background:#e2e8f0;color:#1e293b}
.ok{background:#d1fae5;color:#065f46}
.bad{background:#fee2e2;color:#991b1b}
.warn{background:#fef3c7;color:#92400e}
.ai-badge{background:linear-gradient(135deg,rgba(37,99,235,.2),rgba(6,182,212,.25));color:#38bdf8;border:1px solid rgba(56,189,248,.4)}
.snum-badge{font-family:Consolas,monospace;font-size:12px;font-weight:800;padding:3px 8px;border-radius:6px;background:#1e293b;color:#38bdf8;border:1px solid #334155}
.snum-badge.derived{background:#064e3b;color:#34d399;border-color:#059669}
.actions{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.actions form{margin:0}
.route{direction:ltr;text-align:left;font:12px Consolas,monospace}
.empty{text-align:center;color:var(--muted);padding:28px}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
.field label{display:block;font-size:13px;font-weight:700;margin-bottom:6px}
.field input,.field select,.field textarea{width:100%;padding:10px;border:1px solid #cbd5e1;border-radius:8px;font-size:13.5px}
.field textarea{min-height:80px}
.wide{grid-column:1/-1}
.check{display:flex;align-items:center;gap:8px}
.check input{width:auto}
.hint{font-size:12px;color:var(--muted);margin-top:4px}
.series-choice{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.series-check{width:18px;height:18px;accent-color:var(--blue);cursor:pointer}
.select-study{margin:0}
.bulk-bar{position:sticky;bottom:14px;z-index:30;margin:16px auto 0;padding:14px 18px;display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap;background:var(--card);border:2px solid var(--blue);border-radius:14px;box-shadow:0 12px 32px rgba(0,0,0,.35)}
.bulk-bar[hidden]{display:none}
.bulk-bar select{padding:8px 10px;border:1px solid #cbd5e1;border-radius:8px}
.footer{color:var(--muted);font-size:12px;padding:24px;text-align:center}
.queue-toast{position:fixed;top:76px;left:50%;transform:translateX(-50%);z-index:60;padding:12px 20px;border-radius:10px;box-shadow:0 8px 28px rgba(0,0,0,.35);font-weight:700}
.queue-toast[hidden]{display:none}
.bulk-tools{display:flex;gap:8px;flex-wrap:wrap;max-width:740px}
.tool-check{padding:6px 10px;background:#f1f5f9;border-radius:8px;font-size:12px;white-space:nowrap}
.tool-check input{accent-color:var(--blue)}
.review-form{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:6px}
.review-form select,.review-form input,.queue-filters select{padding:7px 10px;border:1px solid #cbd5e1;border-radius:8px;font-size:13px}
.queue-filters{align-items:center}
.health-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.meter{height:8px;background:#cbd5e1;border-radius:99px;overflow:hidden;margin-top:8px}
.meter>span{display:block;height:100%;background:linear-gradient(90deg,#3b82f6,#06b6d4);width:0;transition:width .4s}
.chart{width:100%;height:140px;background:#070d19;border-radius:10px;border:1px solid #1e293b}
.pipeline{display:grid;grid-template-columns:repeat(7,1fr);gap:8px}
.pipeline .card{text-align:center;padding:12px}
.top-actions{display:flex;gap:8px;align-items:center}
.icon-btn{border:1px solid #334155;background:#0f172a;color:#e2e8f0;border-radius:8px;padding:7px 11px;cursor:pointer;font-size:12.5px;font-weight:600}
.icon-btn:hover{background:#1e293b}
.study-card{margin:14px 0;border:1px solid var(--border);border-radius:14px;background:var(--card);overflow:hidden;transition:box-shadow .2s}
.study-card[open]{box-shadow:0 8px 26px rgba(0,0,0,.18)}
.study-card summary{display:grid;grid-template-columns:100px 1fr auto;gap:16px;align-items:center;padding:14px 18px;cursor:pointer;list-style:none}
.study-card summary::-webkit-details-marker{display:none}
.study-card[open] summary{border-bottom:1px solid var(--border);background:rgba(59,130,246,.04)}
.study-body{padding:16px}
.thumb{width:96px;height:80px;object-fit:contain;background:#020617;border-radius:10px;border:1px solid #1e293b}
.chevron{font-size:13px;font-weight:bold;color:#60a5fa;padding:6px 12px;border-radius:8px;background:rgba(59,130,246,.1)}
.chevron:before{content:'＋ Expand Series'}
.study-card[open] .chevron:before{content:'− Collapse Study'}
.series-grid{display:grid;grid-template-columns:1fr;gap:8px;margin-top:10px}
.series-item{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:11px 14px;border-radius:10px;border:1px solid var(--border);background:rgba(15,23,42,.02);flex-wrap:wrap}
.series-item.derived-item{background:rgba(16,185,129,.06);border-color:rgba(16,185,129,.28)}
.kpi-pills{display:flex;gap:6px;flex-wrap:wrap}
.kpi-pill{font-size:12px;padding:4px 9px;border-radius:7px;background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.25);font-weight:600}
.study-toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;align-items:center}
@media(max-width:900px){.cards,.grid,.health-grid,.pipeline{grid-template-columns:1fr}.table-wrap{overflow:auto}.study-card summary{grid-template-columns:76px 1fr}}
html[data-theme='dark']{--nav:#040914;--bg:#09101d;--card:#111c2d;--text:#f1f5f9;--muted:#94a3b8;--blue:#3b82f6;--border:#1e293b;color-scheme:dark}
html[data-theme='dark'] body,html[data-theme='dark'] .panel,html[data-theme='dark'] .card,html[data-theme='dark'] .study-card,html[data-theme='dark'] summary,html[data-theme='dark'] td{color:var(--text)}
html[data-theme='dark'] .panel,html[data-theme='dark'] .card,html[data-theme='dark'] .bulk-bar,html[data-theme='dark'] .study-card{background:var(--card);border-color:var(--border)}
html[data-theme='dark'] th,html[data-theme='dark'] .gray,html[data-theme='dark'] .tool-check{background:#1e293b;color:#f8fafc}
html[data-theme='dark'] th{color:#cbd5e1;border-color:#334155}
html[data-theme='dark'] td,html[data-theme='dark'] .series-row,html[data-theme='dark'] .study-card[open] summary{border-color:#1e293b}
html[data-theme='dark'] .series-item{background:#0d1626;border-color:#1e293b}
html[data-theme='dark'] .series-item.derived-item{background:rgba(6,78,59,.22);border-color:rgba(16,185,129,.35)}
html[data-theme='dark'] a:not(.btn):not(.nav-link){color:#60a5fa}
html[data-theme='dark'] input,html[data-theme='dark'] select,html[data-theme='dark'] textarea{background:#070d19;color:#fff;border-color:#334155}
html[data-theme='dark'] .badge{background:#1e293b;color:#e2e8f0}
html[data-theme='dark'] .badge.ok,html[data-theme='dark'] .ok{background:rgba(16,185,129,.2);color:#34d399;border:1px solid rgba(16,185,129,.35)}
html[data-theme='dark'] .badge.warn,html[data-theme='dark'] .warn{background:rgba(245,158,11,.2);color:#fbbf24;border:1px solid rgba(245,158,11,.35)}
html[data-theme='dark'] .badge.bad,html[data-theme='dark'] .bad{background:rgba(239,68,68,.2);color:#f87171;border:1px solid rgba(239,68,68,.35)}
html[data-theme='dark'] .notice{background:#172554;color:#dbeafe;border-color:#1e40af}
html[data-theme='dark'] .notice.error,html[data-theme='dark'] .error{background:#450a0a;color:#fecaca;border-color:#991b1b}
"""


def layout(title: str, body: str, message: str = "", error: bool = False) -> str:
    alert = f"<div class='notice {'error' if error else ''}'>{esc(message)}</div>" if message else ""
    ohif_url = os.getenv("OHIF_PUBLIC_URL", "http://192.168.0.207:8042/ohif/viewer")
    shell = """<script>
const root=document.documentElement;
const savedTheme=localStorage.getItem('mri-theme')||'dark';
root.dataset.theme=savedTheme;
function toggleTheme(){const next=root.dataset.theme==='dark'?'light':'dark';root.dataset.theme=next;localStorage.setItem('mri-theme',next)}
function toggleLang(){const next=(localStorage.getItem('mri-lang-v2')||'en')==='en'?'ar':'en';localStorage.setItem('mri-lang-v2',next);applyLang(next)}
function applyLang(lang){root.lang=lang;root.dir=lang==='ar'?'rtl':'ltr';document.querySelectorAll('[data-ar][data-en]').forEach(x=>x.textContent=x.dataset[lang]);const b=document.getElementById('lang-button');if(b)b.textContent=lang==='ar'?'English':'العربية'}
document.addEventListener('DOMContentLoaded',()=>{
  applyLang(localStorage.getItem('mri-lang-v2')||'en');
  const path=location.pathname;
  document.querySelectorAll('.main-nav .nav-link').forEach(a=>{
    const href=a.getAttribute('href');
    if(href===path || (href!=='/admin' && path.startsWith(href))) a.classList.add('active');
  });
});
</script>"""
    nav = (
        "<nav class='main-nav'>"
        "<a class='nav-link' href='/admin' data-en='🏠 Dashboard' data-ar='🏠 الرئيسية'>🏠 Dashboard</a>"
        "<a class='nav-link' href='/admin/studies' data-en='🩻 Studies & Compare' data-ar='🩻 الحالات والمقارنة'>🩻 Studies & Compare</a>"
        "<a class='nav-link' href='/admin/training' data-en='🧠 AI Training' data-ar='🧠 تدريب الذكاء الاصطناعي'>🧠 AI Training</a>"
        "<a class='nav-link' href='/admin/protocols' data-en='📋 Protocol Study' data-ar='📋 بروتوكول الجودة'>📋 Protocol Study</a>"
        "<a class='nav-link' href='/admin/synthetic' data-en='🔬 Synthetic Lab' data-ar='🔬 المختبر الافتراضي'>🔬 Synthetic Lab</a>"
        "<a class='nav-link' href='/admin/tools' data-en='⚙️ AI Tools' data-ar='⚙️ أدوات التحسين'>⚙️ AI Tools</a>"
        "<a class='nav-link' href='/admin/queues' data-en='📥 Queues' data-ar='📥 الطوابير'>📥 Queues</a>"
        f"<a class='nav-link ohif-btn' target='_blank' href='{esc(ohif_url)}' data-en='🖥️ Open OHIF Viewer' data-ar='🖥️ فتح عارض OHIF'>🖥️ Open OHIF Viewer</a>"
        "</nav>"
    )
    return (
        f"<!doctype html><html lang='en' dir='ltr'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{esc(title)} · RAD4AI</title>"
        f"<style>{STYLE}</style>{shell}</head><body>"
        f"<header class='top'><div class='brand'><div class='brand-logo' style='width:auto;padding:0 10px;font-size:15px'>RAD4AI</div><div>"
        f"<h1>RAD4AI 0.3T AI Appliance <span class='badge ai-badge'>RAD4AI-NAFNet v2.0 Active</span></h1>"
        f"<small data-en='RAD4AI MRI Enhancement Gateway · Hitachi AIRIS Vento' data-ar='منظومة تحسين الرنين المغناطيسي بالذكاء الاصطناعي · Hitachi AIRIS Vento'>RAD4AI MRI Enhancement Gateway · Hitachi AIRIS Vento</small>"
        f"</div></div>{nav}<div class='top-actions'>"
        f"<button id='lang-button' class='icon-btn' onclick='toggleLang()'>العربية</button>"
        f"<button class='icon-btn' onclick='toggleTheme()' title='Toggle Theme'>☀ / ☾</button>"
        f"</div></header><main class='wrap'>{alert}{body}"
        f"<div class='footer'>RAD4AI 0.3T AI Enhancer v2.0 — NAFNet-03T Clinical Engine · Powered by ONNX Runtime</div></main></body></html>"
    )


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
        f"<option value='{key}' {'selected' if c.get('auto_profile','AI_SWIFT_DENOISE')==key else ''}>{esc(label)}</option>"
        for key, label in ENHANCE_PROFILES.items() if key in auto_allowed
    )
    auto_strengths = "".join(f"<option value='{key}' {'selected' if c.get('auto_strength','MILD')==key else ''}>{esc(label)}</option>" for key,label in STRENGTHS.items())
    auto_checked = "checked" if c.get("auto_enhance_enabled") else ""
    return f"""<form method='post' action='{action}' class='grid'>
    <div class='field'><label>Center / Branch Name *</label><input name='name' value='{val('name')}' required></div><div class='field'><label>Allowed Source IP / VPN *</label><input name='source_ip' value='{val('source_ip')}' required><div class='hint'>Source IP seen by the appliance (used in DICOM ACL).</div></div>
    <div class='field'><label>Public IP</label><input name='public_ip' value='{val('public_ip')}'></div><div class='field check'><input id='static' type='checkbox' name='public_ip_static' {checked}><label for='static'>Static Public IP</label></div>
    <div class='field'><label>Scanner / PACS Calling AE *</label><input name='source_aet' value='{val('source_aet')}' maxlength='16' pattern='[A-Za-z0-9_]{{1,16}}' required></div><div class='field'><label>Appliance Called AE</label><input value='{esc(os.getenv('ORTHANC_CALLED_AET','MRI03T'))}' disabled></div>
    <div class='field'><label>Return PACS IP *</label><input name='return_ip' value='{val('return_ip')}' required></div><div class='field'><label>Return PACS AE *</label><input name='return_aet' value='{val('return_aet')}' maxlength='16' pattern='[A-Za-z0-9_]{{1,16}}' required></div>
    <div class='field'><label>Return DICOM Port *</label><input name='return_port' type='number' min='1' max='65535' value='{val('return_port','104')}' required></div><div class='field'><label>PACS Internal IP</label><input name='pacs_internal_ip' value='{val('pacs_internal_ip')}'></div>
    <div class='field'><label>Network / NAT Mode *</label><select name='network_mode'>{network_options}</select><div class='hint'>VPN is recommended for reliable bi-directional DICOM routing.</div></div>
    <div class='field'><label>Return Policy *</label><select name='return_policy'>{return_options}</select><div class='hint'>Manual by default. Auto if PASS sends automatically after Quality Gate passes.</div></div>
    <div class='field'><label>VPN CIDR</label><input name='vpn_cidr' value='{val('vpn_cidr')}' placeholder='10.20.0.0/24'></div>
    <div class='field'><label>Maximum Return Attempts</label><input name='max_return_attempts' type='number' min='1' max='10' value='{val('max_return_attempts','3')}'></div>
    <div class='field check'><input id='auto-enhance' type='checkbox' name='auto_enhance_enabled' {auto_checked}><label for='auto-enhance'>Auto-Enhance Incoming Series</label></div>
    <div class='field'><label>Branch Enhancement Profile</label><select name='auto_profile'>{auto_profiles}</select><div class='hint'>Applied automatically to stable incoming series from this Calling AE.</div></div>
    <div class='field'><label>Enhancement Strength</label><select name='auto_strength'>{auto_strengths}</select></div>
    <div class='field'><label>Legacy AE Profile *</label><select name='processing_profile'>{options}</select></div><div class='field'><label>Field Strength (T)</label><input name='field_strength' value='{val('field_strength','0.3')}'></div>
    <div class='field'><label>Device Manufacturer</label><input name='device_manufacturer' value='{val('device_manufacturer')}' placeholder='Hitachi'></div><div class='field'><label>Scanner Model</label><input name='device_model' value='{val('device_model')}' placeholder='AIRIS Vento'></div>
    <div class='field'><label>Anatomies</label><input name='anatomy' value='{val('anatomy')}' placeholder='Brain, Shoulder, Spine, Knee'></div><div class='field'><label>Sequences</label><input name='sequences' value='{val('sequences')}' placeholder='T1, T2, FLAIR, STIR, DWI'></div>
    <div class='field wide'><label>Notes</label><textarea name='notes'>{val('notes')}</textarea></div><div class='wide'><button class='btn'>Save Center</button> <a class='btn gray' href='/admin'>Cancel</a></div></form>"""


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
    centers = "".join(f"<tr><td><b>{esc(c['name'])}</b><br><span class='muted'>{esc(c['device_manufacturer'])} {esc(c['device_model'])} · {esc(c['field_strength'])}T</span></td><td>{esc(c['source_ip'])}<br><b>{esc(c['source_aet'])}</b></td><td><span class='badge'>{esc(c['auto_profile'])}</span><br><span class='muted'>{esc(c['network_mode'])} · {esc(c['return_policy'])}</span></td><td class='route'>{esc(c['return_ip'])}:{c['return_port']} ← {esc(c['return_aet'])}<br><span class='badge {'ok' if c['last_echo_status']=='PASS' else 'bad' if c['last_echo_status']=='FAIL' else 'warn'}'>C-ECHO {esc(c['last_echo_status'])}</span></td><td><span class='badge {'ok' if c['enabled'] else 'bad'}'>{'Active' if c['enabled'] else 'Disabled'}</span></td><td><div class='actions'><a class='btn gray' href='/admin/centers/{c['id']}/edit'>Edit</a><form method='post' action='/admin/centers/{c['id']}/echo'><button class='btn gray'>Test C-ECHO</button></form><form method='post' action='/admin/centers/{c['id']}/toggle'><button class='btn gray'>Enable/Disable</button></form><form method='post' action='/admin/centers/{c['id']}/delete' onsubmit=\"return confirm('Delete this center?')\"><button class='btn red'>Delete</button></form></div></td></tr>" for c in list_centers()) or "<tr><td colspan='6' class='empty'>No centers registered yet.</td></tr>"
    queue = "".join(f"<tr><td class='route'>{esc(q['series_id'])}</td><td>{esc(q['center_name'] or 'Unknown')}</td><td><span class='badge {'ok' if q['status']=='sent' else 'bad' if q['status']=='failed' else 'warn'}'>{esc(q['status'])}</span></td><td>{esc(q['updated_at'])}</td><td>{esc(q['error'])}</td><td><form method='post' action='/admin/queue/{quote(str(q['series_id']),safe='')}/retry'><button class='btn gray'>Retry</button></form></td></tr>" for q in list_queue()) or "<tr><td colspan='6' class='empty'>Processing queue is empty.</td></tr>"
    pending = "".join(f"<tr><td>{esc(p['source_ip'])}</td><td><b>{esc(p['source_aet'])}</b></td><td>{esc(p['called_aet'])}</td><td>{p['attempts']}</td><td>{esc(p['last_seen'])}</td><td><div class='actions'><a class='btn' href='/admin/pending/{p['id']}/register'>Register</a><form method='post' action='/admin/pending/{p['id']}/reject'><button class='btn red'>Reject</button></form></div></td></tr>" for p in list_pending_senders()) or "<tr><td colspan='6' class='empty'>No pending sender attempts.</td></tr>"
    returns = "".join(f"<tr><td>{r['id']}</td><td>{esc(r['center_name'])}</td><td><span class='badge {'ok' if r['status']=='sent' else 'bad' if r['status']=='failed' else 'warn'}'>{esc(r['status'])}</span></td><td>{r['attempts']}</td><td class='route'>{esc(r['return_aet'])}@{esc(r['return_ip'])}:{r['return_port']}</td><td>{esc(r['error'])}</td><td><form method='post' action='/admin/returns/{r['id']}/retry'><button class='btn gray'>Retry</button></form></td></tr>" for r in list_return_jobs()) or "<tr><td colspan='7' class='empty'>No return jobs.</td></tr>"
    inbound = "".join(f"<tr><td class='route'>{esc(i['series_id'])}</td><td>{esc(i['center_name'] or i['source_aet'] or 'Unmatched')}</td><td>{esc(i['description'])}</td><td>{i['instances']}</td><td><span class='badge {'ok' if i['status']=='READY' else 'bad' if i['status']=='FAILED' else 'warn'}'>{esc(i['status'])}</span></td><td>{esc(i['updated_at'])}</td><td>{esc(i['error'])}</td></tr>" for i in list_inbound_series()) or "<tr><td colspan='7' class='empty'>No inbound series received yet.</td></tr>"
    p = pipeline_counts()
    health_panel = f"""<section class='panel'><div class='panel-head'><h2>System Health & Live Progress</h2><span id='orthanc-state' class='badge warn'>Checking…</span></div><p class='muted'>Real-time appliance resource telemetry and DICOM pipeline status.</p><div class='health-grid'><div class='card'><b>CPU Load</b><div class='number' id='cpu-value'>--%</div><div class='meter'><span id='cpu-meter'></span></div></div><div class='card'><b>System RAM</b><div class='number' id='ram-value'>--%</div><div class='meter'><span id='ram-meter'></span></div><span id='ram-detail' class='muted'></span></div><div class='card'><b>Storage Disk</b><div class='number' id='disk-value'>--%</div><div class='meter'><span id='disk-meter'></span></div><span id='disk-detail' class='muted'></span></div><div class='card'><b>AI Engine RAM</b><div class='number' id='process-ram'>-- MB</div></div></div><canvas id='health-chart' class='chart' width='1100' height='150' style='margin-top:14px'></canvas><h3>DICOM & AI Enhancement Pipeline</h3><div class='pipeline'>{''.join(f"<div class='card'><span class='muted'>{label}</span><div class='number' id='pipe-{key}'>{p[key]}</div></div>" for key,label in [('received','Received'),('queued','Queued'),('processing','Processing'),('ready','Ready'),('return_queued','Return Queue'),('sent','Sent'),('failed','Failed')])}</div><div id='active-progress' style='margin-top:14px'></div></section>"""
    body = f"<div class='cards'><div class='card'><span class='muted'>Connected Centers</span><div class='number'>{n['centers']}</div></div><div class='card'><span class='muted'>Pending Senders</span><div class='number'>{n['pending']}</div></div><div class='card'><span class='muted'>Active Processing</span><div class='number'>{n['processing']}</div></div><div class='card'><span class='muted'>Failed Jobs</span><div class='number'>{n['failed']}</div></div></div><section class='panel'><div class='panel-head'><h2>Register New DICOM Sender</h2><span class='badge warn'>Requires Admin Approval</span></div><div class='table-wrap'><table><tr><th>Source IP</th><th>Calling AE</th><th>Called AE</th><th>Attempts</th><th>Last Seen</th><th>Action</th></tr>{pending}</table></div></section><section class='panel'><div class='panel-head'><h2>MRI Centers & Routing Rules</h2><a class='btn' href='/admin/centers/new'>+ Add Center</a></div><div class='table-wrap'><table><tr><th>Center & Scanner</th><th>Source IP / AE</th><th>AI Profile</th><th>Return Route</th><th>Status</th><th>Actions</th></tr>{centers}</table></div></section><section class='panel'><div class='panel-head'><h2>Auto-Processing Queue</h2><span class='badge {'ok' if running else 'warn'}'>{'Auto Processing Active' if running else 'Auto Processing Idle'}</span></div><div class='table-wrap'><table><tr><th>Series ID</th><th>Center</th><th>Status</th><th>Updated</th><th>Error</th><th>Action</th></tr>{queue}</table></div></section><section class='panel'><h2>PACS Return Queue</h2><div class='table-wrap'><table><tr><th>ID</th><th>Center</th><th>Status</th><th>Attempts</th><th>Destination</th><th>Error</th><th>Action</th></tr>{returns}</table></div></section>"
    body = health_panel + body
    body += f"<section class='panel'><h2>Inbound DICOM Receive Queue</h2><div class='table-wrap'><table><tr><th>Series</th><th>Branch</th><th>Description</th><th>Images</th><th>Status</th><th>Updated</th><th>Error</th></tr>{inbound}</table></div></section>" + """<script>
const hs=[];function meter(id,v){document.getElementById(id).style.width=Math.min(100,v)+'%'}function chart(){const c=document.getElementById('health-chart'),g=c.getContext('2d'),w=c.width,h=c.height;g.clearRect(0,0,w,h);g.strokeStyle='#34445a';for(let i=1;i<4;i++){g.beginPath();g.moveTo(0,h*i/4);g.lineTo(w,h*i/4);g.stroke()}[['cpu','#4f9cff'],['ram','#45c486']].forEach(([k,color])=>{g.strokeStyle=color;g.lineWidth=2;g.beginPath();hs.forEach((s,i)=>{const x=i*w/Math.max(hs.length-1,1),y=h-s[k]*h/100;i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()})}async function live(){try{const d=await(await fetch('/admin/api/status',{cache:'no-store'})).json(),s=d.system;document.getElementById('cpu-value').textContent=s.cpu_percent+'%';document.getElementById('ram-value').textContent=s.ram_percent+'%';document.getElementById('disk-value').textContent=s.disk_percent+'%';document.getElementById('ram-detail').textContent=s.ram_used_gb+' / '+s.ram_total_gb+' GB';document.getElementById('disk-detail').textContent=s.disk_free_gb+' GB free';document.getElementById('process-ram').textContent=s.process_ram_mb+' MB';meter('cpu-meter',s.cpu_percent);meter('ram-meter',s.ram_percent);meter('disk-meter',s.disk_percent);const o=document.getElementById('orthanc-state');o.textContent='Orthanc '+s.orthanc;o.className='badge '+(s.orthanc==='UP'?'ok':'bad');Object.entries(d.pipeline).forEach(([k,v])=>{const e=document.getElementById('pipe-'+k);if(e)e.textContent=v});document.getElementById('active-progress').innerHTML=d.active_jobs.map(j=>`<div class='series-row'><b>Job ${j.id} · ${j.profile}/${j.strength}</b> <span class='badge warn'>${j.status} ${j.progress_percent}%</span><div class='meter'><span style='width:${j.progress_percent}%'></span></div></div>`).join('')||'<span class="muted">No active enhancement jobs</span>';hs.push({cpu:s.cpu_percent,ram:s.ram_percent});if(hs.length>60)hs.shift();chart()}catch(e){const o=document.getElementById('orthanc-state');o.textContent='Health unavailable';o.className='badge bad'}}live();setInterval(live,5000);
</script>"""
    return layout("MRI Control Center", body, error or message, bool(error))


@router.get("/centers/new", response_class=HTMLResponse)
def new(_: str = Depends(require_admin)): return layout("Add Center", "<section class='panel'><h2>Add MRI Center & Routing Rule</h2>" + center_form() + "</section>")


@router.get("/pending/{sender_id}/register", response_class=HTMLResponse)
def register_page(sender_id: int, _: str = Depends(require_admin)):
    sender = get_pending_sender(sender_id)
    if sender is None: raise HTTPException(404, "Pending sender not found")
    prefill = {"source_ip": sender["source_ip"], "source_aet": sender["source_aet"],
               "public_ip": sender["source_ip"], "return_ip": sender["source_ip"],
               "return_aet": sender["source_aet"], "return_port": 104,
               "processing_profile": "BASE_ONLY", "field_strength": "0.3"}
    return layout("Register Sender", "<section class='panel'><h2>Register New DICOM Sender</h2><div class='notice'>IP and Calling AE have been pre-filled automatically. Review return settings and enter the Center Name.</div>" + center_form(prefill) + "</section>")


@router.post("/pending/{sender_id}/reject")
def reject_sender(sender_id: int, _: str = Depends(require_admin)):
    with connect() as db: db.execute("UPDATE pending_senders SET status='rejected' WHERE id=?", (sender_id,))
    return RedirectResponse("/admin?message="+quote("Sender rejected"),303)


@router.get("/centers/{center_id}/edit", response_class=HTMLResponse)
def edit_page(center_id: int, _: str = Depends(require_admin)):
    center = get_center(center_id)
    if center is None: raise HTTPException(404, "Center not found")
    return layout("Edit Center", "<section class='panel'><h2>Edit MRI Center & Routing Rule</h2>" + center_form(center) + "</section>")


def values(name, source_ip, public_ip, public_ip_static, source_aet, return_ip, return_aet, return_port, pacs_internal_ip, device_manufacturer, device_model, field_strength, anatomy, sequences, notes, processing_profile, network_mode, return_policy, vpn_cidr, max_return_attempts, auto_enhance_enabled, auto_profile, auto_strength):
    return clean(locals())


@router.post("/centers")
def add(name: str=Form(...), source_ip: str=Form(...), public_ip: str=Form(""), public_ip_static: str|None=Form(None), source_aet: str=Form(...), return_ip: str=Form(...), return_aet: str=Form(...), return_port: int=Form(...), pacs_internal_ip: str=Form(""), device_manufacturer: str=Form(""), device_model: str=Form(""), field_strength: str=Form("0.3"), anatomy: str=Form(""), sequences: str=Form(""), notes: str=Form(""), processing_profile: str=Form("BASE_ONLY"), network_mode: str=Form("UNKNOWN"), return_policy: str=Form("MANUAL"), vpn_cidr: str=Form(""), max_return_attempts: int=Form(3), auto_enhance_enabled: str|None=Form(None), auto_profile: str=Form("EDGE_DENOISE_MILD"), auto_strength: str=Form("MILD"), _: str=Depends(require_admin)):
    try:
        d = values(name,source_ip,public_ip,public_ip_static,source_aet,return_ip,return_aet,return_port,pacs_internal_ip,device_manufacturer,device_model,field_strength,anatomy,sequences,notes,processing_profile,network_mode,return_policy,vpn_cidr,max_return_attempts,auto_enhance_enabled,auto_profile,auto_strength)
        with connect() as db: db.execute(f"INSERT INTO centers({','.join(d)}) VALUES({','.join('?' for _ in d)})", tuple(d.values()))
        with connect() as db: db.execute("UPDATE pending_senders SET status='approved' WHERE source_ip=? AND source_aet=? COLLATE NOCASE", (d["source_ip"], d["source_aet"]))
    except (ValueError, sqlite3.IntegrityError) as exc: return RedirectResponse("/admin?error="+quote(str(exc)),303)
    return RedirectResponse("/admin?message="+quote("Center and routing rule added successfully"),303)


@router.post("/centers/{center_id}/edit")
def edit(center_id: int, name: str=Form(...), source_ip: str=Form(...), public_ip: str=Form(""), public_ip_static: str|None=Form(None), source_aet: str=Form(...), return_ip: str=Form(...), return_aet: str=Form(...), return_port: int=Form(...), pacs_internal_ip: str=Form(""), device_manufacturer: str=Form(""), device_model: str=Form(""), field_strength: str=Form("0.3"), anatomy: str=Form(""), sequences: str=Form(""), notes: str=Form(""), processing_profile: str=Form("BASE_ONLY"), network_mode: str=Form("UNKNOWN"), return_policy: str=Form("MANUAL"), vpn_cidr: str=Form(""), max_return_attempts: int=Form(3), auto_enhance_enabled: str|None=Form(None), auto_profile: str=Form("EDGE_DENOISE_MILD"), auto_strength: str=Form("MILD"), _: str=Depends(require_admin)):
    try:
        d=values(name,source_ip,public_ip,public_ip_static,source_aet,return_ip,return_aet,return_port,pacs_internal_ip,device_manufacturer,device_model,field_strength,anatomy,sequences,notes,processing_profile,network_mode,return_policy,vpn_cidr,max_return_attempts,auto_enhance_enabled,auto_profile,auto_strength)
        with connect() as db: db.execute("UPDATE centers SET "+",".join(f"{k}=?" for k in d)+" WHERE id=?",(*d.values(),center_id))
    except (ValueError, sqlite3.IntegrityError) as exc: return RedirectResponse("/admin?error="+quote(str(exc)),303)
    return RedirectResponse("/admin?message="+quote("Changes saved successfully"),303)


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
    return RedirectResponse("/admin?message="+quote("Series re-queued for processing"),303)


@router.post("/returns/{return_id}/retry")
def retry_return(return_id: int, _: str = Depends(require_admin)):
    with connect() as db:
        result = db.execute("UPDATE return_jobs SET status='queued',error=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='failed'", (return_id,))
        if result.rowcount != 1:
            raise HTTPException(409, "Only failed return jobs can be retried")
    return RedirectResponse("/admin?message=" + quote("Return job re-queued"), 303)


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
