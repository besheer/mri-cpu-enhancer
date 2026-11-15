from __future__ import annotations

from html import escape
from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from app.store import (
    audit,
    connect,
    get_center,
    list_centers,
    list_feedback,
    record_feedback,
    set_center_calibration,
)

router = APIRouter()

VERDICT_LABELS = {
    "APPROVED": "✅ موافق وممتاز",
    "MORE_GRAIN": "🌾 يحتاج خشونة",
    "SMOOTHER": "🧈 يحتاج تنعيم",
    "BRIGHTER_T2": "💧 تباين / سوائل",
    "ADJUST": "✏️ يحتاج تعديل",
    "REJECT": "❌ مرفوض",
    "NOTE": "📝 ملاحظة",
}


class FeedbackPayload(BaseModel):
    series_id: str = ""
    patient_name: str = ""
    study_description: str = ""
    series_description: str = ""
    level: str = ""
    verdict: str = "NOTE"
    note: str = ""


@router.get("/api/v1/feedback")
def api_list_feedback(limit: int = 50):
    return {"items": list_feedback(limit=limit)}


@router.post("/api/v1/feedback")
def api_create_feedback(payload: FeedbackPayload):
    fid = record_feedback(
        source="web",
        series_id=payload.series_id,
        patient_name=payload.patient_name,
        study_description=payload.study_description,
        series_description=payload.series_description,
        level=payload.level,
        verdict=payload.verdict,
        note=payload.note,
    )
    audit("FEEDBACK_SUBMITTED", "feedback", str(fid), f"verdict={payload.verdict};level={payload.level}")
    return JSONResponse({"id": fid, "status": "ok"})


@router.post("/api/v1/feedback/calibration")
def api_toggle_calibration(center_id: int = Form(8), enabled: int = Form(1)):
    set_center_calibration(center_id, bool(enabled))
    audit("CALIBRATION_MODE_CHANGED", "center", str(center_id), f"enabled={bool(enabled)}")
    return RedirectResponse("/feedback", status_code=303)


@router.post("/feedback/submit")
def web_submit_feedback(
    series_info: str = Form(""),
    level: str = Form(""),
    verdict: str = Form("APPROVED"),
    note: str = Form(""),
):
    parts = series_info.split("||") if "||" in series_info else ["", "", series_info]
    sid = parts[0] if len(parts) > 0 else ""
    prof = parts[1] if len(parts) > 1 else ""
    sdesc = parts[2] if len(parts) > 2 else ""
    fid = record_feedback(
        source="web",
        series_id=sid,
        patient_name="",
        study_description=prof,
        series_description=sdesc,
        level=level,
        verdict=verdict,
        note=note.strip(),
    )
    audit("FEEDBACK_SUBMITTED", "feedback", str(fid), f"verdict={verdict};level={level}")
    return RedirectResponse("/feedback?saved=1", status_code=303)


@router.get("/feedback", response_class=HTMLResponse)
def feedback_page(saved: int = 0):
    centers = list_centers()
    target_center = next((c for c in centers if c["id"] == 8), centers[0] if centers else None)
    cal_on = bool(target_center and target_center.get("calibration_mode", 0))
    cid = target_center["id"] if target_center else 8
    cname = target_center["name"] if target_center else "DarElHa"

    with connect() as db:
        recent_jobs = db.execute(
            """
            SELECT cj.id, cj.source_series_id, cj.profile, cj.strength, cj.qc_status,
                   cj.updated_at, COALESCE(ib.description, '') AS description
            FROM comparison_jobs cj
            LEFT JOIN inbound_series ib ON ib.series_id = cj.source_series_id
            WHERE cj.archived = 0
            ORDER BY cj.id DESC LIMIT 15
            """
        ).fetchall()

    options_html = '<option value="||عام / آخر فحص">آخر فحص / ملاحظة عامة</option>'
    for j in recent_jobs:
        val = f"{j['source_series_id']}||{j['profile']} ({j['strength']})||{j['description']}"
        label = f"#{j['id']} | {j['description'] or j['source_series_id'][:12]} | {j['profile']} ({j['strength']}) | {j['qc_status']}"
        options_html += f'<option value="{escape(val)}">{escape(label)}</option>'

    items = list_feedback(limit=25)
    rows_html = ""
    for item in items:
        v_text = VERDICT_LABELS.get(item["verdict"], item["verdict"])
        src_badge = "🖥️ GE DICOM" if item["source"] == "ge_dicom" else "🌐 صفحة التقييم"
        rows_html += f"""
        <tr>
          <td>#{item['id']}</td>
          <td>{src_badge}</td>
          <td>{escape(item['patient_name'] or '-')}</td>
          <td>{escape(item['series_description'] or item['study_description'] or '-')}</td>
          <td><span class="pill">{escape(item['level'] or '-')}</span></td>
          <td>{v_text}</td>
          <td>{escape(item['note'] or '-')}</td>
          <td style="font-size:12px;color:#94a3b8;">{escape(item['created_at'])}</td>
        </tr>
        """

    saved_banner = '<div class="banner-ok">✅ تم حفظ التقييم والملاحظة بنجاح!</div>' if saved else ""
    cal_btn = (
        f'<form method="post" action="/api/v1/feedback/calibration" style="display:inline;">'
        f'<input type="hidden" name="center_id" value="{cid}">'
        f'<input type="hidden" name="enabled" value="{0 if cal_on else 1}">'
        f'<button type="submit" class="btn {"btn-warn" if cal_on else "btn-primary"}">'
        f'{"⏸️ إيقاف وضع الـ 3 مستويات والعودة لنسخة واحدة" if cal_on else "▶️ تفعيل وضع الـ 3 مستويات (L1/L2/L3)"}'
        f'</button></form>'
    )

    return f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>RAD4AI - تقييم وملاحظات الفني</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:18px; }}
    .wrap {{ max-width: 960px; margin: 0 auto; }}
    .card {{ background:#1e293b; border:1px solid #334155; border-radius:12px; padding:18px; margin-bottom:16px; }}
    h1 {{ margin:0 0 8px 0; font-size:22px; color:#38bdf8; }}
    label {{ display:block; margin:10px 0 6px; font-weight:600; font-size:14px; }}
    select, textarea {{ width:100%; box-sizing:border-box; background:#0f172a; color:#f8fafc; border:1px solid #475569; border-radius:8px; padding:10px; font-size:14px; }}
    .btn {{ border:none; border-radius:8px; padding:10px 16px; font-size:14px; font-weight:700; cursor:pointer; color:#fff; }}
    .btn-primary {{ background:#0284c7; }}
    .btn-warn {{ background:#d97706; }}
    .btn-ok {{ background:#16a34a; width:100%; margin-top:12px; padding:12px; font-size:16px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:8px; margin-top:8px; }}
    .radio-btn {{ background:#0f172a; border:1px solid #475569; border-radius:8px; padding:10px; text-align:center; cursor:pointer; font-size:13px; }}
    .radio-btn input {{ margin-left:6px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ padding:8px 10px; border-bottom:1px solid #334155; text-align:right; }}
    th {{ color:#94a3b8; }}
    .pill {{ background:#334155; padding:2px 8px; border-radius:99px; font-size:12px; }}
    .banner-ok {{ background:#14532d; border:1px solid #22c55e; color:#dcfce7; padding:10px 14px; border-radius:8px; margin-bottom:14px; font-weight:700; }}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>🩺 بوابة تقييم صور الرنين — RAD4AI</h1>
    <p style="margin:0 0 12px; color:#cbd5e1;">المركز النشط: <strong>{escape(cname)}</strong> |
      وضع التقييم (3 مستويات): <strong>{"🟢 مفعّل (يرسل L1-SOFT / L2-BALANCED / L3-CRISP)" if cal_on else "⚪ متوقف (يرسل نسخة واحدة فقط)"}</strong>
    </p>
    {cal_btn}
  </div>
  {saved_banner}
  <div class="card">
    <form method="post" action="/feedback/submit">
      <label>1) اختر الفحص / السلسلة:</label>
      <select name="series_info">{options_html}</select>

      <label>2) المستوى المفضّل (إن وجد):</label>
      <div class="grid">
        <label class="radio-btn"><input type="radio" name="level" value="L1-SOFT">L1-SOFT (ناعم)</label>
        <label class="radio-btn"><input type="radio" name="level" value="L2-BALANCED" checked>L2-BALANCED (متوازن)</label>
        <label class="radio-btn"><input type="radio" name="level" value="L3-CRISP">L3-CRISP (خشن وواضح)</label>
        <label class="radio-btn"><input type="radio" name="level" value="ALL">الكل</label>
      </div>

      <label>3) التقييم:</label>
      <div class="grid">
        <label class="radio-btn"><input type="radio" name="verdict" value="APPROVED" checked>✅ موافق وممتاز</label>
        <label class="radio-btn"><input type="radio" name="verdict" value="MORE_GRAIN">🌾 يحتاج خشونة</label>
        <label class="radio-btn"><input type="radio" name="verdict" value="SMOOTHER">🧈 يحتاج تنعيم</label>
        <label class="radio-btn"><input type="radio" name="verdict" value="BRIGHTER_T2">💧 تباين / سوائل</label>
        <label class="radio-btn"><input type="radio" name="verdict" value="REJECT">❌ مرفوض</label>
      </div>

      <label>4) ملاحظات التعديل (بالعربي أو الإنجليزي):</label>
      <textarea name="note" rows="3" placeholder="مثال: الـ T2 ممتاز لكن FLAIR يحتاج سنة خشونة، أو L2 هو الأفضل..."></textarea>

      <button type="submit" class="btn btn-ok">إرسال التقييم والملاحظة</button>
    </form>
  </div>

  <div class="card">
    <h2 style="margin:0 0 10px; font-size:17px; color:#38bdf8;">📋 آخر التقييمات والملاحظات (الويب + المرتجع من GE)</h2>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr><th>#</th><th>المصدر</th><th>المريض</th><th>السلسلة</th><th>المستوى</th><th>التقييم</th><th>الملاحظة</th><th>الوقت</th></tr>
        </thead>
        <tbody>{rows_html or '<tr><td colspan="8" style="text-align:center;color:#94a3b8;">لا توجد تقييمات مسجلة بعد</td></tr>'}</tbody>
      </table>
    </div>
  </div>
</div>
</body>
</html>"""
