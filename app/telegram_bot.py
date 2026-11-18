from __future__ import annotations

import io
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from scipy.ndimage import gaussian_filter, sobel

from app.store import (
    audit,
    connect,
    get_center,
    list_feedback,
    record_feedback,
    set_center_calibration,
    set_setting,
    setting,
)

log = logging.getLogger("telegram-bot")


def _register_chat(chat_id: int) -> None:
    try:
        raw = setting("telegram_chat_ids", "[]")
        ids = json.loads(raw) if raw else []
        if not isinstance(ids, list):
            ids = []
        if chat_id not in ids:
            ids.append(chat_id)
            set_setting("telegram_chat_ids", json.dumps(ids[-20:]))
    except Exception:
        pass


def send_telegram_notification(text: str) -> int:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return 0
    try:
        raw = setting("telegram_chat_ids", "[]")
        ids = json.loads(raw) if raw else []
    except Exception:
        ids = []
    sent = 0
    if not ids:
        return 0
    with httpx.Client(timeout=15) as client:
        for cid in ids:
            try:
                r = client.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": cid, "text": text, "parse_mode": "Markdown"},
                )
                if r.status_code == 200:
                    sent += 1
            except Exception:
                pass
    return sent




def render_before_after_png(source_series_id: str, output_series_id: str, slice_index: int | None = None) -> tuple[bytes, int, int, int]:
    import tempfile
    import pydicom
    import SimpleITK as sitk
    from scipy.ndimage import zoom

    orthanc_url = os.getenv("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
    auth = (os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", ""))

    def _instances_by_num(api: httpx.Client, sid: str) -> list[tuple[int, str]]:
        r = api.get(f"{orthanc_url}/series/{sid}/instances")
        r.raise_for_status()
        items = []
        for inst in r.json():
            num_raw = inst.get("MainDicomTags", {}).get("InstanceNumber", "0")
            try:
                num = int(str(num_raw).strip() or 0)
            except ValueError:
                num = 0
            items.append((num, inst["ID"]))
        items.sort(key=lambda x: x[0])
        return items

    def _read_inst(api: httpx.Client, iid: str) -> tuple[np.ndarray, int]:
        raw = api.get(f"{orthanc_url}/instances/{iid}/file")
        raw.raise_for_status()
        ds = pydicom.dcmread(io.BytesIO(raw.content))
        inum = int(getattr(ds, "InstanceNumber", 0) or 0)
        return ds.pixel_array.astype(np.float32), inum

    with httpx.Client(auth=auth, timeout=30) as api:
        src_items = _instances_by_num(api, source_series_id)
        dst_items = _instances_by_num(api, output_series_id)
        if not src_items or not dst_items:
            raise ValueError("Empty series")
        src_by_num = {num: iid for num, iid in src_items}
        dst_by_num = {num: iid for num, iid in dst_items}
        common_nums = sorted(set(src_by_num.keys()) & set(dst_by_num.keys()))

        if common_nums:
            n = len(common_nums)
            if slice_index is not None:
                best_idx = max(0, min(n - 1, int(slice_index)))
                chosen_num = common_nums[best_idx]
                src_arr, _ = _read_inst(api, src_by_num[chosen_num])
            else:
                lo_idx = n // 4 if n >= 4 else 0
                hi_idx = max(lo_idx + 1, (3 * n) // 4) if n >= 4 else n
                candidates = list(range(lo_idx, hi_idx))
                if len(candidates) > 7:
                    candidates = candidates[:: max(1, len(candidates) // 7)]
                best_idx = n // 2
                best_score = -1.0
                src_arr = None
                chosen_num = common_nums[best_idx]
                for idx in candidates:
                    num = common_nums[idx]
                    arr, _ = _read_inst(api, src_by_num[num])
                    sm = gaussian_filter(arr, sigma=1.0)
                    score = float(np.mean(np.hypot(sobel(sm, axis=0), sobel(sm, axis=1)))) * float(np.std(arr))
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                        src_arr = arr
                        chosen_num = num
                if src_arr is None:
                    src_arr, _ = _read_inst(api, src_by_num[chosen_num])
            dst_arr, _ = _read_inst(api, dst_by_num[chosen_num])
        else:
            n = 1
            best_idx = 0
            src_arr, chosen_num = _read_inst(api, src_items[0][1])
            dst_arr, _ = _read_inst(api, dst_items[0][1])
    if dst_arr.shape != src_arr.shape:
        src_arr = zoom(src_arr, (dst_arr.shape[0] / src_arr.shape[0], dst_arr.shape[1] / src_arr.shape[1]), order=1)

    lo = float(np.percentile(src_arr, 1.0))
    hi = max(float(np.percentile(src_arr, 99.5)), lo + 1.0)
    src_u8 = np.clip((src_arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    dst_u8 = np.clip((dst_arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    divider = np.full((src_u8.shape[0], 4), 220, dtype=np.uint8)
    combined = np.concatenate([src_u8, divider, dst_u8], axis=1)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sitk.WriteImage(sitk.GetImageFromArray(combined), tmp_path)
        return Path(tmp_path).read_bytes(), chosen_num, best_idx, n
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _feedback_keyboard_for_job(job_id: int, slice_idx: int = 0, total_slices: int = 1) -> dict[str, Any]:
    prev_idx = max(0, slice_idx - 1)
    next_idx = min(max(0, total_slices - 1), slice_idx + 1)
    rows = []
    if total_slices > 1:
        rows.append([
            {"text": "⬅️ الشريحة السابقة", "callback_data": f"sl_{job_id}_{prev_idx}"},
            {"text": f"📄 {slice_idx + 1}/{total_slices}", "callback_data": "noop"},
            {"text": "➡️ الشريحة التالية", "callback_data": f"sl_{job_id}_{next_idx}"},
        ])
    rows.extend([
        [
            {"text": "✅ ممتاز وموافق", "callback_data": f"fb_ok_{job_id}"},
            {"text": "🌾 يحتاج خشونة", "callback_data": f"fb_grain_{job_id}"},
        ],
        [
            {"text": "🧈 يحتاج تنعيم", "callback_data": f"fb_smooth_{job_id}"},
            {"text": "❌ مرفوض", "callback_data": f"fb_reject_{job_id}"},
        ],
    ])
    return {"inline_keyboard": rows}


def send_before_after_to_telegram(
    source_series_id: str,
    output_series_id: str,
    caption: str,
    job_id: int = 0,
) -> int:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return 0
    try:
        raw_ids = setting("telegram_chat_ids", "[]")
        ids = json.loads(raw_ids) if raw_ids else []
    except Exception:
        ids = []
    if not ids:
        return 0
    png_bytes, slice_num, slice_idx, total_slices = render_before_after_png(source_series_id, output_series_id)
    caption = f"{caption}\n• شريحة متطابقة: `#{slice_num}` (`{slice_idx + 1}/{total_slices}`)"
    sent = 0
    markup = json.dumps(_feedback_keyboard_for_job(job_id, slice_idx, total_slices)) if job_id else None
    with httpx.Client(timeout=30) as client:
        for cid in ids:
            try:
                data = {"chat_id": str(cid), "caption": caption, "parse_mode": "Markdown"}
                if markup:
                    data["reply_markup"] = markup
                r = client.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data=data,
                    files={"photo": ("before_after.png", png_bytes, "image/png")},
                )
                if r.status_code == 200:
                    sent += 1
            except Exception:
                pass
    return sent




def enhance_slice_with_gemini(source_series_id: str) -> tuple[bytes, int]:
    import base64
    import tempfile
    import pydicom
    import SimpleITK as sitk
    from scipy.ndimage import zoom

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    orthanc_url = os.getenv("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
    auth = (os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", ""))

    with httpx.Client(auth=auth, timeout=30) as api:
        r = api.get(f"{orthanc_url}/series/{source_series_id}/instances")
        r.raise_for_status()
        items = []
        for inst in r.json():
            try:
                num = int(str(inst.get("MainDicomTags", {}).get("InstanceNumber", "0")).strip() or 0)
            except ValueError:
                num = 0
            items.append((num, inst["ID"]))
        items.sort(key=lambda x: x[0])
        if not items:
            raise ValueError("Empty series")
        n = len(items)
        lo_idx = n // 4 if n >= 4 else 0
        hi_idx = max(lo_idx + 1, (3 * n) // 4) if n >= 4 else n
        cands = list(range(lo_idx, hi_idx))
        if len(cands) > 7:
            cands = cands[:: max(1, len(cands) // 7)]
        best_arr, best_num, best_score = None, items[n // 2][0], -1.0
        for idx in cands:
            num, iid = items[idx]
            raw = api.get(f"{orthanc_url}/instances/{iid}/file")
            raw.raise_for_status()
            ds = pydicom.dcmread(io.BytesIO(raw.content))
            arr = ds.pixel_array.astype(np.float32)
            sm = gaussian_filter(arr, sigma=1.0)
            score = float(np.mean(np.hypot(sobel(sm, axis=0), sobel(sm, axis=1)))) * float(np.std(arr))
            if score > best_score:
                best_score, best_arr = score, arr
                best_num = int(getattr(ds, "InstanceNumber", 0) or num)

    lo = float(np.percentile(best_arr, 1.0))
    hi = max(float(np.percentile(best_arr, 99.5)), lo + 1.0)
    src_u8 = np.clip((best_arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        in_path = tmp_in.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
        out_path = tmp_out.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_comb:
        comb_path = tmp_comb.name
    try:
        sitk.WriteImage(sitk.GetImageFromArray(src_u8), in_path)
        b64_in = base64.b64encode(Path(in_path).read_bytes()).decode("ascii")
        prompt = (
            "Enhance this 0.3T MRI grayscale slice to have the crisp clarity, clean denoising, and balanced contrast "
            "of a high-field 3T MRI while strictly preserving every anatomical structure, bone, disc, vessel, and nerve "
            "without adding or removing anything. Output a pure grayscale MRI image with no text overlays."
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/png", "data": b64_in}}]}],
            "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
        }
        resp = None
        last_status = 0
        for model_name in ("gemini-2.5-flash-image", "gemini-3.1-flash-image", "gemini-3-pro-image"):
            r_try = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                headers={"x-goog-api-key": api_key},
                json=payload,
                timeout=90,
            )
            last_status = r_try.status_code
            if r_try.status_code == 200:
                resp = r_try
                break
            if r_try.status_code not in (429, 404, 503):
                r_try.raise_for_status()
        if resp is None:
            if last_status == 429:
                raise RuntimeError("حصة Gemini API للصور ممتلئة حالياً (429 Too Many Requests). استخدم التحسين المحلي أو فعّل الفوترة على Google AI Studio.")
            raise RuntimeError(f"تعذر الاتصال بنموذج Gemini للصور (HTTP {last_status})")
        gem_bytes = None
        for cand in resp.json().get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                idata = part.get("inlineData") or part.get("inline_data")
                if idata and idata.get("data"):
                    gem_bytes = base64.b64decode(idata["data"])
                    break
            if gem_bytes:
                break
        if not gem_bytes:
            raise RuntimeError("Gemini did not return an image")
        Path(out_path).write_bytes(gem_bytes)
        gem_raw = sitk.GetArrayFromImage(sitk.ReadImage(out_path)).astype(np.float32)
        gem_gray = gem_raw.mean(axis=-1) if gem_raw.ndim == 3 else gem_raw
        gem_u8 = np.clip(gem_gray, 0, 255).astype(np.uint8)
        if gem_u8.shape != src_u8.shape:
            gem_u8 = zoom(gem_u8.astype(np.float32), (src_u8.shape[0] / gem_u8.shape[0], src_u8.shape[1] / gem_u8.shape[1]), order=1).astype(np.uint8)
        divider = np.full((src_u8.shape[0], 4), 220, dtype=np.uint8)
        combined = np.concatenate([src_u8, divider, gem_u8], axis=1)
        sitk.WriteImage(sitk.GetImageFromArray(combined), comb_path)
        return Path(comb_path).read_bytes(), best_num
    finally:
        for fp in (in_path, out_path, comb_path):
            Path(fp).unlink(missing_ok=True)


def _detect_verdict_and_level(text: str) -> tuple[str, str]:
    u = text.upper()
    level = (
        "L1-SOFT" if ("L1" in u or "المستوى 1" in text or "الاول" in text)
        else "L2-BALANCED" if ("L2" in u or "المستوى 2" in text or "الثاني" in text)
        else "L3-CRISP" if ("L3" in u or "المستوى 3" in text or "الثالث" in text)
        else "L4-PRINT" if ("L4" in u or "طباع" in text or "PRINT" in u)
        else "L5-AI3T" if ("L5" in u or "3T" in u)
        else ""
    )
    verdict = (
        "APPROVED" if any(k in text for k in ("موافق", "مقبول", "ممتاز", "تمام", "ثبت", "كويس")) or "ACCEPT" in u or "OK" in u
        else "MORE_GRAIN" if any(k in text for k in ("خشون", "خشن")) or "GRAIN" in u
        else "SMOOTHER" if any(k in text for k in ("ناعم", "تنعيم")) or "SMOOTH" in u
        else "BRIGHTER_T2" if any(k in text for k in ("سوائل", "ابيض", "تباين")) or "BRIGHT" in u
        else "REJECT" if any(k in text for k in ("مرفوض", "رفض", "سيء", "بايظ")) or "REJECT" in u
        else "NOTE"
    )
    return verdict, level


def _analyze_reference_image(img_path: Path) -> dict[str, float]:
    import SimpleITK as sitk
    raw = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path))).astype(np.float32)
    arr = raw.mean(axis=-1) if raw.ndim == 3 else raw
    arr = arr / max(float(arr.max()), 1.0)
    smooth = gaussian_filter(arr, sigma=1.0)
    gx, gy = sobel(smooth, axis=1), sobel(smooth, axis=0)
    edge_energy = float(np.mean(np.hypot(gx, gy)))
    noise_est = float(np.std(arr - smooth))
    p10, p95 = float(np.percentile(arr, 10)), float(np.percentile(arr, 95))
    return {
        "width": float(arr.shape[1]),
        "height": float(arr.shape[0]),
        "edge_energy": round(edge_energy, 4),
        "noise_est": round(noise_est, 4),
        "contrast_span": round(p95 - p10, 4),
        "bg_floor_p10": round(p10, 4),
    }


class TelegramBotWorker:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_error: str | None = None
        self.bot_username: str = ""
        self.offset: int = 0

    @property
    def token(self) -> str:
        return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    def start(self) -> None:
        if not self.token:
            log.info("TELEGRAM_BOT_TOKEN not set; Telegram bot disabled")
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.token),
            "running": bool(self.thread and self.thread.is_alive()),
            "username": self.bot_username,
            "last_error": self.last_error,
        }

    def _api(self, client: httpx.Client, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        r = client.post(url, json=payload or {}, timeout=35)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error: {data}")
        return data

    def _menu_keyboard(self) -> dict[str, Any]:
        return {
            "inline_keyboard": [
                [
                    {"text": "📊 حالة السيرفر وآخر فحص", "callback_data": "status"},
                    {"text": "📋 آخر التقييمات", "callback_data": "feedbacks"},
                ],
                [
                    {"text": "🟢 قوة MILD (L1)", "callback_data": "str_MILD"},
                    {"text": "🟡 قوة MEDIUM (L2)", "callback_data": "str_MEDIUM"},
                    {"text": "🔴 قوة STRONG (L3)", "callback_data": "str_STRONG"},
                ],
                [
                    {"text": "☀️ إضاءة +10%", "callback_data": "bc_bright_up"},
                    {"text": "🌙 إضاءة -10%", "callback_data": "bc_bright_down"},
                    {"text": "↩️ افتراضي", "callback_data": "bc_reset"},
                ],
                [
                    {"text": "🌓 كونتراست +10%", "callback_data": "bc_cont_up"},
                    {"text": "🌫️ كونتراست -10%", "callback_data": "bc_cont_down"},
                ],
                [
                    {"text": "🦴 SPINE/ORTHO", "callback_data": "prof_RAD4AI_SPINE_PRO"},
                    {"text": "🧠 BRAIN RICIAN", "callback_data": "prof_RICIAN_NLM_VST"},
                ],
                [
                    {"text": "🎛️ تفعيل الـ 5 مستويات", "callback_data": "cal_on"},
                    {"text": "🔒 نسخة واحدة فقط", "callback_data": "cal_off"},
                ],
                [
                    {"text": "🖼️ قبل وبعد لآخر فحص", "callback_data": "before_after"},
                    {"text": "🔄 إعادة إرسال آخر فحص لـ GE", "callback_data": "reprocess_last"},
                ],
                [
                    {"text": "✨ تحسين خارجي (Gemini AI)", "callback_data": "gemini_enhance"},
                    {"text": "📸 مقارنة صورة مرجعية", "callback_data": "ref_guide"},
                ],
                [
                    {"text": "📤 بطاقات GE", "callback_data": "send_cards"},
                ],

            ]
        }

    def _format_status(self) -> str:
        c = get_center(8)
        cal = bool(c and c["calibration_mode"])
        strength = c["auto_strength"] if c else "STRONG"
        prof = c["auto_profile"] if c else "RAD4AI_SPINE_PRO"
        with connect() as db:
            job = db.execute(
                "SELECT cj.id, cj.profile, cj.strength, cj.status, cj.qc_status, cj.updated_at, "
                "COALESCE(ib.description, '') AS desc "
                "FROM comparison_jobs cj LEFT JOIN inbound_series ib ON ib.series_id=cj.source_series_id "
                "ORDER BY cj.id DESC LIMIT 1"
            ).fetchone()
            ret = db.execute(
                "SELECT id, comparison_job_id, status, updated_at FROM return_jobs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        j_txt = (
            f"#{job['id']} | {job['desc'] or '-'} | {job['profile']} ({job['strength']}) | {job['qc_status']}"
            if job else "لا توجد وظائف"
        )
        r_txt = f"#{ret['id']} (Job #{ret['comparison_job_id']}): {ret['status']} @ {ret['updated_at']}" if ret else "لا يوجد"
        return (
            "🖥️ *حالة سيرفر RAD4AI*\n"
            f"• البروفايل: `{prof}` | القوة: *{strength}*\n"
            f"• وضع المعايرة (5 مستويات): *{'مفعّل 🟢' if cal else 'متوقف (نسخة واحدة) ⚪'}*\n"
            f"• آخر معالجة: `{j_txt}`\n"
            f"• آخر إرسال لـ GE: `{r_txt}`"
        )

    def _format_feedbacks(self) -> str:
        items = list_feedback(limit=5)
        if not items:
            return "لا توجد تقييمات مسجلة حتى الآن."
        lines = ["📋 *آخر 5 تقييمات وملاحظات:*"]
        for it in items:
            lines.append(
                f"• `#{it['id']}` [{it['source']}] {it['series_description'] or '-'} | "
                f"*{it['level'] or '-'}* | {it['verdict']} — {it['note']}"
            )
        return "\n".join(lines)

    def _handle_callback(self, client: httpx.Client, cb: dict[str, Any]) -> None:
        cb_id = cb["id"]
        chat_id = cb["message"]["chat"]["id"]
        _register_chat(int(chat_id))
        data = cb.get("data", "")
        self._api(client, "answerCallbackQuery", {"callback_query_id": cb_id})

        if data == "status":
            reply = self._format_status()
        elif data == "feedbacks":
            reply = self._format_feedbacks()
        elif data.startswith("str_"):
            new_str = data[4:]
            with connect() as db:
                db.execute("UPDATE centers SET auto_strength=? WHERE id=8", (new_str,))
            audit("TELEGRAM_STRENGTH_CHANGED", "center", "8", f"strength={new_str};chat={chat_id}")
            reply = f"✅ تم تغيير قوة التحسين الافتراضية للمركز إلى: *{new_str}*"
        elif data.startswith("prof_"):
            new_prof = data[5:]
            with connect() as db:
                db.execute("UPDATE centers SET auto_profile=? WHERE id=8", (new_prof,))
            audit("TELEGRAM_PROFILE_CHANGED", "center", "8", f"profile={new_prof};chat={chat_id}")
            reply = f"✅ تم تغيير البروفايل الافتراضي للمركز إلى: `{new_prof}`"
        elif data == "cal_on":
            set_center_calibration(8, True)
            audit("TELEGRAM_CALIBRATION_ON", "center", "8", f"chat={chat_id}")
            reply = "🟢 تم تفعيل وضع الـ 5 مستويات (`L1` إلى `L5`)."
        elif data == "cal_off":
            set_center_calibration(8, False)
            audit("TELEGRAM_CALIBRATION_OFF", "center", "8", f"chat={chat_id}")
            reply = "⚪ تم إيقاف وضع المعايرة — سيتم إرسال نسخة واحدة فقط."
        elif data == "send_cards":
            try:
                from app.feedback_cards import send_feedback_cards_to_center
                res = send_feedback_cards_to_center(8)
                reply = f"📤 تم إرسال {res['sent_cards']} بطاقات تقييم إلى `{res['destination']}` بنجاح."
            except Exception as exc:
                reply = f"❌ تعذر إرسال البطاقات: {exc}"
        elif data == "before_after":
            with connect() as db:
                j = db.execute(
                    "SELECT cj.id, cj.source_series_id, cj.output_series_id, cj.profile, cj.strength, cj.qc_status, "
                    "COALESCE(ib.description,'') AS desc "
                    "FROM comparison_jobs cj LEFT JOIN inbound_series ib ON ib.series_id=cj.source_series_id "
                    "WHERE cj.status='ready' AND cj.output_series_id IS NOT NULL ORDER BY cj.id DESC LIMIT 1"
                ).fetchone()
            if j:
                try:
                    cap = (
                        f"🖼️ *قبل (يسار) | بعد (يمين)*\n"
                        f"• السلسلة: `{j['desc'] or '-'}` (`#{j['id']}`)\n"
                        f"• البروفايل: `{j['profile']}` ({j['strength']}) | QC: *{j['qc_status']}*"
                    )
                    send_before_after_to_telegram(j["source_series_id"], j["output_series_id"], cap, job_id=int(j["id"]))
                    return
                except Exception as exc:
                    reply = f"❌ تعذر تجهيز صورة المقارنة: {exc}"
            else:
                reply = "لا توجد حالة مكتملة بعد."
        elif data == "noop":
            return
        elif data.startswith("sl_"):
            parts = data.split("_", 2)
            jid = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            s_idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            with connect() as db:
                j = db.execute(
                    "SELECT cj.id, cj.source_series_id, cj.output_series_id, cj.profile, cj.strength, cj.qc_status, "
                    "COALESCE(ib.description,'') AS desc "
                    "FROM comparison_jobs cj LEFT JOIN inbound_series ib ON ib.series_id=cj.source_series_id "
                    "WHERE cj.id=?",
                    (jid,),
                ).fetchone()
            if j and j["source_series_id"] and j["output_series_id"]:
                try:
                    png_bytes, slice_num, cur_idx, total = render_before_after_png(
                        j["source_series_id"], j["output_series_id"], slice_index=s_idx
                    )
                    cap = (
                        f"🖼️ *قبل (يسار) | بعد (يمين)*\n"
                        f"• السلسلة: `{j['desc'] or '-'}` (`#{j['id']}`)\n"
                        f"• البروفايل: `{j['profile']}` ({j['strength']}) | QC: *{j['qc_status']}*\n"
                        f"• شريحة متطابقة: `#{slice_num}` (`{cur_idx + 1}/{total}`)"
                    )
                    client.post(
                        f"https://api.telegram.org/bot{self.token}/editMessageMedia",
                        data={
                            "chat_id": str(chat_id),
                            "message_id": str(cb["message"]["message_id"]),
                            "media": json.dumps({"type": "photo", "media": "attach://photo", "caption": cap, "parse_mode": "Markdown"}),
                            "reply_markup": json.dumps(_feedback_keyboard_for_job(jid, cur_idx, total)),
                        },
                        files={"photo": ("before_after.png", png_bytes, "image/png")},
                        timeout=30,
                    )
                    return
                except Exception as exc:
                    reply = f"❌ تعذر تحميل الشريحة: {exc}"
            else:
                reply = "السلسلة غير متاحة."
        elif data.startswith("fb_"):
            parts = data.split("_", 2)
            action = parts[1] if len(parts) > 1 else "ok"
            jid = parts[2] if len(parts) > 2 else ""
            verdict_map = {
                "ok": ("APPROVED", "✅ ممتاز وموافق"),
                "grain": ("MORE_GRAIN", "🌾 يحتاج خشونة"),
                "smooth": ("SMOOTHER", "🧈 يحتاج تنعيم"),
                "reject": ("REJECT", "❌ مرفوض"),
            }
            verdict, label = verdict_map.get(action, ("NOTE", action))
            sdesc = ""
            if jid.isdigit():
                with connect() as db:
                    r = db.execute(
                        "SELECT COALESCE(ib.description,'') AS desc FROM comparison_jobs cj "
                        "LEFT JOIN inbound_series ib ON ib.series_id=cj.source_series_id WHERE cj.id=?",
                        (int(jid),),
                    ).fetchone()
                    if r:
                        sdesc = str(r["desc"] or "")
            fid = record_feedback(
                source="telegram",
                series_id=jid,
                patient_name=cb.get("from", {}).get("first_name", "tech"),
                study_description=f"Job #{jid}",
                series_description=sdesc,
                verdict=verdict,
                note=f"تقييم على صورة قبل/بعد (#{jid}): {label}",
                source_aet=f"tg:{chat_id}",
            )
            audit("TELEGRAM_PHOTO_FEEDBACK", "feedback", str(fid), f"job={jid};verdict={verdict}")
            reply = f"{label} — تم تسجيل تقييمك بنجاح برقم `#{fid}`."
        elif data.startswith("bc_"):
            br = float(setting("clinic_brightness", "1.0") or 1.0)
            ct = float(setting("clinic_contrast", "1.0") or 1.0)
            if data == "bc_bright_up":
                br = round(min(1.25, br + 0.10), 2)
            elif data == "bc_bright_down":
                br = round(max(0.75, br - 0.10), 2)
            elif data == "bc_cont_up":
                ct = round(min(1.35, ct + 0.10), 2)
            elif data == "bc_cont_down":
                ct = round(max(0.75, ct - 0.10), 2)
            elif data == "bc_reset":
                br, ct = 1.0, 1.0
            set_setting("clinic_brightness", str(br))
            set_setting("clinic_contrast", str(ct))
            audit("TELEGRAM_BC_CHANGED", "settings", "8", f"brightness={br};contrast={ct};chat={chat_id}")
            reply = (
                f"🎛️ تم تحديث الإضاءة والكونتراست:\n"
                f"• الإضاءة (`Brightness`): *{int(round(br * 100))}%*\n"
                f"• التباين (`Contrast`): *{int(round(ct * 100))}%*\n"
                "اضغط `🔄 إعادة إرسال آخر فحص لـ GE` لتطبيقها ومشاهدتها فوراً."
            )
        elif data == "reprocess_last":
            c = get_center(8)
            prof = c["auto_profile"] if c else "RAD4AI_SPINE_PRO"
            strength = c["auto_strength"] if c else "STRONG"
            with connect() as db:
                last_j = db.execute(
                    "SELECT cj.source_series_id, COALESCE(ib.description,'') AS desc "
                    "FROM comparison_jobs cj LEFT JOIN inbound_series ib ON ib.series_id=cj.source_series_id "
                    "ORDER BY cj.id DESC LIMIT 1"
                ).fetchone()
                if last_j:
                    cur = db.execute(
                        "INSERT INTO comparison_jobs(source_series_id, profile, strength, acknowledged) VALUES(?,?,?,1)",
                        (last_j["source_series_id"], prof, strength),
                    )
                    new_jid = int(cur.lastrowid)
                    audit("TELEGRAM_REPROCESS_LAST", "comparison_job", str(new_jid), f"chat={chat_id}")
                    reply = (
                        f"🔄 تمت جدولة إعادة معالجة وإرسال آخر سلسلة (`{last_j['desc'] or '-'}`) "
                        f"برقم `#{new_jid}` (`{prof}` / `{strength}`).\n"
                        "ستصلك النتيجة وصورة قبل/بعد تلقائياً فور الانتهاء والإرسال لـ GE."
                    )
                else:
                    reply = "لا توجد سلسلة سابقة لإعادة معالجتها."
        elif data == "gemini_enhance":
            with connect() as db:
                j = db.execute(
                    "SELECT cj.id, cj.source_series_id, COALESCE(ib.description,'') AS desc "
                    "FROM comparison_jobs cj LEFT JOIN inbound_series ib ON ib.series_id=cj.source_series_id "
                    "ORDER BY cj.id DESC LIMIT 1"
                ).fetchone()
            if j:
                try:
                    self._api(client, "sendMessage", {"chat_id": chat_id, "text": "⏳ جاري التحسين الخارجي عبر Gemini AI (للعرض فقط)..."})
                    png_bytes, slice_num = enhance_slice_with_gemini(j["source_series_id"])
                    cap = (
                        f"✨ *تحسين خارجي (Gemini AI) — قبل (يسار) | بعد (يمين)*\n"
                        f"• السلسلة: `{j['desc'] or '-'}` (`#{j['id']}`) | شريحة `#{slice_num}`\n"
                        "⚠️ *للعرض والمقارنة البصرية فقط — لا يُستخدم للتشخيص ولا يُرسل للـ PACS.*"
                    )
                    client.post(
                        f"https://api.telegram.org/bot{self.token}/sendPhoto",
                        data={"chat_id": str(chat_id), "caption": cap, "parse_mode": "Markdown"},
                        files={"photo": ("gemini_before_after.png", png_bytes, "image/png")},
                        timeout=30,
                    )
                    audit("TELEGRAM_GEMINI_ENHANCE", "comparison_job", str(j["id"]), f"chat={chat_id}")
                    return
                except Exception as exc:
                    reply = f"❌ تعذر التحسين الخارجي عبر Gemini: {exc}"
            else:
                reply = "لا توجد سلسلة متاحة للتحسين الخارجي."
        elif data == "ref_guide":
            reply = (
                "📸 *إرسال نموذج/صورة مرجعية للمقارنة:*\n"
                "أرسل أي صورة (أو ملف صورة) هنا على التليجرام، واكتب معها نوع الفحص "
                "(مثلاً: `MRV` أو `T2 Knee` أو `STIR` أو `PD`) —\n"
                "سيقوم البوت بتحليل حدة وتباين وضوضاء الصورة المرجعية ومقارنتها فوراً بآخر فحص مرسل وحفظها للمراجعة."
            )
        else:
            reply = "أمر غير معروف."

        self._api(
            client,
            "sendMessage",
            {"chat_id": chat_id, "text": reply, "parse_mode": "Markdown", "reply_markup": self._menu_keyboard()},
        )

    def _download_and_compare_photo(self, client: httpx.Client, chat_id: int, sender: str, file_id: str, caption: str) -> None:
        finfo = self._api(client, "getFile", {"file_id": file_id})
        fpath = finfo["result"]["file_path"]
        dl = client.get(f"https://api.telegram.org/file/bot{self.token}/{fpath}", timeout=30)
        dl.raise_for_status()
        ref_dir = Path("/var/lib/enhancer/telegram_refs")
        ref_dir.mkdir(parents=True, exist_ok=True)
        saved_path = ref_dir / f"ref_{chat_id}_{Path(fpath).name}"
        saved_path.write_bytes(dl.content)

        ref_stats = _analyze_reference_image(saved_path)
        with connect() as db:
            last_job = db.execute(
                "SELECT cj.id, cj.profile, cj.strength, cj.metrics_json, COALESCE(ib.description,'') AS desc "
                "FROM comparison_jobs cj LEFT JOIN inbound_series ib ON ib.series_id=cj.source_series_id "
                "WHERE cj.status='ready' ORDER BY cj.id DESC LIMIT 1"
            ).fetchone()

        verdict, level = _detect_verdict_and_level(caption or "صورة مرجعية للمقارنة")
        fid = record_feedback(
            source="telegram_ref",
            patient_name=sender,
            study_description=caption or "Reference Sample",
            series_description=str(saved_path.name),
            level=level,
            verdict=verdict,
            note=f"{caption or 'Reference Sample'} | stats={json.dumps(ref_stats)}",
            source_aet=f"tg:{chat_id}",
        )
        audit("TELEGRAM_REF_RECEIVED", "feedback", str(fid), f"file={saved_path.name};chat={chat_id}")

        last_line = ""
        if last_job:
            m = json.loads(last_job["metrics_json"] or "{}")
            last_line = (
                f"\n• مقارنة بآخر فحص مرسل (`#{last_job['id']} {last_job['desc']}`):\n"
                f"  حواف المرسل: `{m.get('edge_preservation_percent', '-')}%` | تغير الضوضاء: `{m.get('noise_change_percent', '-')}%`"
            )

        reply = (
            f"📸 *تم استلام الصورة المرجعية وحفظها برقم `#{fid}`*\n"
            f"• النوع/الملاحظة: `{caption or 'بدون وصف'}`\n"
            f"• الأبعاد: `{int(ref_stats['width'])}×{int(ref_stats['height'])}`\n"
            f"• حدة الحواف: `{ref_stats['edge_energy']}` | التباين: `{ref_stats['contrast_span']}` | الضوضاء: `{ref_stats['noise_est']}`"
            f"{last_line}"
        )
        self._api(
            client,
            "sendMessage",
            {"chat_id": chat_id, "text": reply, "parse_mode": "Markdown", "reply_markup": self._menu_keyboard()},
        )

    def _handle_message(self, client: httpx.Client, msg: dict[str, Any]) -> None:
        chat_id = msg["chat"]["id"]
        _register_chat(int(chat_id))
        sender = msg.get("from", {}).get("first_name", "") or msg.get("from", {}).get("username", "tech")
        text = (msg.get("text") or msg.get("caption") or "").strip()

        if msg.get("photo"):
            largest = msg["photo"][-1]
            self._download_and_compare_photo(client, chat_id, sender, largest["file_id"], text)
            return

        if not text:
            return

        if text.startswith("/start") or text.startswith("/help"):
            welcome = (
                "👋 أهلاً بك في *RAD4AI MRI Bot*!\n"
                "• تحكم في جودة الصور (القوة / البروفايل / وضع الـ 5 مستويات) من الأزرار بالأسفل.\n"
                "• أرسل أي ملاحظة أو صورة مرجعية مع نوع الفحص لمقارنتها وحفظها فوراً."
            )
            self._api(
                client,
                "sendMessage",
                {"chat_id": chat_id, "text": welcome, "parse_mode": "Markdown", "reply_markup": self._menu_keyboard()},
            )
            return

        if text == "/status":
            self._api(
                client,
                "sendMessage",
                {"chat_id": chat_id, "text": self._format_status(), "parse_mode": "Markdown", "reply_markup": self._menu_keyboard()},
            )
            return

        if any(k in text for k in ("اخر صور", "قبل وبعد", "اخر حاله", "اخر فحص", "ابعتلي صور")):
            with connect() as db:
                j = db.execute(
                    "SELECT cj.id, cj.source_series_id, cj.output_series_id, cj.profile, cj.strength, cj.qc_status, "
                    "COALESCE(ib.description,'') AS desc "
                    "FROM comparison_jobs cj LEFT JOIN inbound_series ib ON ib.series_id=cj.source_series_id "
                    "WHERE cj.status='ready' AND cj.output_series_id IS NOT NULL ORDER BY cj.id DESC LIMIT 1"
                ).fetchone()
            if j:
                try:
                    cap = (
                        f"🖼️ *قبل (يسار) | بعد (يمين)*\n"
                        f"• السلسلة: `{j['desc'] or '-'}` (`#{j['id']}`)\n"
                        f"• البروفايل: `{j['profile']}` ({j['strength']}) | QC: *{j['qc_status']}*"
                    )
                    send_before_after_to_telegram(j["source_series_id"], j["output_series_id"], cap, job_id=int(j["id"]))
                    return
                except Exception:
                    pass

        verdict, level = _detect_verdict_and_level(text)
        last_sdesc = ""
        with connect() as db:
            row = db.execute("SELECT description FROM inbound_series ORDER BY received_at DESC LIMIT 1").fetchone()
            if row:
                last_sdesc = str(row["description"] or "")

        fid = record_feedback(
            source="telegram",
            patient_name=sender,
            study_description="Telegram Request",
            series_description=last_sdesc,
            level=level,
            verdict=verdict,
            note=text,
            source_aet=f"tg:{chat_id}",
        )
        audit("TELEGRAM_FEEDBACK", "feedback", str(fid), f"chat={chat_id};verdict={verdict};level={level}")
        ack = (
            f"✅ تم استلام وحفظ طلبك برقم `#{fid}`\n"
            f"• آخر سلسلة: `{last_sdesc or '-'}`\n"
            f"• التصنيف: `{verdict}` {f'| {level}' if level else ''}\n"
            "• سيتم مراجعته وتنفيذه."
        )
        self._api(
            client,
            "sendMessage",
            {"chat_id": chat_id, "text": ack, "parse_mode": "Markdown", "reply_markup": self._menu_keyboard()},
        )

    def _run(self) -> None:
        with httpx.Client(timeout=40) as client:
            try:
                me = self._api(client, "getMe")
                self.bot_username = me.get("result", {}).get("username", "")
                log.info("Telegram bot started: @%s", self.bot_username)
            except Exception as exc:
                self.last_error = str(exc)
                log.warning("Telegram getMe failed: %s", exc)

            while not self.stop_event.is_set():
                try:
                    resp = client.get(
                        f"https://api.telegram.org/bot{self.token}/getUpdates",
                        params={"offset": self.offset, "timeout": 25},
                        timeout=35,
                    )
                    resp.raise_for_status()
                    updates = resp.json().get("result", [])
                    self.last_error = None
                    for upd in updates:
                        self.offset = max(self.offset, int(upd["update_id"]) + 1)
                        if "callback_query" in upd:
                            self._handle_callback(client, upd["callback_query"])
                        elif "message" in upd:
                            self._handle_message(client, upd["message"])
                except Exception as exc:
                    self.last_error = str(exc)
                    log.warning("Telegram poll error: %s", exc)
                    self.stop_event.wait(5)
