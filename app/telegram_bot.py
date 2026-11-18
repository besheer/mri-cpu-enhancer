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
                    {"text": "🦴 SPINE/ORTHO", "callback_data": "prof_RAD4AI_SPINE_PRO"},
                    {"text": "🧠 BRAIN RICIAN", "callback_data": "prof_RICIAN_NLM_VST"},
                ],
                [
                    {"text": "🎛️ تفعيل الـ 5 مستويات", "callback_data": "cal_on"},
                    {"text": "🔒 نسخة واحدة فقط", "callback_data": "cal_off"},
                ],
                [
                    {"text": "📤 بطاقات GE", "callback_data": "send_cards"},
                    {"text": "📸 مقارنة صورة مرجعية", "callback_data": "ref_guide"},
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
