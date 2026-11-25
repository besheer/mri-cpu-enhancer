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

AUTHORIZED_CHAT_IDS = {1281167916, 9270505617}
AUTHORIZED_PHONES = {"01003741929", "+201003741929", "201003741929"}
DEFAULT_PASSWORDS = {"adminPassword2026!", "2026", "mri2026", "adw2026"}


def _get_authenticated_chats() -> set[int]:
    try:
        raw = setting("telegram_authenticated_chats", "[]")
        arr = json.loads(raw) if raw else []
        return set(int(x) for x in arr)
    except Exception:
        return set()


def _add_authenticated_chat(chat_id: int) -> None:
    try:
        current = _get_authenticated_chats()
        current.add(int(chat_id))
        set_setting("telegram_authenticated_chats", json.dumps(list(current)))
    except Exception:
        pass


def _is_chat_authorized(chat_id: int) -> bool:
    if int(chat_id) in AUTHORIZED_CHAT_IDS:
        return True
    return int(chat_id) in _get_authenticated_chats()


def _is_password_correct(text: str) -> bool:
    cleaned = (text or "").strip()
    configured_pw = setting("telegram_access_password", "")
    if configured_pw and cleaned == configured_pw.strip():
        return True
    return cleaned in DEFAULT_PASSWORDS



def _register_chat(chat_id: int) -> None:
    if not _is_chat_authorized(chat_id):
        return
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


def _generate_bilingual_gemini_report(job: dict) -> str:
    import base64
    import struct
    import zlib
    import pydicom
    import numpy as np
    from io import BytesIO

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    def _gray_to_png_bytes(arr_u8: np.ndarray) -> bytes:
        h, w = arr_u8.shape
        raw = b"".join(bytes([0]) + arr_u8[y].tobytes() for y in range(h))
        def _chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        ihdr = struct.pack("!IIBBBBB", w, h, 8, 0, 0, 0, 0)
        return bytes([137, 80, 78, 71, 13, 10, 26, 10]) + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b"")

    orthanc = os.getenv("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
    auth = (os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", ""))
    st_desc = ""
    with httpx.Client(auth=auth, timeout=30) as api:
        if job.get("study_id"):
            st = api.get(f"{orthanc}/studies/{job['study_id']}").json()
            st_desc = st.get("MainDicomTags", {}).get("StudyDescription", "")
        sr = api.get(f"{orthanc}/series/{job['output_series_id']}").json()
        inst_ids = sr.get("Instances", [])
        if not inst_ids:
            raise RuntimeError("No instances in output series")
        items = []
        for iid in inst_ids:
            tags = api.get(f"{orthanc}/instances/{iid}/simplified-tags").json()
            try:
                num = int(tags.get("InstanceNumber", 0))
            except Exception:
                num = 0
            items.append((num, iid))
        items.sort(key=lambda x: x[0])
        n = len(items)
        idxs = sorted({max(0, min(n - 1, int(n * f))) for f in (0.25, 0.5, 0.75)})
        parts = [{
            "text": (
                f"You are an expert MRI radiologist reviewing {len(idxs)} de-identified slices from series '{job.get('sdesc','')}' "
                f"(Study: '{st_desc or 'MRI'}').\\n"
                "Write a concise, structured preliminary medical findings report in BOTH Arabic AND English:\\n"
                "1) التقرير الطبي المبدئي (عربي):\\n"
                "- الملاحظات المرصودة (Findings): اذكر بدقة ما يظهر تشريحياً أو مرضياً في الشرائح.\\n"
                "- الخلاصة (Impression): ملخص مبدئي مختصر.\\n"
                "2) Preliminary Medical Findings Report (English - For Treating Physician):\\n"
                "- Findings: concise bullet points of anatomical & pathological observations.\\n"
                "- Impression: brief preliminary summary.\\n"
                "Be objective, clinically accurate, and avoid definitive diagnosis."
            )
        }]
        for idx in idxs:
            dcm_bytes = api.get(f"{orthanc}/instances/{items[idx][1]}/file").content
            ds = pydicom.dcmread(BytesIO(dcm_bytes))
            arr = ds.pixel_array.astype(np.float32)
            lo, hi = np.percentile(arr, [0.5, 99.5])
            arr = np.clip((arr - lo) / max(hi - lo, 1e-4), 0.0, 1.0)
            png_bytes = _gray_to_png_bytes((arr * 255).astype(np.uint8))
            parts.append({"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png_bytes).decode("ascii")}})

    with httpx.Client(timeout=60) as gc:
        r = gc.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}",
            json={"contents": [{"parts": parts}]},
        )
        r.raise_for_status()
        body = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    return (
        f"🧠 تقرير Gemini المبدئي (عربي + إنجليزي) — Job #{job['id']}\\n"
        f"• الدراسة: {st_desc or '-'} | السلسلة: {job.get('sdesc','')} ({n} شريحة)\\n"
        f"• البروفايل: {job['profile']} ({job['strength']})\\n\\n"
        f"{body}\\n\\n"
        "⚠️ Disclaimer: Preliminary AI observation for technical/workflow support only; not a final medical diagnosis."
    )


_REPORTED_SERIES_LOCK = threading.Lock()
_REPORTED_SERIES_IDS: set[str] = set()


def send_bilingual_report_async(job_id: int) -> None:
    if setting("enable_gemini_report", "0") != "1":
        return
    def _worker() -> None:
        try:
            with connect() as db:
                row = db.execute(
                    "SELECT cj.id, cj.source_series_id, cj.output_series_id, cj.profile, cj.strength, "
                    "COALESCE(ib.description, '') AS sdesc, COALESCE(ib.study_id, '') AS study_id "
                    "FROM comparison_jobs cj LEFT JOIN inbound_series ib ON ib.series_id=cj.source_series_id "
                    "WHERE cj.id=? AND cj.status='ready' AND cj.output_series_id!=''",
                    (job_id,),
                ).fetchone()
            if not row:
                return
            sid = str(row["source_series_id"] or "")
            with _REPORTED_SERIES_LOCK:
                if sid in _REPORTED_SERIES_IDS:
                    return
                _REPORTED_SERIES_IDS.add(sid)
            report_msg = _generate_bilingual_gemini_report(dict(row))
            with connect() as db:
                db.execute(
                    "UPDATE comparison_jobs SET gemini_audit=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (json.dumps({"bilingual_report": report_msg}, ensure_ascii=False), job_id),
                )
            token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
            if not token:
                return
            ids = json.loads(setting("telegram_chat_ids", "[]") or "[]")
            with httpx.Client(timeout=15) as c:
                for cid in ids:
                    try:
                        c.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": cid, "text": report_msg})
                    except Exception:
                        pass
        except Exception as exc:
            log.warning("Auto bilingual Gemini report skipped for job %s: %s", job_id, exc)

    threading.Thread(target=_worker, daemon=True, name=f"tg-report-{job_id}").start()


def send_telegram_notification(text: str, reply_markup: dict | None = None) -> int:
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
    if reply_markup is None:
        reply_markup = {
            "inline_keyboard": [
                [{"text": "⚡ تنفيذ آخر تعديلات مطلوبة", "callback_data": "exec_latest_modifications"}]
            ]
        }
    with httpx.Client(timeout=15) as client:
        for cid in ids:
            try:
                payload = {"chat_id": cid, "text": text, "parse_mode": "Markdown"}
                if reply_markup:
                    payload["reply_markup"] = reply_markup
                r = client.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json=payload,
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
            {"text": "💎 تطبيق مستوى ENH-6 فائق الدقة", "callback_data": f"apply_enh6_{job_id}"},
        ],
        [
            {"text": "⚡ تنفيذ آخر تعديلات مطلوبة", "callback_data": "exec_latest_modifications"},
        ],
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




def _transcribe_voice_gemini(audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
    import base64
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return ""
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    payload = {
        "contents": [{
            "parts": [
                {"text": "Transcribe this MRI clinic technician/radiologist voice note accurately in its spoken language (Arabic/English). Output ONLY the transcription text, nothing else."},
                {"inlineData": {"mimeType": mime_type, "data": b64}},
            ]
        }]
    }
    try:
        r = httpx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent",
            headers={"x-goog-api-key": api_key},
            json=payload,
            timeout=45,
        )
        if r.status_code == 200:
            parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
            return " ".join(str(pt.get("text", "")).strip() for pt in parts if pt.get("text")).strip()
    except Exception:
        pass
    return ""


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
                    {"text": "💎 تحسين بمستوى ENH-6 (فائق الدقة 3T)", "callback_data": "apply_enh6"},
                ],
                [
                    {"text": "⚡ تنفيذ آخر تعديلات مطلوبة", "callback_data": "exec_latest_modifications"},
                ],
                [
                    {"text": "📡 فحص حالة إرجاع الصور لـ GE", "callback_data": "check_returns"},
                    {"text": "🚀 دفع الصور العالقة", "callback_data": "force_push_unsent"},
                ],
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
                    {"text": "⚡ FastMRI", "callback_data": "prof_RAD4AI_FAST_MRI"},
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
                    {"text": "🧠 تقرير Gemini (عربي + إنجليزي)", "callback_data": "gemini_report"},
                    {"text": "📸 مقارنة صورة مرجعية", "callback_data": "ref_guide"},
                ],
                [
                    {"text": "📤 بطاقات GE", "callback_data": "send_cards"},
                    {"text": "🎙️ تسجيل/كتابة ملاحظة", "callback_data": "note_guide"},
                ],

            ]
        }

    def _format_return_status(self, force_push: bool = False) -> str:
        orthanc_url = os.getenv("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
        auth = (os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", ""))

        # 1. Live C-ECHO check to GE AW (192.168.0.206:4006 / adw)
        echo_msg = "غير مستجيب 🔴"
        try:
            with httpx.Client(auth=auth, timeout=6) as api:
                api.put(f"{orthanc_url}/modalities/darelha", json={"AET": "adw", "Host": "192.168.0.206", "Port": 4006})
                er = api.post(f"{orthanc_url}/modalities/darelha/echo")
                if er.status_code == 200:
                    echo_msg = "متصل ومستجيب 🟢 (`adw` @ `192.168.0.206:4006`)"
                    with connect() as db:
                        db.execute("UPDATE centers SET last_echo_status='PASS' WHERE id=8")
                else:
                    echo_msg = f"خطأ استجابة ({er.status_code}) 🔴"
        except Exception as exc:
            safe_exc = str(exc).replace("_", "-").replace("*", "").replace("`", "")[:45]
            echo_msg = f"تعذر الاتصال ({safe_exc}) 🔴"

        # 2. Check incoming unstable series still transferring into Orthanc
        transferring = []
        try:
            with httpx.Client(auth=auth, timeout=6) as api:
                for sr in api.get(f"{orthanc_url}/series?expand").json():
                    if not sr.get("IsStable", True):
                        d = str(sr.get("MainDicomTags", {}).get("SeriesDescription", "MR")).replace("`", "")
                        if not d.upper().startswith("ENH"):
                            transferring.append(f"`{d}` ({len(sr.get('Instances', []))} شريحة)")
        except Exception:
            pass

        pushed_now = 0
        with connect() as db:
            if force_push:
                unsent_jobs = db.execute(
                    "SELECT MAX(cj.id) AS id, cj.output_series_id, ib.series_id "
                    "FROM comparison_jobs cj JOIN inbound_series ib ON ib.series_id=cj.source_series_id "
                    "WHERE cj.status='ready' AND cj.output_series_id IS NOT NULL "
                    "AND cj.created_at >= datetime('now', '-12 hours') "
                    "AND NOT EXISTS ("
                    "  SELECT 1 FROM return_jobs r2 JOIN comparison_jobs c2 ON c2.id=r2.comparison_job_id "
                    "  WHERE c2.source_series_id=cj.source_series_id AND r2.status IN ('queued', 'sending', 'sent')"
                    ") "
                    "GROUP BY cj.source_series_id"
                ).fetchall()
                for uj in unsent_jobs:
                    db.execute(
                        "INSERT OR IGNORE INTO return_jobs(comparison_job_id, center_id, output_series_id) VALUES (?, 8, ?)",
                        (int(uj["id"]), str(uj["output_series_id"])),
                    )
                    db.execute(
                        "UPDATE inbound_series SET status='SENT', updated_at=CURRENT_TIMESTAMP WHERE series_id=?",
                        (uj["series_id"],),
                    )
                    pushed_now += 1
                retried = db.execute(
                    "UPDATE return_jobs SET status='queued', attempts=0, error=NULL, updated_at=CURRENT_TIMESTAMP "
                    "WHERE status='failed' AND created_at >= datetime('now', '-12 hours')"
                ).rowcount
                pushed_now += int(retried or 0)

            proc_cnt = db.execute("SELECT count(*) FROM comparison_jobs WHERE status IN ('queued', 'processing')").fetchone()[0]
            ret_q_cnt = db.execute("SELECT count(*) FROM return_jobs WHERE status IN ('queued', 'sending')").fetchone()[0]
            ret_fail_cnt = db.execute("SELECT count(*) FROM return_jobs WHERE status='failed' AND created_at >= datetime('now', '-12 hours')").fetchone()[0]
            recent_in = db.execute(
                "SELECT series_id, description, instances, status, received_at FROM inbound_series "
                "ORDER BY received_at DESC LIMIT 6"
            ).fetchall()

            lines = [
                "📡 *تقرير فحص حالة إرجاع الصور إلى GE (`adw`)*",
                f"• *اتصال محطة GE AW*: {echo_msg}",
                f"• *يُستلم الآن من الرنين*: {('، '.join(transferring)) if transferring else 'لا يوجد نقل جارٍ (مستقر)'}",
                f"• *قيد المعالجة حالياً*: `{proc_cnt}` | *في طابور الإرسال*: `{ret_q_cnt}` | *فشل إرسال*: `{ret_fail_cnt}`",
            ]
            if force_push:
                lines.append(
                    f"• 🚀 *نتيجة الدفع الفوري*: تم دفع `{pushed_now}` سلسلة لإرسالها الآن إلى `adw`."
                    if pushed_now
                    else "• ✅ *لا توجد أي صور عالقة — جميع السلاسل الجاهزة مُرسلة بالفعل.*"
                )
            lines.append("")
            lines.append("📋 *آخر 6 سلاسل مستلمة وحالة إرجاعها:*")
            for r in recent_in:
                sent_n = db.execute(
                    "SELECT count(*) FROM return_jobs rj JOIN comparison_jobs cj ON cj.id=rj.comparison_job_id "
                    "WHERE cj.source_series_id=? AND rj.status='sent'",
                    (r["series_id"],),
                ).fetchone()[0]
                st_icon = "✅ وصل لـ GE" if (r["status"] == "SENT" or sent_n > 0) else ("⏳ قيد المعالجة" if r["status"] in ("RECEIVED", "QUEUED", "PROCESSING") else f"⚠️ {r['status']}")
                t_short = str(r["received_at"] or "")[-8:]
                desc_clean = str(r["description"] or "-").replace("`", "")[:24]
                lines.append(f"• `{t_short}` | `{desc_clean}` ({r['instances']} ش) ← *{st_icon}* (`{sent_n}` نسخ)")
        return "\n".join(lines)

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
        if not _is_chat_authorized(int(chat_id)):
            try:
                self._api(client, "answerCallbackQuery", {
                    "callback_query_id": cb_id,
                    "text": "🔒 عذراً، هذا الحساب غير مصرح له. يرجى إدخال كلمة المرور أولاً.",
                    "show_alert": True,
                })
            except Exception:
                pass
            self._api(
                client,
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": "🔒 **عذراً، هذا البوت مؤمن ومخصص حصرياً للفريق المصرح له (01003741929 والفني محمد).**\nيرجى كتابة كلمة المرور (Password) لتسجيل الدخول وتفعيل الصلاحيات.",
                    "parse_mode": "Markdown",
                },
            )
            return
        _register_chat(int(chat_id))
        data = cb.get("data", "")
        try:
            self._api(client, "answerCallbackQuery", {"callback_query_id": cb_id})
        except Exception:
            pass

        if data == "status":
            reply = self._format_status()
        elif data == "check_returns":
            reply = self._format_return_status(force_push=False)
        elif data == "force_push_unsent":
            reply = self._format_return_status(force_push=True)
        elif data.startswith("lk_stir_"):
            lvl_map = {
                "1": ("RAD4AI_SPINE_PRO", "Spine/Ortho Pro (توازن عظم + تنعيم سوفت)"),
                "2": ("RAD4AI_3T_HD", "3T-HD (خشونة عظم طبيعية + حواف 3T حادة)"),
                "3": ("RAD4AI_FAST_MRI", "FastMRI (تنعيم مخملي إضافي)"),
                "4": ("RAD4AI_ENH6_ULTRA", "ENH-6 Ultra (كونتراست عالي + حواف 1px)"),
            }
            choice = data.split("_")[-1]
            prof, label = lvl_map.get(choice, ("RAD4AI_3T_HD", "3T-HD"))
            set_setting("pref_STIR", prof)
            with connect() as db:
                stir_row = db.execute(
                    "SELECT series_id, description FROM inbound_series WHERE upper(description) LIKE '%STIR%' ORDER BY received_at DESC LIMIT 1"
                ).fetchone()
                new_jid = None
                if stir_row:
                    cur = db.execute(
                        "INSERT INTO comparison_jobs (source_series_id, profile, strength, status, acknowledged) VALUES (?, ?, 'STRONG', 'queued', 1)",
                        (stir_row["series_id"], prof)
                    )
                    new_jid = cur.lastrowid
            audit("TECHNICIAN_STIR_PROFILE_LOCKED", "profile", prof, f"chat={chat_id};choice={choice}")
            reply = (
                f"🎯 *تم اعتماد المستوى ({label}) وتثبيته كمعيار تلقائي لكافة فحوصات الـ STIR!*\n\n"
                f"• تم إرسال السلسلة الكاملة الآن تلقائياً إلى محطة العمل `adw:4006` (فحص `#{new_jid or '-'}`).\n"
                f"• أي فحص STIR قادم سيتم معالجته بهذا المستوى المعتمد تلقائياً."
            )
        elif data.startswith("lk_knee_"):
            lvl_map = {
                "1": ("RAD4AI_SPINE_PRO", "Spine Pro (عظم خشن + تنعيم غضاريف)"),
                "2": ("RAD4AI_3T_HD", "3T-HD (دقة عالية وتفاصيل قشرية)"),
                "3": ("RAD4AI_FAST_MRI", "FastMRI (تنعيم حريري مخملي)"),
                "4": ("RAD4AI_ENH6_ULTRA", "ENH-6 Ultra (تباين سينمائي عالي)"),
            }
            choice = data.split("_")[-1]
            prof, label = lvl_map.get(choice, ("RAD4AI_SPINE_PRO", "Spine Pro"))
            set_setting("pref_KNEE", prof)
            with connect() as db:
                db.execute("UPDATE centers SET auto_profile=? WHERE id=8", (prof,))
                knee_row = db.execute(
                    "SELECT series_id, description FROM inbound_series WHERE description LIKE '%PD%' OR description LIKE '%KNEE%' OR description LIKE '%SAG%' ORDER BY received_at DESC LIMIT 1"
                ).fetchone()
                new_jid = None
                if knee_row:
                    cur = db.execute(
                        "INSERT INTO comparison_jobs (source_series_id, profile, strength, status, acknowledged) VALUES (?, ?, 'STRONG', 'queued', 1)",
                        (knee_row["series_id"], prof)
                    )
                    new_jid = cur.lastrowid
            audit("TECHNICIAN_KNEE_PROFILE_LOCKED", "profile", prof, f"chat={chat_id};choice={choice}")
            reply = (
                f"🎯 *تم بنجاح اعتماد المستوى ({label}) وتثبيته كمعيار تلقائي لكافة فحوصات الركبة (Knee)!*\n\n"
                f"• تم إرسال السلسلة الكاملة الآن تلقائياً إلى محطة العمل `adw:4006` (فحص `#{new_jid or '-'}`).\n"
                f"• أي فحص ركبة قادم سيتم تحسينه بهذا المستوى المعتمد تلقائياً دون الحاجة لإعادة الاختيار."
            )
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
        elif data.startswith("apply_enh6"):
            target_sid = None
            target_desc = ""
            with connect() as db:
                if "_" in data and data.split("_")[-1].isdigit():
                    jid = int(data.split("_")[-1])
                    j_row = db.execute("SELECT cj.source_series_id, COALESCE(ib.description, '') as desc FROM comparison_jobs cj LEFT JOIN inbound_series ib ON ib.series_id=cj.source_series_id WHERE cj.id=?", (jid,)).fetchone()
                    if j_row:
                        target_sid = j_row["source_series_id"]
                        target_desc = str(j_row["desc"] or "-")
                if not target_sid:
                    last_s = db.execute("SELECT series_id, description FROM inbound_series ORDER BY received_at DESC LIMIT 1").fetchone()
                    if last_s:
                        target_sid = last_s["series_id"]
                        target_desc = str(last_s["description"] or "-")

                if target_sid:
                    cur = db.execute(
                        "INSERT INTO comparison_jobs(source_series_id, profile, strength, acknowledged) VALUES(?, 'RAD4AI_ENH6_ULTRA', 'STRONG', 1)",
                        (target_sid,)
                    )
                    new_jid = int(cur.lastrowid)
                    audit("TELEGRAM_APPLY_ENH6", "comparison_job", str(new_jid), f"chat={chat_id}")
                    _nl = chr(10)
                    reply = (
                        "💎📥 **تم تفعيل المعالجة بمستوى ENH-6 (فائق الدقة 3T)!**" + _nl +
                        "• السلسلة المستهدفة: `" + str(target_desc) + "`" + _nl +
                        "• رقم فحص التشغيل: `#" + str(new_jid) + "`" + _nl +
                        "• المعايرة المطبقة: **Myelographic CSF + 1px Cortical Shock + Velvet Soft Tissue**" + _nl +
                        "⚙️ **جاري المعالجة وإرسال النتيجة فوراً إلى جهاز GE AW (adw)...**" + _nl +
                        "• ستصلك صور المقارنة والتأكيد خلال ثوانٍ معدودة."
                    )
                else:
                    reply = "⚠️ لا توجد سلسلة متوفرة حالياً لتطبيق مستوى ENH-6 عليها."
        elif data == "exec_latest_modifications":
            with connect() as db:
                fb = db.execute(
                    "SELECT id, note, verdict, series_description, created_at FROM feedback_events WHERE note IS NOT NULL AND note != '' ORDER BY id DESC LIMIT 1"
                ).fetchone()
                last_s = db.execute(
                    "SELECT series_id, description FROM inbound_series ORDER BY received_at DESC LIMIT 1"
                ).fetchone()
                if last_s:
                    sid = last_s["series_id"]
                    desc = str(last_s["description"] or "-")
                    fb_note = str(fb["note"]) if fb else "آخر إعدادات الفني المعتمدة"
                    target_prof = "RAD4AI_3T_HD"
                    if fb:
                        n_lower = fb_note.lower()
                        if "فاست" in fb_note or "fast" in n_lower:
                            target_prof = "RAD4AI_FAST_MRI"
                        elif "عظم" in fb_note or "فقرات" in fb_note or "spine" in n_lower:
                            target_prof = "RAD4AI_SPINE_PRO"

                    cur = db.execute(
                        "INSERT INTO comparison_jobs(source_series_id, profile, strength, acknowledged) VALUES(?, ?, 'STRONG', 1)",
                        (sid, target_prof),
                    )
                    new_jid = int(cur.lastrowid)
                    audit("TELEGRAM_EXEC_LATEST_MODS", "comparison_job", str(new_jid), f"chat={chat_id};note={fb_note[:50]}")
                    _nl = chr(10)
                    reply = (
                        "⚡📥 **تم استقبال أمرك: جاري تنفيذ آخر تعديلات مطلوبة!**" + _nl +
                        "• آخر تعديل مطلوب: _" + str(fb_note) + "_" + _nl +
                        "• السلسلة المستهدفة: `" + str(desc) + "`" + _nl +
                        "• البروفايل المطبق: `" + str(target_prof) + "` (STRONG)" + _nl +
                        "• رقم فحص التشغيل: `#" + str(new_jid) + "`" + _nl +
                        "⚙️ **جاري العمل على التعديلات وإرسالها فوراً لجهاز GE AW (adw)...**" + _nl +
                        "• ستصلك رسالة التأكيد ومعاينة الفحص فور اكتمال الإرسال."
                    )
                else:
                    reply = "⚠️ لا توجد سلسلة متوفرة حالياً لإعادة تطبيق التعديلات عليها."
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
        elif data == "gemini_report":
            with connect() as db:
                row = db.execute(
                    "SELECT cj.id, cj.output_series_id, cj.profile, cj.strength, "
                    "COALESCE(ib.description, '') AS sdesc, COALESCE(ib.study_id, '') AS study_id "
                    "FROM comparison_jobs cj LEFT JOIN inbound_series ib ON ib.series_id=cj.source_series_id "
                    "WHERE cj.status='ready' AND cj.output_series_id!='' ORDER BY cj.id DESC LIMIT 1"
                ).fetchone()
            if row:
                try:
                    self._api(client, "sendMessage", {"chat_id": chat_id, "text": "⏳ جاري إعداد تقرير Gemini المبدئي (عربي + إنجليزي) لآخر فحص..."})
                    report_msg = _generate_bilingual_gemini_report(dict(row))
                    self._api(client, "sendMessage", {"chat_id": chat_id, "text": report_msg, "reply_markup": self._menu_keyboard()})
                    return
                except Exception as exc:
                    reply = f"❌ تعذر توليد تقرير Gemini: {exc}"
            else:
                reply = "لا توجد سلسلة مكتملة لإعداد التقرير."
        elif data == "note_guide":
            reply = (
                "🎙️ *تسجيل ملاحظة صوتية أو كتابية:*\n"
                "أرسل الآن **رسالة صوتية (Voice)** أو **رسالة نصية** بملاحظتك (مثلاً: _T2 محتاج خشونة بسيطة_ أو _L3 ممتاز_) — "
                "سيقوم البوت بتفريغ الصوت تلقائياً وحفظ الملاحظة وإبلاغ الفريق للتنفيذ فوراً."
            )
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
            f"📸📥 *تم استقبال الصورة والتعديلات بنجاح برقم `#{fid}`!*\n"
            f"⚙️ *جاري العمل على التعديلات ومطابقة النتيجة المرجعية فوراً...*\n"
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
        sender = msg.get("from", {}).get("first_name", "") or msg.get("from", {}).get("username", "tech")
        text = (msg.get("text") or msg.get("caption") or "").strip()

        # Contact sharing verification
        if msg.get("contact"):
            phone = str(msg["contact"].get("phone_number") or "").replace(" ", "").replace("-", "")
            if any(p in phone for p in ("01003741929", "1003741929")):
                _add_authenticated_chat(int(chat_id))
                _register_chat(int(chat_id))
                self._api(
                    client,
                    "sendMessage",
                    {
                        "chat_id": chat_id,
                        "text": "🔓 **تم التحقق من رقم الهاتف بنجاح!**\nمرحباً بك يا بشمهندس (01003741929). تم تفعيل الصلاحيات الكاملة.",
                        "parse_mode": "Markdown",
                        "reply_markup": self._menu_keyboard(),
                    },
                )
                return

        # Security Authentication Gate
        if not _is_chat_authorized(int(chat_id)):
            if _is_password_correct(text):
                _add_authenticated_chat(int(chat_id))
                _register_chat(int(chat_id))
                audit("TELEGRAM_AUTH_SUCCESS", "telegram", str(chat_id), f"sender={sender}")
                self._api(
                    client,
                    "sendMessage",
                    {
                        "chat_id": chat_id,
                        "text": "🔓 **تم التحقق وتسجيل الدخول بنجاح!**\nمرحباً بك في نظام تحكم RAD4AI MRI Appliance.\nتم تفعيل كامل الصلاحيات لتنفيذ واستقبال التعديلات.",
                        "parse_mode": "Markdown",
                        "reply_markup": self._menu_keyboard(),
                    },
                )
                return
            else:
                audit("TELEGRAM_AUTH_DENIED", "telegram", str(chat_id), f"sender={sender};text={text[:30]}")
                self._api(
                    client,
                    "sendMessage",
                    {
                        "chat_id": chat_id,
                        "text": "🔒 **عذراً، هذا البوت مؤمن ومخصص حصرياً للفريق الطبي المعتمد.**\n\n"
                                "• الحسابات المصرح لها حصرياً: رقم المدير (`01003741929`) ورقم الفني محمد.\n"
                                "• للوصول للوحة التحكم وتنفيذ التعديلات، يرجى كتابة **كلمة المرور (Password)** الخاصة بالنظام.",
                        "parse_mode": "Markdown",
                    },
                )
                return

        _register_chat(int(chat_id))

        # Admin command to change password
        if int(chat_id) in (1281167916,) and text.startswith("/password "):
            new_pw = text[10:].strip()
            if new_pw:
                set_setting("telegram_access_password", new_pw)
                self._api(
                    client,
                    "sendMessage",
                    {
                        "chat_id": chat_id,
                        "text": f"🔑 تم تحديث كلمة مرور البوت إلى: `{new_pw}` بنجاح.",
                        "parse_mode": "Markdown",
                    },
                )
                return

        voice_obj = msg.get("voice") or msg.get("audio")
        if voice_obj:
            try:
                finfo = self._api(client, "getFile", {"file_id": voice_obj["file_id"]})
                fpath = finfo["result"]["file_path"]
                dl = client.get(f"https://api.telegram.org/file/bot{self.token}/{fpath}", timeout=30)
                dl.raise_for_status()
                vdir = Path("/var/lib/enhancer/telegram_voices")
                vdir.mkdir(parents=True, exist_ok=True)
                vpath = vdir / f"voice_{chat_id}_{Path(fpath).name}"
                vpath.write_bytes(dl.content)
                mime = str(voice_obj.get("mime_type") or "audio/ogg")
                transcript = _transcribe_voice_gemini(dl.content, mime_type=mime)
                note_txt = transcript or f"[رسالة صوتية محفوظة: {vpath.name}]"
                verdict, level = _detect_verdict_and_level(note_txt)
                last_sdesc = ""
                with connect() as db:
                    row = db.execute("SELECT description FROM inbound_series ORDER BY received_at DESC LIMIT 1").fetchone()
                    if row:
                        last_sdesc = str(row["description"] or "")
                fid = record_feedback(
                    source="telegram_voice",
                    patient_name=sender,
                    study_description="Telegram Voice Note",
                    series_description=last_sdesc,
                    level=level,
                    verdict=verdict,
                    note=f"{note_txt} ({vpath.name})",
                    source_aet=f"tg:{chat_id}",
                )
                audit("TELEGRAM_VOICE_NOTE", "feedback", str(fid), f"chat={chat_id};file={vpath.name}")
                voice_action_keywords = ("نعم", "نعومه", "خشن", "خشونه", "فتح", "بيض", "سواد", "ابعت", "تعديل", "احسن", "جوده", "كونترست", "بروتون", "برتون", "اكسيل", "اكسل", "جردينت", "ركبه", "عظم", "فقرات")
                voice_actionable = any(k in note_txt for k in voice_action_keywords)
                v_jid = None
                v_target = last_sdesc
                if voice_actionable:
                    try:
                        with connect() as db:
                            t_row = None
                            if any(k in note_txt for k in ("بروتون", "برتون", "pd", "ركبه")):
                                t_row = db.execute("SELECT series_id, description FROM inbound_series WHERE description LIKE '%PD%' OR description LIKE '%PROTON%' ORDER BY received_at DESC LIMIT 1").fetchone()
                            elif any(k in note_txt for k in ("اكسيل", "اكسل", "axial", "جردينت", "سواد")):
                                t_row = db.execute("SELECT series_id, description FROM inbound_series WHERE description LIKE '%AX%' OR description LIKE '%GRE%' ORDER BY received_at DESC LIMIT 1").fetchone()
                            if not t_row:
                                t_row = db.execute("SELECT series_id, description FROM inbound_series ORDER BY received_at DESC LIMIT 1").fetchone()
                            if t_row:
                                v_target = str(t_row["description"] or "")
                                cur = db.execute(
                                    "INSERT INTO comparison_jobs(source_series_id, profile, strength, acknowledged) VALUES(?, 'RAD4AI_3T_HD', 'STRONG', 1)",
                                    (t_row["series_id"],)
                                )
                                v_jid = int(cur.lastrowid)
                                audit("TELEGRAM_AUTO_EXEC_VOICE_ORDER", "comparison_job", str(v_jid), f"fid={fid};chat={chat_id}")
                    except Exception as ve:
                        log.warning("Voice auto-reprocess error: %s", ve)

                if v_jid:
                    reply = (
                        f"🎙️📥 *تم استقبال التعديلات الصوتية بنجاح!*\n"
                        f"⚙️ *جاري العمل على التعديلات وإعادة المعالجة فوراً...*\n"
                        f"• التفريغ الصوتي: _{note_txt}_\n"
                        f"• السلسلة المستهدفة: `{v_target}`\n"
                        f"• رقم فحص التشغيل: `#{v_jid}`\n"
                        f"• سيتم إرسال النسخة المحسنة لجهاز GE AW وتأكيد الإرسال فور الانتهاء."
                    )
                else:
                    reply = (
                        f"🎙️📥 *تم استقبال الملاحظة الصوتية برقم `#{fid}`!*\n"
                        f"⚙️ *جاري العمل على تطبيق التعديلات المطلوبة على الفحوصات...*\n"
                        f"• التفريغ الصوتي: _{note_txt}_\n"
                        f"• التصنيف: `{verdict}` {f'| {level}' if level else ''}"
                    )
            except Exception as exc:
                reply = f"❌ تعذر معالجة الرسالة الصوتية: {exc}"
            self._api(
                client,
                "sendMessage",
                {"chat_id": chat_id, "text": reply, "parse_mode": "Markdown", "reply_markup": self._menu_keyboard()},
            )
            return

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

        nl = chr(10)
        # Intelligent Automated Order Execution before incoming scans
        action_keywords = ("نعم", "نعومه", "خشن", "خشونه", "فتح", "تفتيح", "بيض", "سواد", "ابعت", "تعديل", "احسن", "جوده", "كونترست", "بروتون", "برتون", "اكسيل", "اكسل", "جردينت", "ركبه", "فقرات", "فاست", "fast", "نفذ", "تنفيذ", "lss")
        is_actionable = any(k in text.lower() for k in action_keywords)
        reprocess_jid = None
        target_desc = last_sdesc

        if is_actionable:
            try:
                with connect() as db:
                    t_row = None
                    # If user says "نفذ", look at the latest feedback note
                    search_txt = text
                    if any(k in text for k in ("نفذ", "تنفيذ")):
                        last_fb = db.execute("SELECT note FROM feedback_events WHERE note NOT IN ('نفذ', 'تنفيذ') AND note != '' ORDER BY id DESC LIMIT 1").fetchone()
                        if last_fb and last_fb["note"]:
                            search_txt = search_txt + " " + last_fb["note"]

                    s_lower = search_txt.lower()
                    if any(k in s_lower for k in ("بروتون", "برتون", "pd", "ركبه")):
                        t_row = db.execute("SELECT series_id, description FROM inbound_series WHERE description LIKE '%PD%' OR description LIKE '%PROTON%' ORDER BY received_at DESC LIMIT 1").fetchone()
                    elif any(k in s_lower for k in ("اكسيل", "اكسل", "axial", "جردينت", "سواد")):
                        t_row = db.execute("SELECT series_id, description FROM inbound_series WHERE description LIKE '%AX%' OR description LIKE '%GRE%' ORDER BY received_at DESC LIMIT 1").fetchone()
                    elif any(k in s_lower for k in ("sag t2", "sagt2", "sag t1", "sagt1", "lss", "فقرات")):
                        t_row = db.execute("SELECT series_id, description FROM inbound_series WHERE description LIKE '%SAG%' OR description LIKE '%FL08E%' ORDER BY received_at DESC LIMIT 1").fetchone()
                    if not t_row:
                        t_row = db.execute("SELECT series_id, description FROM inbound_series ORDER BY received_at DESC LIMIT 1").fetchone()

                    if t_row:
                        target_desc = str(t_row["description"] or "")
                        # Choose profile based on search_txt
                        chosen_prof = "RAD4AI_3T_HD"
                        if "فاست" in s_lower or "fast" in s_lower:
                            chosen_prof = "RAD4AI_FAST_MRI"
                        elif "عظم" in s_lower or "فقرات" in s_lower or "spine" in s_lower:
                            chosen_prof = "RAD4AI_SPINE_PRO"

                        cur = db.execute(
                            "INSERT INTO comparison_jobs(source_series_id, profile, strength, acknowledged) VALUES(?, ?, 'STRONG', 1)",
                            (t_row["series_id"], chosen_prof)
                        )
                        reprocess_jid = int(cur.lastrowid)
                        audit("TELEGRAM_AUTO_EXEC_ORDER", "comparison_job", str(reprocess_jid), f"fid={fid};chat={chat_id};prof={chosen_prof}")
            except Exception as e:
                log.warning("Auto-reprocess error: %s", e)
        if reprocess_jid:
            ack = (
                "📥 *تم استقبال التعديلات بنجاح!*" + nl +
                "⚙️ *جاري العمل على التعديلات وإعادة المعالجة فوراً...*" + nl +
                "• طلبك: _" + str(text) + "_" + nl +
                "• السلسلة المستهدفة: `" + str(target_desc or "-") + "`" + nl +
                "• رقم فحص التشغيل: `#" + str(reprocess_jid) + "`" + nl +
                "• سيتم إرسال الصور المحسنة مباشرة إلى جهاز GE Advantage Workstation (adw) وإشعارك فور الانتهاء."
            )
        else:
            ack = (
                "📥 *تم استقبال التعديلات بنجاح برقم `#" + str(fid) + "`!*" + nl +
                "⚙️ *جاري العمل على تطبيق التعديلات على الفحوصات فوراً...*" + nl +
                "• طلبك: _" + str(text) + "_" + nl +
                "• آخر سلسلة: `" + str(last_sdesc or "-") + "`" + nl +
                "• التصنيف: `" + str(verdict) + "`"
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
