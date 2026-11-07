from __future__ import annotations

import base64
import json
import logging
import os
import io

import httpx

from app.store import connect

log = logging.getLogger("gemini-auditor")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
ORTHANC_URL = os.getenv("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
ORTHANC_AUTH = (os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", ""))

CANDIDATE_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-flash-latest",
    "gemini-3.8-flash"
]


def audit_job_with_gemini(job_id: int) -> dict:
    key = os.getenv("GEMINI_API_KEY", "").strip() or GEMINI_API_KEY
    if not key:
        return {"error": "GEMINI_API_KEY is not configured in .env"}

    with connect() as db:
        job = db.execute("SELECT * FROM comparison_jobs WHERE id=?", (job_id,)).fetchone()
        if not job or not job["output_series_id"]:
            return {"error": f"Job #{job_id} not found or has no output series"}

    src_sid = job["source_series_id"]
    out_sid = job["output_series_id"]

    try:
        with httpx.Client(auth=ORTHANC_AUTH, timeout=30.0) as api:
            s_info = api.get(f"{ORTHANC_URL}/series/{src_sid}").json()
            o_info = api.get(f"{ORTHANC_URL}/series/{out_sid}").json()

            s_insts = s_info.get("Instances", [])
            o_insts = o_info.get("Instances", [])

            if not s_insts or not o_insts:
                return {"error": "Series instances empty"}

            mid_idx = len(s_insts) // 2
            src_mid_id = s_insts[mid_idx]
            out_mid_id = o_insts[min(mid_idx, len(o_insts) - 1)]

            src_img_resp = api.get(f"{ORTHANC_URL}/instances/{src_mid_id}/preview")
            out_img_resp = api.get(f"{ORTHANC_URL}/instances/{out_mid_id}/preview")

            src_img_resp.raise_for_status()
            out_img_resp.raise_for_status()

            src_b64 = base64.b64encode(src_img_resp.content).decode("ascii")
            out_b64 = base64.b64encode(out_img_resp.content).decode("ascii")
    except Exception as exc:
        log.exception("Failed to fetch slices from Orthanc")
        return {"error": f"Failed to fetch MRI slices from Orthanc: {exc}"}

    prompt = (
        "You are an expert Senior Neuroradiologist and Clinical AI Quality Officer. "
        "You are evaluating an MRI AI enhancement result from an Hitachi AIRIS Vento 0.3T MRI scanner.\n\n"
        "IMAGE 1 is the Original Raw 0.3T MRI scan (low-field, native noise).\n"
        "IMAGE 2 is the RAD4AI Enhanced MRI scan (processed with edge-preserving noise reduction and 3T contrast modeling).\n\n"
        "Carefully compare both images side-by-side:\n"
        "1. Identify the exact anatomical region and sequence (e.g. Sagittal T2 Lumbosacral Spine, Axial T2 Brain, etc.).\n"
        "2. Diagnostic Safety: Verify that all anatomical structures (cortical bone, disc margins, thecal sac, nerve roots, marrow) "
        "are preserved with 100% diagnostic fidelity and NO hallucinations, no fictitious lesions, and no altered pathology.\n"
        "3. Image Quality: Note the noise reduction, soft-tissue contrast, edge crispness, and background suppression.\n"
        "4. Assign a Diagnostic Safety Score (1 to 10) and Verdict (PASS or WARN).\n"
        "5. Draft preliminary clinical findings for the reporting radiologist.\n"
        "6. Provide a concise summary in English, and a clear Arabic summary (ملخص إكلينيكي لطبيب الأشعة باللغة العربية).\n\n"
        "Return STRICTLY a valid JSON object (no markdown code blocks, just raw JSON) matching this exact format:\n"
        "{\n"
        '  "sequence_region": "...",\n'
        '  "safety_score": 10,\n'
        '  "verdict": "PASS",\n'
        '  "hallucinations_detected": false,\n'
        '  "noise_reduction_quality": "...",\n'
        '  "summary_en": "...",\n'
        '  "summary_ar": "...",\n'
        '  "preliminary_findings": ["...", "..."]\n'
        "}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": src_b64,
                        }
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": out_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
        },
    }

    last_error = ""
    for model in CANDIDATE_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if raw_text.startswith("```json"):
                        raw_text = raw_text[7:]
                    if raw_text.startswith("```"):
                        raw_text = raw_text[3:]
                    if raw_text.endswith("```"):
                        raw_text = raw_text[:-3]

                    parsed = json.loads(raw_text.strip())
                    parsed["model_used"] = model

                    # Save in database
                    with connect() as db:
                        db.execute(
                            "UPDATE comparison_jobs SET gemini_audit=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                            (json.dumps(parsed), job_id),
                        )

                    return parsed
                elif resp.status_code == 503:
                    last_error = f"{model} temporarily overloaded (503), trying next model..."
                    log.warning(last_error)
                    continue
                else:
                    last_error = f"Gemini API ({model}) returned {resp.status_code}: {resp.text[:200]}"
                    log.error(last_error)
        except Exception as exc:
            last_error = f"Call to {model} failed: {exc}"
            log.warning(last_error)

    return {"error": last_error or "All candidate Gemini models failed"}
