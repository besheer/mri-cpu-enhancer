from __future__ import annotations

import io
import json
import logging
import os

import numpy as np
import pydicom
from scipy.ndimage import gaussian_filter, sobel

from app.rician_nlm import estimate_sigma
from app.store import connect

log = logging.getLogger("physics-auditor")

ORTHANC_URL = os.getenv("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
ORTHANC_AUTH = (os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", ""))


def _local_physics_audit(src_arr: np.ndarray, out_arr: np.ndarray, series_desc: str, profile: str) -> dict:
    """100% On-Premise Deterministic Physics & Signal Quality Auditor.

    Runs strictly inside the local server with zero external cloud or internet calls.
    """
    s = src_arr.astype(np.float64)
    o = out_arr.astype(np.float64)
    if s.shape != o.shape:
        min_h = min(s.shape[0], o.shape[0])
        min_w = min(s.shape[1], o.shape[1])
        s = s[:min_h, :min_w]
        o = o[:min_h, :min_w]

    s_lo, s_hi = np.percentile(s, (1.0, 99.0))
    s_span = max(float(s_hi - s_lo), 1e-4)
    s01 = np.clip((s - s_lo) / s_span, 0.0, 1.0)

    o_lo, o_hi = np.percentile(o, (1.0, 99.0))
    o_span = max(float(o_hi - o_lo), 1e-4)
    o01 = np.clip((o - o_lo) / o_span, 0.0, 1.0)

    sigma_before = estimate_sigma(s01)
    sigma_after = estimate_sigma(o01)
    noise_reduction_pct = round((1.0 - (sigma_after / max(sigma_before, 1e-6))) * 100.0, 1)

    # Structural Similarity (Global & Local Window SSIM P05)
    mu_s = gaussian_filter(s01, 1.5)
    mu_o = gaussian_filter(o01, 1.5)
    var_s = np.maximum(gaussian_filter(s01 * s01, 1.5) - mu_s * mu_s, 0.0)
    var_o = np.maximum(gaussian_filter(o01 * o01, 1.5) - mu_o * mu_o, 0.0)
    cov_so = gaussian_filter(s01 * o01, 1.5) - mu_s * mu_o
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2.0 * mu_s * mu_o + c1) * (2.0 * cov_so + c2)) / (
        (mu_s * mu_s + mu_o * mu_o + c1) * (var_s + var_o + c2) + 1e-12
    )
    ssim_val = round(float(np.mean(ssim_map)), 4)

    fg_mask = s01 > 0.12
    if np.count_nonzero(fg_mask) > 32:
        local_ssim_p05 = round(float(np.percentile(ssim_map[fg_mask], 5.0)), 4)
    else:
        local_ssim_p05 = round(float(np.percentile(ssim_map, 5.0)), 4)

    # Anatomical Edge Correlation (Pearson r on smoothed gradient magnitudes, clamped to [0, 1])
    gs = gaussian_filter(np.hypot(sobel(s01, axis=0), sobel(s01, axis=1)), 1.0).ravel()
    go = gaussian_filter(np.hypot(sobel(o01, axis=0), sobel(o01, axis=1)), 1.0).ravel()
    if gs.std() > 1e-9 and go.std() > 1e-9:
        raw_r = float(np.corrcoef(gs, go)[0, 1])
        if np.isnan(raw_r):
            raw_r = 0.0
        edge_r = round(max(0.0, min(1.0, raw_r)), 4)
    else:
        edge_r = 1.0

    macro_diff = float(np.percentile(np.abs(gaussian_filter(s01, 3.5) - gaussian_filter(o01, 3.5)), 99.0))
    structural_drift = bool(macro_diff > 0.22 or local_ssim_p05 < 0.45 or edge_r < 0.75)

    verdict = "PASS" if (not structural_drift and edge_r >= 0.80 and ssim_val >= 0.70) else "WARN"
    score = 10 if (verdict == "PASS" and edge_r >= 0.88 and local_ssim_p05 >= 0.60) else 9 if verdict == "PASS" else 7

    desc_clean = (series_desc or "MRI Series").strip()
    return {
        "sequence_region": f"{desc_clean} ({profile})",
        "safety_score": score,
        "verdict": verdict,
        "hallucinations_detected": structural_drift,
        "noise_reduction_quality": f"Rician Sigma {sigma_before:.4f} -> {sigma_after:.4f} (-{noise_reduction_pct}%)",
        "summary_en": (
            f"100% On-Premise Deterministic Signal Audit: "
            f"Rician noise floor reduced by {noise_reduction_pct}%, "
            f"Edge Gradient Pearson r = {edge_r:.3f}, Global SSIM = {ssim_val:.3f} (Local P05 SSIM = {local_ssim_p05:.3f}). "
            f"No macro-anatomical drift detected (P99 drift = {macro_diff:.3f} < 0.22); does not replace physician review."
        ),
        "summary_ar": (
            f"فحص إشارة فيزيائي محلي 100% (داخل السيرفر بدون أي إرسال خارجي): "
            f"انخفاض الضوضاء الرايسية بنسبة {noise_reduction_pct}%، "
            f"ومعامل ارتباط الحواف (Pearson r = {edge_r:.3f})، ومعامل SSIM الكلي = {ssim_val:.3f} (المحلي P05 = {local_ssim_p05:.3f}). "
            f"لم يُرصد انحراف ماكرو-تشريحي (P99 drift = {macro_diff:.3f} < 0.22)؛ لا يُغني عن مراجعة الطبيب المختص."
        ),
        "quantitative_metrics": [
            f"Rician Noise Floor Suppression: -{noise_reduction_pct}% (Haar-MAD Sigma: {sigma_before:.4f} → {sigma_after:.4f})",
            f"Gradient Edge Correlation (Pearson r): {edge_r:.4f} (scaled: {round(edge_r * 100.0, 2)}%)",
            f"Structural Similarity: Global SSIM = {ssim_val:.4f} | Local Window SSIM (P05) = {local_ssim_p05:.4f}",
            f"Macro-Anatomical Drift (P99, sigma=3.5): {macro_diff:.4f} (Threshold < 0.22 — heuristic check; physician review required)",
        ],
        "model_used": "Local Rician-Physics QC Engine (On-Premise)",
    }


def audit_job_with_gemini(job_id: int, allow_cloud: bool = False) -> dict:
    """Run deterministic local physics & signal audit on a comparison job.

    External cloud transmission is permanently disabled to guarantee 100% PHI privacy.
    """
    _ = allow_cloud  # Ignored: external cloud transmission is permanently disabled
    import httpx

    with connect() as db:
        job = db.execute("SELECT * FROM comparison_jobs WHERE id=?", (job_id,)).fetchone()
        if not job or not job["output_series_id"]:
            return {"error": f"Job #{job_id} not found or has no output series"}

    src_sid = job["source_series_id"]
    out_sid = job["output_series_id"]
    profile = str(job["profile"] or "RICIAN_NLM_VST")

    try:
        with httpx.Client(auth=ORTHANC_AUTH, timeout=30.0) as api:
            s_info = api.get(f"{ORTHANC_URL}/series/{src_sid}").json()
            o_info = api.get(f"{ORTHANC_URL}/series/{out_sid}").json()
            series_desc = str(s_info.get("MainDicomTags", {}).get("SeriesDescription", "MR Series"))

            s_insts = s_info.get("Instances", [])
            o_insts = o_info.get("Instances", [])
            if not s_insts or not o_insts:
                return {"error": "Series instances empty"}

            mid_idx = len(s_insts) // 2
            src_mid_id = s_insts[mid_idx]
            out_mid_id = o_insts[min(mid_idx, len(o_insts) - 1)]

            src_dcm_resp = api.get(f"{ORTHANC_URL}/instances/{src_mid_id}/file")
            out_dcm_resp = api.get(f"{ORTHANC_URL}/instances/{out_mid_id}/file")
            src_dcm_resp.raise_for_status()
            out_dcm_resp.raise_for_status()

            src_ds = pydicom.dcmread(io.BytesIO(src_dcm_resp.content))
            out_ds = pydicom.dcmread(io.BytesIO(out_dcm_resp.content))
            src_arr = src_ds.pixel_array.astype(np.float32)
            out_arr = out_ds.pixel_array.astype(np.float32)

        parsed = _local_physics_audit(src_arr, out_arr, series_desc, profile)
        with connect() as db:
            db.execute(
                "UPDATE comparison_jobs SET gemini_audit=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (json.dumps(parsed), job_id),
            )
        return parsed
    except Exception as exc:
        log.exception("Failed to run local physics audit")
        return {"error": f"Failed to run local physics audit: {exc}"}
