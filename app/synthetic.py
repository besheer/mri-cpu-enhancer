from __future__ import annotations

import io
import json
import math
import os
from datetime import datetime, timezone
from urllib.parse import quote

import httpx
import numpy as np
from fastapi import APIRouter, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
from scipy.ndimage import gaussian_filter, shift

from app.admin import esc, layout, require_admin
from app.enhancer import derive_with_pixels
from app.profiles import PREFIXES, PROFILE_INFO, STRENGTHS, process_volume
from app.store import audit, connect, enabled_profile_names

router = APIRouter(prefix="/admin/synthetic", tags=["synthetic-lab"])

TESTS = {
    "RICIAN_NOISE": "Rician noise / ضوضاء MRI",
    "BIAS_FIELD": "Bias field / تفاوت الإضاءة",
    "MOTION": "Motion / حركة",
    "GHOSTING": "Ghosting / أشباح",
    "BLUR": "Spatial blur / ضعف الحدة",
    "LOW_CONTRAST": "Low contrast / تباين منخفض",
    "MIXED_03T": "Mixed 0.3T / Noise + Bias + Blur",
    "EXTREME": "Extreme rejection test / اختبار الرفض",
}
SEVERITIES = {"MILD": 0.65, "MEDIUM": 1.0, "STRONG": 1.5}
RECOMMENDATIONS = {
    "RICIAN_NOISE": "EDGE_DENOISE_MILD",
    "BIAS_FIELD": "N4_BIAS",
    "MOTION": "MANUAL_REVIEW",
    "GHOSTING": "MANUAL_REVIEW",
    "BLUR": "MANUAL_REVIEW",
    "LOW_CONTRAST": "CLAHE_VIEW",
    "MIXED_03T": "N4_EDGE_DENOISE",
    "EXTREME": "REJECT",
}
REVIEW_ONLY_TESTS = {"MOTION", "GHOSTING", "BLUR", "EXTREME"}


def make_phantom(slices: int = 12, size: int = 128) -> np.ndarray:
    """Build a deterministic 3-D MR-like phantom with known ground truth."""
    z, y, x = np.mgrid[-1:1:complex(slices), -1:1:complex(size), -1:1:complex(size)]
    volume = np.full((slices, size, size), 80.0, dtype=np.float32)
    body = (x / 0.72) ** 2 + (y / 0.86) ** 2 + (z / 1.45) ** 2 <= 1
    inner = (x / 0.55) ** 2 + (y / 0.67) ** 2 + (z / 1.30) ** 2 <= 1
    lesion = ((x - 0.22) / 0.095) ** 2 + ((y + 0.12) / 0.075) ** 2 + ((z - 0.05) / 0.55) ** 2 <= 1
    dark = ((x + 0.25) / 0.14) ** 2 + ((y - 0.18) / 0.11) ** 2 + ((z + 0.08) / 0.7) ** 2 <= 1
    volume[body] = 1050
    volume[inner] = 1750
    volume[lesion] = 2550
    volume[dark] = 520
    ring = body & ~(((x / 0.64) ** 2 + (y / 0.77) ** 2 + (z / 1.4) ** 2) <= 1)
    volume[ring] = 2850
    return gaussian_filter(volume, sigma=(0.15, 0.55, 0.55)).astype(np.uint16)


def corrupt_volume(clean: np.ndarray, test_type: str, severity: str, seed: int) -> np.ndarray:
    if test_type not in TESTS or severity not in SEVERITIES:
        raise ValueError("Unknown synthetic test or severity")
    factor = SEVERITIES[severity]
    rng = np.random.default_rng(seed)
    source = clean.astype(np.float32)

    def rician(image: np.ndarray, sigma: float) -> np.ndarray:
        n1 = rng.normal(0, sigma, image.shape)
        n2 = rng.normal(0, sigma, image.shape)
        return np.sqrt((image + n1) ** 2 + n2 ** 2)

    def bias(image: np.ndarray, amount: float) -> np.ndarray:
        yy, xx = np.mgrid[-1:1:complex(image.shape[1]), -1:1:complex(image.shape[2])]
        field = 1 + amount * (0.60 * xx + 0.35 * yy + 0.25 * xx * yy)
        return image * np.clip(field[None, ...], 0.45, 1.65)

    if test_type == "RICIAN_NOISE":
        out = rician(source, 95 * factor)
    elif test_type == "BIAS_FIELD":
        out = bias(source, 0.34 * factor)
    elif test_type == "MOTION":
        out = source.copy()
        for index in range(out.shape[0]):
            if index % 3:
                out[index] = shift(out[index], (factor * (index % 2), factor * 2.2), order=1, mode="nearest")
    elif test_type == "GHOSTING":
        out = source + 0.20 * factor * np.roll(source, int(12 * factor), axis=1)
    elif test_type == "BLUR":
        out = gaussian_filter(source, sigma=(0.2, 1.25 * factor, 1.25 * factor))
    elif test_type == "LOW_CONTRAST":
        mean = float(source.mean())
        out = mean + (source - mean) * max(0.25, 1 - 0.38 * factor)
    elif test_type == "MIXED_03T":
        out = bias(source, 0.25 * factor)
        out = gaussian_filter(out, sigma=(0.15, 0.65 * factor, 0.65 * factor))
        out = rician(out, 80 * factor)
    else:
        out = bias(source, 0.45)
        out = gaussian_filter(out, sigma=(0.3, 1.8, 1.8))
        out = rician(out, 180 * factor)
        out += 0.28 * np.roll(out, 18, axis=1)
    return np.rint(np.clip(out, 0, 4095)).astype(np.uint16)


def _global_ssim(ref: np.ndarray, out: np.ndarray, data_range: float) -> float:
    if ref.size == 0:
        return 0.0
    mu_x, mu_y = float(ref.mean()), float(out.mean())
    var_x, var_y = float(ref.var()), float(out.var())
    covariance = float(np.mean((ref - mu_x) * (out - mu_y)))
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return float(np.clip(((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / max(denominator, 1e-12), -1, 1))


def reference_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = reference.astype(np.float64)
    out = candidate.astype(np.float64)
    data_range = max(float(ref.max() - ref.min()), 1.0)
    error = out - ref
    mse = float(np.mean(error * error))
    psnr = 99.0 if mse == 0 else 20 * math.log10(data_range) - 10 * math.log10(mse)
    low, high = np.percentile(ref, [5, 99])
    foreground = ref > (low + 0.12 * max(high - low, 1.0))
    background = ~foreground
    roi_error = error[foreground]
    roi_mse = float(np.mean(roi_error * roi_error)) if roi_error.size else mse
    roi_psnr = 99.0 if roi_mse == 0 else 20 * math.log10(data_range) - 10 * math.log10(roi_mse)
    slice_scores = []
    for index in range(ref.shape[0]):
        mask = foreground[index]
        if np.count_nonzero(mask) > 16:
            slice_scores.append(_global_ssim(ref[index][mask], out[index][mask], data_range))
    gradients_ref = np.gradient(ref)
    gradients_out = np.gradient(out)
    edge_ref = np.sqrt(sum(axis * axis for axis in gradients_ref))
    edge_out = np.sqrt(sum(axis * axis for axis in gradients_out))
    edge_mask = foreground & (edge_ref >= np.percentile(edge_ref[foreground], 85))
    edge_error = float(np.mean(np.abs(edge_out[edge_mask] - edge_ref[edge_mask])))
    edge_base = max(float(np.mean(edge_ref[edge_mask])), 1e-6)
    background_std = float(np.std(out[background])) if np.any(background) else 0.0
    foreground_mean = float(np.mean(out[foreground])) if np.any(foreground) else 0.0
    return {
        "psnr_db": round(psnr, 3),
        "global_ssim": round(_global_ssim(ref, out, data_range), 5),
        "roi_psnr_db": round(roi_psnr, 3),
        "roi_ssim": round(_global_ssim(ref[foreground], out[foreground], data_range), 5),
        "slice_ssim_mean": round(float(np.mean(slice_scores)) if slice_scores else 0.0, 5),
        "mae": round(float(np.mean(np.abs(error))), 3),
        "roi_mae": round(float(np.mean(np.abs(roi_error))) if roi_error.size else 0.0, 3),
        "normalized_rmse": round(math.sqrt(mse) / data_range, 5),
        "reference_edge_similarity_percent": round(100 * max(0.0, 1.0 - edge_error / edge_base), 2),
        "foreground_snr": round(foreground_mean / max(background_std, 1e-6), 3),
    }


def difference_map(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Absolute error magnified x4 for visual inspection, not quantitative reading."""
    return np.rint(np.clip(np.abs(candidate.astype(np.float32) - reference.astype(np.float32)) * 4.0, 0, 4095)).astype(np.uint16)


def result_decision(test_type: str, input_metrics: dict[str, float], metrics: dict[str, float]) -> tuple[str, float]:
    gain = metrics["roi_psnr_db"] - input_metrics["roi_psnr_db"]
    ssim_gain = metrics["roi_ssim"] - input_metrics["roi_ssim"]
    score = gain + 20.0 * ssim_gain + 0.01 * metrics["reference_edge_similarity_percent"]
    if test_type in REVIEW_ONLY_TESTS:
        return "REVIEW_ONLY", round(score, 5)
    if gain >= 0.10 and ssim_gain >= -0.0005:
        return "IMPROVED", round(score, 5)
    if gain < -0.10 or ssim_gain < -0.002:
        return "WORSE", round(score, 5)
    return "NEUTRAL", round(score, 5)


def dicom_series(volume: np.ndarray, study_uid: str, description: str, series_number: int) -> list[FileDataset]:
    series_uid = generate_uid()
    now = datetime.now(timezone.utc)
    result = []
    for index, pixels in enumerate(volume, start=1):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.PatientName = "SYNTHETIC^PHANTOM"
        ds.PatientID = "SYNTH-03T"
        ds.StudyID = "SYNTH-LAB"
        ds.AccessionNumber = "SYNTHETIC"
        ds.StudyDescription = "Synthetic MRI Enhancement Benchmark"
        ds.SeriesDescription = description[:64]
        ds.Modality = "MR"
        ds.Manufacturer = "SYNTHETIC TEST DATA"
        ds.MagneticFieldStrength = "0.3"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "RESEARCH"]
        ds.SeriesNumber = series_number
        ds.InstanceNumber = index
        ds.StudyDate = now.strftime("%Y%m%d")
        ds.StudyTime = now.strftime("%H%M%S")
        ds.ContentDate = ds.StudyDate
        ds.ContentTime = ds.StudyTime
        ds.Rows, ds.Columns = pixels.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 4.0
        ds.SpacingBetweenSlices = 4.0
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, (index - 1) * 4.0]
        ds.PixelData = pixels.tobytes()
        ds.BurnedInAnnotation = "NO"
        result.append(ds)
    return result


def upload_series(api: httpx.Client, datasets: list[FileDataset]) -> str:
    parent = ""
    for ds in datasets:
        payload = io.BytesIO()
        ds.save_as(payload, enforce_file_format=True)
        response = api.post("/instances", content=payload.getvalue(), headers={"Content-Type": "application/dicom"})
        response.raise_for_status()
        parent = response.json().get("ParentSeries", parent)
    if not parent:
        raise RuntimeError("Orthanc did not return a series identifier")
    return parent


@router.get("", response_class=HTMLResponse)
def lab(message: str = "", error: str = "", _: str = Depends(require_admin)) -> str:
    enabled = enabled_profile_names()
    profile_boxes = "".join(
        f"<label class='tool-check'><input type='checkbox' name='profiles' value='{esc(key)}' "
        f"{'checked' if key in {'BASE_MILD','DENOISE_MEDIUM','EDGE_DENOISE_MILD','N4_EDGE_DENOISE'} else ''} "
        f"{'disabled' if key not in enabled else ''}> <b>{esc(info['label'])}</b><br><span class='muted'>{esc(info['guidance'])}</span></label>"
        for key, info in PROFILE_INFO.items())
    with connect() as db:
        runs = db.execute("SELECT * FROM synthetic_runs ORDER BY id DESC LIMIT 15").fetchall()
        results = db.execute("SELECT * FROM synthetic_results ORDER BY rank_score DESC,id").fetchall()
    by_run: dict[int, list] = {}
    for item in results:
        by_run.setdefault(int(item["run_id"]), []).append(item)
    rows = ""
    for run in runs:
        incoming = json.loads(run["input_metrics_json"] or "{}")
        result_rows = ""
        for rank, r in enumerate(by_run.get(int(run["id"]), []), start=1):
            metric = json.loads(r["metrics_json"])
            decision_class = "ok" if r["decision"] == "IMPROVED" else "bad" if r["decision"] == "WORSE" else "warn"
            result_rows += (f"<tr><td>#{rank}</td><td>{esc(r['profile'])}</td><td>{esc(r['strength'])}</td>"
                            f"<td><span class='badge {decision_class}'>{esc(r['decision'])}</span></td>"
                            f"<td>{metric.get('roi_psnr_db','—')}</td><td>{metric.get('roi_ssim','—')}</td>"
                            f"<td>{metric.get('slice_ssim_mean','—')}</td><td>{metric.get('reference_edge_similarity_percent','—')}%</td>"
                            f"<td>{metric.get('mae','—')}</td><td>{metric.get('seconds','—')}s</td></tr>")
        result_rows = result_rows or "<tr><td colspan='10'>No enhancer results</td></tr>"
        ohif = "/ohif/viewer?StudyInstanceUIDs=" + quote(run["study_uid"], safe=".")
        rows += f"""<details class='study-card'><summary><div class='badge warn'>#{run['id']}</div><div><b>{esc(run['test_type'])} · {esc(run['severity'])}</b><br><span class='muted'>{esc(run['created_at'])} · Input ROI PSNR {incoming.get('roi_psnr_db','—')} · ROI SSIM {incoming.get('roi_ssim','—')} · Suggested: {esc(RECOMMENDATIONS.get(run['test_type']))}</span></div><span class='chevron'></span></summary><div class='study-body'><div class='actions'><a class='btn' target='_blank' href='{ohif}'>Open Reference / Input / Enhanced / Difference in OHIF</a><a class='btn gray' href='/admin/synthetic/{run['id']}/csv'>Export CSV</a></div><div class='notice' style='margin-top:12px'>Synthetic test only. Auto Return is always disabled. Difference maps display absolute error ×4.</div><div class='table-wrap'><table><tr><th>Rank</th><th>Enhancer</th><th>Level</th><th>Decision</th><th>ROI PSNR</th><th>ROI SSIM</th><th>Slice SSIM</th><th>Reference Edges</th><th>MAE</th><th>Time</th></tr>{result_rows}</table></div></div></details>"""
    test_options = "".join(f"<option value='{k}'>{esc(v)}</option>" for k, v in TESTS.items())
    level_options = "".join(f"<option value='{k}'>{esc(v)}</option>" for k, v in STRENGTHS.items())
    body = f"""<div class='actions'><a class='btn gray' href='/admin'>Back to Dashboard</a><a class='btn gray' href='/admin/studies'>Clinical/Research Studies</a></div><section class='panel'><div class='panel-head'><h2>Synthetic MRI Lab / معمل الاختبارات الصناعية</h2><span class='badge warn'>NO PATIENT DATA · NO AUTO RETURN</span></div><p class='muted'>Generate a known reference phantom, add a controlled MRI artifact, run selected enhancers, and rank them with foreground-sensitive metrics.</p><form method='post' action='/admin/synthetic/generate'><div class='grid'><div class='field'><label>Artifact / المشكلة</label><select id='test-type' name='test_type' onchange='recommend()'>{test_options}</select><div id='recommendation' class='hint'></div></div><div class='field'><label>Artifact Severity / شدة المشكلة</label><select name='severity'>{level_options}</select></div><div class='field'><label>Enhancement Level / مستوى التحسين</label><select name='strength'>{level_options}</select></div><div class='field'><label>Random Seed</label><input type='number' name='seed' min='1' max='999999' value='42'></div><div class='wide'><label><b>Enhancers / المحسنات</b></label><div class='bulk-tools'>{profile_boxes}</div></div><div class='wide check'><input id='benchmark-all' type='checkbox' name='benchmark_all'><label for='benchmark-all'><b>Benchmark every level (MILD + MEDIUM + STRONG)</b> — slower; research comparison only</label></div><div class='wide notice'>Recommended first benchmark: MIXED_03T + MEDIUM. Motion, ghosting, blur and extreme artifacts are REVIEW_ONLY because these filters cannot reconstruct lost information.</div><div class='wide'><button class='btn'>Generate, Enhance and Compare</button></div></div></form></section><section class='panel'><h2>Benchmark Runs / نتائج المقارنة</h2>{rows or "<div class='empty'>No synthetic runs yet.</div>"}</section><script>const rec={json.dumps(RECOMMENDATIONS)};function recommend(){{const t=document.getElementById('test-type').value;document.getElementById('recommendation').textContent='Suggested action: '+rec[t]}}recommend();</script>"""
    return layout("Synthetic MRI Lab", body, error or message, bool(error))


@router.get("/{run_id}/csv")
def export_csv(run_id: int, _: str = Depends(require_admin)) -> Response:
    import csv
    with connect() as db:
        run = db.execute("SELECT * FROM synthetic_runs WHERE id=?", (run_id,)).fetchone()
        results = db.execute("SELECT * FROM synthetic_results WHERE run_id=? ORDER BY rank_score DESC,id", (run_id,)).fetchall()
    if run is None:
        return Response("Run not found", status_code=404)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["run_id", "test_type", "artifact_severity", "profile", "strength", "decision", "rank_score",
                     "psnr_db", "roi_psnr_db", "global_ssim", "roi_ssim", "slice_ssim_mean", "mae", "roi_mae",
                     "reference_edge_similarity_percent", "foreground_snr", "seconds"])
    for row in results:
        metric = json.loads(row["metrics_json"])
        writer.writerow([run_id, run["test_type"], run["severity"], row["profile"], row["strength"], row["decision"], row["rank_score"]] +
                        [metric.get(key, "") for key in ("psnr_db", "roi_psnr_db", "global_ssim", "roi_ssim", "slice_ssim_mean",
                                                          "mae", "roi_mae", "reference_edge_similarity_percent", "foreground_snr", "seconds")])
    return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="synthetic-run-{run_id}.csv"'})


@router.post("/generate")
def generate(test_type: str = Form(...), severity: str = Form(...), strength: str = Form("MEDIUM"),
             seed: int = Form(42), profiles: list[str] = Form(default=[]), benchmark_all: str | None = Form(None),
             _: str = Depends(require_admin)):
    try:
        if test_type not in TESTS or severity not in SEVERITIES or strength not in STRENGTHS:
            raise ValueError("Invalid benchmark selection")
        if not 1 <= seed <= 999999:
            raise ValueError("Seed must be between 1 and 999999")
        enabled = enabled_profile_names()
        selected = list(dict.fromkeys(profiles))
        if not selected:
            raise ValueError("Select at least one enhancer")
        if any(profile not in PROFILE_INFO or profile not in enabled for profile in selected):
            raise ValueError("One or more selected enhancers are unavailable or disabled")
        combinations = [(profile, level) for profile in selected
                        for level in (("MILD", "MEDIUM", "STRONG") if benchmark_all else (strength,))]
        if len(combinations) > 24:
            raise ValueError("Benchmark is limited to 24 profile/level combinations")
        clean = make_phantom()
        corrupted = corrupt_volume(clean, test_type, severity, seed)
        input_metrics = reference_metrics(clean, corrupted)
        study_uid = generate_uid()
        clean_ds = dicom_series(clean, study_uid, "SYNTH-GROUND-TRUTH-NOT-FOR-DIAGNOSIS", 1)
        corrupt_ds = dicom_series(corrupted, study_uid, f"SYNTH-{test_type}-{severity}-INPUT", 2)
        base = os.getenv("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
        auth = (os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", ""))
        with httpx.Client(base_url=base, auth=auth, timeout=1800) as api:
            clean_id = upload_series(api, clean_ds)
            corrupt_id = upload_series(api, corrupt_ds)
            with connect() as db:
                cursor = db.execute("INSERT INTO synthetic_runs(test_type,severity,seed,study_uid,clean_series_id,corrupted_series_id,input_metrics_json) VALUES(?,?,?,?,?,?,?)",
                                    (test_type, severity, seed, study_uid, clean_id, corrupt_id, json.dumps(input_metrics)))
                run_id = int(cursor.lastrowid)
            for combo_index, (profile, level) in enumerate(combinations):
                number = 10 + combo_index * 2
                processed, operational = process_volume(corrupted, profile, level, (1.0, 1.0, 4.0))
                output_uid = generate_uid()
                description = f"SYNTH-{PREFIXES[profile]}{level}-{test_type}"[:64]
                derived = [derive_with_pixels(ds, pixels, output_uid, description,
                           f"SYNTHETIC BENCHMARK; {profile}; {level}; NOT FOR DIAGNOSIS")
                           for ds, pixels in zip(corrupt_ds, processed)]
                for ds in derived:
                    ds.SeriesNumber = number
                    ds.PatientID = "SYNTH-03T"
                output_id = upload_series(api, derived)
                metrics = reference_metrics(clean, processed)
                metrics.update(operational)
                decision, score = result_decision(test_type, input_metrics, metrics)
                difference = difference_map(clean, processed)
                diff_ds = dicom_series(difference, study_uid, f"SYNTH-DIFF-X4-{profile}-{level}", number + 1)
                for ds in diff_ds:
                    ds.ImageType = ["DERIVED", "SECONDARY", "RESEARCH", "DIFFERENCE"]
                    ds.DerivationDescription = "ABSOLUTE DIFFERENCE FROM SYNTHETIC GROUND TRUTH MULTIPLIED BY 4"
                difference_id = upload_series(api, diff_ds)
                with connect() as db:
                    db.execute("INSERT INTO synthetic_results(run_id,profile,strength,output_series_id,difference_series_id,metrics_json,decision,rank_score) VALUES(?,?,?,?,?,?,?,?)",
                               (run_id, profile, level, output_id, difference_id, json.dumps(metrics), decision, score))
        audit("SYNTHETIC_BENCHMARK", "synthetic_run", run_id,
              f"test={test_type}; severity={severity}; combinations={len(combinations)}")
        return RedirectResponse("/admin/synthetic?message=" + quote(f"Benchmark #{run_id} completed"), 303)
    except Exception as exc:
        return RedirectResponse("/admin/synthetic?error=" + quote(str(exc)[:500]), 303)
