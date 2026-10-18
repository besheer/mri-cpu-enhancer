from __future__ import annotations

import io
import json
import logging
import os
import threading
import zipfile

import httpx
import numpy as np
import pydicom
from pydicom.uid import generate_uid

from app.enhancer import derive_with_pixels, derived_series_description
from app.profiles import PREFIXES, PROFILES, process_volume, validate_selection
from app.quality import evaluate_quality
from app.store import audit, auto_eligible_profile_names, connect, enabled_profile_names, enqueue_return, get_center_by_aet, setting

log = logging.getLogger("comparison-worker")


class ComparisonWorker:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.url = os.getenv("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
        self.auth = (os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", ""))

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)

    def _next_job(self):
        with connect() as db:
            job = db.execute("SELECT * FROM comparison_jobs WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
            if job:
                db.execute("UPDATE comparison_jobs SET status='processing',processed_instances=0,error=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (job["id"],))
                db.execute("UPDATE inbound_series SET status='PROCESSING',updated_at=CURRENT_TIMESTAMP WHERE job_id=?", (job["id"],))
            return job

    def _run(self) -> None:
        while not self.stop_event.is_set():
            if setting("pause_enhancement") == "1" or setting("maintenance_mode") == "1":
                self.stop_event.wait(2)
                continue
            job = self._next_job()
            if not job:
                self.stop_event.wait(2)
                continue
            try:
                output_id, metrics, qc_status, qc_reasons, center = self._process(
                    job["source_series_id"], job["profile"], job["strength"]
                )
                with connect() as db:
                    db.execute("UPDATE comparison_jobs SET status='ready',output_series_id=?,metrics_json=?,qc_status=?,qc_reasons=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                               (output_id, json.dumps(metrics), qc_status, json.dumps(qc_reasons), job["id"]))
                    db.execute("INSERT INTO processed_series(series_id,status,error) VALUES(?,?,NULL) ON CONFLICT(series_id) DO UPDATE SET status=excluded.status,error=NULL,updated_at=CURRENT_TIMESTAMP",
                               (job["source_series_id"], "comparison_ready"))
                    db.execute("UPDATE inbound_series SET status='READY',updated_at=CURRENT_TIMESTAMP WHERE job_id=?", (job["id"],))
                    protocol_protected = job["protocol_study_id"] is not None or db.execute(
                        "SELECT 1 FROM protocol_studies WHERE archived=0 AND (standard_series_id=? OR fast_series_id=?) LIMIT 1",
                        (job["source_series_id"], job["source_series_id"]),
                    ).fetchone() is not None
                if (not protocol_protected and center
                        and job["profile"] in auto_eligible_profile_names()
                        and job["strength"] != "STRONG"
                        and center["return_policy"] == "AUTO_PASS" and qc_status == "PASS"
                        and center["last_echo_status"] == "PASS"):
                    enqueue_return(job["id"], center["id"], output_id)
                    audit("AUTO_RETURN_QUEUED", "comparison_job", job["id"], f"center={center['name']}")
                elif protocol_protected:
                    audit("PROTOCOL_RETURN_BLOCKED", "comparison_job", job["id"],
                          f"protocol_study={job['protocol_study_id'] or 'series-match'}; comparison-only")
            except Exception as exc:
                log.exception("Comparison job failed")
                with connect() as db:
                    db.execute("UPDATE comparison_jobs SET status='failed',error=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (str(exc)[:2000], job["id"]))
                    db.execute("UPDATE inbound_series SET status='FAILED',error=?,updated_at=CURRENT_TIMESTAMP WHERE job_id=?", (str(exc)[:2000], job["id"]))

    def _process(self, series_id: str, profile: str, strength: str):
        if profile not in PROFILES:
            raise ValueError("Unknown profile")
        if profile not in enabled_profile_names():
            raise ValueError("Profile is disabled by administrator")
        with httpx.Client(auth=self.auth, timeout=1800) as api:
            info = api.get(f"{self.url}/series/{series_id}")
            info.raise_for_status()
            tags = info.json().get("MainDicomTags", {})
            description = str(tags.get("SeriesDescription", "MR SERIES"))
            if tags.get("Modality") != "MR" or description.upper().startswith("ENHANC"):
                raise ValueError("Only original MR series can be processed")
            center = None
            instance_ids = info.json().get("Instances", [])
            if instance_ids:
                origin = api.get(f"{self.url}/instances/{instance_ids[0]}/metadata/1024")
                if origin.status_code == 200:
                    center = get_center_by_aet(origin.text.strip())
            validate_selection(profile, description, len(info.json().get("Instances", [])),
                               acknowledged=(profile == "CLAHE_VIEW" or strength == "STRONG"),
                               strength=strength)
            archive = api.get(f"{self.url}/series/{series_id}/archive")
            archive.raise_for_status()
            datasets = []
            with zipfile.ZipFile(io.BytesIO(archive.content)) as files:
                for name in files.namelist():
                    if not name.endswith("/"):
                        with files.open(name) as source:
                            datasets.append(pydicom.dcmread(source))
            if not datasets:
                raise ValueError("Empty DICOM series")
            datasets.sort(key=lambda ds: (int(ds.get("InstanceNumber", 0)), str(ds.SOPInstanceUID)))
            with connect() as db:
                db.execute("UPDATE comparison_jobs SET total_instances=?,processed_instances=0,updated_at=CURRENT_TIMESTAMP WHERE source_series_id=? AND status='processing'",
                           (len(datasets), series_id))
            shapes = {ds.pixel_array.shape for ds in datasets}
            if len(shapes) != 1:
                raise ValueError("Inconsistent image dimensions in series")
            volume = np.stack([ds.pixel_array for ds in datasets])
            first = datasets[0]
            pixel_spacing = first.get("PixelSpacing", [1.0, 1.0])
            slice_spacing = float(first.get("SpacingBetweenSlices", first.get("SliceThickness", 1.0)))
            spacing = (float(pixel_spacing[1]), float(pixel_spacing[0]), abs(slice_spacing))
            processed, metrics = process_volume(volume, profile, strength, spacing)
            output_uid = generate_uid()
            output_description = derived_series_description(description, f"{PREFIXES[profile]}{strength}-")
            output_series_id = ""
            for index, (dataset, pixels) in enumerate(zip(datasets, processed), start=1):
                derived = derive_with_pixels(dataset, pixels, output_uid, output_description,
                                             f"{profile}; STRENGTH={strength}; RESEARCH ONLY; NOT FOR DIAGNOSIS")
                payload = io.BytesIO()
                derived.save_as(payload, enforce_file_format=True)
                result = api.post(f"{self.url}/instances", content=payload.getvalue(),
                                  headers={"Content-Type": "application/dicom"})
                result.raise_for_status()
                output_series_id = result.json().get("ParentSeries", output_series_id)
                with connect() as db:
                    db.execute("UPDATE comparison_jobs SET processed_instances=?,updated_at=CURRENT_TIMESTAMP WHERE source_series_id=? AND status='processing'",
                               (index, series_id))
            if not output_series_id:
                raise RuntimeError("Orthanc did not return derived series ID")
            metrics["instances"] = len(datasets)
            output_info = api.get(f"{self.url}/series/{output_series_id}")
            output_info.raise_for_status()
            output_instances = len(output_info.json().get("Instances", []))
            metrics["output_instances"] = output_instances
            qc_status, qc_reasons = evaluate_quality(metrics, len(datasets), output_instances)
            return output_series_id, metrics, qc_status, qc_reasons, center
