from __future__ import annotations

import logging
import os
import threading

import httpx

from app.profiles import PROFILES
from app.store import audit, auto_eligible_profile_names, connect, enqueue_comparison, get_center_by_aet, setting

log = logging.getLogger("intake-worker")


class IntakeWorker:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.url = os.getenv("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
        self.auth = (os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", ""))
        self.interval = max(5, int(os.getenv("INTAKE_POLL_SECONDS", "10")))
        self.last_error: str | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)

    def status(self) -> dict[str, object]:
        return {"running": bool(self.thread and self.thread.is_alive()), "last_error": self.last_error}

    def _run(self) -> None:
        while not self.stop_event.is_set():
            if setting("pause_intake") == "1" or setting("maintenance_mode") == "1":
                self.stop_event.wait(self.interval)
                continue
            try:
                self.scan_once()
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                log.exception("Inbound scan failed")
            self.stop_event.wait(self.interval)

    def scan_once(self) -> None:
        with httpx.Client(auth=self.auth, timeout=120) as api:
            response = api.get(f"{self.url}/series?expand")
            response.raise_for_status()
            for series in response.json():
                if not series.get("IsStable", False):
                    continue
                sid = series["ID"]
                tags = series.get("MainDicomTags", {})
                description = str(tags.get("SeriesDescription", ""))
                # Synthetic Lab studies are managed in their own benchmark registry and
                # must never enter branch intake or automatic-return workflows.
                if (tags.get("Modality") != "MR"
                        or description.upper().startswith("ENHANC")
                        or description.upper().startswith("SYNTH-")):
                    continue
                instances = series.get("Instances", [])
                if not instances:
                    continue
                with connect() as db:
                    if db.execute("SELECT 1 FROM inbound_series WHERE series_id=?", (sid,)).fetchone():
                        continue
                origin = api.get(f"{self.url}/instances/{instances[0]}/metadata/1024")
                source_aet = origin.text.strip() if origin.status_code == 200 else ""
                center = get_center_by_aet(source_aet) if source_aet else None
                status = "RECEIVED" if center else "UNMATCHED"
                with connect() as db:
                    db.execute("INSERT OR IGNORE INTO inbound_series(series_id,center_id,source_aet,study_id,description,instances,status) VALUES(?,?,?,?,?,?,?)",
                               (sid, center["id"] if center else None, source_aet,
                                series.get("ParentStudy", ""), description, len(instances), status))
                audit("SERIES_RECEIVED", "series", sid, f"aet={source_aet};instances={len(instances)}")
                if not center or not center["auto_enhance_enabled"]:
                    continue
                profile, strength = center["auto_profile"], center["auto_strength"]
                if profile not in PROFILES or profile not in auto_eligible_profile_names() or strength == "STRONG":
                    with connect() as db:
                        db.execute("UPDATE inbound_series SET status='FAILED',error='Profile is not eligible for automatic processing',updated_at=CURRENT_TIMESTAMP WHERE series_id=?", (sid,))
                    continue
                job_id = enqueue_comparison(sid, profile, strength)
                with connect() as db:
                    db.execute("UPDATE inbound_series SET status='QUEUED',job_id=?,updated_at=CURRENT_TIMESTAMP WHERE series_id=?", (job_id, sid))
                audit("AUTO_ENHANCE_QUEUED", "series", sid, f"job={job_id};profile={profile};strength={strength}")
