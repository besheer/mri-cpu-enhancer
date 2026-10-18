from __future__ import annotations

import io
import logging
import os
import sqlite3
import threading
import time
import zipfile

import httpx
import pydicom
from pydicom.uid import generate_uid

from app.enhancer import derived_series_description, enhance_dataset
from app.store import connect, get_center_by_aet

log = logging.getLogger("mri-worker")


class AutoProcessor:
    def __init__(self) -> None:
        # Legacy path has no Quality Gate and is intentionally opt-in under a new
        # variable. AUTO_PROCESS_ENABLED from older releases is ignored safely.
        self.enabled = os.getenv("LEGACY_UNSAFE_AUTO_PROCESS_ENABLED", "false").lower() == "true"
        self.interval = max(5, int(os.getenv("POLL_SECONDS", "10")))
        self.prefix = os.getenv("ENHANC_PREFIX", "ENHANC-")
        self.url = os.getenv("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
        self.auth = (os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", ""))
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_error: str | None = None
        self.last_series: str | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)

    def status(self) -> dict[str, object]:
        return {"enabled": self.enabled, "running": bool(self.thread and self.thread.is_alive()),
                "last_series": self.last_series, "last_error": self.last_error}

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.process_once()
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                log.exception("Automatic processing cycle failed")
            self.stop_event.wait(self.interval)

    @staticmethod
    def _handled(series_id: str) -> bool:
        with connect() as db:
            row = db.execute("SELECT status FROM processed_series WHERE series_id=?", (series_id,)).fetchone()
        return bool(row and row[0] in {"processing", "sent", "skipped", "comparison_ready"})

    @staticmethod
    def _mark(series_id: str, status: str, center_id=None, error=None) -> None:
        with connect() as db:
            db.execute(
                "INSERT INTO processed_series(series_id,center_id,status,error) VALUES(?,?,?,?) "
                "ON CONFLICT(series_id) DO UPDATE SET center_id=excluded.center_id,status=excluded.status,error=excluded.error,updated_at=CURRENT_TIMESTAMP",
                (series_id, center_id, status, error),
            )

    def process_once(self) -> None:
        with httpx.Client(auth=self.auth, timeout=120) as client:
            response = client.get(f"{self.url}/series?expand")
            response.raise_for_status()
            for series in response.json():
                sid = series["ID"]
                tags = series.get("MainDicomTags", {})
                description = str(tags.get("SeriesDescription", ""))
                if not series.get("IsStable", False) or self._handled(sid):
                    continue
                if tags.get("Modality") != "MR" or description.upper().startswith("ENHANC"):
                    self._mark(sid, "skipped")
                    continue
                instances = series.get("Instances", [])
                if not instances:
                    continue
                origin = client.get(f"{self.url}/instances/{instances[0]}/metadata/1024")
                if origin.status_code != 200:
                    self._mark(sid, "skipped", error="No C-STORE origin metadata")
                    continue
                source_aet = origin.text.strip()
                center = get_center_by_aet(source_aet)
                if center is None:
                    self._mark(sid, "skipped", error="Center disabled or unknown")
                    continue
                self._process(client, sid, description, center)

    def _process(self, client: httpx.Client, sid: str, description: str, center) -> None:
        center_id = center["id"]
        self._mark(sid, "processing", center_id)
        self.last_series = sid
        try:
            client.put(
                f"{self.url}/modalities/{center['slug']}",
                json={"AET": center["return_aet"], "Host": center["return_ip"], "Port": center["return_port"]},
            ).raise_for_status()
            archive = client.get(f"{self.url}/series/{sid}/archive")
            archive.raise_for_status()
            new_uid = generate_uid()
            new_description = derived_series_description(description, self.prefix)
            output_series = None
            with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                for name in names:
                    with zf.open(name) as stream:
                        source = pydicom.dcmread(stream)
                    output = enhance_dataset(source, new_uid, new_description)
                    payload = io.BytesIO()
                    output.save_as(payload, enforce_file_format=True)
                    uploaded = client.post(
                        f"{self.url}/instances", content=payload.getvalue(),
                        headers={"Content-Type": "application/dicom"},
                    )
                    uploaded.raise_for_status()
                    output_series = uploaded.json().get("ParentSeries", output_series)
            if not output_series:
                raise RuntimeError("No derived instances were created")
            client.post(
                f"{self.url}/modalities/{center['slug']}/store", json=output_series, timeout=1800
            ).raise_for_status()
            self._mark(sid, "sent", center_id)
        except Exception as exc:
            self._mark(sid, "failed", center_id, str(exc))
            raise
