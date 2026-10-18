from __future__ import annotations

import logging
import os
import threading

import httpx

from app.store import audit, connect, setting

log = logging.getLogger("return-worker")


class ReturnWorker:
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

    def status(self) -> dict[str, object]:
        return {"running": bool(self.thread and self.thread.is_alive())}

    def _next(self):
        with connect() as db:
            job = db.execute(
                "SELECT r.*,c.slug,c.return_aet,c.return_ip,c.return_port,c.max_return_attempts,c.enabled,c.last_echo_status "
                "FROM return_jobs r JOIN centers c ON c.id=r.center_id "
                "WHERE r.status='queued' ORDER BY r.id LIMIT 1"
            ).fetchone()
            if job:
                db.execute("UPDATE return_jobs SET status='sending',attempts=attempts+1,updated_at=CURRENT_TIMESTAMP WHERE id=?", (job["id"],))
            return job

    def _run(self) -> None:
        while not self.stop_event.is_set():
            if setting("pause_return") == "1" or setting("maintenance_mode") == "1":
                self.stop_event.wait(2)
                continue
            job = self._next()
            if not job:
                self.stop_event.wait(2)
                continue
            try:
                if not job["enabled"]:
                    raise RuntimeError("Center is disabled")
                if job["last_echo_status"] != "PASS":
                    raise RuntimeError("C-ECHO must pass before DICOM return")
                with httpx.Client(auth=self.auth, timeout=1800) as api:
                    api.put(f"{self.url}/modalities/{job['slug']}", json={
                        "AET": job["return_aet"], "Host": job["return_ip"], "Port": job["return_port"]
                    }).raise_for_status()
                    echo = api.post(f"{self.url}/modalities/{job['slug']}/echo")
                    echo.raise_for_status()
                    sent = api.post(f"{self.url}/modalities/{job['slug']}/store", json=job["output_series_id"])
                    sent.raise_for_status()
                with connect() as db:
                    db.execute("UPDATE return_jobs SET status='sent',error=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (job["id"],))
                    db.execute("UPDATE inbound_series SET status='SENT',error=NULL,updated_at=CURRENT_TIMESTAMP WHERE job_id=?", (job["comparison_job_id"],))
                audit("RETURN_SENT", "return_job", job["id"], f"destination={job['return_aet']}@{job['return_ip']}:{job['return_port']}")
            except Exception as exc:
                retry = job["attempts"] + 1 < int(job["max_return_attempts"])
                with connect() as db:
                    db.execute("UPDATE return_jobs SET status=?,error=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                               ("queued" if retry else "failed", str(exc)[:2000], job["id"]))
                    if not retry:
                        db.execute("UPDATE inbound_series SET status='RETURN_FAILED',error=?,updated_at=CURRENT_TIMESTAMP WHERE job_id=?",
                                   (str(exc)[:2000], job["comparison_job_id"]))
                audit("RETURN_RETRY" if retry else "RETURN_FAILED", "return_job", job["id"], str(exc))
                log.exception("DICOM return failed")
