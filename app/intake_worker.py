from __future__ import annotations

import logging
import os
import threading

import httpx

from app.enhancer import is_derived_series
from app.profiles import PROFILES
from app.store import audit, auto_eligible_profile_names, connect, enabled_profile_names, enqueue_comparison, get_center_by_aet, record_feedback, setting

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
                # Synthetic Lab studies and already-enhanced series are managed in their own registry
                # and must never enter branch intake or automatic-return workflows.
                u_desc = description.upper()
                if u_desc.startswith("SYNTH-"):
                    continue
                if u_desc.startswith("FB-"):
                    instances = series.get("Instances", [])
                    if instances:
                        origin = api.get(f"{self.url}/instances/{instances[0]}/metadata/1024")
                        ret_aet = origin.text.strip() if origin.status_code == 200 else ""
                        if ret_aet:
                            card_map = {
                                "FB-L3-ACCEPTED": ("L3-CRISP", "APPROVED", "GE Card: L3 ACCEPTED"),
                                "FB-L3-REJECTED": ("L3-CRISP", "REJECT", "GE Card: L3 REJECTED"),
                                "FB-MORE-GRAIN": ("", "MORE_GRAIN", "GE Card: MORE GRAIN"),
                                "FB-SMOOTHER": ("", "SMOOTHER", "GE Card: SMOOTHER"),
                                "FB-HIGHER-CLARITY": ("", "ADJUST", "GE Card: HIGHER CLARITY"),
                            }
                            level, verdict, note = card_map.get(u_desc, ("", "NOTE", f"GE Card: {description}"))
                            last_pt, last_sdesc = "", ""
                            with connect() as db:
                                last_row = db.execute(
                                    "SELECT study_id, description FROM inbound_series ORDER BY received_at DESC LIMIT 1"
                                ).fetchone()
                                if last_row:
                                    last_sdesc = str(last_row["description"] or "")
                                    if last_row["study_id"]:
                                        st_r = api.get(f"{self.url}/studies/{last_row['study_id']}")
                                        if st_r.status_code == 200:
                                            last_pt = str(st_r.json().get("PatientMainDicomTags", {}).get("PatientName", ""))
                            record_feedback(
                                source="ge_dicom",
                                series_id=sid,
                                patient_name=last_pt,
                                study_description=description,
                                series_description=last_sdesc or description,
                                level=level,
                                verdict=verdict,
                                note=note,
                                source_aet=ret_aet,
                            )
                            audit("GE_CARD_FEEDBACK", "series", sid, f"card={u_desc};verdict={verdict};aet={ret_aet}")
                            api.delete(f"{self.url}/series/{sid}")
                    continue
                modality = str(tags.get("Modality", ""))
                is_derived = is_derived_series(description)
                if modality != "MR" or is_derived:
                    # Capture returned enhanced series or Secondary Capture notes sent back from GE AW
                    instances = series.get("Instances", [])
                    if instances:
                        with connect() as db:
                            locally_created = db.execute(
                                "SELECT 1 FROM comparison_jobs WHERE output_series_id=?", (sid,)
                            ).fetchone()
                            already_logged = db.execute(
                                "SELECT 1 FROM feedback_events WHERE series_id=?", (sid,)
                            ).fetchone()
                        if not locally_created and not already_logged:
                            origin = api.get(f"{self.url}/instances/{instances[0]}/metadata/1024")
                            ret_aet = origin.text.strip() if origin.status_code == 200 else ""
                            if ret_aet:
                                pt_name = ""
                                st_desc = ""
                                parent_study = series.get("ParentStudy", "")
                                if parent_study:
                                    st_r = api.get(f"{self.url}/studies/{parent_study}")
                                    if st_r.status_code == 200:
                                        st_j = st_r.json()
                                        pt_name = str(st_j.get("PatientMainDicomTags", {}).get("PatientName", ""))
                                        st_desc = str(st_j.get("MainDicomTags", {}).get("StudyDescription", ""))
                                img_comment = ""
                                st_comment = ""
                                sim_r = api.get(f"{self.url}/instances/{instances[0]}/simplified-tags")
                                if sim_r.status_code == 200:
                                    sim_j = sim_r.json()
                                    img_comment = str(sim_j.get("ImageComments", ""))
                                    st_comment = str(sim_j.get("StudyComments", ""))
                                combined_note = " | ".join(p for p in (description, img_comment, st_comment) if p)
                                upper_note = combined_note.upper()
                                level = (
                                    "L1-SOFT" if ("L1-" in upper_note or "MILD" in upper_note)
                                    else "L2-BALANCED" if ("L2-" in upper_note or "MEDIUM" in upper_note)
                                    else "L3-CRISP" if ("L3-" in upper_note or "STRONG" in upper_note)
                                    else ""
                                )
                                verdict = (
                                    "REJECT" if any(k in upper_note for k in ("REJECT", "BAD", "REDO", "NO"))
                                    else "MORE_GRAIN" if "GRAIN" in upper_note
                                    else "SMOOTHER" if "SOFT" in upper_note and "L1-SOFT" not in upper_note
                                    else "ADJUST" if any(k in upper_note for k in ("ADJUST", "FIX", "CRISP", "DARK", "BRIGHT"))
                                    else "APPROVED"
                                )
                                record_feedback(
                                    source="ge_dicom",
                                    series_id=sid,
                                    patient_name=pt_name,
                                    study_description=st_desc,
                                    series_description=description,
                                    level=level,
                                    verdict=verdict,
                                    note=combined_note,
                                    source_aet=ret_aet,
                                )
                                audit("GE_FEEDBACK_CAPTURED", "series", sid, f"aet={ret_aet};verdict={verdict};level={level}")
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
                if any(w in u_desc for w in ("LOCALIZER", "SCANO", "SURVEY", "3PLANE", "3-PLANE", "CALIB")):
                    with connect() as db:
                        db.execute("UPDATE inbound_series SET status='SKIPPED_LOCALIZER',updated_at=CURRENT_TIMESTAMP WHERE series_id=?", (sid,))
                    audit("LOCALIZER_SKIPPED", "series", sid, f"desc={description}")
                    continue
                study_desc = ""
                parent_study = series.get("ParentStudy", "")
                if parent_study:
                    try:
                        st_resp = api.get(f"{self.url}/studies/{parent_study}")
                        if st_resp.status_code == 200:
                            study_desc = str(st_resp.json().get("MainDicomTags", {}).get("StudyDescription", "")).upper()
                    except Exception:
                        study_desc = ""
                combined_desc = f"{study_desc} {u_desc}"
                profile, strength = center["auto_profile"], center["auto_strength"]
                auto_allowed = auto_eligible_profile_names()
                enabled_set = enabled_profile_names()
                if any(w in combined_desc for w in ("BRAIN", "HEAD", "NEURO", "SKULL", "FLAIR")) and "RICIAN_NLM_VST" in (auto_allowed & enabled_set):
                    profile = "RICIAN_NLM_VST"
                if profile not in PROFILES or (profile not in auto_allowed and profile not in enabled_set):
                    with connect() as db:
                        db.execute("UPDATE inbound_series SET status='FAILED',error='Profile is not eligible for automatic processing',updated_at=CURRENT_TIMESTAMP WHERE series_id=?", (sid,))
                    continue
                if center["calibration_mode"]:
                    job_ids = [enqueue_comparison(sid, profile, lvl) for lvl in ("MILD", "MEDIUM", "STRONG")]
                    job_id = job_ids[0]
                    with connect() as db:
                        db.execute("UPDATE inbound_series SET status='QUEUED',job_id=?,updated_at=CURRENT_TIMESTAMP WHERE series_id=?", (job_id, sid))
                    audit("AUTO_CALIBRATION_QUEUED", "series", sid, f"jobs={job_ids};profile={profile}")
                else:
                    job_id = enqueue_comparison(sid, profile, strength)
                    with connect() as db:
                        db.execute("UPDATE inbound_series SET status='QUEUED',job_id=?,updated_at=CURRENT_TIMESTAMP WHERE series_id=?", (job_id, sid))
                    audit("AUTO_ENHANCE_QUEUED", "series", sid, f"job={job_id};profile={profile};strength={strength}")
