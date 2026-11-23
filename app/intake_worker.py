from __future__ import annotations

import logging
import os
import threading

import httpx

from app.enhancer import is_derived_series
from app.profiles import PROFILE_INFO, PROFILES
from app.store import audit, auto_eligible_profile_names, connect, enabled_profile_names, enqueue_comparison, get_center_by_aet, record_feedback, setting

log = logging.getLogger("intake-worker")




def resolve_scan_profile_and_strength(
    study_desc: str,
    series_desc: str,
    default_profile: str,
    default_strength: str,
    auto_allowed: set[str],
    enabled_set: set[str],
    is_fast_scan: bool = False,
) -> tuple[str, str]:
    """Dedicated per-scan Smart Routing based on clinic-validated preferences."""
    import re
    u = f"{study_desc} {series_desc}".upper()
    profile, strength = default_profile, default_strength

    is_mrv = any(k in u for k in ("MRV", "MRA", "TOF", "ANGIO", "VENO", "PROJECTION", "MIP", "INFLOW", "PCA"))
    is_brain = any(w in u for w in ("BRAIN", "HEAD", "NEURO", "SKULL", "FLAIR"))
    is_pd = ("PD" in u or "PROTON" in u)
    is_stir = any(k in u for k in ("STIR", "FAT", "SPAIR", "FATSAT")) or bool(re.search(r"(?:^|[\s_\-/])FS(?:$|[\s_\-/])", u))

    is_fast_mri = is_fast_scan or any(k in u for k in ("FASTMRI", "FAST_MRI", "FAST ", "QUICK"))
    if is_fast_mri and not is_mrv and "RAD4AI_FAST_MRI" in enabled_set:
        return "RAD4AI_FAST_MRI", "STRONG"
    if is_mrv and "RICIAN_NLM_VST" in (auto_allowed & enabled_set):
        return "RICIAN_NLM_VST", "STRONG"
    if is_brain and "RICIAN_NLM_VST" in (auto_allowed & enabled_set):
        return "RICIAN_NLM_VST", "STRONG"
    if "RAD4AI_SPINE_PRO" in (auto_allowed & enabled_set):
        if is_pd:
            return "RAD4AI_SPINE_PRO", "STRONG"
        if is_stir and "FLAIR" not in u:
            if "RAD4AI_PRINT_MASTER" in enabled_set:
                return "RAD4AI_PRINT_MASTER", "MEDIUM"  # Technician-approved L4 high-contrast S-curve + bone grain for STIR
            return "RAD4AI_SPINE_PRO", "MILD"
        if "T1" in u and "RAD4AI_PRINT_MASTER" in enabled_set:
            return "RAD4AI_PRINT_MASTER", "MEDIUM"
        if "T2" in u or "T1" in u:
            return "RAD4AI_SPINE_PRO", "STRONG"
    return profile, strength


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
                self._check_and_execute_pending_orders()
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                log.exception("Inbound scan failed")
            self.stop_event.wait(self.interval)

    def _check_and_execute_pending_orders(self) -> None:
        """Periodically checks and auto-executes technician orders between scans."""
        try:
            last_id = int(setting("last_server_executed_feedback_id", "0") or "0")
            with connect() as db:
                rows = db.execute(
                    "SELECT id, patient_name, verdict, note, series_description FROM feedback_events WHERE id > ? ORDER BY id ASC LIMIT 5",
                    (last_id,)
                ).fetchall()
                if not rows:
                    return

                from app.store import set_setting
                for r in rows:
                    fid = int(r["id"])
                    note = str(r["note"] or "")
                    set_setting("last_server_executed_feedback_id", str(fid))

                    action_kws = ("نعم", "نعومه", "خشن", "خشونه", "فتح", "بيض", "سواد", "ابعت", "تعديل", "احسن", "جوده", "كونترست", "بروتون", "برتون", "اكسيل", "اكسل", "جردينت", "ركبه", "عظم", "فقرات")
                    if not any(k in note for k in action_kws):
                        continue

                    t_row = None
                    if any(k in note for k in ("بروتون", "برتون", "pd", "ركبه")):
                        t_row = db.execute("SELECT series_id, description FROM inbound_series WHERE description LIKE '%PD%' OR description LIKE '%PROTON%' ORDER BY received_at DESC LIMIT 1").fetchone()
                    elif any(k in note for k in ("اكسيل", "اكسل", "axial", "جردينت", "سواد")):
                        t_row = db.execute("SELECT series_id, description FROM inbound_series WHERE description LIKE '%AX%' OR description LIKE '%GRE%' ORDER BY received_at DESC LIMIT 1").fetchone()
                    elif any(k in note for k in ("فقرات", "عنقيه", "قطنيه", "t1", "t2")):
                        t_row = db.execute("SELECT series_id, description FROM inbound_series WHERE description LIKE '%SAG%' OR description LIKE '%FL08%' ORDER BY received_at DESC LIMIT 1").fetchone()
                    if not t_row:
                        t_row = db.execute("SELECT series_id, description FROM inbound_series ORDER BY received_at DESC LIMIT 1").fetchone()

                    if t_row:
                        sid = t_row["series_id"]
                        db.execute("UPDATE comparison_jobs SET archived=1, status='superseded' WHERE source_series_id=? AND status IN ('queued','processing')", (sid,))
                        cur = db.execute(
                            "INSERT INTO comparison_jobs(source_series_id, profile, strength, acknowledged) VALUES(?, 'RAD4AI_3T_HD', 'STRONG', 1)",
                            (sid,)
                        )
                        new_jid = int(cur.lastrowid)
                        audit("AUTO_SERVER_EXEC_ORDER", "comparison_job", str(new_jid), f"fid={fid};note={note[:50]}")
                        log.info("Auto-executed technician order from feedback #%d on series %s -> Job #%d", fid, t_row["description"], new_jid)
                        try:
                            from app.telegram_bot import send_telegram_notification
                            send_telegram_notification(
                                f"📥 *تم استقبال التعديلات برقم `#{fid}`!*\n"
                                f"⚙️ *جاري العمل على التعديلات فوراً على السلسلة:* `{t_row['description']}`\n"
                                f"• طلب الفني: _{note}_\n"
                                f"• رقم فحص التشغيل: `#{new_jid}`\n"
                                f"• سيتم إرسال الصور المعالجة تلقائياً لجهاز GE AW وتأكيد الإرسال."
                            )
                        except Exception:
                            pass
        except Exception as exc:
            log.warning("Server periodic feedback check error: %s", exc)

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
                grew_from = None
                with connect() as db:
                    existing_row = db.execute(
                        "SELECT instances, status FROM inbound_series WHERE series_id=?",
                        (sid,),
                    ).fetchone()
                    if existing_row:
                        prev_inst = int(existing_row["instances"] or 0)
                        if len(instances) > prev_inst and str(existing_row["status"] or "") not in ("SKIPPED_LOCALIZER",):
                            db.execute("DELETE FROM inbound_series WHERE series_id=?", (sid,))
                            grew_from = prev_inst
                        else:
                            continue
                if grew_from is not None:
                    audit("SERIES_GREW_REPROCESS", "series", sid, f"prev={grew_from};now={len(instances)}")
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
                auto_allowed = auto_eligible_profile_names()
                enabled_set = enabled_profile_names()
                is_fast_scan = False
                try:
                    t_r = api.get(f"{self.url}/instances/{instances[0]}/simplified-tags")
                    if t_r.status_code == 200:
                        tj = t_r.json()
                        nex = float(tj.get("NumberOfAverages") or 2.0)
                        dur = float(tj.get("AcquisitionDuration") or 999999999.0)
                        if False:
                            is_fast_scan = True
                except Exception:
                    pass
                profile, strength = resolve_scan_profile_and_strength(
                    study_desc,
                    u_desc,
                    center["auto_profile"],
                    center["auto_strength"],
                    auto_allowed,
                    enabled_set,
                    is_fast_scan=is_fast_scan,
                )
                if profile not in PROFILES or (profile not in auto_allowed and profile not in enabled_set):
                    with connect() as db:
                        db.execute("UPDATE inbound_series SET status='FAILED',error='Profile is not eligible for automatic processing',updated_at=CURRENT_TIMESTAMP WHERE series_id=?", (sid,))
                    continue
                if center["calibration_mode"]:
                    cal_specs = [
                        (profile, "MILD"),
                        (profile, "MEDIUM"),
                        (profile, "STRONG"),
                        ("RAD4AI_PRINT_MASTER", "MEDIUM"),
                        ("RAD4AI_3T_HD", "STRONG"),
                    ]
                    job_ids = [
                        enqueue_comparison(
                            sid,
                            p,
                            lvl,
                            acknowledged=(lvl == "STRONG" or bool(PROFILE_INFO.get(p, {}).get("research_only"))),
                        )
                        for p, lvl in cal_specs
                    ]
                    job_id = job_ids[0]
                    with connect() as db:
                        db.execute("UPDATE inbound_series SET status='QUEUED',job_id=?,updated_at=CURRENT_TIMESTAMP WHERE series_id=?", (job_id, sid))
                    audit("AUTO_CALIBRATION_QUEUED", "series", sid, f"jobs={job_ids};profile={profile}")
                else:
                    specs = [(profile, strength)]
                    candidates = [
                        ("RAD4AI_3T_HD", "STRONG"),
                        ("RAD4AI_FAST_MRI", "STRONG"),
                        ("RAD4AI_PRINT_MASTER", "MEDIUM"),
                        ("RAD4AI_SPINE_PRO", "STRONG"),
                    ]
                    for cand_prof, cand_str in candidates:
                        if len(specs) >= 3:
                            break
                        if cand_prof != profile and cand_prof in enabled_set:
                            specs.append((cand_prof, cand_str))
                    job_ids = [
                        enqueue_comparison(
                            sid,
                            p,
                            lvl,
                            acknowledged=(lvl == "STRONG" or bool(PROFILE_INFO.get(p, {}).get("research_only"))),
                        )
                        for p, lvl in specs
                    ]
                    job_id = job_ids[0]
                    with connect() as db:
                        db.execute("UPDATE inbound_series SET status='QUEUED',job_id=?,updated_at=CURRENT_TIMESTAMP WHERE series_id=?", (job_id, sid))
                    audit("AUTO_ENHANCE_QUEUED", "series", sid, f"jobs={job_ids};profile={profile};strength={strength}")
