from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import httpx
import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
from pynetdicom import AE
from pynetdicom.sop_class import MRImageStorage as MRStorage, Verification

from app.store import audit, connect, enqueue_return, initialize, slugify

ORTHANC = "orthanc"
OUT = Path("/e2e")
SOURCE_AET = "E2E_SENDER"
RETURN_AET = "E2E_RETURN"


def wait_for_receiver(timeout: int = 60) -> str:
    """Wait for Compose DNS, TCP listener, and DICOM Verification readiness."""
    deadline = time.time() + timeout
    last_error = "not ready"
    while time.time() < deadline:
        try:
            receiver_ip = socket.gethostbyname("e2e-pacs")
            with socket.create_connection((receiver_ip, 11112), timeout=3):
                pass
            echo_ae = AE(ae_title=SOURCE_AET)
            echo_ae.add_requested_context(Verification)
            assoc = echo_ae.associate(receiver_ip, 11112, ae_title=RETURN_AET)
            if assoc.is_established:
                status = assoc.send_c_echo()
                assoc.release()
                if status and status.Status == 0x0000:
                    print("E2E_PACS_READY=PASS", flush=True)
                    return receiver_ip
            last_error = "DICOM C-ECHO not ready"
        except (OSError, AttributeError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"Mock PACS readiness timed out: {last_error}")


def cleanup_e2e() -> None:
    with connect() as db:
        centers = [row["id"] for row in db.execute("SELECT id FROM centers WHERE source_aet=?", (SOURCE_AET,))]
        for center_id in centers:
            db.execute("UPDATE comparison_jobs SET archived=1 WHERE source_series_id IN (SELECT series_id FROM inbound_series WHERE center_id=?)", (center_id,))
            db.execute("UPDATE inbound_series SET archived=1 WHERE center_id=?", (center_id,))
            db.execute("UPDATE return_jobs SET archived=1 WHERE center_id=?", (center_id,))
            db.execute("DELETE FROM centers WHERE id=?", (center_id,))


def make_slice(study_uid: str, series_uid: str, index: int) -> FileDataset:
    sop_uid = generate_uid()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, sop_uid
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
    ds.PatientName, ds.PatientID = "E2E^SYNTHETIC", "E2E-TEST"
    ds.Modality, ds.SeriesDescription = "MR", "E2E SYNTHETIC 0.3T"
    ds.MagneticFieldStrength, ds.InstanceNumber = "0.3", index
    ds.Rows = ds.Columns = 128
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelSpacing, ds.SliceThickness, ds.SpacingBetweenSlices = [1.0, 1.0], "4", "4"
    y, x = np.ogrid[-1:1:128j, -1:1:128j]
    rng = np.random.default_rng(100 + index)
    image = 350 + 1600 * np.exp(-(x*x + y*y) * 4) + rng.normal(0, 90, (128, 128))
    ds.PixelData = np.clip(image, 0, 4095).astype(np.uint16).tobytes()
    return ds


def main():
    initialize()
    enhancer_ip = socket.gethostbyname(socket.gethostname())
    receiver_ip = wait_for_receiver()
    cleanup_e2e()
    with connect() as db:
        cur = db.execute("INSERT INTO centers(name,slug,source_ip,source_aet,return_ip,return_aet,return_port,network_mode,return_policy,last_echo_status,auto_enhance_enabled,auto_profile,auto_strength) VALUES(?,?,?,?,?,?,11112,'VPN','AUTO_PASS','PASS',1,'EDGE_DENOISE_MILD','MILD')",
                         ("E2E Synthetic", slugify("E2E Synthetic"), enhancer_ip, SOURCE_AET, receiver_ip, RETURN_AET))
        center_id = cur.lastrowid
    study_uid, series_uid = generate_uid(), generate_uid()
    sender = AE(ae_title=SOURCE_AET)
    sender.add_requested_context(MRStorage, ExplicitVRLittleEndian)
    assoc = sender.associate(ORTHANC, 4242, ae_title="MRI03T")
    assert assoc.is_established, "Orthanc association failed"
    for index in range(1, 4):
        status = assoc.send_c_store(make_slice(study_uid, series_uid, index))
        assert status and status.Status == 0x0000, f"C-STORE failed at {index}"
    assoc.release()
    deadline = time.time() + 240
    state = ""
    while time.time() < deadline:
        with connect() as db:
            row = db.execute("SELECT status,error,job_id FROM inbound_series WHERE center_id=? AND description='E2E SYNTHETIC 0.3T' ORDER BY received_at DESC LIMIT 1", (center_id,)).fetchone()
        if row:
            state = row["status"]
            print("PIPELINE_STATUS=" + state, flush=True)
            if state == "READY":
                with connect() as db:
                    job = db.execute("SELECT id,qc_status,output_series_id FROM comparison_jobs WHERE id=?", (row["job_id"],)).fetchone()
                    returned = db.execute("SELECT 1 FROM return_jobs WHERE comparison_job_id=?", (row["job_id"],)).fetchone()
                if job and job["qc_status"] == "WARN" and not returned:
                    print("QC_WARN_AUTO_RETURN_BLOCK=PASS", flush=True)
                    enqueue_return(job["id"], center_id, job["output_series_id"])
                    audit("E2E_MANUAL_RETURN", "comparison_job", job["id"], "Synthetic WARN approved only to test transport")
                    print("E2E_MANUAL_RETURN_QUEUED=PASS", flush=True)
                elif job and job["qc_status"] == "FAIL":
                    raise RuntimeError("Synthetic result failed Quality Gate")
            if state == "SENT":
                break
            if state in {"FAILED", "RETURN_FAILED"}:
                raise RuntimeError(row["error"] or state)
        time.sleep(3)
    assert state == "SENT", f"Timed out; final state={state}"
    status = json.loads((OUT / "status.json").read_text())
    assert status["received"] == 3, status
    files = list(OUT.glob("*.dcm"))
    import pydicom
    result = pydicom.dcmread(files[0], stop_before_pixels=True)
    assert str(result.SeriesDescription).startswith("ENHANC-"), result.SeriesDescription
    print("RETURNED_INSTANCES=3")
    print("DERIVED_SERIES=" + str(result.SeriesDescription))
    audit("E2E_RESULT", "system", "pipeline", "PASS; profile=EDGE_DENOISE_MILD; returned=3")
    cleanup_e2e()
    print("E2E_RESULT=PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        audit("E2E_RESULT", "system", "pipeline", f"FAIL; {str(exc)[:500]}")
        cleanup_e2e()
        print("E2E_RESULT=FAIL", flush=True)
        raise
