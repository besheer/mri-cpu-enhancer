from __future__ import annotations

import datetime as dt
import io
import os
import httpx
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from app.store import audit, get_center

# Minimal 5x7 uppercase bitmap font for crisp DICOM card rendering (zero external font deps)
FONT_5X7: dict[str, list[str]] = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11100", "10010", "10001", "10001", "10001", "10010", "11100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10011", "10101", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["01110", "10000", "11110", "10001", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}

CARDS = [
    ("FB-L3-ACCEPTED", 901, "L3 ACCEPTED", "SEND TO MRI03T TO APPROVE L3"),
    ("FB-L3-REJECTED", 902, "L3 REJECTED", "SEND TO MRI03T TO REJECT L3"),
    ("FB-MORE-GRAIN", 903, "MORE GRAIN", "SEND TO MRI03T FOR MORE GRAIN"),
    ("FB-SMOOTHER", 904, "SMOOTHER", "SEND TO MRI03T FOR SMOOTHER"),
    ("FB-HIGHER-CLARITY", 905, "HIGHER CLARITY", "SEND TO MRI03T FOR MORE CLARITY"),
]


def _draw_text(canvas: np.ndarray, text: str, y0: int, scale: int, val: int) -> None:
    text = text.upper()
    total_w = len(text) * 6 * scale - scale
    x0 = max(12, (canvas.shape[1] - total_w) // 2)
    for idx, ch in enumerate(text):
        glyph = FONT_5X7.get(ch, FONT_5X7[" "])
        gx = x0 + idx * 6 * scale
        for r, row_bits in enumerate(glyph):
            for c, bit in enumerate(row_bits):
                if bit == "1":
                    y_s = y0 + r * scale
                    x_s = gx + c * scale
                    if y_s + scale < canvas.shape[0] and x_s + scale < canvas.shape[1]:
                        canvas[y_s : y_s + scale, x_s : x_s + scale] = val


def render_card_pixels(title: str, subtitle: str) -> np.ndarray:
    img = np.zeros((512, 512), dtype=np.uint16)
    img[18:24, 18:494] = 2800
    img[488:494, 18:494] = 2800
    img[18:494, 18:24] = 2800
    img[18:494, 488:494] = 2800
    _draw_text(img, "RAD4AI FEEDBACK", 70, 4, 2400)
    _draw_text(img, title, 210, 6, 3600)
    _draw_text(img, subtitle, 360, 2, 2000)
    return img


def build_card_dicom(study_uid: str, series_desc: str, series_num: int, title: str, subtitle: str) -> bytes:
    pixels = render_card_pixels(title, subtitle)
    now = dt.datetime.now(dt.timezone.utc)
    meta = Dataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset("feedback_card.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientName = "RAD4AI^FEEDBACK CARDS"
    ds.PatientID = "RAD4AI_FB"
    ds.StudyDescription = "RAD4AI FEEDBACK CARDS"
    ds.SeriesDescription = series_desc
    ds.StudyID = "RAD4AI_FB"
    ds.SeriesNumber = series_num
    ds.InstanceNumber = 1
    ds.Modality = "MR"
    ds.ImageType = ["DERIVED", "SECONDARY", "FEEDBACK_CARD"]
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.Manufacturer = "RAD4AI"
    ds.FrameOfReferenceUID = generate_uid()
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 5.0
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.ScanningSequence = "SE"
    ds.SequenceVariant = "NONE"
    ds.ScanOptions = "NONE"
    ds.MRAcquisitionType = "2D"
    ds.RepetitionTime = 500.0
    ds.EchoTime = 15.0
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.WindowCenter = 1800
    ds.WindowWidth = 3600
    ds.PixelData = pixels.tobytes()

    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def send_feedback_cards_to_center(center_id: int = 8) -> dict:
    center = get_center(center_id)
    if not center:
        raise ValueError(f"Center {center_id} not found")
    url = os.getenv("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
    auth = (os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", ""))
    study_uid = generate_uid()
    uploaded_series: list[str] = []

    with httpx.Client(auth=auth, timeout=60) as api:
        api.put(
            f"{url}/modalities/{center['slug']}",
            json={"AET": center["return_aet"], "Host": center["return_ip"], "Port": center["return_port"]},
        ).raise_for_status()
        api.post(f"{url}/modalities/{center['slug']}/echo").raise_for_status()

        for sdesc, snum, title, sub in CARDS:
            raw = build_card_dicom(study_uid, sdesc, snum, title, sub)
            r = api.post(f"{url}/instances", content=raw, headers={"Content-Type": "application/dicom"})
            r.raise_for_status()
            sid = r.json().get("ParentSeries")
            if sid:
                uploaded_series.append(sid)

        try:
            for sid in uploaded_series:
                api.post(f"{url}/modalities/{center['slug']}/store", json=sid).raise_for_status()
        finally:
            # Remove transient cards from Orthanc so re-sending any card from GE triggers a fresh intake capture
            for sid in uploaded_series:
                try:
                    api.delete(f"{url}/series/{sid}")
                except Exception:
                    pass

    audit("FEEDBACK_CARDS_SENT", "center", str(center_id), f"cards={len(uploaded_series)};dest={center['return_aet']}")
    return {"sent_cards": len(uploaded_series), "destination": f"{center['return_aet']}@{center['return_ip']}:{center['return_port']}"}
