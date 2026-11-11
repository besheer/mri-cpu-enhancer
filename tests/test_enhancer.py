from __future__ import annotations

import io

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from app.enhancer import (
    UnsupportedDicom,
    derived_series_description,
    enhance_dataset,
    is_derived_series,
)
from app.profiles import PREFIXES


def synthetic_mr(modality: str = "MR") -> FileDataset:
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = modality
    ds.PatientName = "ANON"
    ds.PatientID = "TEST"
    ds.SeriesDescription = "Synthetic 0.3T Test"
    ds.Rows = 64
    ds.Columns = 64
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    rng = np.random.default_rng(7)
    pixels = np.clip(900 + rng.normal(0, 80, (64, 64)), 0, 4095).astype(np.uint16)
    ds.PixelData = pixels.tobytes()
    return ds


def test_creates_distinct_research_series() -> None:
    source = synthetic_mr()
    original_uid = source.SOPInstanceUID
    target_series = generate_uid()
    output = enhance_dataset(source, target_series)

    assert output.SeriesInstanceUID == target_series
    assert output.SOPInstanceUID != original_uid
    assert output.SeriesDescription == "ENHANC-Synthetic 0.3T Test [ENHANCED]"
    assert output.SeriesDescription.startswith("ENHANC-")
    assert is_derived_series(output.SeriesDescription) is True
    assert output.ImageType[0] == "DERIVED"
    assert source.SOPInstanceUID == original_uid
    assert output.pixel_array.shape == source.pixel_array.shape


def test_output_can_be_serialized() -> None:
    output = enhance_dataset(synthetic_mr(), generate_uid())
    buffer = io.BytesIO()
    output.save_as(buffer, enforce_file_format=True)
    assert len(buffer.getvalue()) > 1024


def test_rejects_non_mr() -> None:
    with pytest.raises(UnsupportedDicom):
        enhance_dataset(synthetic_mr("CT"), generate_uid())


def test_description_is_prefixed_and_limited() -> None:
    result = derived_series_description("A" * 100)
    assert result.startswith("ENHANC-")
    assert result.endswith("[ENHANCED]")
    assert len(result) == 64


def test_derived_series_is_detected() -> None:
    desc = derived_series_description("AX T2", "ENHANC-RNLM-")
    assert desc.startswith("ENHANC-")
    assert desc.endswith("[RAD4AI-RNLM]")
    assert is_derived_series(desc) is True
    for profile_key, prefix in PREFIXES.items():
        d = derived_series_description(f"SAG T2 {profile_key}", prefix)
        assert d.startswith("ENHANC-")
        assert is_derived_series(d) is True
    assert is_derived_series("AX T2 FSE") is False
    assert is_derived_series("ENH STIR") is False
    assert is_derived_series("ENH AX T2 [ENHANCED]") is True
