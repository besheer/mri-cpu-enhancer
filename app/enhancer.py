from __future__ import annotations

import copy
from datetime import datetime, timezone

import numpy as np
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from scipy.ndimage import gaussian_filter

DERIVATION = "RESEARCH DENOISING; NOT FOR DIAGNOSIS"
DEFAULT_PREFIX = "ENHANC-"


class UnsupportedDicom(ValueError):
    pass


def _validate(ds: FileDataset) -> None:
    if str(ds.get("Modality", "")) != "MR":
        raise UnsupportedDicom("Only MR DICOM instances are accepted")
    if "PixelData" not in ds:
        raise UnsupportedDicom("DICOM instance has no Pixel Data")
    if int(ds.get("SamplesPerPixel", 1)) != 1:
        raise UnsupportedDicom("Only monochrome images are supported")
    if int(ds.get("NumberOfFrames", 1)) != 1:
        raise UnsupportedDicom("Multi-frame DICOM is not supported in the MVP")
    if getattr(ds.file_meta, "TransferSyntaxUID", ExplicitVRLittleEndian).is_compressed:
        raise UnsupportedDicom("Compressed transfer syntax is not supported in the MVP")


def conservative_baseline(image: np.ndarray) -> np.ndarray:
    """Conservative reference filter, not a trained diagnostic model."""
    x = image.astype(np.float32)
    low, high = np.percentile(x, (0.5, 99.5))
    if high <= low:
        return image.copy()

    clipped = np.clip(x, low, high)
    smooth = gaussian_filter(clipped, sigma=0.75)
    denoised = 0.75 * clipped + 0.25 * smooth

    # Mild low-frequency bias correction; deliberately capped.
    field = gaussian_filter(np.maximum(denoised, 0), sigma=max(8.0, min(x.shape) / 12))
    positive = field[field > 0]
    if positive.size:
        reference = float(np.median(positive))
        scale = np.clip(reference / np.maximum(field, reference * 0.25), 0.8, 1.2)
        denoised *= scale

    info = np.iinfo(image.dtype)
    return np.rint(np.clip(denoised, info.min, info.max)).astype(image.dtype)


def derived_series_description(original: str, prefix: str = DEFAULT_PREFIX) -> str:
    return f"{prefix}{original.strip() or 'MR SERIES'}"[:64]


def enhance_dataset(
    ds: FileDataset,
    series_uid: str,
    series_description: str | None = None,
) -> FileDataset:
    _validate(ds)
    # Dataset.copy() is shallow and would mutate shared DataElements in the source.
    output = copy.deepcopy(ds)
    output.PixelData = conservative_baseline(ds.pixel_array).tobytes()
    output.SeriesInstanceUID = series_uid
    output.SOPInstanceUID = generate_uid()
    output.file_meta = ds.file_meta.copy()
    output.file_meta.MediaStorageSOPInstanceUID = output.SOPInstanceUID
    output.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    output.SeriesDescription = series_description or derived_series_description(
        str(ds.get("SeriesDescription", "MR SERIES"))
    )
    output.ImageType = ["DERIVED", "SECONDARY", "RESEARCH"]
    output.DerivationDescription = DERIVATION
    output.BurnedInAnnotation = "NO"
    now = datetime.now(timezone.utc)
    output.ContentDate = now.strftime("%Y%m%d")
    output.ContentTime = now.strftime("%H%M%S.%f")
    return output


def derive_with_pixels(ds: FileDataset, pixels: np.ndarray, series_uid: str,
                       series_description: str, derivation: str) -> FileDataset:
    """Create a traceable derived DICOM instance without changing its geometry."""
    _validate(ds)
    if pixels.shape != ds.pixel_array.shape:
        raise UnsupportedDicom("Processed pixels do not match source geometry")
    output = copy.deepcopy(ds)
    output.PixelData = pixels.astype(ds.pixel_array.dtype, copy=False).tobytes()
    output.SeriesInstanceUID = series_uid
    output.SOPInstanceUID = generate_uid()
    output.file_meta = ds.file_meta.copy()
    output.file_meta.MediaStorageSOPInstanceUID = output.SOPInstanceUID
    output.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    output.SeriesDescription = series_description[:64]
    output.ImageType = ["DERIVED", "SECONDARY", "RESEARCH"]
    output.DerivationDescription = derivation[:1024]
    output.SpatialLocationsPreserved = "YES"
    reference = Dataset()
    reference.ReferencedSOPClassUID = ds.SOPClassUID
    reference.ReferencedSOPInstanceUID = ds.SOPInstanceUID
    output.SourceImageSequence = Sequence([reference])
    output.BurnedInAnnotation = "NO"
    now = datetime.now(timezone.utc)
    output.ContentDate = now.strftime("%Y%m%d")
    output.ContentTime = now.strftime("%H%M%S.%f")
    return output
