from __future__ import annotations

import io
import os
import secrets
import zipfile
from contextlib import asynccontextmanager

import pydicom
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydicom.uid import generate_uid

from app.enhancer import UnsupportedDicom, enhance_dataset
from app.admin import router as admin_router
from app.viewer import router as viewer_router
from app.store import authorize, initialize
from app.worker import AutoProcessor
from app.comparison_worker import ComparisonWorker
from app.return_worker import ReturnWorker
from app.training import router as training_router
from app.intake_worker import IntakeWorker
from app.system_health import snapshot
from app.operations import router as operations_router
from app.synthetic import router as synthetic_router
from app.protocol_study import router as protocol_study_router

worker = AutoProcessor()
comparison_worker = ComparisonWorker()
return_worker = ReturnWorker()
intake_worker = IntakeWorker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize()
    worker.start()
    comparison_worker.start()
    return_worker.start()
    intake_worker.start()
    yield
    comparison_worker.stop()
    return_worker.stop()
    intake_worker.stop()
    worker.stop()

app = FastAPI(
    title="MRI 0.3T CPU Enhancer — Research MVP",
    version="1.5.0",
    description="Research-only DICOM enhancement service. Not for diagnosis.",
    lifespan=lifespan,
)
app.include_router(admin_router)
app.include_router(viewer_router)
app.include_router(training_router)
app.include_router(operations_router)
app.include_router(synthetic_router)
app.include_router(protocol_study_router)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "degraded" if worker.last_error or intake_worker.last_error else "ok",
            "mode": "research-cpu-baseline", "worker": worker.status(),
            "intake_worker": intake_worker.status(),
            "return_worker": return_worker.status(), "system": snapshot()}


@app.get("/internal/dicom-authorize", response_class=PlainTextResponse, include_in_schema=False)
def dicom_authorize(ip: str, aet: str, called_aet: str = "", x_internal_token: str = Header(default="")) -> str:
    expected = os.getenv("INTERNAL_TOKEN", "")
    if not expected or not secrets.compare_digest(x_internal_token, expected):
        raise HTTPException(403, "Forbidden")
    return "ALLOW" if authorize(ip, aet, called_aet) else "DENY"


@app.post("/v1/enhance")
async def enhance(files: list[UploadFile] = File(...)) -> StreamingResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one DICOM file is required")
    series_uid = generate_uid()
    archive = io.BytesIO()
    try:
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for index, upload in enumerate(files, start=1):
                raw = await upload.read()
                ds = pydicom.dcmread(io.BytesIO(raw))
                output = enhance_dataset(ds, series_uid)
                buffer = io.BytesIO()
                output.save_as(buffer, enforce_file_format=True)
                zf.writestr(f"IM{index:05d}.dcm", buffer.getvalue())
    except (pydicom.errors.InvalidDicomError, UnsupportedDicom, AttributeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    archive.seek(0)
    return StreamingResponse(
        archive,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="enhanced-series.zip"'},
    )
