from __future__ import annotations

import base64
import copy
import io
import os
import struct
import time
import zlib
import zipfile
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from scipy.ndimage import zoom
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from app.admin import require_admin, layout, esc
from app.ai_engine import ai_engine
from app.profiles import PROFILES, STRENGTHS

router = APIRouter(tags=["Print Studio"])


def to_png_bytes(arr_2d: np.ndarray) -> bytes:
    # Normalize 2D array to uint8 RGB
    a = arr_2d.astype(np.float32)
    p1 = float(np.percentile(a, 1.0))
    p99 = float(np.percentile(a, 99.5))
    span = max(p99 - p1, 1e-4)
    norm = np.clip((a - p1) / span, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    h, w = u8.shape
    rgb = np.stack([u8, u8, u8], axis=-1)
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def enhance_dicom_dataset(ds: pydicom.Dataset, profile: str = "RAD4AI_PRINT_MASTER",
                          strength: str = "STRONG", upscale: bool = False) -> tuple[pydicom.Dataset, dict]:
    started = time.perf_counter()
    orig_pixels = ds.pixel_array
    orig_dtype = orig_pixels.dtype
    arr = orig_pixels.astype(np.float32)
    
    strength_val = 1.0 if strength == "STRONG" else 0.8 if strength == "MEDIUM" else 0.5
    mode = "rician_nlm" if profile == "RICIAN_NLM_VST" else "3t_hd" if profile == "RAD4AI_3T_HD" else "print_master" if profile == "RAD4AI_PRINT_MASTER" else "ge_aw" if profile == "RAD4AI_GE_AW" else "spine_pro" if profile == "RAD4AI_SPINE_PRO" else "standard"
    
    if arr.ndim == 2:
        enhanced = ai_engine.enhance_slice(arr, strength=strength_val, mode=mode, upscale=upscale, description=str(ds.get("SeriesDescription", "")))
    elif arr.ndim == 3:
        enhanced = ai_engine.enhance_volume(arr, strength=strength_val, mode=mode, description=str(ds.get("SeriesDescription", "")))
    else:
        raise ValueError(f"Unsupported DICOM pixel dimension: {arr.ndim}")
        
    out_ds = copy.deepcopy(ds)
    if hasattr(out_ds, "_pixel_array"):
        delattr(out_ds, "_pixel_array")
    out_ds.SeriesInstanceUID = generate_uid()
    out_ds.SOPInstanceUID = generate_uid()
    orig_desc = str(ds.get("SeriesDescription", "MR SERIES")).strip()
    tag = "[RAD4AI-RNLM]" if profile == "RICIAN_NLM_VST" else "[RAD4AI-3THD]" if profile == "RAD4AI_3T_HD" else "[RAD4AI-PRINT]" if profile == "RAD4AI_PRINT_MASTER" else "[RAD4AI-GEAW]" if profile == "RAD4AI_GE_AW" else "[RAD4AI-PRO]"
    out_ds.SeriesDescription = f"ENH {orig_desc} {tag}"[:64]
    try:
        out_ds.SeriesNumber = int(ds.get("SeriesNumber", 0)) + 100
    except Exception:
        out_ds.SeriesNumber = 100
        
    if upscale and arr.ndim == 2 and enhanced.shape != arr.shape:
        out_ds.Rows, out_ds.Columns = enhanced.shape
        if hasattr(out_ds, "PixelSpacing"):
            try:
                ps = [float(x) for x in out_ds.PixelSpacing]
                out_ds.PixelSpacing = [ps[0] / 2.0, ps[1] / 2.0]
            except Exception:
                pass

    info = np.iinfo(orig_dtype) if np.issubdtype(orig_dtype, np.integer) else None
    if info:
        final_pixels = np.rint(np.clip(enhanced, info.min, info.max)).astype(orig_dtype)
    else:
        final_pixels = enhanced.astype(orig_dtype)
        
    out_ds.PixelData = final_pixels.tobytes()
    elapsed = round(time.perf_counter() - started, 3)
    metrics = {
        "profile": profile,
        "strength": strength,
        "upscaled": upscale,
        "original_shape": list(arr.shape),
        "enhanced_shape": list(enhanced.shape),
        "seconds": elapsed
    }
    return out_ds, metrics


@router.get("/admin/print-studio", response_class=HTMLResponse)
def print_studio_page(_: str = Depends(require_admin)):
    profile_options = "".join(f"<option value='{k}' {'selected' if k=='RAD4AI_PRINT_MASTER' else ''}>{v}</option>"
                               for k, v in PROFILES.items() if k.startswith(("RAD4AI", "RICIAN", "AI_")))
    strength_options = "".join(f"<option value='{k}' {'selected' if k=='STRONG' else ''}>{v}</option>"
                                for k, v in STRENGTHS.items())
                                
    body = """
<section class='panel' style='background:linear-gradient(180deg,#0f172a 0%,#090d16 100%);border:1px solid rgba(16,185,129,0.25);border-radius:12px;padding:24px'>
  <div class='panel-head' style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px'>
    <div>
      <h2 style='font-size:22px;color:#f8fafc;margin:0'>🖨️ RAD4AI Print-Master Studio</h2>
      <p style='color:#94a3b8;margin:4px 0 0'>Rician-VST Physics Restoration · Zero-Grain Velvet Tissue · Pure Black Air Background</p>
    </div>
    <span class='badge ok' style='font-size:13px;padding:6px 12px'>✓ Print Ultra Active</span>
  </div>

  <div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px'>
    <!-- Upload Box -->
    <div style='background:rgba(30,41,59,0.6);border:2px dashed #3b82f6;border-radius:10px;padding:24px;text-align:center'>
      <input type='file' id='file-input' accept='.dcm,.zip,.png,.jpg,.jpeg' style='display:none' onchange='handleFileSelect(this.files)'>
      <div id='drop-zone' onclick="document.getElementById('file-input').click()" style='cursor:pointer;padding:20px'>
        <div style='font-size:40px;margin-bottom:10px'>📂</div>
        <b style='color:#e2e8f0;font-size:16px'>Click or Drag & Drop MRI file here</b>
        <p style='color:#64748b;font-size:12px;margin:8px 0 0'>Supports DICOM (.dcm), Series (.zip), or Image (.png, .jpg)</p>
        <div id='selected-file-name' style='color:#38bdf8;margin-top:10px;font-weight:bold'></div>
      </div>
    </div>

    <!-- Options Box -->
    <div style='background:rgba(30,41,59,0.6);border:1px solid #334155;border-radius:10px;padding:20px;display:flex;flex-direction:column;justify-content:space-between'>
      <div>
        <div class='field' style='margin-bottom:14px'>
          <label style='color:#cbd5e1;font-weight:bold;margin-bottom:6px;display:block'>Enhancer Profile</label>
          <select id='profile-select' style='width:100%;padding:10px;background:#0f172a;border:1px solid #475569;border-radius:6px;color:#f8fafc'>{{PROFILE_OPTIONS}}</select>
        </div>
        <div class='field' style='margin-bottom:14px'>
          <label style='color:#cbd5e1;font-weight:bold;margin-bottom:6px;display:block'>Print Strength</label>
          <select id='strength-select' style='width:100%;padding:10px;background:#0f172a;border:1px solid #475569;border-radius:6px;color:#f8fafc'>{{STRENGTH_OPTIONS}}</select>
        </div>
        <div style='margin-bottom:14px;display:flex;align-items:center;gap:10px'>
          <input type='checkbox' id='upscale-check' checked style='width:18px;height:18px'>
          <label for='upscale-check' style='color:#e2e8f0;cursor:pointer'>2x Bicubic Print Interpolation (512×512 Display Boost)</label>
        </div>
        <div style='display:flex;align-items:center;gap:10px'>
          <input type='checkbox' id='send-ge-check' style='width:18px;height:18px'>
          <label for='send-ge-check' style='color:#e2e8f0;cursor:pointer'>Also push directly to GE Workstation (192.168.0.206)</label>
        </div>
      </div>
      <button id='enhance-btn' onclick='processUpload()' class='btn' style='background:linear-gradient(135deg,#059669,#10b981);font-size:15px;padding:12px;margin-top:16px;width:100%;font-weight:bold'>✨ Enhance for Printing</button>
    </div>
  </div>

  <!-- Progress Bar -->
  <div id='progress-wrap' style='display:none;margin-top:20px'>
    <div style='color:#38bdf8;font-weight:bold;margin-bottom:6px' id='progress-text'>Processing with RAD4AI Print-Master Engine…</div>
    <div style='width:100%;background:#1e293b;border-radius:4px;overflow:hidden;height:8px'>
      <div id='progress-bar' style='width:100%;background:#10b981;height:100%;animation:pulse 1.5s infinite'></div>
    </div>
  </div>

  <!-- Comparison & Result Display -->
  <div id='result-box' style='display:none;margin-top:24px;border-top:1px solid #334155;padding-top:20px'>
    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:16px'>
      <h3 style='margin:0;color:#f8fafc'>Visual Comparison (Original vs. RAD4AI Restored)</h3>
      <div style='display:flex;gap:10px'>
        <button onclick='window.print()' class='btn' style='background:#3b82f6'>🖨️ Print Now</button>
        <a id='download-link' class='btn' style='background:#059669' download='enhanced_print.dcm'>⬇️ Download Enhanced File</a>
      </div>
    </div>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>
      <div style='background:#000;border:1px solid #334155;border-radius:8px;padding:10px;text-align:center'>
        <div style='color:#94a3b8;font-weight:bold;margin-bottom:8px'>Original 0.3T Raw Scan</div>
        <img id='orig-img' style='max-width:100%;max-height:480px;border-radius:4px' src=''>
      </div>
      <div style='background:#000;border:1px solid #10b981;border-radius:8px;padding:10px;text-align:center;box-shadow:0 0 15px rgba(16,185,129,0.2)'>
        <div style='color:#10b981;font-weight:bold;margin-bottom:8px'>✨ RAD4AI Restored (Film/Paper Quality)</div>
        <img id='enh-img' style='max-width:100%;max-height:480px;border-radius:4px' src=''>
      </div>
    </div>
    <div id='metrics-detail' style='margin-top:14px;color:#94a3b8;font-size:13px'></div>
  </div>
</section>

<script>
let selectedFile = null;
function handleFileSelect(files){
  if(files && files[0]){
    selectedFile = files[0];
    document.getElementById('selected-file-name').textContent = 'Selected: ' + selectedFile.name + ' (' + (selectedFile.size/1024).toFixed(1) + ' KB)';
  }
}
const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover', (e)=>{ e.preventDefault(); dropZone.style.background='rgba(59,130,246,0.1)'; });
dropZone.addEventListener('dragleave', (e)=>{ e.preventDefault(); dropZone.style.background='transparent'; });
dropZone.addEventListener('drop', (e)=>{
  e.preventDefault();
  dropZone.style.background='transparent';
  if(e.dataTransfer.files && e.dataTransfer.files[0]){
    handleFileSelect(e.dataTransfer.files);
  }
});

async function processUpload(){
  if(!selectedFile){
    alert('Please select or drop an MRI file first.');
    return;
  }
  const btn = document.getElementById('enhance-btn');
  const prog = document.getElementById('progress-wrap');
  btn.disabled = true;
  prog.style.display = 'block';
  
  const fd = new FormData();
  fd.append('file', selectedFile);
  fd.append('profile', document.getElementById('profile-select').value);
  fd.append('strength', document.getElementById('strength-select').value);
  fd.append('upscale', document.getElementById('upscale-check').checked ? 'true' : 'false');
  fd.append('send_ge', document.getElementById('send-ge-check').checked ? 'true' : 'false');
  
  try{
    const res = await fetch('/api/v1/enhance-studio', { method:'POST', body:fd });
    if(!res.ok){
      const err = await res.json();
      throw new Error(err.detail || 'Enhancement failed');
    }
    const data = await res.json();
    document.getElementById('orig-img').src = data.orig_preview;
    document.getElementById('enh-img').src = data.enh_preview;
    document.getElementById('download-link').href = data.download_url;
    document.getElementById('download-link').setAttribute('download', data.download_filename);
    document.getElementById('metrics-detail').innerHTML = 
      '<b>Processing Time:</b> ' + data.metrics.seconds + 's · <b>Dimensions:</b> ' + data.metrics.original_shape.join('×') + ' → <b>' + data.metrics.enhanced_shape.join('×') + '</b>' +
      (data.sent_ge ? ' · <span style="color:#10b981">✓ Pushed to GE Workstation (192.168.0.206:4006)</span>' : '');
    document.getElementById('result-box').style.display = 'block';
    document.getElementById('result-box').scrollIntoView({ behavior:'smooth' });
  }catch(e){
    alert('Error: ' + e.message);
  }finally{
    btn.disabled = false;
    prog.style.display = 'none';
  }
}
</script>
"""
    html = body.replace("{{PROFILE_OPTIONS}}", profile_options).replace("{{STRENGTH_OPTIONS}}", strength_options)
    return layout("Print Studio", html)


@router.post("/api/v1/enhance-studio")
async def enhance_studio(file: UploadFile = File(...), profile: str = Form("RAD4AI_PRINT_MASTER"),
                         strength: str = Form("STRONG"), upscale: str = Form("false"),
                         send_ge: str = Form("false")):
    do_upscale = upscale.lower() in ("true", "1", "on")
    do_send_ge = send_ge.lower() in ("true", "1", "on")
    raw = await file.read()
    
    is_dcm = False
    try:
        ds = pydicom.dcmread(io.BytesIO(raw))
        is_dcm = True
    except Exception:
        is_dcm = False

    if is_dcm:
        orig_slice = ds.pixel_array.astype(np.float32)
        orig_slice_2d = orig_slice[orig_slice.shape[0]//2] if orig_slice.ndim == 3 else orig_slice
        orig_png = to_png_bytes(orig_slice_2d)
        
        out_ds, metrics = enhance_dicom_dataset(ds, profile=profile, strength=strength, upscale=do_upscale)
        enh_arr = out_ds.pixel_array.astype(np.float32)
        enh_slice_2d = enh_arr[enh_arr.shape[0]//2] if enh_arr.ndim == 3 else enh_arr
        enh_png = to_png_bytes(enh_slice_2d)
        
        # Save to temp buffer for download
        tmp_id = f"print_{int(time.time()*1000)}"
        tmp_path = f"/tmp/{tmp_id}.dcm"
        out_ds.save_as(tmp_path, enforce_file_format=True)
        download_url = f"/api/v1/download-temp?file={tmp_id}&ext=dcm"
        download_filename = f"ENH_{file.filename or 'print.dcm'}"
        
        # Push to GE Workstation if requested
        sent_ge = False
        if do_send_ge:
            try:
                from pynetdicom import AE
                ae = AE(ae_title=b"MRI03T")
                ae.add_requested_context(out_ds.SOPClassUID)
                assoc = ae.associate("192.168.0.206", 4006, ae_title=b"adw")
                if assoc.is_established:
                    st = assoc.send_c_store(out_ds)
                    sent_ge = (st.Status == 0)
                    assoc.release()
            except Exception:
                sent_ge = False
                
        return JSONResponse({
            "status": "success",
            "metrics": metrics,
            "orig_preview": f"data:image/png;base64,{base64.b64encode(orig_png).decode('ascii')}",
            "enh_preview": f"data:image/png;base64,{base64.b64encode(enh_png).decode('ascii')}",
            "download_url": download_url,
            "download_filename": download_filename,
            "sent_ge": sent_ge
        })
    else:
        # Standard image (PNG/JPG)
        import SimpleITK as sitk
        tmp_in = f"/tmp/upload_{int(time.time()*1000)}"
        try:
            with open(tmp_in, "wb") as f_in:
                f_in.write(raw)
            itk_img = sitk.ReadImage(tmp_in)
            img_arr = sitk.GetArrayFromImage(itk_img).astype(np.float32)
        except Exception as exc:
            raise HTTPException(422, f"Unsupported image file format: {exc}")
        finally:
            if os.path.exists(tmp_in):
                try:
                    os.remove(tmp_in)
                except Exception:
                    pass
            
        strength_val = 1.0 if strength == "STRONG" else 0.8 if strength == "MEDIUM" else 0.5
        mode = "rician_nlm" if profile == "RICIAN_NLM_VST" else "3t_hd" if profile == "RAD4AI_3T_HD" else "print_master" if profile == "RAD4AI_PRINT_MASTER" else "ge_aw" if profile == "RAD4AI_GE_AW" else "spine_pro" if profile == "RAD4AI_SPINE_PRO" else "standard"
        
        if img_arr.ndim == 3 and img_arr.shape[0] in (3, 4):
            # RGB to Gray
            gray = 0.299 * img_arr[0] + 0.587 * img_arr[1] + 0.114 * img_arr[2]
        elif img_arr.ndim == 3 and img_arr.shape[-1] in (3, 4):
            gray = 0.299 * img_arr[:,:,0] + 0.587 * img_arr[:,:,1] + 0.114 * img_arr[:,:,2]
        else:
            gray = img_arr if img_arr.ndim == 2 else img_arr[0]
            
        orig_png = to_png_bytes(gray)
        started = time.perf_counter()
        enh = ai_engine.enhance_slice(gray, strength=strength_val, mode=mode, upscale=do_upscale)
        elapsed = round(time.perf_counter() - started, 3)
        enh_png = to_png_bytes(enh)
        
        tmp_id = f"print_{int(time.time()*1000)}"
        with open(f"/tmp/{tmp_id}.png", "wb") as f:
            f.write(enh_png)
        download_url = f"/api/v1/download-temp?file={tmp_id}&ext=png"
        download_filename = f"ENH_{os.path.splitext(file.filename or 'print')[0]}.png"
        
        return JSONResponse({
            "status": "success",
            "metrics": {
                "profile": profile,
                "strength": strength,
                "upscaled": do_upscale,
                "original_shape": list(gray.shape),
                "enhanced_shape": list(enh.shape),
                "seconds": elapsed
            },
            "orig_preview": f"data:image/png;base64,{base64.b64encode(orig_png).decode('ascii')}",
            "enh_preview": f"data:image/png;base64,{base64.b64encode(enh_png).decode('ascii')}",
            "download_url": download_url,
            "download_filename": download_filename,
            "sent_ge": False
        })


@router.get("/api/v1/download-temp")
def download_temp(file: str, ext: str):
    safe_name = os.path.basename(file)
    safe_ext = "dcm" if ext == "dcm" else "png"
    path = f"/tmp/{safe_name}.{safe_ext}"
    if not os.path.exists(path):
        raise HTTPException(404, "File not found or expired")
    with open(path, "rb") as f:
        data = f.read()
    media = "application/dicom" if safe_ext == "dcm" else "image/png"
    return Response(data, media_type=media, headers={"Content-Disposition": f'attachment; filename="ENH_{safe_name}.{safe_ext}"'})


@router.post("/api/v1/enhance-dicom")
async def api_enhance_dicom(file: UploadFile = File(...), profile: str = Form("RAD4AI_PRINT_MASTER"),
                            strength: str = Form("STRONG"), upscale: bool = Form(False)):
    raw = await file.read()
    try:
        ds = pydicom.dcmread(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Invalid DICOM file: {exc}")
    out_ds, metrics = enhance_dicom_dataset(ds, profile=profile, strength=strength, upscale=upscale)
    buf = io.BytesIO()
    out_ds.save_as(buf, enforce_file_format=True)
    return Response(
        buf.getvalue(),
        media_type="application/dicom",
        headers={
            "Content-Disposition": f'attachment; filename="ENH_{file.filename or "image.dcm"}"',
            "X-Profile": profile,
            "X-Strength": strength,
            "X-Seconds": str(metrics["seconds"])
        }
    )
