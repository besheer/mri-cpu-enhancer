from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, sobel

from app.rician_nlm import (
    estimate_sigma,
    rician_forward_vst,
    non_local_means,
    structure_tensor_tangent_restore,
)


class MRIAIEngine:
    def __init__(self):
        self.session = None
        self.model_mtime = 0.0
        self._load()

    def _find_model_path(self) -> Path | None:
        paths = [
            Path("/app/models/swift_nafnet_03t.onnx"),
            Path("models/swift_nafnet_03t.onnx"),
            Path("/models/swift_nafnet_03t.onnx"),
        ]
        for p in paths:
            if p.exists():
                return p
        return None

    def _get_mtime(self, p: Path) -> float:
        mtime = p.stat().st_mtime
        data_p = Path(str(p) + ".data")
        if data_p.exists():
            mtime = max(mtime, data_p.stat().st_mtime)
        return mtime

    def _load(self):
        p = self._find_model_path()
        if p is not None:
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = int(os.getenv("OMP_NUM_THREADS", "4"))
                self.session = ort.InferenceSession(str(p), sess_options=opts, providers=["CPUExecutionProvider"])
                self.model_mtime = self._get_mtime(p)
                print(f"✓ AI Model Loaded Successfully from: {p} (mtime={self.model_mtime})")
                return
            except Exception as e:
                print(f"Error loading ONNX from {p}: {e}")
        print("! AI Engine: ONNX model not loaded; falling back to Rician-VST + Structure-Tensor NLM physics core.")

    @property
    def is_available(self) -> bool:
        p = self._find_model_path()
        if p is not None and (self.session is None or self._get_mtime(p) > self.model_mtime):
            self._load()
        return self.session is not None

    @staticmethod
    def detect_sequence_type(description: str = "") -> str:
        u = (description or "").upper()
        if any(k in u for k in ("MRV", "MRA", "TOF", "ANGIO", "VENO", "FLOW", "PROJECTION", "MIP", "INFLOW", "PCA")):
            return "ANGIO_MRV"
        if "PD" in u or "PROTON" in u:
            return "PD_ORTHO"
        import re as _re
        if any(k in u for k in ("STIR", "FLAIR", "FAT", "SPAIR", "FATSAT")) or _re.search(r"(?:^|[\s_\-/])FS(?:$|[\s_\-/])", u):
            return "STIR_FLAIR"
        if "T2*" in u or "GRE" in u or "FE" in u:
            return "T2_STAR_GRE"
        if "T2" in u:
            return "T2_FSE"
        if "T1" in u:
            return "T1_SE"
        return "GENERAL_MR"

    def enhance_volume(self, volume: np.ndarray, strength: float = 0.8, mode: str = "standard",
                       description: str = "") -> np.ndarray:
        seq_type = self.detect_sequence_type(description)
        u_desc_init = (description or "").upper()
        is_axial = any(k in u_desc_init for k in ("AX", "OAX", "TRANS"))

        # Calibrated clinical strength mapping (0.5=MILD, 0.8=MEDIUM, 1.0=STRONG)
        if mode in ("print_master", "3t_hd", "enh6_ultra"):
            flat_denoise = float(np.clip(0.76 + 0.16 * strength, 0.78, 0.94))
        elif mode == "ge_aw":
            flat_denoise = float(np.clip(0.56 + 0.16 * strength, 0.62, 0.74))
        elif mode == "sr_15t":
            flat_denoise = float(np.clip(0.54 + 0.16 * strength, 0.60, 0.72))
        elif mode == "fast_mri":
            flat_denoise = float(np.clip(0.84 + 0.12 * strength, 0.84, 0.96))
        elif mode == "spine_pro":
            flat_denoise = float(np.clip(0.84 + 0.12 * strength, 0.84, 0.96))
        else:
            flat_denoise = float(np.clip(0.45 + 0.28 * strength, 0.52, 0.74))

        edge_protect_factor = 0.94
        bg_suppress_factor = 0.96 if mode == "print_master" else 0.35 if is_axial else 0.72 if (mode == "fast_mri" or seq_type in ("T2_FSE", "T2_STAR_GRE")) else 0.45 if mode in ("ge_aw", "3t_hd") else 0.18 if mode == "rician_nlm" else 0.35
        crisp_boost = 0.036 if mode == "rician_nlm" else (0.054 if seq_type in ("T2_STAR_GRE", "T2_FSE", "T1_SE") else 0.048) if mode in ("print_master", "3t_hd", "enh6_ultra", "sr_15t", "spine_pro", "ge_aw", "fast_mri") else 0.020

        if mode == "rician_nlm":
            edge_protect_factor = 0.98
            flat_denoise = min(flat_denoise, 0.60)
        if seq_type == "ANGIO_MRV":
            edge_protect_factor = 0.99
            bg_suppress_factor = 0.68
            crisp_boost = 0.050
        elif seq_type == "STIR_FLAIR":
            edge_protect_factor = 0.99
            bg_suppress_factor = 0.18 if mode == "rician_nlm" else 0.85
            flat_denoise = min(flat_denoise, 0.78) if mode == "print_master" else min(flat_denoise, 0.68)
            crisp_boost = max(crisp_boost, 0.062)
        elif seq_type == "T1_SE":
            edge_protect_factor = 0.985
            flat_denoise = min(flat_denoise, 0.78)
            crisp_boost = max(crisp_boost, 0.048)
        elif seq_type == "PD_ORTHO":
            edge_protect_factor = 0.985
            flat_denoise = min(flat_denoise, 0.85)
            crisp_boost = max(crisp_boost, 0.045)

        num_slices = volume.shape[0]
        has_onnx = self.is_available
        input_name = self.session.get_inputs()[0].name if has_onnx else ""

        # Compute one global 3D volume intensity window preserving full T2 fluid dynamic range (no 99% clipping)
        vol_f = volume.astype(np.float32)
        vol_low = float(np.percentile(vol_f, 0.5))
        p999 = float(np.percentile(vol_f, 99.95))
        v_max = float(vol_f.max())
        vol_high = v_max if v_max <= p999 * 1.8 else p999
        vol_span = max(float(vol_high - vol_low), 1e-4)

        # Estimate one global Rician noise sigma on the middle slice using the unified volume window
        mid_idx = num_slices // 2
        mid_src = vol_f[mid_idx]
        mid_norm = np.clip((mid_src - vol_low) / vol_span, 0.0, 1.0)
        global_sigma = estimate_sigma(mid_norm)

        norm_slices = []
        raw_norm_slices = []
        base_denoised_slices = []

        for i in range(num_slices):
            src = vol_f[i]
            norm = np.ascontiguousarray(np.clip((src - vol_low) / vol_span, 0.0, 1.0), dtype=np.float32)
            raw_norm_slices.append(norm)

            if mode in ("sr_15t", "ge_aw", "print_master", "3t_hd", "spine_pro", "fast_mri"):
                # GE PURE / SCIC Surface Coil Shading Equalization (lifts posterior/inferior coil drop-off without brightening air)
                illum = gaussian_filter(norm, sigma=max(14.0, min(norm.shape) / 8.0))
                fg = norm > 0.08
                if np.any(fg):
                    ref_illum = float(np.median(illum[fg]))
                    coil_gain = np.clip(ref_illum / np.maximum(illum, ref_illum * 0.32), 0.88, 1.80)
                    gain_weight = 0.65 if mode in ("print_master", "3t_hd", "fast_mri") else 0.62 if mode in ("ge_aw", "spine_pro") else 0.35
                    fg_soft = np.clip((norm - 0.04) / 0.06, 0.0, 1.0)
                    eff_gain = 1.0 + fg_soft * gain_weight * (coil_gain - 1.0)
                    norm_work = np.clip(norm * eff_gain, 0.0, 1.0).astype(np.float32)
                else:
                    norm_work = norm
            else:
                norm_work = norm

            norm_slices.append(norm_work)

            # Physics-safe Rician-VST + Structure-Tensor Tangent-Guided Restoration
            max_eff = 1.10 if seq_type in ("PD_ORTHO", "STIR_FLAIR") else 1.35
            eff_strength = min(max_eff, strength * (1.35 if (global_sigma >= 0.024 and strength > 0.6) else 1.0))
            phys_clean, _ = structure_tensor_tangent_restore(
                norm_work, strength=eff_strength, sigma_est=global_sigma
            )

            if has_onnx and mode != "rician_nlm":
                inp = norm_work[np.newaxis, np.newaxis, :, :]
                pred = self.session.run(None, {input_name: inp})[0]
                onnx_out = np.clip(pred[0, 0].astype(np.float32), 0.0, 1.0)
                smooth_sigma = 0.35 if mode in ("spine_pro", "ge_aw") else 0.50
                onnx_clean = gaussian_filter(onnx_out, sigma=smooth_sigma)
                # Hybrid fusion: combine ONNX neural prediction with Rician-VST Structure-Tensor physics restoration
                w_phys = 0.65 if mode in ("3t_hd", "print_master", "spine_pro", "rician_nlm") else 0.45
                combined_clean = (1.0 - w_phys) * onnx_clean + w_phys * phys_clean
            else:
                combined_clean = phys_clean

            base_denoised_slices.append(combined_clean)

        results = []
        for i in range(num_slices):
            slice_2d = volume[i]
            norm_work = norm_slices[i]
            curr_clean = base_denoised_slices[i]

            # Multi-directional anatomical edge map (Sobel gradient magnitude on lightly smoothed slice)
            g_smooth = gaussian_filter(norm_work, sigma=1.1)
            gx = sobel(g_smooth, axis=1)
            gy = sobel(g_smooth, axis=0)
            grad = np.sqrt(gx * gx + gy * gy)
            p80 = max(float(np.percentile(grad, 80.0)), 1e-4)
            edge_mask = np.clip(grad / p80, 0.0, 1.0) ** 1.2

            # 2.5D Multi-Slice Context (z-1, z, z+1) Structure-Aware Coherence
            if num_slices >= 3:
                prev_idx = max(0, i - 1)
                next_idx = min(num_slices - 1, i + 1)
                n_prev = base_denoised_slices[prev_idx]
                n_next = base_denoised_slices[next_idx]
                diff_prev = np.abs(norm_work - norm_slices[prev_idx])
                diff_next = np.abs(norm_work - norm_slices[next_idx])
                flat_zone = (1.0 - edge_mask) ** 2
                w_prev = 0.06 * flat_zone * np.exp(- (diff_prev / 0.06) ** 2)
                w_next = 0.06 * flat_zone * np.exp(- (diff_next / 0.06) ** 2)
                w_curr = 1.0 - w_prev - w_next
                vol_smooth = w_curr * curr_clean + w_prev * n_prev + w_next * n_next
            else:
                vol_smooth = curr_clean

            # Anisotropic fusion: in flat regions use 100% Rician-debias + NLM clean; on true edges blend coherently
            edge_prot = 0.72 if seq_type in ("T2_FSE", "T2_STAR_GRE") else 0.55
            local_alpha = flat_denoise * (1.0 - edge_prot * edge_protect_factor * edge_mask)
            enhanced = (1.0 - local_alpha) * curr_clean + local_alpha * vol_smooth

            # Coherent Edge Deblurring (computed on the denoised signal curr_clean, NEVER on raw noisy norm_work!)
            if crisp_boost > 0:
                sigma_detail = 0.75 if mode in ("sr_15t", "3t_hd") else 0.85
                # Clamp positive detail tightly (+0.015) and roll off near peak brightness to prevent hyperintense/bleed-like spots
                clean_detail = np.clip(curr_clean - gaussian_filter(curr_clean, sigma=sigma_detail), -0.030, 0.015)
                peak_rolloff = np.clip((1.02 - curr_clean) / 0.25, 0.25, 1.0)
                crisp_gate = edge_mask * peak_rolloff
                crisp_mult = 1.2 if strength <= 0.6 else 1.6 if strength <= 0.85 else 2.0
                enhanced = np.clip(enhanced + (crisp_boost * crisp_mult) * clean_detail * crisp_gate, 0.0, 1.0)

            # T2 Fluid Purity & Whiteness Boost (+24% CSF-only lift above 0.48 so spinal cord stays dark & crisp against bright CSF)
            if seq_type in ("T2_FSE", "T2_STAR_GRE"):
                fluid_mask = np.clip((enhanced - 0.46) / 0.26, 0.0, 1.0) ** 1.1
                enhanced = np.clip(enhanced * (1.0 + 0.26 * fluid_mask), 0.0, 1.0)

            # Calibrated Natural Grain + Structure-Selective Bone/Cartilage Texture (soft tissue stays smooth)
            fast_scan_scale = 0.52 if global_sigma >= 0.024 else 1.0
            u_desc = (description or "").upper()
            is_axial = any(k in u_desc for k in ("AX", "OAX", "TRANS"))
            is_sagittal = any(k in u_desc for k in ("SAG", "OSAG"))
            if mode == "fast_mri":
                grain_ratio = 0.00
            elif mode == "print_master":
                grain_ratio = 0.24 if seq_type == "STIR_FLAIR" else 0.12 if seq_type == "T1_SE" else 0.22 if seq_type == "PD_ORTHO" else 0.04
            elif mode == "3t_hd":
                grain_ratio = 0.34 if seq_type == "STIR_FLAIR" else 0.32 if seq_type == "T1_SE" else 0.08 if seq_type == "PD_ORTHO" else (0.20 if is_axial else 0.28) if seq_type in ("T2_STAR_GRE", "T2_FSE") else 0.25
            elif strength <= 0.6:  # MILD / L1
                if seq_type in ("T2_FSE", "T2_STAR_GRE"):
                    grain_ratio = 0.28
                elif seq_type == "STIR_FLAIR":
                    grain_ratio = 0.27
                elif seq_type == "T1_SE":
                    grain_ratio = 0.26
                elif seq_type == "PD_ORTHO":
                    grain_ratio = 0.28
                else:
                    grain_ratio = 0.22
            elif strength <= 0.85:  # MEDIUM / L2
                base_g = 0.28 if seq_type in ("PD_ORTHO", "STIR_FLAIR") else 0.22 if mode == "rician_nlm" else 0.18 if seq_type in ("T2_FSE", "T2_STAR_GRE", "T1_SE") else 0.14
                grain_ratio = base_g * fast_scan_scale
            else:  # STRONG / L3 (PD cartilage: 36%; T1/STIR: 26%; T2/GRE: 20%; ANGIO_MRV MIP: 6%)
                fast_l3_scale = 0.85 if global_sigma >= 0.035 else 1.0
                base_g = 0.06 if seq_type == "ANGIO_MRV" else 0.34 if seq_type == "STIR_FLAIR" else 0.32 if seq_type == "T1_SE" else 0.08 if seq_type == "PD_ORTHO" else 0.26 if mode == "rician_nlm" else (0.20 if is_axial else 0.28) if seq_type in ("T2_STAR_GRE", "T2_FSE") else 0.25
                grain_ratio = base_g * fast_l3_scale
            texture_residual = norm_work - enhanced
            tissue_mask = np.clip(norm_work / 0.08, 0.0, 1.0)
            if seq_type == "PD_ORTHO":
                # Dual-tissue adaptive grain (#152: تنعم البروتون وتخشن العظم)
                # Bone marrow gets authentic 32% trabecular grain; meniscus/cartilage/ligaments get 6% velvet smooth
                bone_marrow_mask = np.clip((norm_work - 0.20) / 0.12, 0.0, 1.0) * np.clip((0.74 - norm_work) / 0.12, 0.0, 1.0)
                effective_grain = (1.0 - bone_marrow_mask) * 0.06 + bone_marrow_mask * 0.32
                pd_gate = np.clip(edge_mask ** 0.85, 0.08, 1.0) * (1.0 - bone_marrow_mask) + bone_marrow_mask * np.clip(0.55 + 0.45 * edge_mask, 0.55, 1.0)
                enhanced = np.clip(enhanced + (effective_grain * tissue_mask * pd_gate) * texture_residual, 0.0, 1.0)
            else:
                texture_gate = np.clip(0.55 + 0.45 * edge_mask, 0.55, 1.0) if seq_type in ("T2_STAR_GRE", "T2_FSE", "T1_SE") else np.clip(edge_mask ** 0.70, 0.12, 1.0) if seq_type == "STIR_FLAIR" else np.clip(0.35 + 0.65 * edge_mask, 0.35, 1.0)
                enhanced = np.clip(enhanced + (grain_ratio * tissue_mask * texture_gate) * texture_residual, 0.0, 1.0)

            # Display-only Film S-Curve is applied ONLY in print_master mode (never in diagnostic modes!)
            if mode == "print_master":
                k = 4.2 if seq_type in ("STIR_FLAIR", "T1_SE") else 3.5
                x0 = 0.28 if seq_type == "STIR_FLAIR" else 0.36 if seq_type == "T1_SE" else 0.38
                scurve = 1.0 / (1.0 + np.exp(-k * (enhanced - x0)))
                s_min = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
                s_max = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))
                scurve = np.clip((scurve - s_min) / (s_max - s_min), 0.0, 1.0)
                enhanced = 0.55 * scurve + 0.45 * enhanced

            # Clean MIP Vessel Contrast Lift for ANGIO_MRV (keeps thin cortical veins & distal branches crisp)
            if seq_type == "ANGIO_MRV":
                vessel_mask = np.clip((enhanced - 0.18) / 0.25, 0.0, 1.0)
                enhanced = np.clip(enhanced * (1.0 - 0.12 * (1.0 - vessel_mask)) + 0.06 * vessel_mask * (enhanced ** 0.85), 0.0, 1.0)

            # Clinical axial & T1 adjustments requested by technician (#120 & #124)
            if is_axial:
                # Reduce noise & brighten spinal canal / foramina in axial views
                enhanced = np.clip(np.power(enhanced, 0.82) * 1.08, 0.0, 1.0)
            if seq_type == "T1_SE" and mode != "print_master":
                # Boost T1 contrast: rich bright marrow/fat and deep dark CSF/discs
                enhanced = np.clip(np.power(enhanced, 1.10) * 1.08, 0.0, 1.0)
            if seq_type == "PD_ORTHO" and mode != "print_master":
                # Boost PD contrast & smooth cartilage/meniscus tissue (#127)
                enhanced = np.clip(np.power(enhanced, 1.08) * 1.06, 0.0, 1.0)

            # Technician-calibrated Brightness & Contrast adjustment (default 1.0 / 1.0 = no-op)
            try:
                from app.store import setting as _get_setting
                _br = float(np.clip(float(_get_setting("clinic_brightness", "1.0") or 1.0), 0.75, 1.25))
                _ct = float(np.clip(float(_get_setting("clinic_contrast", "1.0") or 1.0), 0.75, 1.35))
            except Exception:
                _br, _ct = 1.0, 1.0
            if abs(_br - 1.0) > 1e-3 or abs(_ct - 1.0) > 1e-3:
                enhanced = np.clip((enhanced - 0.5) * _ct + 0.5 + (_br - 1.0) * 0.5, 0.0, 1.0)

            # Rician Background Air Gate
            bg_threshold = 0.035 if is_axial else (0.14 if seq_type == "ANGIO_MRV" else 0.055 if seq_type == "STIR_FLAIR" else 0.085 if mode == "print_master" else 0.070)
            bg_mask = np.clip((bg_threshold - norm_work) / bg_threshold, 0.0, 1.0)
            enhanced = enhanced * (1.0 - bg_suppress_factor * bg_mask)

            overflow = np.maximum(vol_f[i] - vol_high, 0.0)
            restored = enhanced * vol_span + vol_low + overflow

            # Auto-upscale all clinical scans to 1024x1024 + Post-Upscale Micro-Acutance to remove bicubic softness
            h_s, w_s = slice_2d.shape
            if 200 <= h_s <= 512 and 200 <= w_s <= 512:
                from scipy.ndimage import zoom as _zoom
                scale_factor = 4.0 if (h_s <= 256 and w_s <= 256) else 2.0
                restored = _zoom(restored, scale_factor, order=3)
                blur_hi = gaussian_filter(restored, sigma=1.1)
                detail_hi = restored - blur_hi
                limit_hi = max(vol_span * 0.030, 1e-4)
                restored = restored + 0.48 * np.clip(detail_hi, -limit_hi, limit_hi)
            if np.issubdtype(slice_2d.dtype, np.integer):
                info = np.iinfo(slice_2d.dtype)
                results.append(np.rint(np.clip(restored, info.min, info.max)).astype(slice_2d.dtype))
            else:
                results.append(restored.astype(slice_2d.dtype))

        return np.stack(results)

    def enhance_slice(self, slice_2d: np.ndarray, strength: float = 1.0, mode: str = "print_master",
                      upscale: bool = False, description: str = "") -> np.ndarray:
        vol = slice_2d[np.newaxis, :, :]
        enhanced_vol = self.enhance_volume(vol, strength=strength, mode=mode, description=description)
        out = enhanced_vol[0]
        if upscale and out.shape[0] <= 256 and out.shape[1] <= 256:
            from scipy.ndimage import zoom
            out = zoom(out, 2.0, order=3)
        return out


ai_engine = MRIAIEngine()
