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
        if any(k in u for k in ("MRV", "MRA", "TOF", "ANGIO", "VENO")):
            return "ANGIO_MRV"
        if any(k in u for k in ("STIR", "FLAIR", "FAT", "FS", "SPAIR")):
            return "STIR_FLAIR"
        if "T2*" in u or "GRE" in u or "FE" in u:
            return "T2_STAR_GRE"
        if "T2" in u:
            return "T2_FSE"
        if "T1" in u:
            return "T1_SE"
        if "PD" in u:
            return "PD_ORTHO"
        return "GENERAL_MR"

    def enhance_volume(self, volume: np.ndarray, strength: float = 0.8, mode: str = "standard",
                       description: str = "") -> np.ndarray:
        seq_type = self.detect_sequence_type(description)

        # Calibrated clinical strength mapping (0.5=MILD, 0.8=MEDIUM, 1.0=STRONG)
        if mode in ("print_master", "3t_hd"):
            flat_denoise = float(np.clip(0.76 + 0.16 * strength, 0.78, 0.94))
        elif mode == "ge_aw":
            flat_denoise = float(np.clip(0.56 + 0.16 * strength, 0.62, 0.74))
        elif mode == "sr_15t":
            flat_denoise = float(np.clip(0.54 + 0.16 * strength, 0.60, 0.72))
        elif mode == "spine_pro":
            flat_denoise = float(np.clip(0.58 + 0.16 * strength, 0.64, 0.76))
        else:
            flat_denoise = float(np.clip(0.45 + 0.28 * strength, 0.52, 0.74))

        edge_protect_factor = 0.94
        bg_suppress_factor = 0.96 if mode == "print_master" else 0.45 if mode in ("ge_aw", "3t_hd") else 0.35
        crisp_boost = 0.05 if mode in ("print_master", "3t_hd", "sr_15t") else 0.03 if mode in ("spine_pro", "ge_aw") else 0.015

        if seq_type == "ANGIO_MRV":
            edge_protect_factor = 0.98
            bg_suppress_factor = 0.45
            crisp_boost = 0.04
        elif seq_type == "STIR_FLAIR":
            edge_protect_factor = 0.97
            bg_suppress_factor = 0.38
            flat_denoise = min(flat_denoise, 0.64)
        elif seq_type == "T1_SE":
            edge_protect_factor = 0.95
            crisp_boost = max(crisp_boost, 0.03)

        num_slices = volume.shape[0]
        has_onnx = self.is_available
        input_name = self.session.get_inputs()[0].name if has_onnx else ""

        # Estimate one global Rician noise sigma on the middle slice so inter-slice appearance is consistent
        mid_idx = num_slices // 2
        mid_src = volume[mid_idx].astype(np.float32)
        mid_low, mid_high = np.percentile(mid_src, (1.0, 99.0))
        mid_span = max(float(mid_high - mid_low), 1e-4)
        mid_norm = np.clip((mid_src - mid_low) / mid_span, 0.0, 1.0)
        global_sigma = estimate_sigma(mid_norm)

        norm_slices = []
        raw_norm_slices = []
        base_denoised_slices = []
        spans_lows = []

        for i in range(num_slices):
            src = volume[i].astype(np.float32)
            low = float(np.percentile(src, 1.0))
            high = float(np.percentile(src, 99.0))
            span = max(float(high - low), 1e-4)
            spans_lows.append((low, span))

            norm = np.ascontiguousarray(np.clip((src - low) / span, 0.0, 1.0), dtype=np.float32)
            raw_norm_slices.append(norm)

            if mode in ("spine_pro", "sr_15t", "ge_aw", "print_master", "3t_hd"):
                # Anti-Burnout Surface Coil Shading Equalization (GE PURE / SCIC homomorphic curve)
                illum = gaussian_filter(norm, sigma=max(14.0, min(norm.shape) / 8.0))
                fg = norm > 0.10
                if np.any(fg):
                    ref_illum = float(np.median(illum[fg]))
                    coil_gain = np.clip(ref_illum / np.maximum(illum, ref_illum * 0.55), 0.90, 1.08)
                    gain_weight = 0.24 if mode in ("print_master", "3t_hd") else 0.20 if mode == "ge_aw" else 0.18
                    norm_work = np.clip(norm * ((1.0 - gain_weight) + gain_weight * coil_gain), 0.0, 1.0).astype(np.float32)
                else:
                    norm_work = norm
            else:
                norm_work = norm

            norm_slices.append(norm_work)

            # Physics-safe Rician-VST + Structure-Tensor Tangent-Guided Restoration
            phys_clean, _ = structure_tensor_tangent_restore(
                norm_work, strength=strength, sigma_est=global_sigma
            )

            if has_onnx:
                inp = norm_work[np.newaxis, np.newaxis, :, :]
                pred = self.session.run(None, {input_name: inp})[0]
                onnx_out = np.clip(pred[0, 0].astype(np.float32), 0.0, 1.0)
                smooth_sigma = 0.90 if mode == "spine_pro" else 0.85
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
            low, span = spans_lows[i]
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
            local_alpha = flat_denoise * (1.0 - 0.55 * edge_protect_factor * edge_mask)
            enhanced = (1.0 - local_alpha) * curr_clean + local_alpha * vol_smooth

            # Coherent Edge Deblurring (computed on the denoised signal curr_clean, NEVER on raw noisy norm_work!)
            if crisp_boost > 0:
                sigma_detail = 0.75 if mode in ("sr_15t", "3t_hd") else 0.90
                clean_detail = np.clip(curr_clean - gaussian_filter(curr_clean, sigma=sigma_detail), -0.05, 0.05)
                bright_suppress = np.clip((0.90 - curr_clean) / 0.22, 0.0, 1.0)
                crisp_gate = edge_mask * bright_suppress
                enhanced = np.clip(enhanced + (crisp_boost * 2.5) * clean_detail * crisp_gate, 0.0, 1.0)

            # Display-only Film S-Curve is applied ONLY in print_master mode (never in diagnostic modes!)
            if mode == "print_master":
                k = 3.5
                x0 = 0.38
                scurve = 1.0 / (1.0 + np.exp(-k * (enhanced - x0)))
                s_min = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
                s_max = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))
                scurve = np.clip((scurve - s_min) / (s_max - s_min), 0.0, 1.0)
                enhanced = 0.55 * scurve + 0.45 * enhanced

            # Rician Background Air Gate
            bg_threshold = 0.09 if mode == "print_master" else 0.075
            bg_mask = np.clip((bg_threshold - norm_work) / bg_threshold, 0.0, 1.0)
            enhanced = enhanced * (1.0 - bg_suppress_factor * bg_mask)

            restored = enhanced * span + low

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
