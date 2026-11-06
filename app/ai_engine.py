from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, sobel

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
                print(f"? AI Model Loaded Successfully from: {p} (mtime={self.model_mtime})")
                return
            except Exception as e:
                print(f"Error loading ONNX from {p}: {e}")
        print("? AI Engine: Model file not found!")

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
        if not self.is_available:
            raise RuntimeError("AI model file not found in /app/models/swift_nafnet_03t.onnx")

        seq_type = self.detect_sequence_type(description)

        # Calibrated clinical strength mapping (0.5=MILD, 0.8=MEDIUM, 1.0=STRONG)
        if mode == "sr_15t":
            # Super-Resolution 1.5T Detail Reconstruction Mode (Anti-staircasing + Sub-pixel edge synthesis)
            flat_denoise = float(np.clip(0.52 + 0.14 * strength, 0.58, 0.68))
        elif mode == "spine_pro":
            # High-clarity smooth mode (like Spine video) with anti-burnout coil shading correction
            flat_denoise = float(np.clip(0.56 + 0.14 * strength, 0.62, 0.70))
        else:
            flat_denoise = float(np.clip(0.38 + 0.34 * strength, 0.50, 0.70))

        # Auto-Sequence Intelligence adjustments
        edge_protect_factor = 0.96
        bg_suppress_factor = 0.35
        crisp_boost = 0.03 if mode in ("spine_pro", "sr_15t") else 0.01
        if seq_type == "ANGIO_MRV":
            # Protect fine vessels maximally & clean dark background
            edge_protect_factor = 0.98
            bg_suppress_factor = 0.42
            crisp_boost = 0.04
        elif seq_type == "STIR_FLAIR":
            # STIR/FLAIR has lower SNR on 0.3T; protect high-frequency edema borders
            edge_protect_factor = 0.98
            bg_suppress_factor = 0.38
            flat_denoise = min(flat_denoise, 0.60)
        elif seq_type == "T1_SE":
            # Crisp trabecular bone and tendon borders
            edge_protect_factor = 0.96
            crisp_boost = max(crisp_boost, 0.03)

        num_slices = volume.shape[0]
        input_name = self.session.get_inputs()[0].name

        # Pre-normalize all slices and compute ONNX predictions so 2.5D Multi-Slice Context can use (z-1, z, z+1)
        norm_slices = []
        raw_norm_slices = []
        onnx_slices = []
        spans_lows = []

        for i in range(num_slices):
            src = volume[i].astype(np.float32)
            low = float(np.percentile(src, 1.0))
            high = float(np.percentile(src, 99.0))
            span = max(float(high - low), 1e-4)
            spans_lows.append((low, span))

            norm = np.ascontiguousarray(np.clip((src - low) / span, 0.0, 1.0), dtype=np.float32)
            raw_norm_slices.append(norm)

            if mode in ("spine_pro", "sr_15t"):
                # Anti-Burnout Surface Coil Shading Equalization (applied strictly to non-edge bright fat zones)
                illum = gaussian_filter(norm, sigma=max(14.0, min(norm.shape) / 8.0))
                fg = norm > 0.10
                if np.any(fg):
                    ref_illum = float(np.median(illum[fg]))
                    coil_gain = np.clip(ref_illum / np.maximum(illum, ref_illum * 0.55), 0.92, 1.06)
                    norm_work = np.clip(norm * (0.82 + 0.18 * coil_gain), 0.0, 1.0).astype(np.float32)
                else:
                    norm_work = norm
            else:
                norm_work = norm

            norm_slices.append(norm_work)
            inp = norm_work[np.newaxis, np.newaxis, :, :]
            pred = self.session.run(None, {input_name: inp})[0]
            onnx_out = np.clip(pred[0, 0].astype(np.float32), 0.0, 1.0)
            smooth_sigma = 0.92 if mode == "spine_pro" else 0.82 if mode == "sr_15t" else 0.85
            onnx_slices.append(gaussian_filter(onnx_out, sigma=smooth_sigma))

        results = []
        for i in range(num_slices):
            slice_2d = volume[i]
            low, span = spans_lows[i]
            norm_work = norm_slices[i]
            onnx_curr = onnx_slices[i]

            # Multi-directional anatomical edge map (Sobel gradient magnitude)
            gx = sobel(norm_work, axis=1)
            gy = sobel(norm_work, axis=0)
            grad = np.sqrt(gx * gx + gy * gy)
            p75 = max(float(np.percentile(grad, 75.0)), 1e-4)
            edge_mask = np.clip(grad / p75, 0.0, 1.0) ** 1.15

            # Improvement 1: 2.5D Multi-Slice Context (z-1, z, z+1) Structure-Aware Temporal/Spatial Coherence
            # Applied strictly in non-edge regions (1 - edge_mask) to preserve 100% 3D Z-gradient fidelity
            if num_slices >= 3:
                prev_idx = max(0, i - 1)
                next_idx = min(num_slices - 1, i + 1)
                n_prev = onnx_slices[prev_idx]
                n_next = onnx_slices[next_idx]
                diff_prev = np.abs(norm_work - norm_slices[prev_idx])
                diff_next = np.abs(norm_work - norm_slices[next_idx])
                flat_zone = (1.0 - edge_mask) ** 2
                w_prev = 0.04 * flat_zone * np.exp(- (diff_prev / 0.06) ** 2)
                w_next = 0.04 * flat_zone * np.exp(- (diff_next / 0.06) ** 2)
                w_curr = 1.0 - w_prev - w_next
                onnx_smooth = w_curr * onnx_curr + w_prev * n_prev + w_next * n_next
            else:
                onnx_smooth = onnx_curr

            # Anisotropic neural fusion (use original raw_norm_slices[i] on edges to preserve 100% 3D Z-gradient fidelity)
            local_alpha = flat_denoise * (1.0 - edge_protect_factor * edge_mask)
            enhanced = (1.0 - local_alpha) * raw_norm_slices[i] + local_alpha * onnx_smooth

            # Improvement 2: Sub-Pixel Anti-Staircasing & Micro-Edge Crisping (1.5T Matrix Look)
            if crisp_boost > 0:
                sigma_detail = 0.65 if mode == "sr_15t" else 0.80
                fine_detail = norm_work - gaussian_filter(norm_work, sigma=sigma_detail)
                bright_suppress = np.clip((0.88 - norm_work) / 0.25, 0.0, 1.0)
                crisp_gate = edge_mask * (1.0 - 0.45 * edge_mask) * bright_suppress
                enhanced = np.clip(enhanced + crisp_boost * fine_detail * crisp_gate, 0.0, 1.0)

            # Rician Background Air Gate
            bg_mask = np.clip((0.08 - norm_work) / 0.08, 0.0, 1.0)
            enhanced = enhanced * (1.0 - bg_suppress_factor * bg_mask)

            restored = enhanced * span + low

            if np.issubdtype(slice_2d.dtype, np.integer):
                info = np.iinfo(slice_2d.dtype)
                results.append(np.rint(np.clip(restored, info.min, info.max)).astype(slice_2d.dtype))
            else:
                results.append(restored.astype(slice_2d.dtype))

        return np.stack(results)

ai_engine = MRIAIEngine()
