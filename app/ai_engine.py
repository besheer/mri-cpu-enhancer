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

    @staticmethod
    def _estimate_kspace_sigma(img01: np.ndarray) -> float:
        """Multi-stride Haar diagonal MAD across strides (1, 2, 3) to capture
        zero-filled 0.3T k-space correlated Rician speckle accurately."""
        a = img01.astype(np.float64, copy=False)
        sigmas = []
        for stride in (1, 2, 3):
            sub = a[::stride, ::stride]
            h = (sub.shape[0] // 2) * 2
            w = (sub.shape[1] // 2) * 2
            if h < 8 or w < 8:
                continue
            b = sub[:h, :w].reshape(h // 2, 2, w // 2, 2)
            valid = (b[:, 0, :, 0] > 0.005) & (b[:, 1, :, 1] > 0.005)
            hh = (b[:, 0, :, 0] - b[:, 0, :, 1] - b[:, 1, :, 0] + b[:, 1, :, 1]) / 2.0
            vals = hh[valid]
            if vals.size > 32:
                sigmas.append(float(np.median(np.abs(vals)) / 0.6745))
        return float(np.clip(max(sigmas) if sigmas else 0.035, 0.018, 0.14))

    def _enhance_stir(self, volume: np.ndarray, strength: float = 1.0, mode: str = "spine_pro") -> np.ndarray:
        """True 1.5T-Grade Clinical STIR / Fat-Suppressed Engine:
        - Multi-scale k-space Rician despeckling in flat muscle/fat (eliminates 3px wormy 0.3T clumps).
        - Per-border air-floor detection & Rician magnitude debiasing (pure black air even on Shoulder/Pelvis).
        - Dark charcoal muscle vs bright cancellous bone marrow contrast + sharp 1px dark cortical rims.
        - Preserves real trabecular bone architecture + 1024x1024 1px 1.5T MRI micro-grain (zero plastic blur).
        - Preserves 100% unclipped joint effusion, CSF, and bone marrow edema dynamic range.
        """
        from scipy.ndimage import zoom as _zoom
        vol_f = volume.astype(np.float32)
        pos = vol_f[vol_f > 0]
        if pos.size < 100:
            return volume.copy()
        ref_scale = max(float(np.percentile(pos, 99.4)), 1.0)

        num_slices = volume.shape[0]
        mid_norm = np.clip(vol_f[num_slices // 2] / ref_scale, 0.0, 1.0)
        k_sigma = self._estimate_kspace_sigma(mid_norm)
        high_noise = float(np.clip((k_sigma - 0.035) / 0.055, 0.0, 1.0))

        has_onnx = self.is_available
        input_name = self.session.get_inputs()[0].name if has_onnx else ""

        if mode == "fast_mri":
            real_tex_soft, real_tex_bone, hi_grain, gamma_c, local_c, crisp_w = 0.02, 0.18, 0.022, 1.22, 0.20, 0.64
        elif mode == "3t_hd":
            real_tex_soft, real_tex_bone, hi_grain, gamma_c, local_c, crisp_w = 0.06, 0.35, 0.044, 1.28, 0.28, 0.88
        elif mode == "enh6_ultra":
            real_tex_soft, real_tex_bone, hi_grain, gamma_c, local_c, crisp_w = 0.04, 0.26, 0.030, 1.34, 0.32, 0.94
        else:  # spine_pro / default clinical 1.5T STIR
            real_tex_soft, real_tex_bone, hi_grain, gamma_c, local_c, crisp_w = 0.04, 0.28, 0.034, 1.26, 0.26, 0.80

        eff_tex_soft = real_tex_soft * (1.0 - 0.75 * high_noise)

        norm_slices = []
        clean_slices = []
        edge_gates = []
        body_gates = []

        for i in range(num_slices):
            src = vol_f[i]
            norm01 = np.ascontiguousarray(np.clip(src / ref_scale, 0.0, 1.0), dtype=np.float32)
            fov_mask = (src > 0).astype(np.float32)
            norm_slices.append(norm01)

            g_macro = gaussian_filter(norm01, sigma=4.2)
            strips = [
                g_macro[4:22, :][src[4:22, :] > 0],
                g_macro[-22:-4, :][src[-22:-4, :] > 0],
                g_macro[:, 4:22][src[:, 4:22] > 0],
                g_macro[:, -22:-4][src[:, -22:-4] > 0],
                g_macro[4:36, 4:36][src[4:36, 4:36] > 0],
                g_macro[4:36, -36:-4][src[4:36, -36:-4] > 0],
            ]
            air_candidates = []
            for st in strips:
                if st.size > 30:
                    med_s = float(np.median(st))
                    std_s = float(np.std(st))
                    if med_s < min(3.6 * k_sigma, 0.36) and std_s < max(0.85 * k_sigma, 0.055):
                        air_candidates.append(float(np.percentile(st, 82.0)))
            air_floor = max(air_candidates) if air_candidates else (0.95 * k_sigma)

            fine_clean, _ = structure_tensor_tangent_restore(norm01, strength=0.90, sigma_est=0.80 * k_sigma)
            if has_onnx and k_sigma < 0.055:
                onnx_out = np.clip(self.session.run(None, {input_name: norm01[np.newaxis, np.newaxis, :, :]})[0][0, 0].astype(np.float32), 0.0, 1.0)
                fine_clean = 0.78 * fine_clean + 0.22 * onnx_out

            if k_sigma > 0.038:
                half = _zoom(fine_clean, 0.5, order=1)
                half_nlm = non_local_means(
                    half, sigma=k_sigma * 0.88, patch_radius=2, search_radius=3,
                    h_factor=1.15 + 0.22 * high_noise, guide=gaussian_filter(half, 0.85)
                )
                coarse_clean = np.clip(_zoom(half_nlm.astype(np.float32), (norm01.shape[0] / half_nlm.shape[0], norm01.shape[1] / half_nlm.shape[1]), order=3), 0.0, 1.0)
            else:
                coarse_clean = fine_clean

            g_sm = gaussian_filter(fine_clean, sigma=1.30)
            grad = np.hypot(sobel(g_sm, axis=1), sobel(g_sm, axis=0))
            p78 = max(float(np.percentile(grad[fov_mask > 0], 78.0)) if np.any(fov_mask > 0) else 1e-3, 1e-4)
            e_gate = np.clip((grad - 0.28 * p78) / max(0.85 * p78, 1e-4), 0.0, 1.0)
            fluid_spot = np.clip((fine_clean - 0.68) / 0.16, 0.0, 1.0)
            struct_w = np.clip(np.maximum(e_gate, fluid_spot), 0.0, 1.0)

            coarse_mix = 0.32 + 0.42 * high_noise
            fused = (1.0 - struct_w) * ((1.0 - coarse_mix) * fine_clean + coarse_mix * coarse_clean) + struct_w * fine_clean

            deb = air_floor * 0.96
            debiased = np.sqrt(np.clip(fused * fused - deb * deb, 0.0, None))
            dark_gate = np.clip((0.48 - fused) / 0.30, 0.0, 1.0)
            clean_deb = fused * (1.0 - 0.80 * dark_gate) + debiased * (0.80 * dark_gate)

            b_raw = np.clip((g_macro - air_floor * 1.12) / max(0.55 * k_sigma, 0.035), 0.0, 1.0) * fov_mask
            b_raw = np.maximum(b_raw, e_gate * np.clip((g_macro - air_floor * 0.90) / 0.05, 0.0, 1.0))
            b_gate = gaussian_filter(b_raw, sigma=1.6) * fov_mask

            clean_slices.append(clean_deb.astype(np.float32))
            edge_gates.append(e_gate.astype(np.float32))
            body_gates.append(b_gate.astype(np.float32))

        results = []
        for i in range(num_slices):
            slice_2d = volume[i]
            src = vol_f[i]
            norm01 = norm_slices[i]
            curr = clean_slices[i]
            e_gate = edge_gates[i]
            b_gate = body_gates[i]

            if num_slices >= 3:
                pi, ni = max(0, i - 1), min(num_slices - 1, i + 1)
                flat = (1.0 - e_gate) ** 2.0
                w_z = 0.08 + 0.06 * high_noise
                wp = w_z * flat * np.exp(-((np.abs(curr - clean_slices[pi]) / max(1.25 * k_sigma, 0.04)) ** 2))
                wn = w_z * flat * np.exp(-((np.abs(curr - clean_slices[ni]) / max(1.25 * k_sigma, 0.04)) ** 2))
                base = (1.0 - wp - wn) * curr + wp * clean_slices[pi] + wn * clean_slices[ni]
            else:
                base = curr

            illum = gaussian_filter(base, sigma=min(base.shape) / 5.0)
            inside = b_gate > 0.65
            if np.sum(inside) > 400:
                med_il = float(np.median(illum[inside]))
                coil_eq = np.clip(med_il / np.maximum(illum, med_il * 0.52), 0.74, 1.18)
                base = np.clip(base * (1.0 + 0.42 * b_gate * (coil_eq - 1.0)), 0.0, 1.0)

            toned = np.power(np.clip(base, 0.0, 1.0), gamma_c)
            local_band = base - gaussian_filter(base, sigma=6.0)
            cort_band = np.clip(base - gaussian_filter(base, sigma=0.90), -0.055, 0.042)
            toned = np.clip(toned + local_c * b_gate * local_band + crisp_w * cort_band * (0.25 + 0.75 * e_gate), 0.0, 1.0)

            if mode == "enh6_ultra":
                from scipy.ndimage import grey_dilation, grey_erosion, laplace
                lap = laplace(toned)
                shock = np.where(lap < -0.005, grey_dilation(toned, size=(3, 3)), np.where(lap > 0.005, grey_erosion(toned, size=(3, 3)), toned))
                toned = np.clip((1.0 - 0.35 * e_gate) * toned + (0.35 * e_gate) * shock, 0.0, 1.0)

            real_hp = np.clip(norm01 - gaussian_filter(norm01, sigma=0.95), -0.80 * k_sigma, 0.80 * k_sigma)
            bone_or_edge = np.clip(np.maximum((base - 0.34) / 0.20, e_gate), 0.0, 1.0)
            tex_w = eff_tex_soft * (1.0 - bone_or_edge) + real_tex_bone * bone_or_edge
            textured = np.clip(toned + b_gate * tex_w * real_hp, 0.0, 1.0)

            textured = textured * (b_gate ** 1.45)

            super_bright = np.maximum(src - ref_scale, 0.0)
            restored256 = np.maximum(textured * ref_scale + super_bright * b_gate, 0.0)

            h_s, w_s = slice_2d.shape
            if 200 <= h_s <= 512 and 200 <= w_s <= 512:
                scale_factor = 4.0 if (h_s <= 256 and w_s <= 256) else 2.0
                up_cub = _zoom(restored256, scale_factor, order=3)
                up_lin = _zoom(restored256, scale_factor, order=1)
                restored = np.clip(up_cub, np.minimum(up_lin, up_cub) * 0.97, np.maximum(up_lin, up_cub) * 1.03)
                restored = np.maximum(restored, 0.0)

                hi_det = restored - gaussian_filter(restored, sigma=1.10)
                lim = ref_scale * 0.032
                up_egate = _zoom(e_gate, scale_factor, order=1)
                restored = np.maximum(restored + 0.52 * crisp_w * np.clip(hi_det, -lim, lim) * (0.22 + 0.78 * up_egate), 0.0)

                up_bgate = np.clip(_zoom(b_gate, scale_factor, order=1), 0.0, 1.0)
                up_bone = np.clip(_zoom(bone_or_edge, scale_factor, order=1), 0.0, 1.0)
                seed = (int(abs(float(np.sum(src)))) * 2654435761 + i * 97531) & 0xFFFFFFFF
                rng = np.random.default_rng(seed)
                fine_1px = gaussian_filter(rng.standard_normal(restored.shape, dtype=np.float32), sigma=0.42) * 1.50
                grain_amp = hi_grain * (0.50 + 0.85 * up_bone) * ref_scale
                fluid_fade = np.clip((0.92 - restored / ref_scale) / 0.20, 0.30, 1.0)
                restored = np.maximum(restored + (up_bgate ** 1.6) * fluid_fade * grain_amp * fine_1px, 0.0)
            else:
                restored = restored256

            if np.issubdtype(slice_2d.dtype, np.integer):
                info = np.iinfo(slice_2d.dtype)
                results.append(np.rint(np.clip(restored, 0, info.max)).astype(slice_2d.dtype))
            else:
                results.append(restored.astype(slice_2d.dtype))

        return np.stack(results, axis=0)

    def enhance_volume(self, volume: np.ndarray, strength: float = 0.8, mode: str = "standard",
                       description: str = "") -> np.ndarray:
        seq_type = self.detect_sequence_type(description)
        u_desc_init = (description or "").upper()
        if seq_type == "STIR_FLAIR" and "FLAIR" not in u_desc_init:
            return self._enhance_stir(volume, strength=strength, mode=mode)
        is_axial = any(k in u_desc_init for k in ("AX", "OAX", "TRANS"))

        # Calibrated clinical strength mapping (0.5=MILD, 0.8=MEDIUM, 1.0=STRONG)
        if mode == "enh6_ultra":
            flat_denoise = float(np.clip(0.88 + 0.10 * strength, 0.90, 0.98))
            edge_protect_factor = 0.985
            bg_suppress_factor = 0.98
            crisp_boost = 0.065
        elif mode in ("print_master", "3t_hd"):
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
        crisp_boost = 0.036 if mode == "rician_nlm" else (0.054 if seq_type in ("T2_STAR_GRE", "T2_FSE", "T1_SE") else 0.048) if mode in ("print_master", "3t_hd", "sr_15t", "spine_pro", "ge_aw", "fast_mri") else 0.020

        if mode == "rician_nlm":
            edge_protect_factor = 0.98
            flat_denoise = min(flat_denoise, 0.60)
        if seq_type == "ANGIO_MRV":
            edge_protect_factor = 0.99
            bg_suppress_factor = 0.68
            crisp_boost = 0.050
        elif seq_type == "STIR_FLAIR":
            edge_protect_factor = 0.95
            bg_suppress_factor = 0.50
            flat_denoise = 0.92
            crisp_boost = 0.0
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

            if seq_type != "STIR_FLAIR" and mode in ("sr_15t", "ge_aw", "print_master", "3t_hd", "spine_pro", "fast_mri", "enh6_ultra"):
                # True Normalized-Convolution GE PURE / SCIC Coil Equalization (zero black-air dilution at skin borders!)
                fg = (norm > 0.07).astype(np.float32)
                if np.sum(fg) > 200:
                    sig_il = max(12.0, min(norm.shape) / 9.5)
                    num_il = gaussian_filter(norm * fg, sigma=sig_il)
                    den_il = np.maximum(gaussian_filter(fg, sigma=sig_il), 0.08)
                    illum = num_il / den_il
                    ref_illum = float(np.median(illum[fg > 0.5]))
                    # Compress blinding near-coil fat hot-spots (down to 0.56) & gently lift shaded anatomy (up to 1.32)
                    coil_gain = np.clip(ref_illum / np.maximum(illum, ref_illum * 0.45), 0.56, 1.32)
                    gain_weight = np.where(coil_gain < 1.0, 0.72, 0.42 if mode in ("print_master", "3t_hd", "fast_mri", "spine_pro", "ge_aw") else 0.28)
                    fg_soft = np.clip((norm - 0.04) / 0.06, 0.0, 1.0)
                    hi_protect = np.where(coil_gain > 1.0, np.clip((0.88 - norm) / 0.36, 0.0, 1.0), 1.0)
                    eff_gain = 1.0 + fg_soft * gain_weight * hi_protect * (coil_gain - 1.0)
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
                raw_det = curr_clean - gaussian_filter(curr_clean, sigma=sigma_detail)
                # Keep full negative detail (-0.048) inside bright CSF/fat so dark nerve roots & septa stay crisp, while rolling off positive bright spikes!
                pos_rolloff = np.clip((0.95 - curr_clean) / 0.30, 0.15, 1.0)
                clean_detail = np.where(raw_det < 0, np.clip(raw_det, -0.048, 0.0), np.clip(raw_det, 0.0, 0.014) * pos_rolloff)
                crisp_mult = 1.2 if strength <= 0.6 else 1.6 if strength <= 0.85 else 2.0
                enhanced = np.clip(enhanced + (crisp_boost * crisp_mult) * clean_detail * (0.20 + 0.80 * edge_mask), 0.0, 1.0)

            # T2 Fluid Purity & Cord Contrast with Highlight-Preserving Soft Shoulder (zero white burnout!)
            if seq_type in ("T2_FSE", "T2_STAR_GRE"):
                if mode == "enh6_ultra":
                    csf_fluid_mask = np.clip((enhanced - 0.45) / 0.28, 0.0, 1.0) ** 1.3
                    # Soft-shoulder lift: (1 - enhanced) preserves 100% of internal nerve-root & fat-septa detail without clipping
                    enhanced = np.clip(enhanced + 0.45 * csf_fluid_mask * (1.0 - enhanced), 0.0, 1.0)
                else:
                    fluid_mask = np.clip((enhanced - 0.56) / 0.28, 0.0, 1.0) ** 1.35
                    enhanced = np.clip(enhanced + 0.12 * fluid_mask * (1.0 - enhanced), 0.0, 1.0)
                # Cord & cauda equina nerve-root dark delineation inside bright CSF
                cord_mask = np.clip((enhanced - 0.16) / 0.16, 0.0, 1.0) * np.clip((0.62 - enhanced) / 0.18, 0.0, 1.0)
                enhanced = np.clip(enhanced * (1.0 - 0.10 * cord_mask * edge_mask), 0.0, 1.0)

            # ENH-6 Kramer-Bruckner Morphological Shock Filter for Sub-pixel 1px Cortical/Disc Rim Sharpness
            if mode == "enh6_ultra":
                from scipy.ndimage import grey_dilation, grey_erosion, laplace
                lap = laplace(enhanced)
                dil = grey_dilation(enhanced, size=(3, 3))
                ero = grey_erosion(enhanced, size=(3, 3))
                shock = np.where(lap < -0.005, dil, np.where(lap > 0.005, ero, enhanced))
                enhanced = np.clip((1.0 - 0.42 * edge_mask) * enhanced + (0.42 * edge_mask) * shock, 0.0, 1.0)

            # Calibrated Natural Grain + Structure-Selective Bone/Cartilage Texture (soft tissue stays smooth)
            fast_scan_scale = 0.52 if global_sigma >= 0.024 else 1.0
            u_desc = (description or "").upper()
            is_axial = any(k in u_desc for k in ("AX", "OAX", "TRANS"))
            is_sagittal = any(k in u_desc for k in ("SAG", "OSAG"))
            if mode in ("fast_mri", "enh6_ultra"):
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
                bone_marrow_mask = np.clip((norm_work - 0.20) / 0.12, 0.0, 1.0) * np.clip((0.74 - norm_work) / 0.12, 0.0, 1.0)
                effective_grain = (1.0 - bone_marrow_mask) * 0.06 + bone_marrow_mask * 0.32
                pd_gate = np.clip(edge_mask ** 0.85, 0.08, 1.0) * (1.0 - bone_marrow_mask) + bone_marrow_mask * np.clip(0.55 + 0.45 * edge_mask, 0.55, 1.0)
                enhanced = np.clip(enhanced + (effective_grain * tissue_mask * pd_gate) * texture_residual, 0.0, 1.0)
            elif seq_type in ("T2_FSE", "T2_STAR_GRE"):
                # Technician feedback 2026-11-25: عايز تى تو خشنه شويه العظم (Bone marrow 36% grain, CSF/soft tissue smooth)
                bone_marrow_mask = np.clip((norm_work - 0.18) / 0.14, 0.0, 1.0) * np.clip((0.68 - norm_work) / 0.14, 0.0, 1.0)
                effective_grain = (1.0 - bone_marrow_mask) * 0.10 + bone_marrow_mask * 0.36
                t2_gate = np.clip(edge_mask ** 0.85, 0.10, 1.0) * (1.0 - bone_marrow_mask) + bone_marrow_mask * np.clip(0.55 + 0.45 * edge_mask, 0.55, 1.0)
                enhanced = np.clip(enhanced + (effective_grain * tissue_mask * t2_gate) * texture_residual, 0.0, 1.0)
            elif seq_type == "STIR_FLAIR":
                # Technician feedback 2026-11-25: Stair عايز ننعم السوفت ونعمل كونترست العظم
                # Soft tissue gets velvet smooth (0.04 grain), bone marrow edema gets authentic high-contrast definition
                bone_edema_mask = np.clip((norm_work - 0.20) / 0.16, 0.0, 1.0) * np.clip((0.85 - norm_work) / 0.16, 0.0, 1.0)
                effective_grain = (1.0 - bone_edema_mask) * 0.04 + bone_edema_mask * 0.26
                stir_gate = np.clip(edge_mask ** 0.70, 0.08, 1.0)
                enhanced = np.clip(enhanced + (effective_grain * tissue_mask * stir_gate) * texture_residual, 0.0, 1.0)
                # Boost bone marrow contrast in STIR
                enhanced = np.clip(enhanced * (1.0 + 0.18 * bone_edema_mask), 0.0, 1.0)
            else:
                texture_gate = np.clip(0.55 + 0.45 * edge_mask, 0.55, 1.0) if seq_type == "T1_SE" else np.clip(0.35 + 0.65 * edge_mask, 0.35, 1.0)
                enhanced = np.clip(enhanced + (grain_ratio * tissue_mask * texture_gate) * texture_residual, 0.0, 1.0)

            # ENH-6 3T High-Field Dynamic S-Curve (Rich marrow + Brilliant CSF + Pure black air)
            if mode == "enh6_ultra":
                k_3t = 4.2 if seq_type in ("T2_FSE", "T2_STAR_GRE") else 3.8
                x0_3t = 0.35 if seq_type in ("T2_FSE", "T2_STAR_GRE") else 0.38
                sc_raw = 1.0 / (1.0 + np.exp(-k_3t * (enhanced - x0_3t)))
                sc_0 = 1.0 / (1.0 + np.exp(-k_3t * (0.0 - x0_3t)))
                sc_1 = 1.0 / (1.0 + np.exp(-k_3t * (1.0 - x0_3t)))
                sc_norm = np.clip((sc_raw - sc_0) / (sc_1 - sc_0), 0.0, 1.0)
                enhanced = 0.38 * sc_norm + 0.62 * enhanced

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

            # Highlight-safe clinical tone curves (zero white clipping — preserves all fat septa & nerve roots!)
            if is_axial:
                # Lift ONLY dark canal/foraminal shadows (<0.52), gently compress glaring fat highlights (>0.75)
                shadow_lift = 0.045 * np.sin(np.pi * np.clip(enhanced / 0.52, 0.0, 1.0)) * (enhanced < 0.52)
                hi_tame = -0.045 * ((np.clip((enhanced - 0.72) / 0.28, 0.0, 1.0)) ** 1.4)
                enhanced = np.clip(enhanced + shadow_lift + hi_tame, 0.0, 1.0)
            if seq_type == "T1_SE" and mode != "print_master":
                # Anchored contrast (0->0, 1->1) with soft highlight shoulder so subcutaneous coil fat never burns out
                enhanced = np.clip(np.power(enhanced, 1.08), 0.0, 1.0)
                enhanced = enhanced - 0.05 * ((np.clip((enhanced - 0.75) / 0.25, 0.0, 1.0)) ** 1.5)
            if seq_type == "PD_ORTHO" and mode != "print_master":
                enhanced = np.clip(np.power(enhanced, 1.06), 0.0, 1.0)

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
            restored = np.maximum(enhanced * vol_span + vol_low + overflow, 0.0)

            # Auto-upscale all clinical scans to 1024x1024 + Post-Upscale Micro-Acutance to remove bicubic softness
            h_s, w_s = slice_2d.shape
            if 200 <= h_s <= 512 and 200 <= w_s <= 512:
                from scipy.ndimage import zoom as _zoom
                scale_factor = 1.0 if seq_type == "STIR_FLAIR" else (4.0 if (h_s <= 256 and w_s <= 256) else 2.0)
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
