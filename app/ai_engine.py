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

    def _enhance_axial(self, volume: np.ndarray, strength: float = 1.0, mode: str = "spine_pro",
                       seq_type: str = "T2_FSE") -> np.ndarray:
        """True 1.5T/3T-Grade Clinical Axial MRI Engine v3 (Spine, Knee/Ortho, Neuro):
        - Tissue-selective dual-scale Rician NLM: silky smooth subcutaneous/retroperitoneal fat & CSF,
          while preserving real trabecular bone marrow architecture & white intramuscular fascial striations.
        - Adaptive p99 foreground tissue normalization + highlight-safe CSF shoulder so Axial Spine & Knee
          open bright, high-contrast, and punchy on GE AW native Window/Level (never dark/muddy gray).
        - Zero 256x256 stair-stepping; smooth 1024x1024 anti-ringing upscale + 1024px cortical/nerve-root
          acutance + authentic 1.5T 1px cancellous bone micro-grain.
        """
        from scipy.ndimage import zoom as _zoom

        vol_f = volume.astype(np.float32)
        num_slices = vol_f.shape[0]

        if mode == "fast_mri":
            fat_denoise = 0.92
            bone_denoise = 0.68
            scurve_mix = 0.30
            hi_crisp = 0.85
            bone_tex_256 = 0.16
            bone_grain_1k = 0.0080
        elif mode == "enh6_ultra":
            fat_denoise = 0.90
            bone_denoise = 0.58
            scurve_mix = 0.40
            hi_crisp = 1.18
            bone_tex_256 = 0.24
            bone_grain_1k = 0.0150
        elif mode == "3t_hd":
            fat_denoise = 0.86
            bone_denoise = 0.52
            scurve_mix = 0.36
            hi_crisp = 1.12
            bone_tex_256 = 0.30
            bone_grain_1k = 0.0195
        elif mode == "print_master":
            fat_denoise = 0.90
            bone_denoise = 0.62
            scurve_mix = 0.42
            hi_crisp = 1.00
            bone_tex_256 = 0.20
            bone_grain_1k = 0.0110
        else:  # spine_pro / standard / ge_aw
            fat_denoise = 0.88
            bone_denoise = 0.56
            scurve_mix = 0.35
            hi_crisp = 1.06
            bone_tex_256 = 0.28
            bone_grain_1k = 0.0170

        v_pos = vol_f[vol_f > 0]
        vol_p99 = float(np.percentile(v_pos, 99.0)) if v_pos.size > 100 else float(np.max(vol_f))
        vol_p999 = float(np.percentile(v_pos, 99.9)) if v_pos.size > 100 else float(np.max(vol_f))

        results = []
        for i in range(num_slices):
            slice_2d = volume[i]
            s_raw = np.maximum(vol_f[i], 0.0)
            h_s, w_s = s_raw.shape

            s_pos = s_raw[s_raw > 0]
            s_p99 = float(np.percentile(s_pos, 99.0)) if s_pos.size > 64 else vol_p99
            s_p999 = float(np.percentile(s_pos, 99.9)) if s_pos.size > 64 else vol_p999
            # Tissue scale anchored at p99.2 so 99% of anatomy (fat/marrow/muscle) fills dynamic range cleanly,
            # while super-bright CSF (>s_scale) is preserved via soft shoulder up to s_peak!
            s_scale = max(0.60 * vol_p99 + 0.40 * s_p99, 1.0)
            s_peak = max(0.60 * vol_p999 + 0.40 * s_p999, s_scale * 1.05)

            # Soft-shoulder normalization: maps [0..s_scale] -> [0..0.88] and [s_scale..s_peak] -> [0.88..1.0]
            base_n = np.clip(s_raw / s_scale, 0.0, 1.0) * 0.88
            hi_excess = np.clip((s_raw - s_scale) / max(s_peak - s_scale, 1.0), 0.0, 1.5)
            norm01 = np.clip(base_n + 0.12 * (hi_excess / (1.0 + 0.5 * hi_excess)), 0.0, 1.0).astype(np.float32)

            # 1. Smooth per-border Rician air-floor debiasing (zero 256px stair-stepping!)
            bh = max(4, h_s // 24)
            bw = max(4, w_s // 24)
            strips = [norm01[:bh, :].ravel(), norm01[-bh:, :].ravel(), norm01[:, :bw].ravel(), norm01[:, -bw:].ravel()]
            strip_p75 = [float(np.percentile(st, 75.0)) for st in strips if st.size > 0]
            air_floor = float(np.clip(min(strip_p75) if strip_p75 else 0.018, 0.005, 0.060))
            k_sigma = self._estimate_kspace_sigma(norm01)

            debias = np.sqrt(np.maximum(norm01 * norm01 - (1.25 * air_floor) ** 2, 0.0))
            air_gate = np.clip((2.0 * air_floor - norm01) / max(1.5 * air_floor, 1e-4), 0.0, 1.0) ** 1.5
            work = norm01 * (1.0 - 0.82 * air_gate) + debias * (0.82 * air_gate)

            # 2. Gentle macro illumination balance (lifts shaded Axial flanks without flattening muscle vs fat!)
            fg = np.clip(gaussian_filter((work > max(air_floor * 1.85, 0.042)).astype(np.float32), sigma=1.4), 0.0, 1.0)
            if float(np.sum(fg > 0.5)) > 250:
                sig_macro = max(38.0, min(h_s, w_s) / 3.6)
                num_il = gaussian_filter(work * fg, sigma=sig_macro)
                den_il = np.maximum(gaussian_filter(fg, sigma=sig_macro), 0.10)
                illum = num_il / den_il
                ref_il = max(float(np.percentile(illum[fg > 0.5], 68.0)), 0.08)
                c_gain = np.clip(ref_il / np.maximum(illum, ref_il * 0.48), 0.90, 1.25)
                hi_guard = np.where(c_gain > 1.0, np.clip((0.90 - work) / 0.30, 0.15, 1.0), 0.40)
                work = np.clip(work * (1.0 + fg * 0.65 * hi_guard * (c_gain - 1.0)), 0.0, 1.0).astype(np.float32)

            # 3. Tissue-Selective Dual-Scale Rician NLM:
            #    - Fine 3x3 NLM preserves real trabecular marrow & muscle fascicles
            #    - Coarse half-scale NLM dissolves 3px k-space blotches ONLY in homogeneous fat & dark haze!
            vst_w = rician_forward_vst(work, k_sigma * 0.55, debias_factor=0.55)
            w_in = (0.65 * work + 0.35 * vst_w).astype(np.float32)

            nlm_fine = non_local_means(
                w_in,
                sigma=max(k_sigma, 0.018),
                patch_radius=1,
                search_radius=2,
                h_factor=0.80 + 0.18 * strength,
                guide=gaussian_filter(w_in, sigma=0.60),
            ).astype(np.float32)

            w_half = _zoom(nlm_fine, 0.5, order=1)
            nlm_half = non_local_means(
                w_half,
                sigma=max(k_sigma * 0.85, 0.018),
                patch_radius=1,
                search_radius=3,
                h_factor=0.92 + 0.20 * strength,
                guide=gaussian_filter(w_half, sigma=0.65),
            ).astype(np.float32)
            nlm_coarse = _zoom(nlm_half, (h_s / w_half.shape[0], w_s / w_half.shape[1]), order=1).astype(np.float32)

            # 4. Structure-Tensor Anisotropy & Tissue Zone Routing
            g07 = gaussian_filter(work, sigma=0.72)
            g20 = gaussian_filter(work, sigma=2.00)
            gy, gx = np.gradient(g07)
            jxx = gaussian_filter(gx * gx, sigma=1.8)
            jyy = gaussian_filter(gy * gy, sigma=1.8)
            jxy = gaussian_filter(gx * gy, sigma=1.8)
            trace = jxx + jyy
            aniso = np.hypot(jxx - jyy, 2.0 * jxy) / (trace + 1e-5)

            fg_tr = trace[fg > 0.5] if np.any(fg > 0.5) else trace.ravel()
            p78_tr = max(float(np.percentile(fg_tr, 78.0)), 1e-5)
            macro_g = np.hypot(*np.gradient(g20))
            fg_mg = macro_g[fg > 0.5] if np.any(fg > 0.5) else macro_g.ravel()
            p80_mg = max(float(np.percentile(fg_mg, 80.0)), 1e-4)

            macro_edge = np.clip(macro_g / (p80_mg * 1.02), 0.0, 1.0)
            aniso_line = np.clip((trace / p78_tr) * ((np.clip((aniso - 0.24) / 0.50, 0.0, 1.0)) ** 1.15), 0.0, 1.0)
            # White intramuscular fascial striations inside dark muscle (psoas / paraspinal / calf)
            muscle_zone = np.clip((g20 - 0.05) / 0.05, 0.0, 1.0) * np.clip((0.34 - g20) / 0.10, 0.0, 1.0)
            fascia_line = np.clip((work - g20) / max(1.05 * k_sigma, 0.015), 0.0, 1.0) * muscle_zone * np.clip(aniso / 0.35, 0.3, 1.0)
            # Dark cauda equina nerve roots & meniscal clefts
            nerve_root = np.clip((g20 - 0.38) / 0.16, 0.0, 1.0) * np.clip((g20 - g07 - 0.009) / 0.040, 0.0, 1.0)

            struct_gate = np.clip(np.maximum.reduce([macro_edge, aniso_line, fascia_line, nerve_root]), 0.0, 1.0)

            # Marrow zone (0.20..0.56): keep fine trabecular structure (light 3x3 NLM + real trabecular detail, ZERO coarse blur!)
            marrow_zone = np.clip((g20 - 0.18) / 0.09, 0.0, 1.0) * np.clip((0.58 - g20) / 0.11, 0.0, 1.0) * fg
            # Fat & fluid zone (>0.56) or sub-muscle shadow (<0.09): dissolve 3px k-space mottling into silky smooth cream!
            fat_or_haze = np.clip((g20 - 0.54) / 0.12, 0.0, 1.0) + np.clip((0.10 - g20) / 0.05, 0.0, 1.0)
            fat_or_haze = np.clip(fat_or_haze, 0.0, 1.0)

            nlm_mixed = (1.0 - 0.62 * fat_or_haze) * nlm_fine + (0.62 * fat_or_haze) * nlm_coarse
            zone_denoise = marrow_zone * bone_denoise + (1.0 - marrow_zone) * fat_denoise
            eff_alpha = zone_denoise * ((1.0 - 0.85 * struct_gate) ** 1.3)
            clean256 = (1.0 - eff_alpha) * work + eff_alpha * nlm_mixed

            # Preserve gentle band-limited trabecular bone texture (sigma 0.55..1.8 so it's crisp trabeculae, not blocky pixels)
            trab_band = gaussian_filter(work, sigma=0.48) - gaussian_filter(work, sigma=1.65)
            trab_cap = max(1.35 * k_sigma, 0.028)
            clean256 = np.clip(clean256 + bone_tex_256 * marrow_zone * np.clip(trab_band, -trab_cap, trab_cap), 0.0, 1.0)

            # 5. Crisp 1.5T/3T Axial Contrast Curve (Deep charcoal psoas/cortex + bright creamy fat + brilliant CSF!)
            lifted = np.power(np.clip(clean256, 0.0, 1.0), 0.92)
            k_ax, x0_ax = (3.25, 0.34) if seq_type in ("T2_FSE", "T2_STAR_GRE") else (2.95, 0.35)
            sc_raw = 1.0 / (1.0 + np.exp(-k_ax * (lifted - x0_ax)))
            sc_0 = 1.0 / (1.0 + np.exp(-k_ax * (0.0 - x0_ax)))
            sc_1 = 1.0 / (1.0 + np.exp(-k_ax * (1.0 - x0_ax)))
            sc_norm = np.clip((sc_raw - sc_0) / (sc_1 - sc_0), 0.0, 1.0)
            toned = (1.0 - scurve_mix) * lifted + scurve_mix * sc_norm

            if seq_type in ("T2_FSE", "T2_STAR_GRE"):
                fluid_mask = np.clip((toned - 0.62) / 0.25, 0.0, 1.0) ** 1.2
                toned = np.clip(toned + 0.14 * fluid_mask * (1.0 - toned), 0.0, 1.0)
                toned = np.clip(toned * (1.0 - 0.15 * nerve_root), 0.0, 1.0)

            # Feathered air gate (smooth outer skin contour, pure black air)
            bg_soft = gaussian_filter(np.clip((1.35 * air_floor - norm01) / max(1.35 * air_floor, 1e-4), 0.0, 1.0), sigma=0.80)
            toned = toned * (1.0 - 0.92 * (bg_soft ** 1.35))

            try:
                from app.store import setting as _get_setting
                _br = float(np.clip(float(_get_setting("clinic_brightness", "1.0") or 1.0), 0.75, 1.25))
                _ct = float(np.clip(float(_get_setting("clinic_contrast", "1.0") or 1.0), 0.75, 1.35))
            except Exception:
                _br, _ct = 1.0, 1.0
            if abs(_br - 1.0) > 1e-3 or abs(_ct - 1.0) > 1e-3:
                toned = np.clip((toned - 0.5) * _ct + 0.5 + (_br - 1.0) * 0.5, 0.0, 1.0)

            # Map [0..0.88] -> [0..s_scale] and [0.88..1.0] -> [s_scale..s_peak] so GE AW Window/Level opens bright & crisp!
            low_part = np.minimum(toned / 0.88, 1.0) * s_scale
            hi_part = np.clip((toned - 0.88) / 0.12, 0.0, 1.0) * max(s_peak - s_scale, 0.0)
            restored256 = np.maximum(low_part + hi_part * fg, 0.0)

            # 6. Smooth Anti-Ringing 1024x1024 Upscale + High-Res Cortical/Root Acutance + 1.5T Trabecular Micro-Grain
            if 200 <= h_s <= 512 and 200 <= w_s <= 512:
                scale_factor = 4.0 if (h_s <= 256 and w_s <= 256) else 2.0
                up_cub = _zoom(restored256, scale_factor, order=3)
                up_lin = _zoom(restored256, scale_factor, order=1)
                lo_c = np.minimum(up_lin, up_cub) * 0.982
                hi_c = np.maximum(up_lin, up_cub) * 1.018
                restored = np.maximum(np.clip(up_cub, lo_c, hi_c), 0.0)

                up_sg = np.clip(_zoom(struct_gate, scale_factor, order=1), 0.0, 1.0)
                up_marrow = np.clip(_zoom(marrow_zone, scale_factor, order=1), 0.0, 1.0)
                det_1k = restored - gaussian_filter(restored, sigma=1.25)
                neg_lim = s_scale * 0.048
                pos_lim = s_scale * 0.025
                det_clamped = np.where(det_1k < 0, np.clip(det_1k, -neg_lim, 0.0), np.clip(det_1k, 0.0, pos_lim))
                restored = np.maximum(restored + hi_crisp * det_clamped * (0.20 + 0.80 * np.maximum(up_sg, 0.45 * up_marrow)), 0.0)

                if bone_grain_1k > 0:
                    up_musc = np.clip(_zoom(muscle_zone * fg, scale_factor, order=1), 0.0, 1.0)
                    tex_w = np.clip(up_marrow + 0.32 * up_musc, 0.0, 1.0)
                    seed = (int(abs(float(np.sum(s_raw)))) * 2654435761 + i * 97531) & 0xFFFFFFFF
                    rng = np.random.default_rng(seed)
                    fine_1px = gaussian_filter(rng.standard_normal(restored.shape, dtype=np.float32), sigma=0.48) * 1.65
                    restored = np.maximum(restored + tex_w * (bone_grain_1k * s_scale) * fine_1px, 0.0)
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
        if mode in ("print_master", "3t_hd"):
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
