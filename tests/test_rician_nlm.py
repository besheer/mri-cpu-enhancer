from __future__ import annotations

import math
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

from app.gemini_auditor import _local_physics_audit
from app.profiles import PROFILE_INFO, process_volume, recommendation
from app.rician_nlm import (
    anisotropic_diffusion,
    estimate_sigma,
    non_local_means,
    rician_forward_vst,
    structure_tensor_tangent_restore,
)
from app.synthetic import corrupt_volume, make_phantom


def _psnr(ref: np.ndarray, test: np.ndarray) -> float:
    mse = float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _edge_corr(ref: np.ndarray, test: np.ndarray) -> float:
    gr = gaussian_filter(np.hypot(sobel(ref, axis=0), sobel(ref, axis=1)), 1.0).ravel()
    gt = gaussian_filter(np.hypot(sobel(test, axis=0), sobel(test, axis=1)), 1.0).ravel()
    return float(np.corrcoef(gr, gt)[0, 1])


def test_rician_vst_and_nlm_improve_psnr_and_preserve_edges() -> None:
    clean_vol = make_phantom(4, 64)
    noisy_vol = corrupt_volume(clean_vol, "RICIAN_NOISE", "MEDIUM", seed=42)

    clean01 = np.clip(clean_vol[2].astype(np.float32) / 3000.0, 0.0, 1.0)
    noisy01 = np.clip(noisy_vol[2].astype(np.float32) / 3000.0, 0.0, 1.0)

    sigma_clean = estimate_sigma(clean01)
    sigma_noisy = estimate_sigma(noisy01)
    assert sigma_noisy > sigma_clean

    vst = rician_forward_vst(noisy01, sigma_noisy)
    assert vst.shape == noisy01.shape
    assert float(vst.min()) >= 0.0 and float(vst.max()) <= 1.0

    nlm_out = non_local_means(noisy01, sigma=sigma_noisy, patch_radius=1, search_radius=3)
    ad_out = anisotropic_diffusion(
        noisy01,
        sigma=sigma_noisy,
        niter=4,
        kappa=max(sigma_noisy * 1.6, 0.02),
    )
    restored, info = structure_tensor_tangent_restore(noisy01, strength=0.8, sigma_est=sigma_noisy)

    expected_clamped = float(np.clip(sigma_noisy, 0.012, 0.14))
    assert isinstance(info, dict)
    assert abs(info["sigma_estimated"] - expected_clamped) < 1e-4
    assert _psnr(clean01, nlm_out) > _psnr(clean01, noisy01)
    assert _psnr(clean01, ad_out) > _psnr(clean01, noisy01)
    assert _psnr(clean01, restored) > _psnr(clean01, noisy01)
    assert _edge_corr(clean01, restored) > _edge_corr(clean01, noisy01)
    assert float(restored.min()) >= 0.0 and float(restored.max()) <= 1.0


def test_global_volume_normalization_preserves_inter_slice_contrast() -> None:
    clean = make_phantom(4, 48).astype(np.float32)
    # Create deliberate inter-slice brightness scaling across slices
    clean[0] *= 0.55
    clean[3] *= 1.15
    vol = np.rint(np.clip(clean, 0, 4095)).astype(np.uint16)

    out, metrics = process_volume(vol, "RICIAN_NLM_VST", "MEDIUM", description="SAG T2 FSE")
    assert out.shape == vol.shape
    assert float(out[3].mean()) > float(out[0].mean()) * 1.4
    assert 0.0 <= float(metrics["edge_preservation_percent"]) <= 100.0


def test_default_recommendation_and_auto_eligibility() -> None:
    prof, _ = recommendation("AX T2 FSE BRAIN", 16)
    assert prof == "RICIAN_NLM_VST"
    loc_prof, _ = recommendation("SURVEY LOCALIZER", 3)
    assert loc_prof == "BASE_MILD"
    assert PROFILE_INFO["RICIAN_NLM_VST"]["auto_eligible"] is True
    assert PROFILE_INFO["RAD4AI_3T_HD"]["research_only"] is True
    assert PROFILE_INFO["RAD4AI_3T_HD"]["auto_eligible"] is False
    assert PROFILE_INFO["RAD4AI_PRINT_MASTER"]["research_only"] is True
    assert PROFILE_INFO["RAD4AI_PRINT_MASTER"]["auto_eligible"] is False
    assert PROFILE_INFO["RAD4AI_SR_15T"]["auto_eligible"] is False
    assert PROFILE_INFO["RAD4AI_GE_AW"]["auto_eligible"] is False
    assert PROFILE_INFO["RAD4AI_SPINE_PRO"]["auto_eligible"] is False


def test_local_physics_audit_never_imports_or_calls_network(monkeypatch) -> None:
    # Block httpx in sys.modules to guarantee _local_physics_audit makes zero network calls
    monkeypatch.setitem(sys.modules, "httpx", None)
    clean_vol = make_phantom(4, 48)
    noisy_vol = corrupt_volume(clean_vol, "RICIAN_NOISE", "MILD", seed=11)
    audit_res = _local_physics_audit(noisy_vol[2], clean_vol[2], "AX T2", "RICIAN_NLM_VST")
    assert "quantitative_metrics" in audit_res
    assert "preliminary_findings" not in audit_res
    assert audit_res["verdict"] in {"PASS", "WARN"}
