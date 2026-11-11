"""Physics-accurate Rician MRI noise estimation, Variance-Stabilising Transform (VST),
Vectorized Non-Local Means (Integral-Image NLM), and Anisotropic Structure-Tensor
Tangent-Guided Restoration (NumPy/SciPy).

Design Rules (SwiftMR-Compatible):
1. Strictly remove noise & Rician positive magnitude bias. Never synthesize or inject
   high-frequency texture/grain that was not measured in the patient scan.
2. Estimate Rician sigma automatically via Multi-Scale Donoho Haar-Diagonal MAD
   (accounts for 0.3T k-space zero-filling correlation).
3. Preserve thin fascial planes, cortical bone rims, and cauda equina nerve roots via
   Structure-Tensor Coherence and 1-D Tangent-Guided directional filtering.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def estimate_sigma(img: np.ndarray) -> float:
    """Robust Rician noise estimate using Multi-Scale Donoho Haar-diagonal MAD.

    On 0.3T scanners with zero-filled k-space interpolation, adjacent pixels are
    spatially correlated, so stride-1 MAD underestimates noise. Taking the max of
    stride-1 and stride-2 Haar diagonal MAD gives an accurate, stable estimate.
    """
    a = img.astype(np.float64, copy=False)
    sigmas = []
    for stride in (1, 2):
        sub = a[::stride, ::stride]
        h = (sub.shape[0] // 2) * 2
        w = (sub.shape[1] // 2) * 2
        if h < 4 or w < 4:
            continue
        blocks = sub[:h, :w].reshape(h // 2, 2, w // 2, 2)
        hh = (blocks[:, 0, :, 0] - blocks[:, 0, :, 1] - blocks[:, 1, :, 0] + blocks[:, 1, :, 1]) / 2.0
        if hh.size > 0:
            sigmas.append(float(np.median(np.abs(hh)) / 0.6745))
    return max(sigmas) if sigmas else 0.02


def rician_forward_vst(z: np.ndarray, sigma: float, debias_factor: float = 1.85) -> np.ndarray:
    """Forward VST & Rician magnitude debiasing: sqrt(max(z^2 - debias_factor * sigma^2, 0)).

    Removes the positive Rician noise floor (+sigma^2 / 2A) in dark muscles (psoas),
    annulus fibrosus, and air background so they return to their true dark signal.
    """
    return np.sqrt(np.clip(z * z - debias_factor * (sigma * sigma), 0.0, None))


def _mean_patch(img: np.ndarray, radius: int) -> np.ndarray:
    """Fast O(1) per-pixel box mean over a (2*radius+1)^2 window via 2-D integral image."""
    if radius <= 0:
        return img
    k = 2 * radius + 1
    p = np.pad(img, radius, mode="reflect")
    s = p.cumsum(axis=0).cumsum(axis=1)
    s = np.pad(s, ((1, 0), (1, 0)), mode="constant")
    out = s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
    return out / float(k * k)


def non_local_means(
    img: np.ndarray,
    sigma: float,
    patch_radius: int = 2,
    search_radius: int = 3,
    h_factor: float = 1.05,
    guide: np.ndarray | None = None,
) -> np.ndarray:
    """Fast vectorised Non-Local Means for 2-D MRI slices using integral images."""
    a = img.astype(np.float64, copy=False)
    g = guide.astype(np.float64, copy=False) if guide is not None else a
    h, w = a.shape
    sr = int(search_radius)
    pr = int(patch_radius)
    p_a = np.pad(a, sr, mode="reflect")
    p_g = np.pad(g, sr, mode="reflect")

    acc = np.zeros_like(a)
    wsum = np.zeros_like(a)
    h2 = max((h_factor * sigma) ** 2, 1e-6)
    noise2 = 1.6 * sigma * sigma

    for dy in range(-sr, sr + 1):
        ys = slice(sr + dy, sr + dy + h)
        for dx in range(-sr, sr + 1):
            xs = slice(sr + dx, sr + dx + w)
            shifted_a = p_a[ys, xs]
            if dy == 0 and dx == 0:
                weight = np.full_like(a, 0.45)
            else:
                shifted_g = p_g[ys, xs]
                diff2 = (g - shifted_g) ** 2
                dist2 = _mean_patch(diff2, pr)
                spatial_decay = (dx * dx + dy * dy) / float(sr * sr * 3.0)
                weight = np.exp(-np.maximum(dist2 - noise2, 0.0) / h2 - spatial_decay)
            acc += weight * shifted_a
            wsum += weight
    return acc / np.maximum(wsum, 1e-8)


def anisotropic_diffusion(
    img: np.ndarray,
    sigma: float = 0.04,
    niter: int = 8,
    kappa: float | None = None,
    gamma: float = 0.15,
) -> np.ndarray:
    """Perona-Malik edge-preserving anisotropic diffusion."""
    out = img.astype(np.float64, copy=True)
    k = kappa if kappa is not None else max(2.2 * sigma, 0.025)

    for _ in range(int(niter)):
        p = np.pad(out, 1, mode="reflect")
        d_n = p[:-2, 1:-1] - out
        d_s = p[2:, 1:-1] - out
        d_w = p[1:-1, :-2] - out
        d_e = p[1:-1, 2:] - out
        g_n = 1.0 / (1.0 + (d_n / k) ** 2)
        g_s = 1.0 / (1.0 + (d_s / k) ** 2)
        g_w = 1.0 / (1.0 + (d_w / k) ** 2)
        g_e = 1.0 / (1.0 + (d_e / k) ** 2)
        out += gamma * (g_n * d_n + g_s * d_s + g_w * d_w + g_e * d_e)
    return out


def structure_tensor_tangent_restore(
    img01: np.ndarray,
    strength: float = 0.8,
    sigma_est: float | None = None,
) -> tuple[np.ndarray, dict]:
    """SwiftMR-style Physics Restoration:
    1. Multi-Scale Donoho Sigma Estimation & Unbiased Rician Forward VST.
    2. Vectorized Integral-Image Non-Local Means on pre-smoothed guide.
    3. Structure-Tensor Anisotropy & Coherence gate (distinguishes 4-6px circular
       Rician speckle clumps from elongated anatomical boundaries, fascial planes,
       and cauda equina nerve roots).
    4. 1-D Directional Tangent Filtering along anatomical curves (zero blockiness,
       zero fake grain).
    """
    a = np.clip(img01.astype(np.float64, copy=False), 0.0, 1.0)
    sigma = sigma_est if sigma_est is not None else estimate_sigma(a)
    sigma = float(np.clip(sigma, 0.012, 0.14))

    # 1. Rician Forward VST (removes positive Rician noise bias in dark muscles & air)
    debias_strength = 0.55 + 0.35 * float(strength)
    work = rician_forward_vst(a, sigma, debias_factor=debias_strength)
    # Blend smoothly so bright CSF/fat stays at exact original scale while dark muscle/air is debiased
    dark_gate = np.clip((0.42 - a) / 0.35, 0.0, 1.0)
    work = a * (1.0 - 0.65 * dark_gate) + work * (0.65 * dark_gate)

    # 2. Multi-scale Gaussian pyramid & Adaptive Guide for Vectorized NLM
    g08 = gaussian_filter(work, sigma=0.65)
    g14 = gaussian_filter(work, sigma=1.15)
    g40 = gaussian_filter(work, sigma=2.80)

    # Adaptive NLM: use patch_radius=2 on high-noise 0.3T scans (sigma >= 0.032) and patch_radius=1 on lower noise
    pr = 2 if sigma >= 0.032 else 1
    nlm_guide = g14 if sigma >= 0.038 else g08
    nlm_clean = non_local_means(
        work,
        sigma=max(sigma, 0.02),
        patch_radius=pr,
        search_radius=3,
        h_factor=0.75 + 0.30 * float(strength),
        guide=nlm_guide,
    )

    # 3. Structure Tensor (Jxx, Jxy, Jyy) integrated at scale rho=2.4
    #    Cancels random circular 4-6px speckle blobs (anisotropy -> 0) while reinforcing
    #    real anatomical borders & thin fascial lines (anisotropy -> 1).
    gy, gx = np.gradient(g08)
    jxx = gaussian_filter(gx * gx, sigma=2.4)
    jyy = gaussian_filter(gy * gy, sigma=2.4)
    jxy = gaussian_filter(gx * gy, sigma=2.4)

    trace = jxx + jyy
    det_diff = np.hypot(jxx - jyy, 2.0 * jxy)
    anisotropy = det_diff / (trace + 1e-4)

    # Macro gradient (true organ/disc boundaries, immune to speckle)
    gy40, gx40 = np.gradient(g40)
    grad40 = np.hypot(gx40, gy40)
    p85_g40 = max(float(np.percentile(grad40, 85.0)), 1e-4)
    macro_edge = np.clip(grad40 / (p85_g40 * 1.15), 0.0, 1.0)

    p80_tr = max(float(np.percentile(trace, 80.0)), 1e-5)
    line_edge = np.clip((trace / p80_tr) * (anisotropy ** 1.2), 0.0, 1.0)

    # Thin bright fascial ridge & dark nerve root protection inside bright CSF
    ridge_mask = np.clip((g08 - g40 - 0.012) / 0.055, 0.0, 1.0) * (0.35 + 0.65 * anisotropy)
    nerve_root_mask = np.where((g40 > 0.42) & (g08 < g40 - 0.012), 0.88, 0.0)

    struct_gate = np.clip(
        np.maximum.reduce([macro_edge, line_edge, ridge_mask * 0.85, nerve_root_mask]),
        0.0,
        1.0,
    )

    # 4. Vectorized 1-D Tangent-Guided Curve Smoothing along (-sin(theta), cos(theta))
    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    tx = -np.sin(theta)
    ty = np.cos(theta)

    h, w = a.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    def sample_at(step: float) -> np.ndarray:
        sx = np.clip(np.rint(xx + tx * step).astype(np.intp), 0, w - 1)
        sy = np.clip(np.rint(yy + ty * step).astype(np.intp), 0, h - 1)
        return g08[sy, sx]

    v_tangent = (
        0.50 * g08
        + 0.18 * (sample_at(1.0) + sample_at(-1.0))
        + 0.07 * (sample_at(2.0) + sample_at(-2.0))
    )

    # Homogeneous tissue interior (anisotropic bilateral smoothing around NLM to suppress mottled clumps without blurring)
    diff_g14 = np.abs(nlm_clean - g14)
    w_g14 = np.exp(-((diff_g14 / max(1.5 * sigma, 0.03)) ** 2)) * ((1.0 - struct_gate) ** 1.5)
    v_interior = nlm_clean * (1.0 - 0.35 * strength * w_g14) + g14 * (0.35 * strength * w_g14)

    # Coherent edge restoration (tangent-smoothed curve + subtle deblur normal to edge, clamped to prevent halos)
    deblur_delta = np.clip(v_tangent - g14, -0.025, 0.025)
    v_edge = 0.72 * nlm_clean + 0.28 * (v_tangent + deblur_delta * (0.25 * float(strength)))

    out = v_interior * (1.0 - struct_gate) + v_edge * struct_gate
    out = np.clip(out, 0.0, 1.0)

    info = {
        "sigma_estimated": round(float(sigma), 5),
        "mean_anisotropy": round(float(np.mean(anisotropy)), 4),
    }
    return out.astype(np.float32), info
