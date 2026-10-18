from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

from app.enhancer import conservative_baseline

PROFILE_INFO = {
    "BASE_MILD": {"label": "Baseline محافظ", "guidance": "مرجع آمن نسبيًا ومناسب للاختبار الأول.", "min_slices": 1, "research_only": False},
    "DENOISE_MEDIUM": {"label": "Denoise متوسط", "guidance": "للصور ذات الضوضاء الواضحة مع مراجعة الحواف الدقيقة.", "min_slices": 1, "research_only": False},
    "EDGE_DENOISE_MILD": {"label": "Edge Denoise محافظ", "guidance": "تقليل ضوضاء محافظ مع حماية أفضل للحواف الدقيقة.", "min_slices": 1, "research_only": False},
    "N4_BIAS": {"label": "N4 Bias فقط", "guidance": "لتفاوت الإضاءة التدريجي، ويحتاج Volume متناسقًا.", "min_slices": 8, "research_only": False},
    "N4_DENOISE": {"label": "N4 + Denoise", "guidance": "عند اجتماع تفاوت الإضاءة والضوضاء؛ قارن قبل الاعتماد.", "min_slices": 8, "research_only": False},
    "N4_EDGE_DENOISE": {"label": "N4 + Edge Denoise", "guidance": "Bias واضح مع ضوضاء؛ يحافظ على الحواف أكثر من Gaussian.", "min_slices": 8, "research_only": False},
    "CLAHE_VIEW": {"label": "CLAHE للعرض البحثي", "guidance": "قد يضخم الضوضاء والتباين؛ لا يُستخدم كنتيجة تشخيصية.", "min_slices": 1, "research_only": True},
}
PROFILES = {key: value["label"] for key, value in PROFILE_INFO.items()}
STRENGTHS = {"MILD": "محافظ", "MEDIUM": "متوسط", "STRONG": "قوي — بحثي"}


def recommendation(description: str, instances: int) -> tuple[str, str]:
    name = description.upper()
    if any(word in name for word in ("SCANO", "LOCALIZER", "SURVEY")):
        return "BASE_MILD", "Localizer/Scano: لا تستخدم N4 أو تحسينًا قويًا."
    if instances < 8:
        return "DENOISE_MEDIUM", "Series قصيرة: استخدم Denoise أو Baseline فقط."
    return "EDGE_DENOISE_MILD", "ابدأ بتقليل ضوضاء محافظ للحواف؛ استخدم N4 فقط عند وجود Bias مرئي."


def validate_selection(profile: str, description: str, instances: int, acknowledged: bool = False,
                       strength: str = "MEDIUM") -> None:
    if profile not in PROFILE_INFO:
        raise ValueError("Processing profile غير معروف")
    if strength not in STRENGTHS:
        raise ValueError("Processing level غير معروف")
    info = PROFILE_INFO[profile]
    if instances < int(info["min_slices"]):
        raise ValueError(f"{profile} يحتاج {info['min_slices']} شرائح على الأقل")
    if profile.startswith("N4") and any(x in description.upper() for x in ("SCANO", "LOCALIZER", "SURVEY")):
        raise ValueError("لا يُسمح بتطبيق N4 على Localizer/Scano")
    if info["research_only"] and not acknowledged:
        raise ValueError("يجب تأكيد أن CLAHE للعرض البحثي فقط")
    if strength == "STRONG" and not acknowledged:
        raise ValueError("المستوى القوي بحثي ويحتاج تأكيدًا صريحًا")

PREFIXES = {
    "BASE_MILD": "ENHANC-BASE-",
    "DENOISE_MEDIUM": "ENHANC-DENOISE-",
    "EDGE_DENOISE_MILD": "ENHANC-EDGE-",
    "N4_BIAS": "ENHANC-N4-",
    "N4_DENOISE": "ENHANC-N4DENOISE-",
    "N4_EDGE_DENOISE": "ENHANC-N4EDGE-",
    "CLAHE_VIEW": "ENHANC-CLAHE-VIEW-",
}


def _cast_like(values: np.ndarray, source: np.ndarray) -> np.ndarray:
    info = np.iinfo(source.dtype)
    return np.rint(np.clip(values, info.min, info.max)).astype(source.dtype)


def _n4(volume: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(volume.astype(np.float32))
    image.SetSpacing(spacing)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    shrink = [2 if size >= 32 else 1 for size in image.GetSize()]
    small_image = sitk.Shrink(image, shrink)
    small_mask = sitk.Shrink(mask, shrink)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([30, 20, 10])
    corrector.Execute(small_image, small_mask)
    log_field = corrector.GetLogBiasFieldAsImage(image)
    corrected = image / sitk.Exp(log_field)
    output = sitk.GetArrayFromImage(corrected)
    foreground = sitk.GetArrayFromImage(mask).astype(bool)
    if foreground.any():
        old_median = float(np.median(volume[foreground]))
        new_median = float(np.median(output[foreground]))
        if new_median:
            output *= old_median / new_median
    return output


def _edge_denoise(volume: np.ndarray, spacing: tuple[float, float, float], strength: str) -> np.ndarray:
    import SimpleITK as sitk

    source = volume.astype(np.float32)
    low, high = np.percentile(source, [1, 99])
    span = max(float(high - low), 1.0)
    normalized = np.clip((source - low) / span, 0.0, 1.0)
    image = sitk.GetImageFromArray(normalized)
    image.SetSpacing(spacing)
    iterations = {"MILD": 3, "MEDIUM": 5, "STRONG": 7}[strength]
    blend = {"MILD": 0.25, "MEDIUM": 0.40, "STRONG": 0.55}[strength]
    filtered = sitk.CurvatureAnisotropicDiffusion(image, timeStep=0.04,
                                                  conductanceParameter=1.5,
                                                  numberOfIterations=iterations)
    restored = sitk.GetArrayFromImage(filtered) * span + low
    return (1.0 - blend) * source + blend * restored


def process_volume(volume: np.ndarray, profile: str, strength: str = "MEDIUM",
                   spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> tuple[np.ndarray, dict[str, float | str]]:
    if profile not in PROFILES:
        raise ValueError("Unknown processing profile")
    if strength not in STRENGTHS:
        raise ValueError("Unknown processing strength")
    if volume.ndim != 3 or volume.shape[0] < 1:
        raise ValueError("A 3D DICOM series is required")
    started = time.perf_counter()
    source = volume
    if profile == "BASE_MILD":
        output = np.stack([conservative_baseline(x) for x in source])
    elif profile == "DENOISE_MEDIUM":
        blend = {"MILD": 0.35, "MEDIUM": 0.55, "STRONG": 0.70}[strength]
        smooth = gaussian_filter(source.astype(np.float32), sigma=(0.35, 0.9, 0.9))
        output = _cast_like((1.0 - blend) * source + blend * smooth, source)
    elif profile == "EDGE_DENOISE_MILD":
        output = _cast_like(_edge_denoise(source, spacing, strength), source)
    elif profile in {"N4_BIAS", "N4_DENOISE", "N4_EDGE_DENOISE"}:
        if source.shape[0] < 3:
            raise ValueError("N4 requires at least 3 consistently ordered slices")
        corrected = _n4(source, spacing)
        n4_blend = {"MILD": 0.60, "MEDIUM": 0.80, "STRONG": 1.0}[strength]
        output = (1.0 - n4_blend) * source + n4_blend * corrected
        if profile == "N4_DENOISE":
            blend = {"MILD": 0.30, "MEDIUM": 0.50, "STRONG": 0.65}[strength]
            smooth = gaussian_filter(output, sigma=(0.35, 0.8, 0.8))
            output = (1.0 - blend) * output + blend * smooth
        elif profile == "N4_EDGE_DENOISE":
            output = _edge_denoise(output, spacing, strength)
        output = _cast_like(output, source)
    else:
        import SimpleITK as sitk
        image = sitk.GetImageFromArray(source.astype(np.float32))
        alpha, beta = {"MILD": (0.15, 0.15), "MEDIUM": (0.3, 0.3), "STRONG": (0.45, 0.45)}[strength]
        enhanced = sitk.AdaptiveHistogramEqualization(image, alpha=alpha, beta=beta, radius=[8, 8, 2])
        output = _cast_like(sitk.GetArrayFromImage(enhanced), source)

    src = source.astype(np.float32)
    dst = output.astype(np.float32)
    span = max(float(np.percentile(src, 99) - np.percentile(src, 1)), 1.0)
    src_noise = float(np.std(src - gaussian_filter(src, sigma=(0, 1, 1))))
    dst_noise = float(np.std(dst - gaussian_filter(dst, sigma=(0, 1, 1))))
    src_gradient = np.sqrt(sum(sobel(src, axis=axis) ** 2 for axis in range(3)))
    dst_gradient = np.sqrt(sum(sobel(dst, axis=axis) ** 2 for axis in range(3)))
    edge_mask = src_gradient >= np.percentile(src_gradient, 90)
    edge_error = float(np.mean(np.abs(dst_gradient[edge_mask] - src_gradient[edge_mask])))
    edge_reference = max(float(np.mean(src_gradient[edge_mask])), 1e-6)
    metrics = {
        "strength": strength,
        "seconds": round(time.perf_counter() - started, 3),
        "normalized_mae": round(float(np.mean(np.abs(dst - src))) / span, 5),
        "noise_before": round(src_noise, 3),
        "noise_after": round(dst_noise, 3),
        "noise_change_percent": round((dst_noise - src_noise) * 100 / max(src_noise, 1e-6), 2),
        "edge_preservation_percent": round(100 * max(0.0, 1.0 - edge_error / edge_reference), 2),
    }
    return output, metrics
