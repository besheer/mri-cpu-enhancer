import numpy as np

from app.synthetic import (corrupt_volume, difference_map, make_phantom,
                           reference_metrics, result_decision)


def test_phantom_and_corruption_are_deterministic():
    clean = make_phantom(8, 64)
    first = corrupt_volume(clean, "MIXED_03T", "MEDIUM", 42)
    second = corrupt_volume(clean, "MIXED_03T", "MEDIUM", 42)
    assert clean.shape == (8, 64, 64)
    assert clean.dtype == np.uint16
    assert np.array_equal(first, second)
    assert not np.array_equal(clean, first)


def test_stronger_noise_reduces_reference_quality():
    clean = make_phantom(8, 64)
    mild = corrupt_volume(clean, "RICIAN_NOISE", "MILD", 7)
    strong = corrupt_volume(clean, "RICIAN_NOISE", "STRONG", 7)
    mild_metrics = reference_metrics(clean, mild)
    strong_metrics = reference_metrics(clean, strong)
    assert mild_metrics["psnr_db"] > strong_metrics["psnr_db"]
    assert mild_metrics["global_ssim"] > strong_metrics["global_ssim"]


def test_reference_identity_metrics():
    clean = make_phantom(4, 32)
    metrics = reference_metrics(clean, clean.copy())
    assert metrics["psnr_db"] == 99.0
    assert metrics["global_ssim"] == 1.0
    assert metrics["roi_ssim"] == 1.0
    assert metrics["slice_ssim_mean"] == 1.0
    assert metrics["mae"] == 0.0


def test_difference_and_decision_safety():
    clean = make_phantom(8, 64)
    noisy = corrupt_volume(clean, "RICIAN_NOISE", "MEDIUM", 9)
    input_metrics = reference_metrics(clean, noisy)
    identity_metrics = reference_metrics(clean, clean)
    diff = difference_map(clean, noisy)
    decision, _ = result_decision("RICIAN_NOISE", input_metrics, identity_metrics)
    review, _ = result_decision("MOTION", input_metrics, identity_metrics)
    assert diff.dtype == np.uint16
    assert diff.max() > 0
    assert decision == "IMPROVED"
    assert review == "REVIEW_ONLY"
