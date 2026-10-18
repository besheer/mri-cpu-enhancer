from app.quality import evaluate_quality


def test_quality_gate_passes_conservative_result() -> None:
    status, reasons = evaluate_quality({
        "edge_preservation_percent": 93,
        "normalized_mae": 0.05,
        "noise_change_percent": -20,
    }, 15, 15)
    assert status == "PASS"
    assert reasons


def test_quality_gate_warns_for_borderline_result() -> None:
    status, _ = evaluate_quality({
        "edge_preservation_percent": 82,
        "normalized_mae": 0.13,
        "noise_change_percent": -55,
    }, 15, 15)
    assert status == "WARN"


def test_quality_gate_fails_instance_mismatch_and_excessive_change() -> None:
    status, reasons = evaluate_quality({
        "edge_preservation_percent": 70,
        "normalized_mae": 0.20,
        "noise_change_percent": -70,
    }, 15, 14)
    assert status == "FAIL"
    assert any("Instance count mismatch" in reason for reason in reasons)

