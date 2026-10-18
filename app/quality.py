from __future__ import annotations


def evaluate_quality(metrics: dict[str, float | str], source_instances: int,
                     output_instances: int) -> tuple[str, list[str]]:
    """Conservative research QC gate; it is not a diagnostic-quality claim."""
    failures: list[str] = []
    warnings: list[str] = []
    edge = float(metrics.get("edge_preservation_percent", 0))
    change = float(metrics.get("normalized_mae", 1))
    noise_change = float(metrics.get("noise_change_percent", 0))

    if output_instances != source_instances:
        failures.append(f"Instance count mismatch: {source_instances}->{output_instances}")
    if edge < 75:
        failures.append(f"Edge preservation too low: {edge:.2f}%")
    elif edge < 85:
        warnings.append(f"Review edge preservation: {edge:.2f}%")
    if change > 0.18:
        failures.append(f"Image change too high: {change:.4f}")
    elif change > 0.12:
        warnings.append(f"Review image change: {change:.4f}")
    if noise_change < -65:
        failures.append(f"Noise reduction may be excessive: {noise_change:.2f}%")
    elif noise_change < -50:
        warnings.append(f"Strong noise reduction: {noise_change:.2f}%")

    if failures:
        return "FAIL", failures + warnings
    if warnings:
        return "WARN", warnings
    return "PASS", ["Automated research checks passed; physician review is still required"]
