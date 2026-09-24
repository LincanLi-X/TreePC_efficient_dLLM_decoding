from __future__ import annotations

import statistics


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "latency_mean_s": statistics.mean(values),
        "latency_median_s": statistics.median(values),
        "latency_p90_s": percentile(values, 0.9),
        "latency_total_s": sum(values),
    }
