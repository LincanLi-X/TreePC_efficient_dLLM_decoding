from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_aggregator():
    path = Path(__file__).parents[2] / "scripts" / "16_aggregate_results.py"
    spec = importlib.util.spec_from_file_location("aggregate_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(sample_id: str, passed: bool, latency: float) -> dict[str, object]:
    return {
        "dataset": "gsm8k",
        "sample_id": sample_id,
        "steps": 4,
        "method": "pc_only",
        "passed": passed,
        "latency_s": latency,
        "output_tokens_per_s": 10.0,
        "peak_gpu_memory_mib": 100.0,
        "dependency_head_s": 0.0,
        "mst_s": 0.0,
        "correction_head_s": 0.0,
        "corrected_token_flips": 0,
        "candidate_size_mean": 0.0,
        "tree_depth_max": 0,
        "tree_used_rate": 0.0,
        "dream_nfe": 4,
    }


def test_sharded_results_are_aggregated() -> None:
    summary = load_aggregator().summarize([row("a", True, 1.0), row("b", False, 3.0)])[0]
    assert summary["sample_size"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["latency_mean_s"] == 2.0
    assert summary["dream_nfe_exact"] is True


def test_duplicate_result_row_is_rejected() -> None:
    value = row("a", True, 1.0)
    with pytest.raises(ValueError, match="Duplicate"):
        load_aggregator().summarize([value, value])
