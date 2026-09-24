#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from treepc.utils.io import write_json


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, int, str]] = set()
    for row in rows:
        key = (row["dataset"], row["sample_id"], row["steps"], row["method"])
        if key in seen:
            raise ValueError(f"Duplicate experiment row: {key}")
        seen.add(key)
        grouped[(row["dataset"], row["steps"], row["method"])].append(row)
    result = []
    for (dataset, steps, method), values in sorted(grouped.items()):
        result.append(
            {
                "dataset": dataset,
                "steps": steps,
                "method": method,
                "sample_size": len(values),
                "accuracy": statistics.mean(row["passed"] for row in values),
                "latency_mean_s": statistics.mean(row["latency_s"] for row in values),
                "latency_median_s": statistics.median(
                    row["latency_s"] for row in values
                ),
                "output_tokens_per_s_mean": statistics.mean(
                    row["output_tokens_per_s"] for row in values
                ),
                "peak_gpu_memory_mib_max": max(
                    row["peak_gpu_memory_mib"] for row in values
                ),
                "dependency_head_mean_s": statistics.mean(
                    row["dependency_head_s"] for row in values
                ),
                "mst_mean_s": statistics.mean(row["mst_s"] for row in values),
                "correction_head_mean_s": statistics.mean(
                    row["correction_head_s"] for row in values
                ),
                "corrected_token_flips_mean": statistics.mean(
                    row["corrected_token_flips"] for row in values
                ),
                "candidate_size_mean": statistics.mean(
                    row["candidate_size_mean"] for row in values
                ),
                "tree_depth_max": max(row["tree_depth_max"] for row in values),
                "tree_used_rate_mean": statistics.mean(
                    row["tree_used_rate"] for row in values
                ),
                "dream_nfe_exact": all(row["dream_nfe"] == steps for row in values),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate sharded TreePC result rows")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument(
        "--input-row-limits",
        nargs="+",
        type=int,
        help="Optional per-input row limits, in the same order as --inputs.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    if args.input_row_limits and len(args.input_row_limits) != len(args.inputs):
        parser.error("--input-row-limits must match the number of --inputs")
    limits = args.input_row_limits or [-1] * len(args.inputs)
    for value, limit in zip(args.inputs, limits):
        with Path(value).open(encoding="utf-8") as handle:
            selected = (line for line in handle if line.strip())
            rows.extend(
                json.loads(line)
                for index, line in enumerate(selected)
                if limit < 0 or index < limit
            )
    summary_rows = summarize(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "summary.json",
        {
            "schema": "treepc.stage4_micro.v1",
            "internal_poc_only": True,
            "standard_benchmark_claim_allowed": False,
            "rows": summary_rows,
        },
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
