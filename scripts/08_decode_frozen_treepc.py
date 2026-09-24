#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from treepc.data.datasets import BenchmarkDataset
from treepc.decoding.learned_treepc import learned_treepc_generate
from treepc.dream.adapter import DreamAdapter
from treepc.dream.generation import custom_independent_generate
from treepc.dream.loader import load_dream
from treepc.evaluation.humaneval import evaluate_humaneval
from treepc.evaluation.task_metrics import gsm8k_exact_match
from treepc.models.treepc_bundle import TreePCBundle
from treepc.utils.io import append_jsonl, write_json
from treepc.utils.seed import seed_everything

MAX_NEW_TOKENS = {"gsm8k": 256, "humaneval": 512}


def score(dataset: str, sample: dict[str, Any], output: str) -> tuple[bool, str | None]:
    if dataset == "gsm8k":
        return gsm8k_exact_match(output, sample["answer"]), None
    result = evaluate_humaneval(sample["prompt"], output, sample["test"], sample["entry_point"])
    return bool(result["passed"]), result["error"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen-backbone learned TreePC")
    parser.add_argument("--dependency-checkpoint", required=True)
    parser.add_argument("--correction-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--per-dataset", type=int, default=2)
    parser.add_argument("--dependency-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3030)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    rows_path = output_dir / "rows.jsonl"
    if rows_path.exists():
        raise FileExistsError(rows_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    adapter = DreamAdapter(load_dream(args.device))
    bundle = TreePCBundle.from_checkpoints(
        adapter,
        args.dependency_checkpoint,
        args.correction_checkpoint,
        dependency_threshold=args.dependency_threshold,
    )
    rows: list[dict[str, Any]] = []
    try:
        for dataset_name in ("gsm8k", "humaneval"):
            dataset = BenchmarkDataset(dataset_name)
            indices = manifest["datasets"][dataset_name]["quality_indices"][: args.per_dataset]
            for steps in args.steps:
                for ordinal, index in enumerate(indices, 1):
                    sample = dataset[index]
                    prompt = dataset.prompt(index)
                    seed_everything(args.seed + index + steps * 10_000)
                    baseline = custom_independent_generate(
                        adapter,
                        prompt,
                        steps=steps,
                        max_new_tokens=MAX_NEW_TOKENS[dataset_name],
                    )
                    seed_everything(args.seed + index + steps * 10_000)
                    treepc, trace, timings = learned_treepc_generate(
                        adapter,
                        bundle,
                        prompt,
                        steps=steps,
                        max_new_tokens=MAX_NEW_TOKENS[dataset_name],
                    )
                    baseline_pass, baseline_error = score(dataset_name, sample, baseline.texts[0])
                    treepc_pass, treepc_error = score(dataset_name, sample, treepc.texts[0])
                    row = {
                        "dataset": dataset_name,
                        "sample_id": sample["sample_id"],
                        "dataset_index": index,
                        "steps": steps,
                        "baseline_pass": baseline_pass,
                        "treepc_pass": treepc_pass,
                        "quality_delta": int(treepc_pass) - int(baseline_pass),
                        "baseline_latency_s": baseline.latency_seconds,
                        "treepc_latency_s": treepc.latency_seconds,
                        "latency_overhead_s": treepc.latency_seconds - baseline.latency_seconds,
                        "dependency_head_s": timings["dependency_head_s"],
                        "mst_s": timings["mst_s"],
                        "correction_head_s": timings["correction_head_s"],
                        "global_scale": bundle.correction_head.global_scale_value(),
                        "dream_nfe": int(timings["dream_forwards"]),
                        "tree_steps_used": sum(item["tree_used"] for item in trace),
                        "corrected_token_flips": sum(
                            item["corrected_token_flips"] for item in trace
                        ),
                        "mean_tree_depth": statistics.mean(
                            item["tree_depth"] for item in trace if item["candidate_size"] > 0
                        ),
                        "baseline_output": baseline.texts[0],
                        "treepc_output": treepc.texts[0],
                        "baseline_scoring_error": baseline_error,
                        "treepc_scoring_error": treepc_error,
                        "trace": trace,
                    }
                    rows.append(row)
                    append_jsonl(rows_path, row)
                    print(
                        f"learned TreePC {dataset_name} {steps} NFE {ordinal}/{len(indices)} "
                        f"base={baseline_pass} treepc={treepc_pass} "
                        f"flips={row['corrected_token_flips']}",
                        flush=True,
                    )
    finally:
        adapter.close()
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["steps"])].append(row)
    summary_rows = []
    for (dataset, steps), group in sorted(grouped.items()):
        summary_rows.append(
            {
                "dataset": dataset,
                "steps": steps,
                "sample_size": len(group),
                "baseline_accuracy": statistics.mean(row["baseline_pass"] for row in group),
                "treepc_accuracy": statistics.mean(row["treepc_pass"] for row in group),
                "quality_delta": statistics.mean(row["quality_delta"] for row in group),
                "baseline_latency_mean_s": statistics.mean(
                    row["baseline_latency_s"] for row in group
                ),
                "treepc_latency_mean_s": statistics.mean(row["treepc_latency_s"] for row in group),
                "dependency_head_mean_s": statistics.mean(
                    row["dependency_head_s"] for row in group
                ),
                "mst_mean_s": statistics.mean(row["mst_s"] for row in group),
                "correction_head_mean_s": statistics.mean(
                    row["correction_head_s"] for row in group
                ),
                "mean_corrected_token_flips": statistics.mean(
                    row["corrected_token_flips"] for row in group
                ),
            }
        )
    write_json(
        output_dir / "summary.json",
        {
            "schema": "treepc.stage3_online_micro.v1",
            "internal_poc_only": True,
            "standard_benchmark_claim_allowed": False,
            "dependency_threshold": args.dependency_threshold,
            "global_scale": bundle.correction_head.global_scale_value(),
            "rows": summary_rows,
        },
    )


if __name__ == "__main__":
    main()
