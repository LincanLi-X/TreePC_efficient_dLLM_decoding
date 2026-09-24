#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from treepc.data.datasets import BenchmarkDataset, validate_large_scale_manifest
from treepc.decoding.learned_treepc import learned_treepc_generate
from treepc.dream.adapter import DreamAdapter
from treepc.dream.generation import custom_independent_generate
from treepc.dream.loader import load_dream
from treepc.evaluation.humaneval import evaluate_humaneval
from treepc.evaluation.task_metrics import gsm8k_exact_match
from treepc.models.pc_lora import load_pc_lora
from treepc.models.treepc_bundle import TreePCBundle
from treepc.utils.io import append_jsonl, write_json
from treepc.utils.seed import seed_everything

MAX_NEW_TOKENS = {"gsm8k": 256, "humaneval": 512}


def task_score(dataset: str, sample: dict[str, Any], output: str) -> tuple[bool, str | None]:
    if dataset == "gsm8k":
        return gsm8k_exact_match(output, sample["answer"]), None
    result = evaluate_humaneval(sample["prompt"], output, sample["test"], sample["entry_point"])
    return bool(result["passed"]), result["error"]


def run_phase(
    args: argparse.Namespace,
    *,
    pc: bool,
    manifest: dict[str, Any],
    rows_path: Path,
    completed: set[tuple[str, str, int, str]],
) -> list[dict[str, Any]]:
    loaded = load_dream(args.device)
    if pc:
        loaded.model = load_pc_lora(loaded.model, args.pc_lora)
    adapter = DreamAdapter(loaded)
    dependency_checkpoint = (
        args.dependency_checkpoint
        if pc or args.frozen_dependency_checkpoint is None
        else args.frozen_dependency_checkpoint
    )
    correction_checkpoint = (
        args.correction_checkpoint
        if pc or args.frozen_correction_checkpoint is None
        else args.frozen_correction_checkpoint
    )
    dependency_threshold = (
        args.dependency_threshold
        if pc or args.frozen_dependency_threshold is None
        else args.frozen_dependency_threshold
    )
    bundle = TreePCBundle.from_checkpoints(
        adapter,
        dependency_checkpoint,
        correction_checkpoint,
        dependency_threshold=dependency_threshold,
    )
    rows: list[dict[str, Any]] = []
    available_methods = (
        [("pc_only", None), ("pc_local_treepc", "local"), ("pc_learned_treepc", "learned")]
        if pc
        else [("dream_baseline", None), ("frozen_treepc", "learned")]
    )
    methods = [item for item in available_methods if not args.methods or item[0] in args.methods]
    try:
        for dataset_name in args.datasets:
            dataset = BenchmarkDataset(dataset_name)
            if manifest.get("schema_version") == 4:
                indices = manifest["datasets"][dataset_name][f"{args.evaluation_split}_indices"]
            else:
                indices = manifest["datasets"][dataset_name]["quality_indices"]
            if args.per_dataset is not None:
                indices = indices[args.sample_offset : args.sample_offset + args.per_dataset]
            else:
                indices = indices[args.sample_offset :]
            indices = indices[args.shard_index :: args.num_shards]
            for steps in args.steps:
                for index in indices:
                    sample = dataset[index]
                    prompt = dataset.prompt(index)
                    for method, edge_source in methods:
                        key = (dataset_name, sample["sample_id"], steps, method)
                        if key in completed:
                            continue
                        seed_everything(args.seed + index + steps * 10_000)
                        torch.cuda.reset_peak_memory_stats(adapter.device)
                        if edge_source is None:
                            result = custom_independent_generate(
                                adapter,
                                prompt,
                                steps=steps,
                                max_new_tokens=MAX_NEW_TOKENS[dataset_name],
                            )
                            trace: list[dict[str, Any]] = []
                            timings = {
                                "dependency_head_s": 0.0,
                                "mst_s": 0.0,
                                "correction_head_s": 0.0,
                                "dream_forwards": float(steps),
                            }
                        else:
                            result, trace, timings = learned_treepc_generate(
                                adapter,
                                bundle,
                                prompt,
                                steps=steps,
                                max_new_tokens=MAX_NEW_TOKENS[dataset_name],
                                edge_source=edge_source,
                            )
                        passed, error = task_score(dataset_name, sample, result.texts[0])
                        output_tokens = len(
                            adapter.tokenizer.encode(
                                result.texts[0], add_special_tokens=False
                            )
                        )
                        row = {
                            "dataset": dataset_name,
                            "sample_id": sample["sample_id"],
                            "dataset_index": index,
                            "steps": steps,
                            "method": method,
                            "manifest_fingerprint": manifest.get("fingerprint"),
                            "dependency_threshold": bundle.dependency_threshold,
                            "global_scale": bundle.correction_head.global_scale_value(),
                            "gpu_name": torch.cuda.get_device_name(adapter.device),
                            "passed": passed,
                            "latency_s": result.latency_seconds,
                            "output_tokens": output_tokens,
                            "output_tokens_per_s": output_tokens
                            / max(result.latency_seconds, 1e-12),
                            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(
                                adapter.device
                            )
                            / (1024**2),
                            "dream_nfe": int(timings["dream_forwards"]),
                            "dependency_head_s": timings["dependency_head_s"],
                            "mst_s": timings["mst_s"],
                            "correction_head_s": timings["correction_head_s"],
                            "corrected_token_flips": sum(
                                item["corrected_token_flips"] for item in trace
                            ),
                            "candidate_size_mean": statistics.mean(
                                item["candidate_size"] for item in trace
                            )
                            if trace
                            else 0.0,
                            "tree_depth_max": max(
                                (item["tree_depth"] for item in trace), default=0
                            ),
                            "tree_used_rate": statistics.mean(
                                item["tree_used"] for item in trace
                            )
                            if trace
                            else 0.0,
                            "output": result.texts[0],
                            "scoring_error": error,
                            "trace": trace,
                        }
                        rows.append(row)
                        append_jsonl(rows_path, row)
                        completed.add(key)
                        print(
                            f"stage4 {dataset_name} {steps} {method} "
                            f"sample={sample['sample_id']} pass={passed}",
                            flush=True,
                        )
    finally:
        adapter.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Step-4 PC and TreePC micro comparison")
    parser.add_argument("--pc-lora", required=True)
    parser.add_argument("--dependency-checkpoint", required=True)
    parser.add_argument("--correction-checkpoint", required=True)
    parser.add_argument("--frozen-dependency-checkpoint")
    parser.add_argument("--frozen-correction-checkpoint")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--per-dataset", type=int)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--evaluation-split", choices=("head_test", "pc_test"), default="head_test")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--phase", choices=("both", "base", "pc"), default="both")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "dream_baseline",
            "frozen_treepc",
            "pc_only",
            "pc_local_treepc",
            "pc_learned_treepc",
        ),
        help="Optional method subset; incompatible methods are ignored in each phase.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("gsm8k", "humaneval"),
        default=["gsm8k", "humaneval"],
    )
    parser.add_argument(
        "--dependency-threshold",
        type=float,
        help="Override the validation-calibrated threshold stored in the correction checkpoint.",
    )
    parser.add_argument("--frozen-dependency-threshold", type=float)
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid --shard-index/--num-shards")
    output_dir = Path(args.output_dir)
    rows_path = output_dir / "rows.jsonl"
    if rows_path.exists() and not args.resume:
        raise FileExistsError(rows_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("schema_version") == 4:
        validate_large_scale_manifest(manifest)
    rows: list[dict[str, Any]] = []
    if rows_path.exists():
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    completed = {
        (row["dataset"], row["sample_id"], int(row["steps"]), row["method"])
        for row in rows
    }
    if args.phase in {"both", "base"}:
        rows.extend(
            run_phase(
                args,
                pc=False,
                manifest=manifest,
                rows_path=rows_path,
                completed=completed,
            )
        )
    if args.phase in {"both", "pc"}:
        rows.extend(
            run_phase(
                args,
                pc=True,
                manifest=manifest,
                rows_path=rows_path,
                completed=completed,
            )
        )
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["steps"], row["method"])].append(row)
    summary_rows = []
    for (dataset, steps, method), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "dataset": dataset,
                "steps": steps,
                "method": method,
                "sample_size": len(values),
                "accuracy": statistics.mean(row["passed"] for row in values),
                "latency_mean_s": statistics.mean(row["latency_s"] for row in values),
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
            }
        )
    write_json(
        output_dir / "summary.json",
        {
            "schema": "treepc.stage4_micro.v1",
            "internal_poc_only": True,
            "standard_benchmark_claim_allowed": False,
            "rows": summary_rows,
        },
    )


if __name__ == "__main__":
    main()
