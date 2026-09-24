#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from treepc.data.datasets import BenchmarkDataset, build_subset_manifest, load_manifest
from treepc.dream.adapter import DreamAdapter
from treepc.dream.generation import compare_generation_traces, custom_independent_generate, official_generate
from treepc.dream.loader import load_dream
from treepc.evaluation.humaneval import evaluate_humaneval
from treepc.evaluation.statistics import latency_summary
from treepc.evaluation.task_metrics import gsm8k_exact_match
from treepc.utils.io import append_jsonl, write_json
from treepc.utils.seed import seed_everything

STEPS = (4, 8, 16, 32)
MAX_NEW_TOKENS = {"gsm8k": 256, "humaneval": 512}


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=["gsm8k", "humaneval"], required=True)
    parser.add_argument("--steps", choices=STEPS, type=int, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-revision")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Revised TreePC stage-1 Dream baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--gsm8k-size", type=int, default=20)
    prepare.add_argument("--humaneval-size", type=int, default=10)
    prepare.add_argument("--validate-existing", action="store_true")
    parity = sub.add_parser("parity")
    add_run_args(parity)
    quality = sub.add_parser("quality")
    add_run_args(quality)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--input-dir", required=True)
    aggregate.add_argument("--output-markdown", required=True)
    aggregate.add_argument("--output-csv", required=True)
    return parser.parse_args()


def selected_indices(args: argparse.Namespace, partition: str) -> tuple[dict[str, Any], list[int]]:
    manifest = load_manifest(args.manifest)
    if manifest["seed"] != args.seed:
        raise ValueError("CLI seed does not match the frozen subset manifest")
    indices = list(manifest["datasets"][args.dataset][f"{partition}_indices"])
    if args.max_examples is not None:
        indices = indices[: args.max_examples]
    return manifest, indices


def output_paths(args: argparse.Namespace, kind: str) -> tuple[Path, Path]:
    directory = Path(args.output_dir) / kind / args.dataset / f"steps_{args.steps}"
    return directory / "rows.jsonl", directory / "summary.json"


def run_parity(args: argparse.Namespace) -> None:
    manifest, indices = selected_indices(args, "parity")
    raw_path, summary_path = output_paths(args, "parity")
    if summary_path.exists() and not args.resume:
        raise FileExistsError(summary_path)
    if args.dry_run:
        print(json.dumps({"indices": indices, "raw": str(raw_path)}, indent=2))
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and not args.resume:
        raise FileExistsError(raw_path)
    dataset = BenchmarkDataset(args.dataset)
    loaded = load_dream(args.device, model_revision=args.model_revision)
    adapter = DreamAdapter(loaded)
    rows: list[dict[str, Any]] = []
    try:
        for ordinal, index in enumerate(indices, 1):
            seed_everything(args.seed + index)
            prompt = dataset.prompt(index)
            official = official_generate(
                adapter,
                prompt,
                steps=args.steps,
                max_new_tokens=MAX_NEW_TOKENS[args.dataset],
                capture_trace=True,
            )
            seed_everything(args.seed + index)
            custom = custom_independent_generate(
                adapter,
                prompt,
                steps=args.steps,
                max_new_tokens=MAX_NEW_TOKENS[args.dataset],
                capture_trace=True,
            )
            comparison = compare_generation_traces(official, custom)
            row = {
                "ordinal": ordinal,
                "dataset_index": index,
                "sample_id": dataset[index]["sample_id"],
                "dataset": args.dataset,
                "steps": args.steps,
                **comparison,
                "official_latency_s": official.latency_seconds,
                "custom_latency_s": custom.latency_seconds,
            }
            rows.append(row)
            append_jsonl(raw_path, row)
            print(
                f"parity {args.dataset} {args.steps}: {ordinal}/{len(indices)} "
                f"passed={row['passed']}",
                flush=True,
            )
    finally:
        adapter.close()
    summary = {
        "kind": "official_custom_parity",
        "dataset": args.dataset,
        "steps": args.steps,
        "sample_size": len(rows),
        "passed_examples": sum(row["passed"] for row in rows),
        "pass_rate": sum(row["passed"] for row in rows) / len(rows),
        "all_passed": all(row["passed"] for row in rows),
        "sample_ids": [row["sample_id"] for row in rows],
        "manifest_fingerprint": manifest["fingerprint"],
        "model": loaded.metadata,
        "full_step_reference_enabled": False,
    }
    write_json(summary_path, summary)
    if not summary["all_passed"]:
        raise SystemExit(2)


def run_quality(args: argparse.Namespace) -> None:
    manifest, indices = selected_indices(args, "quality")
    raw_path, summary_path = output_paths(args, "quality")
    if summary_path.exists() and not args.resume:
        raise FileExistsError(summary_path)
    if args.dry_run:
        print(json.dumps({"indices": indices, "raw": str(raw_path)}, indent=2))
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and not args.resume:
        raise FileExistsError(raw_path)
    dataset = BenchmarkDataset(args.dataset)
    loaded = load_dream(args.device, model_revision=args.model_revision)
    adapter = DreamAdapter(loaded)
    rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(adapter.device)
    run_started = time.perf_counter()
    try:
        # One untimed warm-up with the exact workload shape.
        seed_everything(args.seed + indices[0])
        warmup = official_generate(
            adapter,
            dataset.prompt(indices[0]),
            steps=args.steps,
            max_new_tokens=MAX_NEW_TOKENS[args.dataset],
            capture_trace=True,
        )
        commit_count_by_step = [int(trace.commit_positions.numel()) for trace in warmup.traces]
        for ordinal, index in enumerate(indices, 1):
            sample = dataset[index]
            generation_error = None
            seed_everything(args.seed + index)
            try:
                result = official_generate(
                    adapter,
                    dataset.prompt(index),
                    steps=args.steps,
                    max_new_tokens=MAX_NEW_TOKENS[args.dataset],
                    capture_trace=False,
                )
                output = result.texts[0]
            except Exception as error:  # preserve the row and expose the failure in the summary
                generation_error = repr(error)
                output = ""
                result = None
            scoring_started = time.perf_counter()
            if args.dataset == "gsm8k":
                passed = gsm8k_exact_match(output, sample["answer"])
                scoring_error = None
                completion = None
            else:
                score = evaluate_humaneval(
                    sample["prompt"], output, sample["test"], sample["entry_point"], timeout_seconds=5
                )
                passed = bool(score["passed"])
                scoring_error = score["error"]
                completion = score["completion"]
            scoring_latency = time.perf_counter() - scoring_started
            generation_latency = result.latency_seconds if result is not None else 0.0
            token_count = len(adapter.tokenizer.encode(output, add_special_tokens=False))
            row = {
                "ordinal": ordinal,
                "dataset_index": index,
                "sample_id": sample["sample_id"],
                "dataset": args.dataset,
                "model": "Dream-org/Dream-v0-Instruct-7B",
                "decoder": "official_diffusion_generate",
                "steps": args.steps,
                "max_new_tokens": MAX_NEW_TOKENS[args.dataset],
                "passed": passed,
                "generation_error": generation_error,
                "scoring_error": scoring_error,
                "generation_latency_s": generation_latency,
                "scoring_latency_s": scoring_latency,
                "output_tokens": token_count,
                "model_output": output,
                "completion": completion,
            }
            rows.append(row)
            append_jsonl(raw_path, row)
            print(
                f"quality {args.dataset} {args.steps}: {ordinal}/{len(indices)} "
                f"passed={passed} latency={generation_latency:.3f}s",
                flush=True,
            )
    finally:
        peak_memory = torch.cuda.max_memory_allocated(adapter.device) / 2**30
        adapter.close()
    latencies = [row["generation_latency_s"] for row in rows if row["generation_error"] is None]
    total_tokens = sum(row["output_tokens"] for row in rows)
    total_latency = sum(latencies)
    correct = sum(row["passed"] for row in rows)
    metric_name = "exact_match" if args.dataset == "gsm8k" else "pass_at_1"
    summary = {
        "kind": "dream_baseline_quality",
        "dataset": args.dataset,
        "model": "Dream-org/Dream-v0-Instruct-7B",
        "decoder": "official_diffusion_generate",
        "steps": args.steps,
        "nfe": args.steps,
        "sample_size": len(rows),
        "correct": correct,
        metric_name: correct / len(rows),
        "accuracy": correct / len(rows),
        "generation_errors": sum(row["generation_error"] is not None for row in rows),
        "scoring_failures": sum(row["scoring_error"] is not None for row in rows),
        **latency_summary(latencies),
        "wall_runtime_s": time.perf_counter() - run_started,
        "output_tokens_total": total_tokens,
        "output_tokens_mean": statistics.mean(row["output_tokens"] for row in rows),
        "output_tokens_per_s": total_tokens / total_latency if total_latency else 0.0,
        "commit_count_by_step": commit_count_by_step,
        "mean_commit_per_step": statistics.mean(commit_count_by_step),
        "peak_memory_gib": peak_memory,
        "sample_ids": [row["sample_id"] for row in rows],
        "selected_indices": indices,
        "manifest_fingerprint": manifest["fingerprint"],
        "model_metadata": loaded.metadata,
        "algorithm": "entropy",
        "temperature": 0.0,
        "batch_size": 1,
        "max_new_tokens": MAX_NEW_TOKENS[args.dataset],
        "full_step_reference_enabled": False,
    }
    write_json(summary_path, summary)


def aggregate(args: argparse.Namespace) -> None:
    root = Path(args.input_dir)
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in root.glob("quality/*/steps_*/summary.json")
    ]
    parity = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in root.glob("parity/*/steps_*/summary.json")
    ]
    if len(summaries) != 8 or len(parity) != 8:
        raise RuntimeError(
            f"Expected 8 quality and 8 parity summaries, found {len(summaries)} and {len(parity)}"
        )
    summaries.sort(key=lambda row: (row["dataset"], row["steps"]))
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "steps", "sample_size", "correct", "accuracy", "latency_mean_s",
        "latency_median_s", "latency_p90_s", "output_tokens_per_s", "peak_memory_gib",
        "mean_commit_per_step", "generation_errors", "scoring_failures",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in summaries)
    lines = [
        "# Stage 1 — Dream-7B-Instruct micro baseline",
        "",
        "No 512-step reference was run. Quality uses the official `diffusion_generate()` path; "
        "the custom loop is accepted only after deterministic step parity.",
        "",
        "## Quality",
        "",
        "| Dataset | NFE | Correct | Metric | Mean latency | Median | P90 | Tok/s | "
        "Mean commits/step | Peak memory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset']} | {row['steps']} | {row['correct']}/{row['sample_size']} | "
            f"{100 * row['accuracy']:.2f}% | {row['latency_mean_s']:.3f}s | "
            f"{row['latency_median_s']:.3f}s | {row['latency_p90_s']:.3f}s | "
            f"{row['output_tokens_per_s']:.2f} | {row['mean_commit_per_step']:.2f} | "
            f"{row['peak_memory_gib']:.2f} GiB |"
        )
    lines.extend(
        [
            "", "## Official/custom parity", "",
            "| Dataset | NFE | Passed examples | Pass rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sorted(parity, key=lambda value: (value["dataset"], value["steps"])):
        lines.append(
            f"| {row['dataset']} | {row['steps']} | {row['passed_examples']}/{row['sample_size']} | "
            f"{100 * row['pass_rate']:.1f}% |"
        )
    lines.extend(
        [
            "", "## Audit", "",
            f"- Quality summaries: {len(summaries)}/8.",
            f"- Parity summaries: {len(parity)}/8.",
            f"- Generation errors: {sum(row['generation_errors'] for row in summaries)}.",
            f"- All parity groups passed: {all(row['all_passed'] for row in parity)}.",
            "- GSM8K and HumanEval are tiny deterministic subsets; these values are calibration "
            "data, not official scores.",
        ]
    )
    markdown_path = Path(args.output_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {markdown_path} and {csv_path}")


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        manifest = build_subset_manifest(args.seed, args.gsm8k_size, args.humaneval_size)
        output = Path(args.output)
        if args.validate_existing:
            if not output.is_file():
                raise FileNotFoundError(output)
            existing = json.loads(output.read_text(encoding="utf-8"))
            if existing != manifest:
                raise ValueError(
                    "Existing subset manifest does not match the requested seed, sample sizes, "
                    "or uploaded datasets; use a new TREEPC_RUN_DIR."
                )
            print(f"validated existing subset manifest: {output}")
        else:
            if output.exists():
                raise FileExistsError(output)
            write_json(output, manifest)
    elif args.command == "parity":
        run_parity(args)
    elif args.command == "quality":
        run_quality(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
