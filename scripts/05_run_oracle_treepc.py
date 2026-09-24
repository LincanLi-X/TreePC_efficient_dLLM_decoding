#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from treepc.data.cache_schema import validate_counterfactual_bundle, validate_trajectory_bundle
from treepc.data.datasets import BenchmarkDataset
from treepc.decoding.oracle_treepc import online_oracle_generate
from treepc.dream.adapter import DreamAdapter
from treepc.dream.generation import custom_independent_generate
from treepc.dream.loader import load_dream
from treepc.evaluation.humaneval import evaluate_humaneval
from treepc.evaluation.oracle_metrics import evaluate_cached_record
from treepc.evaluation.task_metrics import gsm8k_exact_match
from treepc.utils.io import append_jsonl, write_json
from treepc.utils.seed import seed_everything

MAX_NEW_TOKENS = {"gsm8k": 256, "humaneval": 512}


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def run_cached(args: argparse.Namespace) -> None:
    bundle = torch.load(args.cache, map_location="cpu", weights_only=False)
    validate_counterfactual_bundle(bundle)
    rows = [evaluate_cached_record(record, args.seed) for record in bundle["records"]]
    raw = Path(args.output_dir) / "cached" / bundle["dataset"] / "rows.jsonl"
    if raw.exists():
        raise FileExistsError(raw)
    for row in rows:
        append_jsonl(raw, row)
    summary = {
        "kind": "cached_state_oracle",
        "dataset": bundle["dataset"],
        "record_count": len(rows),
        "oracle_dependency_sum_mean": mean(rows, "oracle_dependency_sum"),
        "random_dependency_sum_mean": mean(rows, "random_dependency_sum"),
        "local_dependency_sum_mean": mean(rows, "local_dependency_sum"),
        "oracle_beats_random_rate": mean(
            [{"v": row["oracle_over_random_gain"] >= -1e-7} for row in rows], "v"
        ),
        "oracle_beats_local_rate": mean(
            [{"v": row["oracle_over_local_gain"] >= -1e-7} for row in rows], "v"
        ),
        "independent_conditional_kl_mean": mean(rows, "independent_conditional_kl"),
        "oracle_conditional_kl_mean": 0.0,
        "base_teacher_token_nll_mean": mean(rows, "base_teacher_token_nll"),
        "conditional_teacher_token_nll_mean": mean(rows, "conditional_teacher_token_nll"),
        "base_teacher_top1_agreement_mean": mean(rows, "base_teacher_top1_agreement"),
        "conditional_teacher_top1_agreement_mean": mean(
            rows, "conditional_teacher_top1_agreement"
        ),
        "mean_tail_mass": mean(rows, "mean_tail_mass"),
        "diagnostic_only": True,
    }
    write_json(Path(args.output_dir) / "cached" / bundle["dataset"] / "summary.json", summary)


def task_pass(dataset: str, sample: dict[str, Any], output: str) -> tuple[bool, str | None]:
    if dataset == "gsm8k":
        return gsm8k_exact_match(output, sample["answer"]), None
    score = evaluate_humaneval(sample["prompt"], output, sample["test"], sample["entry_point"])
    return bool(score["passed"]), score["error"]


def run_online(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    indices = manifest["datasets"][args.dataset]["indices"]
    trajectory = torch.load(args.trajectory, map_location="cpu", weights_only=False)
    validate_trajectory_bundle(trajectory)
    final_by_id = {}
    for record in trajectory["records"]:
        final_by_id.setdefault(record["sample_id"], record["final_teacher_tokens"])
    dataset = BenchmarkDataset(args.dataset)
    adapter = DreamAdapter(load_dream(args.device))
    rows = []
    raw = Path(args.output_dir) / "online" / args.dataset / "rows.jsonl"
    if raw.exists():
        raise FileExistsError(raw)
    try:
        for ordinal, index in enumerate(indices, 1):
            sample = dataset[index]
            prompt = dataset.prompt(index)
            seed_everything(args.seed + index)
            independent = custom_independent_generate(
                adapter,
                prompt,
                steps=args.steps,
                max_new_tokens=MAX_NEW_TOKENS[args.dataset],
            )
            seed_everything(args.seed + index)
            oracle, trace, extra_forwards = online_oracle_generate(
                adapter,
                prompt,
                steps=args.steps,
                max_new_tokens=MAX_NEW_TOKENS[args.dataset],
                seed=args.seed + index,
            )
            independent_pass, independent_error = task_pass(args.dataset, sample, independent.texts[0])
            oracle_pass, oracle_error = task_pass(args.dataset, sample, oracle.texts[0])
            teacher_final = final_by_id[sample["sample_id"]]
            prompt_length = adapter.encode_prompt(prompt)["input_ids"].shape[1]
            independent_agreement = independent.sequences.cpu()[0, prompt_length:].eq(
                teacher_final[prompt_length:]
            ).float().mean()
            oracle_agreement = oracle.sequences.cpu()[0, prompt_length:].eq(
                teacher_final[prompt_length:]
            ).float().mean()
            row = {
                "ordinal": ordinal,
                "dataset": args.dataset,
                "sample_id": sample["sample_id"],
                "dataset_index": index,
                "steps": args.steps,
                "independent_pass": independent_pass,
                "oracle_pass": oracle_pass,
                "task_delta": int(oracle_pass) - int(independent_pass),
                "independent_teacher_token_agreement": float(independent_agreement.item()),
                "oracle_teacher_token_agreement": float(oracle_agreement.item()),
                "teacher_agreement_delta": float((oracle_agreement - independent_agreement).item()),
                "independent_latency_s": independent.latency_seconds,
                "oracle_latency_s": oracle.latency_seconds,
                "oracle_extra_teacher_forwards": extra_forwards,
                "mean_tree_dependency_sum": statistics.mean(
                    item["tree_dependency_sum"] for item in trace
                ),
                "mean_tree_depth": statistics.mean(item["tree_depth"] for item in trace),
                "corrected_token_flips": sum(item["corrected_token_flips"] for item in trace),
                "independent_output": independent.texts[0],
                "oracle_output": oracle.texts[0],
                "independent_scoring_error": independent_error,
                "oracle_scoring_error": oracle_error,
                "trace": trace,
            }
            rows.append(row)
            append_jsonl(raw, row)
            print(
                f"online oracle {args.dataset}: {ordinal}/{len(indices)} "
                f"base={independent_pass} oracle={oracle_pass} extra={extra_forwards}",
                flush=True,
            )
    finally:
        adapter.close()
    summary = {
        "kind": "online_oracle",
        "dataset": args.dataset,
        "steps": args.steps,
        "sample_size": len(rows),
        "independent_correct": sum(row["independent_pass"] for row in rows),
        "oracle_correct": sum(row["oracle_pass"] for row in rows),
        "independent_accuracy": mean(rows, "independent_pass"),
        "oracle_accuracy": mean(rows, "oracle_pass"),
        "task_accuracy_delta": mean(rows, "oracle_pass") - mean(rows, "independent_pass"),
        "independent_teacher_token_agreement": mean(rows, "independent_teacher_token_agreement"),
        "oracle_teacher_token_agreement": mean(rows, "oracle_teacher_token_agreement"),
        "teacher_token_agreement_delta": mean(rows, "teacher_agreement_delta"),
        "independent_latency_mean_s": mean(rows, "independent_latency_s"),
        "oracle_latency_mean_s": mean(rows, "oracle_latency_s"),
        "oracle_extra_teacher_forwards_mean": mean(rows, "oracle_extra_teacher_forwards"),
        "diagnostic_only": True,
        "online_efficiency_claim": False,
    }
    write_json(Path(args.output_dir) / "online" / args.dataset / "summary.json", summary)


def aggregate(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    cached = [json.loads(path.read_text()) for path in root.glob("cached/*/summary.json")]
    online = [json.loads(path.read_text()) for path in root.glob("online/*/summary.json")]
    trajectory = [
        json.loads(path.read_text())
        for path in (root.parent / "trajectories").glob("*_summary.json")
    ]
    if len(cached) != 2 or len(online) != 2:
        raise RuntimeError(f"Expected two cached and online summaries, got {len(cached)}, {len(online)}")
    cached.sort(key=lambda row: row["dataset"])
    online.sort(key=lambda row: row["dataset"])
    trajectory.sort(key=lambda row: row["dataset"])
    teacher_quality = []
    for summary in trajectory:
        dataset = BenchmarkDataset(summary["dataset"])
        passed = [
            task_pass(summary["dataset"], dataset[item["dataset_index"]], item["final_text"])[0]
            for item in summary["examples"]
        ]
        teacher_quality.append(
            {
                "dataset": summary["dataset"],
                "correct": sum(passed),
                "sample_size": len(passed),
            }
        )
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "steps", "sample_size", "independent_accuracy", "oracle_accuracy",
        "task_accuracy_delta", "independent_teacher_token_agreement",
        "oracle_teacher_token_agreement", "teacher_token_agreement_delta",
        "oracle_extra_teacher_forwards_mean",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in online)
    lines = [
        "# Stage 2 — Oracle TreePC micro diagnostic",
        "",
        "> Diagnostic-only test-split samples. These caches must not train Step-3 heads.",
        "",
        "## Cached-state mechanism",
        "",
        "| Dataset | States | Independent conditional KL | Oracle conditional KL | "
        "Oracle edge sum | Random edge sum | Local edge sum |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cached:
        lines.append(
            f"| {row['dataset']} | {row['record_count']} | "
            f"{row['independent_conditional_kl_mean']:.6f} | "
            f"{row['oracle_conditional_kl_mean']:.6f} | "
            f"{row['oracle_dependency_sum_mean']:.6f} | "
            f"{row['random_dependency_sum_mean']:.6f} | "
            f"{row['local_dependency_sum_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Dataset | Base Teacher-token NLL | Conditional Teacher-token NLL | "
            "Base top-1 agreement | Conditional top-1 agreement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in cached:
        lines.append(
            f"| {row['dataset']} | {row['base_teacher_token_nll_mean']:.6f} | "
            f"{row['conditional_teacher_token_nll_mean']:.6f} | "
            f"{100 * row['base_teacher_top1_agreement_mean']:.2f}% | "
            f"{100 * row['conditional_teacher_top1_agreement_mean']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 128-step Teacher quality",
            "",
            "| Dataset | Correct |",
            "|---|---:|",
        ]
    )
    for row in teacher_quality:
        lines.append(f"| {row['dataset']} | {row['correct']}/{row['sample_size']} |")
    lines.extend(
        [
            "",
            "## Online task diagnostic (16 NFE)",
            "",
            "| Dataset | Samples | Independent | Oracle | Task delta | Teacher-token agreement delta | "
            "Base latency | Oracle latency | Extra Teacher forwards |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in online:
        lines.append(
            f"| {row['dataset']} | {row['sample_size']} | "
            f"{100 * row['independent_accuracy']:.1f}% | {100 * row['oracle_accuracy']:.1f}% | "
            f"{100 * row['task_accuracy_delta']:+.1f} pp | "
            f"{100 * row['teacher_token_agreement_delta']:+.2f} pp | "
            f"{row['independent_latency_mean_s']:.2f}s | {row['oracle_latency_mean_s']:.2f}s | "
            f"{row['oracle_extra_teacher_forwards_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The cached Oracle conditional is the stored Teacher conditional, so its conditional KL "
            "is zero by construction.",
            "- Oracle latency includes diagnostic Teacher counterfactual forwards and is not an online "
            "efficiency result.",
            "- The sample is intentionally tiny and can only establish a mechanism smoke signal, not a "
            "paper claim.",
        ]
    )
    report = Path(args.output_markdown)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_cached_expansion(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    cached = sorted(
        [json.loads(path.read_text()) for path in root.glob("cached/*/summary.json")],
        key=lambda row: row["dataset"],
    )
    trajectory = sorted(
        [
            json.loads(path.read_text())
            for path in (root.parent / "trajectories").glob("*_summary.json")
        ],
        key=lambda row: row["dataset"],
    )
    if len(cached) != 2 or len(trajectory) != 2:
        raise RuntimeError("Expected two cached and trajectory summaries")
    teacher_quality = {}
    for summary in trajectory:
        dataset = BenchmarkDataset(summary["dataset"])
        teacher_quality[summary["dataset"]] = sum(
            task_pass(summary["dataset"], dataset[item["dataset_index"]], item["final_text"])[0]
            for item in summary["examples"]
        )
    fields = [
        "dataset",
        "sample_size",
        "teacher_correct",
        "record_count",
        "independent_conditional_kl_mean",
        "oracle_conditional_kl_mean",
        "oracle_dependency_sum_mean",
        "random_dependency_sum_mean",
        "local_dependency_sum_mean",
        "base_teacher_token_nll_mean",
        "conditional_teacher_token_nll_mean",
    ]
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in cached:
            writer.writerow(
                {
                    **{key: row[key] for key in fields if key in row},
                    "sample_size": 10,
                    "teacher_correct": teacher_quality[row["dataset"]],
                }
            )
    lines = [
        "# Stage 2 — Expanded 128-step Teacher/cached Oracle diagnostic",
        "",
        "> Ten test-split examples per dataset; diagnostic only and ineligible for head training.",
        "",
        "| Dataset | Teacher quality | States | Independent conditional KL | Oracle edge sum | "
        "Random edge sum | Local edge sum |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cached:
        lines.append(
            f"| {row['dataset']} | {teacher_quality[row['dataset']]}/10 | {row['record_count']} | "
            f"{row['independent_conditional_kl_mean']:.6f} | "
            f"{row['oracle_dependency_sum_mean']:.6f} | "
            f"{row['random_dependency_sum_mean']:.6f} | "
            f"{row['local_dependency_sum_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The earlier two-example online 16-NFE Oracle diagnostic was not rerun; this expansion "
            "covers the requested complete 128-step Teacher trajectories and cached counterfactual labels.",
        ]
    )
    report = Path(args.output_markdown)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cached and online Oracle TreePC diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)
    cached = sub.add_parser("cached")
    cached.add_argument("--cache", required=True)
    cached.add_argument("--output-dir", required=True)
    cached.add_argument("--seed", type=int, default=2026)
    online = sub.add_parser("online")
    online.add_argument("--dataset", choices=["gsm8k", "humaneval"], required=True)
    online.add_argument("--manifest", required=True)
    online.add_argument("--trajectory", required=True)
    online.add_argument("--output-dir", required=True)
    online.add_argument("--device", default="auto")
    online.add_argument("--steps", type=int, default=16)
    online.add_argument("--seed", type=int, default=2026)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--output-dir", required=True)
    aggregate_parser.add_argument("--output-markdown", required=True)
    aggregate_parser.add_argument("--output-csv", required=True)
    cached_aggregate = sub.add_parser("aggregate-cached")
    cached_aggregate.add_argument("--output-dir", required=True)
    cached_aggregate.add_argument("--output-markdown", required=True)
    cached_aggregate.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "cached":
        run_cached(args)
    elif args.command == "online":
        run_online(args)
    elif args.command == "aggregate":
        aggregate(args)
    else:
        aggregate_cached_expansion(args)


if __name__ == "__main__":
    main()
