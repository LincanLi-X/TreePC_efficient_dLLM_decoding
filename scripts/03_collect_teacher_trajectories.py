#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from treepc.data.cache_schema import validate_trajectory_bundle
from treepc.data.datasets import (
    BenchmarkDataset,
    build_oracle_manifest,
    build_stage3_manifest,
    extend_oracle_manifest,
    validate_large_scale_manifest,
)
from treepc.data.trajectory import collect_teacher_trajectory
from treepc.dream.adapter import DreamAdapter
from treepc.dream.loader import load_dream
from treepc.utils.io import sha256_file, write_json
from treepc.utils.seed import seed_everything

MAX_NEW_TOKENS = {"gsm8k": 256, "humaneval": 512}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect frozen Dream teacher trajectories")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--stage1-manifest", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int, default=2026)
    prepare.add_argument("--gsm8k-size", type=int, default=2)
    prepare.add_argument("--humaneval-size", type=int, default=2)
    prepare.add_argument("--existing-manifest")
    prepare_stage3 = sub.add_parser("prepare-stage3")
    prepare_stage3.add_argument("--stage1-manifest", required=True)
    prepare_stage3.add_argument("--stage2-manifest", required=True)
    prepare_stage3.add_argument("--output", required=True)
    prepare_stage3.add_argument("--seed", type=int, default=3030)
    prepare_stage3.add_argument("--gsm8k-train-size", type=int, default=24)
    prepare_stage3.add_argument("--gsm8k-validation-size", type=int, default=8)
    prepare_stage3.add_argument("--humaneval-train-size", type=int, default=24)
    prepare_stage3.add_argument("--humaneval-validation-size", type=int, default=8)
    collect = sub.add_parser("collect")
    collect.add_argument("--dataset", choices=["gsm8k", "humaneval"], required=True)
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--summary", required=True)
    collect.add_argument("--device", default="auto")
    collect.add_argument("--teacher-steps", type=int, default=256)
    collect.add_argument("--states-per-example", type=int, default=4)
    collect.add_argument("--candidate-size", type=int, default=8)
    collect.add_argument("--top-k", type=int, default=16)
    collect.add_argument("--seed", type=int, default=2026)
    collect.add_argument("--resume", action="store_true")
    collect.add_argument("--partition", choices=["all", "added"], default="all")
    collect.add_argument("--split", choices=["train", "validation", "test"])
    collect.add_argument("--shard-index", type=int, default=0)
    collect.add_argument("--num-shards", type=int, default=1)
    merge = sub.add_parser("merge")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--summaries", nargs="+", required=True)
    merge.add_argument("--manifest", required=True)
    merge.add_argument("--dataset", choices=["gsm8k", "humaneval"], required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--summary", required=True)
    return parser.parse_args()


def collect(args: argparse.Namespace) -> None:
    output = Path(args.output)
    summary_path = Path(args.summary)
    if output.exists() or summary_path.exists():
        if args.resume and output.exists() and summary_path.exists():
            print(f"trajectory already complete: {output}")
            return
        raise FileExistsError(output if output.exists() else summary_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    training_eligible = manifest.get("eligible_for_head_training") is True
    if schema_version == 2 and training_eligible:
        raise ValueError("Stage-2 manifests cannot be training eligible")
    if schema_version == 3 and not training_eligible:
        raise ValueError("Stage-3 manifests must be training eligible")
    if schema_version not in {2, 3, 4}:
        raise ValueError("Expected a Stage-2, Stage-3, or large-scale manifest")
    if schema_version == 4:
        validate_large_scale_manifest(manifest)
        if args.split not in {"train", "validation", "test"}:
            raise ValueError("--split train|validation|test is required for a large-scale manifest")
        indices = manifest["datasets"][args.dataset][f"pc_{args.split}_indices"]
        partition = f"pc_{args.split}"
        training_eligible = True
    elif schema_version == 3:
        if args.split not in {"train", "validation"}:
            raise ValueError("--split train|validation is required for a Stage-3 manifest")
        indices = manifest["datasets"][args.dataset][f"{args.split}_indices"]
        partition = args.split
    else:
        partition_key = "indices" if args.partition == "all" else "added_indices"
        indices = manifest["datasets"][args.dataset][partition_key]
        partition = args.partition
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index/count")
    indices = indices[args.shard_index :: args.num_shards]
    dataset = BenchmarkDataset(args.dataset)
    adapter = DreamAdapter(load_dream(args.device))
    records = []
    examples = []
    started = time.perf_counter()
    try:
        for ordinal, index in enumerate(indices, 1):
            sample = dataset[index]
            seed_everything(args.seed + index)
            example_records, final_tokens, final_text = collect_teacher_trajectory(
                adapter,
                dataset.prompt(index),
                sample["sample_id"],
                index,
                teacher_steps=args.teacher_steps,
                max_new_tokens=MAX_NEW_TOKENS[args.dataset],
                states_per_example=args.states_per_example,
                candidate_size=args.candidate_size,
                top_k=args.top_k,
                seed=args.seed + index,
            )
            records.extend(example_records)
            examples.append(
                {
                    "sample_id": sample["sample_id"],
                    "dataset_index": index,
                    "final_text": final_text,
                    "final_token_count": int(final_tokens.numel()),
                }
            )
            print(
                f"teacher {args.dataset}: {ordinal}/{len(indices)} records={len(example_records)}",
                flush=True,
            )
    finally:
        adapter.close()
    bundle = {
        "schema": "treepc.teacher_trajectory.v1",
        "dataset": args.dataset,
        "diagnostic_only": not training_eligible,
        "eligible_for_head_training": training_eligible,
        "teacher_steps": args.teacher_steps,
        "states_per_example": args.states_per_example,
        "candidate_size": args.candidate_size,
        "top_k": args.top_k,
        "rq1_marginal_targets": "all_masked_positions_topk_plus_other",
        "manifest_fingerprint": manifest["fingerprint"],
        "partition": partition,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "records": records,
    }
    validate_trajectory_bundle(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    summary = {
        "kind": "teacher_trajectory",
        "dataset": args.dataset,
        "sample_size": len(indices),
        "record_count": len(records),
        "teacher_steps": args.teacher_steps,
        "states_per_example": args.states_per_example,
        "candidate_size": args.candidate_size,
        "rq1_marginal_targets": "all_masked_positions_topk_plus_other",
        "runtime_s": time.perf_counter() - started,
        "cache_path": str(output),
        "cache_sha256": sha256_file(output),
        "diagnostic_only": not training_eligible,
        "eligible_for_head_training": training_eligible,
        "partition": partition,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "examples": examples,
    }
    write_json(summary_path, summary)


def merge(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    first_probe = torch.load(args.inputs[0], map_location="cpu", weights_only=False)
    partition = first_probe.get("partition", "all")
    if manifest.get("schema_version") == 4:
        validate_large_scale_manifest(manifest)
        desired_ids = manifest["datasets"][args.dataset][f"{partition}_sample_ids"]
    elif manifest.get("schema_version") == 3:
        desired_ids = manifest["datasets"][args.dataset][f"{partition}_sample_ids"]
    else:
        desired_ids = manifest["datasets"][args.dataset]["sample_ids"]
    bundles = [torch.load(path, map_location="cpu", weights_only=False) for path in args.inputs]
    for bundle in bundles:
        validate_trajectory_bundle(bundle)
        if bundle["dataset"] != args.dataset:
            raise ValueError("Cannot merge different datasets")
    record_by_key = {}
    for bundle in bundles:
        for record in bundle["records"]:
            key = (record["sample_id"], int(record["step_index"]))
            record_by_key[key] = record
    order = {sample_id: index for index, sample_id in enumerate(desired_ids)}
    records = sorted(record_by_key.values(), key=lambda row: (order[row["sample_id"]], row["step_index"]))
    actual_ids = {record["sample_id"] for record in records}
    if actual_ids != set(desired_ids):
        raise ValueError(f"Merged sample IDs differ from manifest: {actual_ids ^ set(desired_ids)}")
    summaries = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.summaries]
    example_by_id = {
        example["sample_id"]: example
        for summary in summaries
        for example in summary["examples"]
    }
    examples = [example_by_id[sample_id] for sample_id in desired_ids]
    first = bundles[0]
    bundle = {
        "schema": "treepc.teacher_trajectory.v1",
        "dataset": args.dataset,
        "diagnostic_only": bool(first.get("diagnostic_only", True)),
        "eligible_for_head_training": bool(first.get("eligible_for_head_training", False)),
        "partition": partition,
        "teacher_steps": first["teacher_steps"],
        "states_per_example": first["states_per_example"],
        "candidate_size": first["candidate_size"],
        "top_k": first["top_k"],
        "rq1_marginal_targets": first.get(
            "rq1_marginal_targets", "legacy_candidate_positions_only"
        ),
        "manifest_fingerprint": manifest["fingerprint"],
        "records": records,
    }
    validate_trajectory_bundle(bundle)
    output = Path(args.output)
    if output.exists() or Path(args.summary).exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    write_json(
        args.summary,
        {
            "kind": "teacher_trajectory_merged",
            "dataset": args.dataset,
            "sample_size": len(desired_ids),
            "record_count": len(records),
            "teacher_steps": first["teacher_steps"],
            "states_per_example": first["states_per_example"],
            "candidate_size": first["candidate_size"],
            "rq1_marginal_targets": first.get(
                "rq1_marginal_targets", "legacy_candidate_positions_only"
            ),
            "cache_path": str(output),
            "cache_sha256": sha256_file(output),
            "diagnostic_only": bool(first.get("diagnostic_only", True)),
            "eligible_for_head_training": bool(first.get("eligible_for_head_training", False)),
            "partition": partition,
            "examples": examples,
        },
    )


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        if args.existing_manifest:
            manifest = extend_oracle_manifest(
                args.existing_manifest,
                args.stage1_manifest,
                args.seed,
                args.gsm8k_size,
                args.humaneval_size,
            )
        else:
            manifest = build_oracle_manifest(
                args.stage1_manifest, args.seed, args.gsm8k_size, args.humaneval_size
            )
        write_json(args.output, manifest)
        print(f"wrote diagnostic manifest: {args.output}")
    elif args.command == "prepare-stage3":
        manifest = build_stage3_manifest(
            args.stage1_manifest,
            args.stage2_manifest,
            args.seed,
            args.gsm8k_train_size,
            args.gsm8k_validation_size,
            args.humaneval_train_size,
            args.humaneval_validation_size,
        )
        write_json(args.output, manifest)
        print(f"wrote internal Stage-3 manifest: {args.output}")
    elif args.command == "collect":
        collect(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
