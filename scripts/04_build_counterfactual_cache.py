#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from treepc.data.cache_schema import validate_counterfactual_bundle, validate_trajectory_bundle
from treepc.data.counterfactual import build_counterfactual_record
from treepc.dream.adapter import DreamAdapter
from treepc.dream.loader import load_dream
from treepc.utils.io import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build multi-sample Dream counterfactual dependency labels"
    )
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--parent-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-inputs", nargs="+")
    args = parser.parse_args()
    output = Path(args.output)
    summary_path = Path(args.summary)
    if output.exists() or summary_path.exists():
        if args.resume and output.exists() and summary_path.exists():
            existing = torch.load(output, map_location="cpu", weights_only=False)
            if int(existing.get("parent_samples", 1)) != args.parent_samples:
                raise ValueError(
                    "Existing counterfactual cache uses a different parent sample count; "
                    "use a new output path"
                )
            print(f"counterfactual cache already complete: {output}")
            return
        raise FileExistsError(output if output.exists() else summary_path)
    if args.merge_inputs:
        bundles = [torch.load(path, map_location="cpu", weights_only=False) for path in args.merge_inputs]
        for bundle in bundles:
            validate_counterfactual_bundle(bundle)
        sample_counts = {int(bundle.get("parent_samples", 1)) for bundle in bundles}
        if len(sample_counts) != 1:
            raise ValueError("Cannot merge counterfactual caches with different sample counts")
        records = [record for bundle in bundles for record in bundle["records"]]
        records.sort(key=lambda row: (row["sample_id"], int(row["step_index"])))
        first = bundles[0]
        merged = {
            key: value
            for key, value in first.items()
            if key not in {"records", "shard_index", "num_shards"}
        }
        merged["records"] = records
        validate_counterfactual_bundle(merged)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged, output)
        write_json(
            summary_path,
            {
                "kind": "counterfactual_cache_merged",
                "dataset": merged["dataset"],
                "record_count": len(records),
                "counterfactual_forwards": sum(
                    record["candidate_positions"].numel()
                    * int(record.get("parent_samples", 1))
                    for record in records
                ),
                "cache_path": str(output),
                "cache_sha256": sha256_file(output),
                "diagnostic_only": merged["diagnostic_only"],
                "eligible_for_head_training": merged["eligible_for_head_training"],
                "partition": merged.get("partition"),
            },
        )
        return
    trajectory = torch.load(args.trajectory, map_location="cpu", weights_only=False)
    validate_trajectory_bundle(trajectory)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index/count")
    selected_records = trajectory["records"][args.shard_index :: args.num_shards]
    adapter = DreamAdapter(load_dream(args.device))
    records = []
    started = time.perf_counter()
    try:
        for ordinal, record in enumerate(selected_records, 1):
            records.append(
                build_counterfactual_record(
                    adapter,
                    record,
                    top_k=args.top_k,
                    seed=args.seed,
                    parent_samples=args.parent_samples,
                )
            )
            print(
                f"counterfactual {trajectory['dataset']}: {ordinal}/{len(selected_records)}",
                flush=True,
            )
    finally:
        adapter.close()
    bundle = {
        "schema": "treepc.counterfactual.v1",
        "dataset": trajectory["dataset"],
        "diagnostic_only": bool(trajectory.get("diagnostic_only", True)),
        "eligible_for_head_training": bool(trajectory.get("eligible_for_head_training", False)),
        "target_type": f"mc_kl_q_conditional_to_q_base_s{args.parent_samples}",
        "parent_sampling_distribution": "teacher_posterior",
        "parent_samples": args.parent_samples,
        "top_k": args.top_k,
        "trajectory_sha256": sha256_file(args.trajectory),
        "partition": trajectory.get("partition"),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "records": records,
    }
    validate_counterfactual_bundle(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    directed_values = torch.cat(
        [record["directed_dependency"].flatten() for record in records]
    )
    summary = {
        "kind": "counterfactual_cache",
        "dataset": trajectory["dataset"],
        "record_count": len(records),
        "parent_samples": args.parent_samples,
        "parent_sampling_distribution": "teacher_posterior",
        "counterfactual_forwards": sum(
            record["candidate_positions"].numel() * args.parent_samples for record in records
        ),
        "directed_dependency_mean": float(directed_values.mean().item()),
        "directed_dependency_max": float(directed_values.max().item()),
        "runtime_s": time.perf_counter() - started,
        "cache_path": str(output),
        "cache_sha256": sha256_file(output),
        "diagnostic_only": bool(trajectory.get("diagnostic_only", True)),
        "eligible_for_head_training": bool(trajectory.get("eligible_for_head_training", False)),
        "partition": trajectory.get("partition"),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
