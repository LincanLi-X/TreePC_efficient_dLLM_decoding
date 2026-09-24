#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from treepc.data.cache_dataset import (
    CorrectionCacheDataset,
    DependencyCacheDataset,
    load_dream_embedding_weights,
    load_training_records,
)
from treepc.data.cache_schema import validate_counterfactual_bundle
from treepc.data.datasets import BenchmarkDataset, validate_large_scale_manifest
from treepc.data.onpolicy import collect_pc_mid_state, label_pc_onpolicy_record
from treepc.dream.adapter import DreamAdapter
from treepc.dream.loader import load_dream, load_model_config, resolve_model_dir
from treepc.models.correction_head import ConditionalCorrectionHead
from treepc.models.dependency_head import DependencyHead
from treepc.models.pc_lora import load_pc_lora
from treepc.training.correction_trainer import (
    calibrate_tree_gate,
    evaluate_correction_head,
    select_learned_tree_pairs,
    train_correction_head,
)
from treepc.training.dependency_trainer import evaluate_dependency_head, train_dependency_head
from treepc.utils.io import sha256_file, write_json
from treepc.utils.seed import seed_everything

MAX_NEW_TOKENS = {"gsm8k": 256, "humaneval": 512}


def collect(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    validate_large_scale_manifest(manifest)
    entry = manifest["datasets"][args.dataset]
    all_indices = entry[f"head_{args.split}_indices"]
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index/count")
    indices = all_indices[args.shard_index :: args.num_shards]
    output = Path(args.output)
    if output.exists():
        if args.resume:
            print(f"PC on-policy shard already exists: {output}")
            return
        raise FileExistsError(output)
    loaded = load_dream(args.device)
    loaded.model = load_pc_lora(loaded.model, args.pc_lora)
    adapter = DreamAdapter(loaded)
    records = []
    dataset = BenchmarkDataset(args.dataset)
    started = time.perf_counter()
    try:
        for ordinal, index in enumerate(indices, 1):
            for steps in args.steps:
                seed = args.seed + index + steps * 10_000
                seed_everything(seed)
                record = collect_pc_mid_state(
                    adapter,
                    dataset.prompt(index),
                    dataset[index]["sample_id"],
                    index,
                    steps=steps,
                    max_new_tokens=MAX_NEW_TOKENS[args.dataset],
                    seed=seed,
                    calibration_fold=args.split,
                    candidate_size=max(2, MAX_NEW_TOKENS[args.dataset] // steps),
                )
                record["dataset"] = args.dataset
                records.append(record)
            print(
                f"PC on-policy {args.dataset} {args.split}: {ordinal}/{len(indices)}",
                flush=True,
            )
    finally:
        adapter.close()
    bundle = {
        "schema": "treepc.pc_onpolicy.v1",
        "eligible_for_head_recalibration": True,
        "dataset": args.dataset,
        "partition": f"head_{args.split}",
        "manifest_fingerprint": manifest["fingerprint"],
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    if args.summary:
        write_json(
            args.summary,
            {
                "kind": "pc_onpolicy_states",
                "dataset": args.dataset,
                "partition": f"head_{args.split}",
                "sample_count": len(indices),
                "state_count": len(records),
                "steps": args.steps,
                "runtime_s": time.perf_counter() - started,
                "cache_path": str(output),
                "cache_sha256": sha256_file(output),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            },
        )


def label(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists():
        if args.resume:
            existing = torch.load(output, map_location="cpu", weights_only=False)
            if int(existing.get("parent_samples", 1)) != args.parent_samples:
                raise ValueError(
                    "Existing label cache uses a different parent sample count; "
                    "use a new output path or run directory"
                )
            print(f"PC on-policy label shard already exists: {output}")
            return
        raise FileExistsError(output)
    bundle = torch.load(args.input, map_location="cpu", weights_only=False)
    if bundle.get("schema") != "treepc.pc_onpolicy.v1":
        raise ValueError("Unsupported PC on-policy bundle")
    teacher = DreamAdapter(load_dream(args.device))
    records = []
    try:
        for ordinal, record in enumerate(bundle["records"], 1):
            records.append(
                label_pc_onpolicy_record(
                    teacher,
                    record,
                    top_k=args.top_k,
                    parent_samples=args.parent_samples,
                )
            )
            print(f"PC on-policy labels {ordinal}/{len(bundle['records'])}", flush=True)
    finally:
        teacher.close()
    labeled_bundle = {
        "schema": "treepc.counterfactual.v1",
        "dataset": bundle.get("dataset", "mixed"),
        "diagnostic_only": False,
        "eligible_for_head_training": True,
        "partition": bundle.get("partition", "pc_onpolicy_calibration"),
        "manifest_fingerprint": bundle.get("manifest_fingerprint"),
        "shard_index": bundle.get("shard_index", 0),
        "num_shards": bundle.get("num_shards", 1),
        "target_type": f"pc_onpolicy_mc_teacher_counterfactual_s{args.parent_samples}",
        "parent_sampling_distribution": "teacher_posterior",
        "parent_samples": args.parent_samples,
        "records": records,
    }
    validate_counterfactual_bundle(labeled_bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(labeled_bundle, output)
    if args.summary:
        write_json(
            args.summary,
            {
                "kind": "pc_onpolicy_teacher_labels",
                "dataset": bundle.get("dataset", "mixed"),
                "partition": bundle.get("partition"),
                "state_count": len(records),
                "top_k": args.top_k,
                "parent_samples": args.parent_samples,
                "parent_sampling_distribution": "teacher_posterior",
                "counterfactual_forwards": sum(
                    int(record["candidate_positions"].numel()) * args.parent_samples
                    for record in records
                ),
                "cache_path": str(output),
                "cache_sha256": sha256_file(output),
            },
        )


def recalibrate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    bundle = torch.load(args.onpolicy_cache, map_location="cpu", weights_only=False)
    onpolicy_records = bundle["records"]
    train_records = [record for record in onpolicy_records if record["calibration_fold"] == "train"]
    validation_records = [
        record for record in onpolicy_records if record["calibration_fold"] == "validation"
    ]
    all_base_records = load_training_records(
        args.base_trajectory, args.base_counterfactual
    )
    gsm_base = [
        record for record in all_base_records if not record["sample_id"].startswith("HumanEval/")
    ]
    humaneval_base = [
        record for record in all_base_records if record["sample_id"].startswith("HumanEval/")
    ]
    gsm_count = len(train_records) // 2
    base_records = gsm_base[:gsm_count] + humaneval_base[: len(train_records) - gsm_count]
    dependency_value = torch.load(
        args.dependency_checkpoint, map_location="cpu", weights_only=False
    )
    dependency = DependencyHead(**dependency_value["config"])
    dependency.load_compatible_state_dict(dependency_value["state_dict"])
    dependency.to(device)
    before_dependency = evaluate_dependency_head(
        dependency,
        DataLoader(DependencyCacheDataset(validation_records), batch_size=1),
        device,
    )
    dependency_report = train_dependency_head(
        dependency,
        ConcatDataset(
            [DependencyCacheDataset(train_records), DependencyCacheDataset(base_records)]
        ),
        DependencyCacheDataset(validation_records),
        device=device,
        output=args.output_dependency,
        epochs=10,
        batch_size=1,
        learning_rate=5e-5,
        ranking_weight=0.25,
        symmetry_weight=0.01,
        seed=args.seed,
    )
    final_dependency_value = torch.load(
        args.output_dependency, map_location="cpu", weights_only=False
    )
    dependency.load_compatible_state_dict(final_dependency_value["state_dict"])

    model_config = load_model_config()
    input_embedding, output_embedding = load_dream_embedding_weights(
        resolve_model_dir(model_config), device
    )
    correction_value = torch.load(
        args.correction_checkpoint, map_location="cpu", weights_only=False
    )
    correction = ConditionalCorrectionHead(
        input_embedding, output_embedding, **correction_value["config"]
    )
    correction.load_compatible_state_dict(correction_value["state_dict"])
    correction.to(device)
    learned_train_pairs = select_learned_tree_pairs(dependency, train_records, device)
    learned_base_pairs = select_learned_tree_pairs(dependency, base_records, device)
    learned_validation_pairs = select_learned_tree_pairs(
        dependency, validation_records, device
    )
    validation_dataset = CorrectionCacheDataset(
        validation_records, selected_pairs=learned_validation_pairs
    )
    before_correction = evaluate_correction_head(
        correction, DataLoader(validation_dataset, batch_size=64), device
    )
    correction_report = train_correction_head(
        correction,
        ConcatDataset(
            [
                CorrectionCacheDataset(
                    train_records, selected_pairs=learned_train_pairs
                ),
                CorrectionCacheDataset(
                    base_records, selected_pairs=learned_base_pairs
                ),
            ]
        ),
        validation_dataset,
        device=device,
        output=args.output_correction,
        epochs=3,
        batch_size=64,
        learning_rate=5e-5,
        delta_weight=1e-3,
        seed=args.seed,
    )
    final_correction_value = torch.load(
        args.output_correction, map_location="cpu", weights_only=False
    )
    correction.load_state_dict(final_correction_value["state_dict"])
    gate = calibrate_tree_gate(
        dependency, correction, validation_records, device, batch_size=64
    )
    hidden_cosines = torch.cat(
        [record["pc_teacher_hidden_cosine"].float() for record in onpolicy_records]
    )
    posterior_kls = torch.cat(
        [record["teacher_to_pc_posterior_kl"].float() for record in onpolicy_records]
    )
    write_json(
        args.report,
        {
            "schema": "treepc.pc_head_recalibration.v1",
            "onpolicy_train_states": len(train_records),
            "onpolicy_validation_states": len(validation_records),
            "base_replay_states": len(base_records),
            "distribution_shift": {
                "pc_teacher_hidden_cosine_mean": float(hidden_cosines.mean()),
                "pc_teacher_hidden_cosine_min": float(hidden_cosines.min()),
                "teacher_to_pc_posterior_kl_mean": float(posterior_kls.mean()),
                "teacher_to_pc_posterior_kl_median": float(posterior_kls.median()),
            },
            "dependency_before": before_dependency,
            "dependency_after": dependency_report["best_validation"],
            "correction_before": before_correction,
            "correction_after": correction_report["best_validation"],
            "correction_adaptation_pair_source": "learned_tree",
            "tree_gate_calibration": gate,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PC on-policy states and recalibrate heads")
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--pc-lora", required=True)
    collect_parser.add_argument("--manifest", required=True)
    collect_parser.add_argument("--dataset", choices=["gsm8k", "humaneval"], required=True)
    collect_parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--summary")
    collect_parser.add_argument("--device", default="cuda:0")
    collect_parser.add_argument("--steps", type=int, nargs="+", default=[8, 16, 32, 64])
    collect_parser.add_argument("--seed", type=int, default=4041)
    collect_parser.add_argument("--shard-index", type=int, default=0)
    collect_parser.add_argument("--num-shards", type=int, default=1)
    collect_parser.add_argument("--resume", action="store_true")
    label_parser = sub.add_parser("label")
    label_parser.add_argument("--input", required=True)
    label_parser.add_argument("--output", required=True)
    label_parser.add_argument("--summary")
    label_parser.add_argument("--device", default="cuda:0")
    label_parser.add_argument("--top-k", type=int, default=16)
    label_parser.add_argument("--parent-samples", type=int, default=3)
    label_parser.add_argument("--resume", action="store_true")
    recalibrate_parser = sub.add_parser("recalibrate")
    recalibrate_parser.add_argument("--onpolicy-cache", required=True)
    recalibrate_parser.add_argument("--base-trajectory", nargs="+", required=True)
    recalibrate_parser.add_argument("--base-counterfactual", nargs="+", required=True)
    recalibrate_parser.add_argument("--dependency-checkpoint", required=True)
    recalibrate_parser.add_argument("--correction-checkpoint", required=True)
    recalibrate_parser.add_argument("--output-dependency", required=True)
    recalibrate_parser.add_argument("--output-correction", required=True)
    recalibrate_parser.add_argument("--report", required=True)
    recalibrate_parser.add_argument("--device", default="cuda:0")
    recalibrate_parser.add_argument("--seed", type=int, default=4041)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    elif args.command == "label":
        label(args)
    else:
        recalibrate(args)


if __name__ == "__main__":
    main()
