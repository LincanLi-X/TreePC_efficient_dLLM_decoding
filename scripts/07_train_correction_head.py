#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from treepc.data.cache_dataset import (
    CorrectionCacheDataset,
    load_dream_embedding_weights,
    load_labeled_records,
    load_training_records,
)
from treepc.dream.loader import load_model_config, resolve_model_dir
from treepc.models.correction_head import ConditionalCorrectionHead
from treepc.models.dependency_head import DependencyHead
from treepc.training.correction_trainer import (
    calibrate_tree_gate,
    evaluate_correction_head,
    select_learned_tree_pairs,
    select_oracle_tree_pairs,
    train_correction_head,
)
from treepc.utils.io import write_json
from treepc.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen-backbone TreePC correction head")
    parser.add_argument("--train-trajectories", nargs="+")
    parser.add_argument("--train-counterfactuals", nargs="+")
    parser.add_argument("--validation-trajectories", nargs="+")
    parser.add_argument("--validation-counterfactuals", nargs="+")
    parser.add_argument("--train-labeled", nargs="+")
    parser.add_argument("--validation-labeled", nargs="+")
    parser.add_argument("--dependency-checkpoint", required=True)
    parser.add_argument("--config", default="configs/stage3_frozen/correction_head.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Step-3 head training requires CUDA")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if float(config["adaptation_learning_rate"]) >= float(config["learning_rate"]):
        raise ValueError("Learned-tree adaptation learning rate must be smaller than Oracle stage")
    seed_everything(int(config["seed"]))
    if bool(args.train_labeled) != bool(args.validation_labeled):
        raise ValueError("Provide both --train-labeled and --validation-labeled")
    if args.train_labeled:
        train_records = load_labeled_records(args.train_labeled, expected_partition="head_train")
        validation_records = load_labeled_records(
            args.validation_labeled, expected_partition="head_validation"
        )
    else:
        required = (
            args.train_trajectories,
            args.train_counterfactuals,
            args.validation_trajectories,
            args.validation_counterfactuals,
        )
        if not all(required):
            raise ValueError("Provide labeled caches or all trajectory/counterfactual inputs")
        train_records = load_training_records(args.train_trajectories, args.train_counterfactuals)
        validation_records = load_training_records(
            args.validation_trajectories, args.validation_counterfactuals
        )
    model_config = load_model_config()
    input_embedding, output_embedding = load_dream_embedding_weights(
        resolve_model_dir(model_config), device
    )
    model = ConditionalCorrectionHead(
        input_embedding,
        output_embedding,
        hidden_size=int(config["hidden_size"]),
        rank=int(config["rank"]),
        feature_size=int(config["feature_size"]),
        global_scale_init=float(config.get("global_scale_init", 0.05)),
    )
    oracle_train_pairs = select_oracle_tree_pairs(train_records)
    oracle_validation_pairs = select_oracle_tree_pairs(validation_records)
    train_dataset = CorrectionCacheDataset(
        train_records,
        identity_fraction=float(config["identity_fraction"]),
        selected_pairs=oracle_train_pairs,
    )
    validation_dataset = CorrectionCacheDataset(
        validation_records,
        identity_fraction=float(config["identity_fraction"]),
        selected_pairs=oracle_validation_pairs,
    )
    report = train_correction_head(
        model,
        train_dataset,
        validation_dataset,
        device=device,
        output=args.checkpoint,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        delta_weight=float(config["delta_weight"]),
        seed=int(config["seed"]),
    )
    best_value = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(best_value["state_dict"])
    dependency_value = torch.load(
        args.dependency_checkpoint, map_location="cpu", weights_only=False
    )
    dependency_head = DependencyHead(**dependency_value["config"])
    dependency_head.load_compatible_state_dict(dependency_value["state_dict"])
    dependency_head.to(device)
    train_pairs = select_learned_tree_pairs(dependency_head, train_records, device)
    validation_pairs = select_learned_tree_pairs(dependency_head, validation_records, device)
    adapted_train = CorrectionCacheDataset(
        train_records,
        identity_fraction=float(config["identity_fraction"]),
        selected_pairs=train_pairs,
    )
    adapted_validation = CorrectionCacheDataset(
        validation_records,
        identity_fraction=float(config["identity_fraction"]),
        selected_pairs=validation_pairs,
    )
    before_adaptation = evaluate_correction_head(
        model, DataLoader(adapted_validation, batch_size=int(config["batch_size"])), device
    )
    adapted_checkpoint = str(Path(args.checkpoint).with_suffix(".adapted.pt"))
    adaptation = train_correction_head(
        model,
        adapted_train,
        adapted_validation,
        device=device,
        output=adapted_checkpoint,
        epochs=int(config["adaptation_epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["adaptation_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        delta_weight=float(config["delta_weight"]),
        seed=int(config["seed"]) + 1,
    )
    keep_adaptation = (
        adaptation["best_validation"]["corrected_conditional_kl"]
        <= before_adaptation["corrected_conditional_kl"]
    )
    if keep_adaptation:
        shutil.copy2(adapted_checkpoint, args.checkpoint)
    Path(adapted_checkpoint).unlink(missing_ok=True)
    report["learned_tree_domain_adaptation"] = {
        "train_tree_edge_count": sum(len(pairs) for pairs in train_pairs.values()),
        "validation_tree_edge_count": sum(
            len(pairs) for pairs in validation_pairs.values()
        ),
        "train_example_count": len(adapted_train),
        "validation_example_count": len(adapted_validation),
        "before": before_adaptation,
        "after": adaptation["best_validation"],
        "kept": keep_adaptation,
    }
    final_value = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(final_value["state_dict"])
    report["tree_gate_calibration"] = calibrate_tree_gate(
        dependency_head,
        model,
        validation_records,
        device,
        batch_size=int(config["batch_size"]),
    )
    final_value["tree_gate_calibration"] = report["tree_gate_calibration"]
    torch.save(final_value, args.checkpoint)
    report.update(
        {
            "schema": "treepc.correction_training_report.v1",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "oracle_tree_training": {
                "train_tree_edge_count": sum(
                    len(pairs) for pairs in oracle_train_pairs.values()
                ),
                "validation_tree_edge_count": sum(
                    len(pairs) for pairs in oracle_validation_pairs.values()
                ),
                "train_example_count": len(train_dataset),
                "validation_example_count": len(validation_dataset),
            },
            "config": config,
        }
    )
    write_json(args.report, report)


if __name__ == "__main__":
    main()
