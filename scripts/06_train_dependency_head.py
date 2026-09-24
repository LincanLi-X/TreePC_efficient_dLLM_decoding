#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from treepc.data.cache_dataset import DependencyCacheDataset, load_labeled_records, load_training_records
from treepc.models.dependency_head import DependencyHead
from treepc.training.dependency_trainer import train_dependency_head
from treepc.utils.io import write_json
from treepc.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen-backbone TreePC dependency head")
    parser.add_argument("--train-trajectories", nargs="+")
    parser.add_argument("--train-counterfactuals", nargs="+")
    parser.add_argument("--validation-trajectories", nargs="+")
    parser.add_argument("--validation-counterfactuals", nargs="+")
    parser.add_argument("--train-labeled", nargs="+")
    parser.add_argument("--validation-labeled", nargs="+")
    parser.add_argument("--config", default="configs/stage3_frozen/dependency_head.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Step-3 head training requires CUDA")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
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
    model_keys = {
        "hidden_size",
        "projection_size",
        "timestep_size",
        "relative_position_size",
        "max_relative_position",
        "pair_hidden_size",
        "pair_chunk_size",
    }
    model = DependencyHead(**{key: config[key] for key in model_keys})
    report = train_dependency_head(
        model,
        DependencyCacheDataset(train_records),
        DependencyCacheDataset(validation_records),
        device=device,
        output=args.checkpoint,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        ranking_weight=float(config["ranking_weight"]),
        symmetry_weight=float(config["symmetry_weight"]),
        seed=int(config["seed"]),
    )
    report.update(
        {
            "schema": "treepc.dependency_training_report.v1",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "train_state_count": len(train_records),
            "validation_state_count": len(validation_records),
            "config": config,
        }
    )
    write_json(args.report, report)


if __name__ == "__main__":
    main()
