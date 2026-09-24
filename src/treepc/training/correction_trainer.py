from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from treepc.data.cache_dataset import CorrectionCacheDataset
from treepc.graph.chow_liu import maximum_spanning_tree
from treepc.graph.orientation import orient_tree
from treepc.models.correction_head import ConditionalCorrectionHead
from treepc.models.dependency_head import DependencyHead
from treepc.training.losses import conditional_kl_loss


def _batch_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _prediction(
    model: ConditionalCorrectionHead, batch: dict[str, Tensor]
) -> tuple[Tensor, Tensor, Tensor]:
    base = batch["base_support_log_probs"].masked_fill(~batch["support_mask"], -torch.inf)
    return model(
        child_hidden=batch["child_hidden"],
        parent_hidden=batch["parent_hidden"],
        parent_tokens=batch["parent_token"],
        relative_position=batch["relative_position"],
        sequence_length=batch["sequence_length"],
        timestep=batch["timestep"],
        dependency_weight=batch["dependency_weight"],
        token_ids=batch["token_ids"],
        base_logits=base,
    )


def _oriented_tree_pairs(weights: Tensor, confidence: Tensor) -> set[tuple[int, int]]:
    prim_parent, _ = maximum_spanning_tree(weights)
    root = int(confidence.argmax().item())
    parent, _ = orient_tree(prim_parent, root)
    return {
        (int(parent[child].item()), child)
        for child in range(parent.numel())
        if parent[child] >= 0
    }


@torch.inference_mode()
def select_oracle_tree_pairs(
    records: list[dict[str, Any]],
) -> dict[int, set[tuple[int, int]]]:
    """Select directed MST pairs from cached Teacher dependency targets."""
    return {
        index: _oriented_tree_pairs(
            record["symmetric_dependency"].float(),
            record["candidate_confidence"].float(),
        )
        for index, record in enumerate(records)
    }


@torch.inference_mode()
def select_learned_tree_pairs(
    dependency_head: DependencyHead,
    records: list[dict[str, Any]],
    device: torch.device,
) -> dict[int, set[tuple[int, int]]]:
    dependency_head.eval()
    result: dict[int, set[tuple[int, int]]] = {}
    for index, record in enumerate(records):
        hidden = record["aligned_hidden"].float().to(device)
        positions = record["candidate_positions"].long().to(device)
        timestep = torch.tensor(record["timestep"], device=device)
        weights = dependency_head(hidden, positions, timestep)
        result[index] = _oriented_tree_pairs(
            weights, record["candidate_confidence"].to(device)
        )
    return result


@torch.inference_mode()
def calibrate_tree_gate(
    dependency_head: DependencyHead,
    correction_head: ConditionalCorrectionHead,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, Any]:
    selected = select_learned_tree_pairs(dependency_head, records, device)
    rows: list[dict[str, float]] = []
    for index, record in enumerate(records):
        hidden = record["aligned_hidden"].float().to(device)
        positions = record["candidate_positions"].long().to(device)
        timestep = torch.tensor(record["timestep"], device=device)
        weights = dependency_head(hidden, positions, timestep)
        _, edge_weight = maximum_spanning_tree(weights)
        dataset = CorrectionCacheDataset(
            [record], identity_fraction=0.0, selected_pairs={0: selected[index]}
        )
        metrics = evaluate_correction_head(
            correction_head, DataLoader(dataset, batch_size=batch_size), device
        )
        rows.append(
            {
                "dependency_sum": float(edge_weight.sum().item()),
                "conditional_kl_reduction": metrics["conditional_kl_reduction"],
            }
        )
    scores = sorted({row["dependency_sum"] for row in rows})
    candidates = [0.0, *scores, (scores[-1] + 1e-6 if scores else 1e-6)]
    objectives = [
        sum(
            row["conditional_kl_reduction"] if row["dependency_sum"] >= threshold else 0.0
            for row in rows
        )
        / max(len(rows), 1)
        for threshold in candidates
    ]
    best_index = max(range(len(candidates)), key=lambda index: objectives[index])
    return {
        "dependency_threshold": candidates[best_index],
        "validation_objective_kl_reduction": objectives[best_index],
        "tree_enabled_state_rate": sum(
            row["dependency_sum"] >= candidates[best_index] for row in rows
        )
        / max(len(rows), 1),
        "state_count": len(rows),
        "candidates": [
            {"threshold": threshold, "objective": objective}
            for threshold, objective in zip(candidates, objectives)
        ],
    }


def _kl_per_item(logits: Tensor, batch: dict[str, Tensor]) -> Tensor:
    _, per_item = conditional_kl_loss(
        logits,
        batch["base_other_log_prob"],
        batch["target_support_log_probs"],
        batch["target_other_log_prob"],
        batch["support_mask"],
        torch.zeros_like(logits),
        delta_weight=0.0,
    )
    return per_item


@torch.inference_mode()
def evaluate_correction_head(
    model: ConditionalCorrectionHead,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    base_kls: list[Tensor] = []
    corrected_kls: list[Tensor] = []
    identity_kls: list[Tensor] = []
    gates: list[Tensor] = []
    effective_correction_norms: list[Tensor] = []
    dependencies: list[Tensor] = []
    top1_matches = 0
    top1_total = 0
    for raw_batch in loader:
        batch = _batch_to_device(raw_batch, device)
        corrected, gate, effective_correction = _prediction(model, batch)
        base = batch["base_support_log_probs"]
        base_kl = _kl_per_item(base, batch)
        corrected_kl = _kl_per_item(corrected, batch)
        actual = ~batch["identity"].bool()
        identity = ~actual
        if actual.any():
            base_kls.append(base_kl[actual].cpu())
            corrected_kls.append(corrected_kl[actual].cpu())
            support_mask = batch["support_mask"][actual]
            target = batch["target_support_log_probs"][actual].masked_fill(~support_mask, -torch.inf)
            predicted = corrected[actual].masked_fill(~support_mask, -torch.inf)
            top1_matches += int(predicted.argmax(-1).eq(target.argmax(-1)).sum().item())
            top1_total += int(actual.sum().item())
            gates.append(gate[actual].cpu())
            masked_correction = effective_correction[actual].masked_fill(
                ~batch["support_mask"][actual], 0.0
            )
            effective_correction_norms.append(masked_correction.norm(dim=-1).cpu())
            dependencies.append(batch["dependency_weight"][actual].cpu())
        if identity.any():
            identity_kls.append(corrected_kl[identity].cpu())
    base_values = torch.cat(base_kls)
    corrected_values = torch.cat(corrected_kls)
    gate_values = torch.cat(gates)
    dependency_values = torch.cat(dependencies)
    correction_norm_values = torch.cat(effective_correction_norms)
    centered_gate = gate_values - gate_values.mean()
    centered_dependency = dependency_values - dependency_values.mean()
    correlation = (centered_gate * centered_dependency).mean() / (
        centered_gate.square().mean().sqrt() * centered_dependency.square().mean().sqrt()
    ).clamp_min(1e-8)
    identity_value = torch.cat(identity_kls).mean() if identity_kls else torch.tensor(float("nan"))
    return {
        "base_conditional_kl": float(base_values.mean().item()),
        "corrected_conditional_kl": float(corrected_values.mean().item()),
        "conditional_kl_reduction": float((base_values.mean() - corrected_values.mean()).item()),
        "conditional_kl_improved_pair_rate": float(
            corrected_values.lt(base_values).float().mean().item()
        ),
        "teacher_top1_agreement": top1_matches / max(top1_total, 1),
        "identity_kl": float(identity_value.item()),
        "mean_gate": float(gate_values.mean().item()),
        "global_scale": model.global_scale_value(),
        "mean_effective_gate": float(gate_values.mean().item()) * model.global_scale_value(),
        "mean_effective_correction_l2": float(correction_norm_values.mean().item()),
        "mean_effective_correction_l2_squared": float(
            correction_norm_values.square().mean().item()
        ),
        "gate_dependency_correlation": float(correlation.item()),
    }


def train_correction_head(
    model: ConditionalCorrectionHead,
    train_dataset: Dataset[dict[str, Tensor]],
    validation_dataset: Dataset[dict[str, Tensor]],
    *,
    device: torch.device,
    output: str | Path,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    delta_weight: float = 1e-3,
    seed: int = 3030,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_kl = math.inf
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for raw_batch in train_loader:
            batch = _batch_to_device(raw_batch, device)
            corrected, _, effective_correction = _prediction(model, batch)
            loss, _ = conditional_kl_loss(
                corrected,
                batch["base_other_log_prob"],
                batch["target_support_log_probs"],
                batch["target_other_log_prob"],
                batch["support_mask"],
                effective_correction,
                delta_weight=delta_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        scheduler.step()
        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            metrics = evaluate_correction_head(model, validation_loader, device)
            row = {"epoch": float(epoch), "train_loss": total_loss / batches, **metrics}
            history.append(row)
            print(
                f"correction epoch={epoch}/{epochs} loss={row['train_loss']:.5f} "
                f"base_kl={metrics['base_conditional_kl']:.5f} "
                f"corrected_kl={metrics['corrected_conditional_kl']:.5f} "
                f"global_scale={metrics['global_scale']:.5f}",
                flush=True,
            )
            if metrics["corrected_conditional_kl"] < best_kl:
                best_kl = metrics["corrected_conditional_kl"]
                best_metrics = metrics
                torch.save(
                    {
                        "schema": "treepc.correction_head.v1",
                        "config": model.config_dict(),
                        "state_dict": {
                            key: value.detach().cpu() for key, value in model.state_dict().items()
                        },
                        "validation_metrics": metrics,
                        "epoch": epoch,
                    },
                    output,
                )
    return {"best_validation": best_metrics, "history": history, "checkpoint": str(output)}
