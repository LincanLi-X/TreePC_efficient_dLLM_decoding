from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from treepc.graph.chow_liu import maximum_spanning_tree, tree_weight
from treepc.models.dependency_head import DependencyHead
from treepc.training.losses import dependency_loss


def _rank(values: Tensor) -> Tensor:
    """Average ranks, including correct handling of tied edge weights."""
    sorted_values, order = values.flatten().sort()
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    starts = counts.cumsum(0) - counts
    average = starts.float() + (counts.float() - 1.0) / 2.0
    sorted_ranks = torch.repeat_interleave(average, counts)
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _spearman(left: Tensor, right: Tensor) -> Tensor:
    left_rank = _rank(left)
    right_rank = _rank(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = left_rank.square().sum().sqrt() * right_rank.square().sum().sqrt()
    return (left_rank * right_rank).sum() / denominator.clamp_min(1e-8)


def _kendall_tau_b(left: Tensor, right: Tensor) -> float:
    """Tie-aware Kendall tau-b in O(E log E) for upper-triangle edge scores."""
    x = [float(value) for value in left.detach().float().cpu().flatten()]
    y = [float(value) for value in right.detach().float().cpu().flatten()]
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    order = sorted(range(len(x)), key=lambda index: (x[index], y[index]))
    y_values = sorted(set(y))
    y_rank = {value: index + 1 for index, value in enumerate(y_values)}
    fenwick = [0] * (len(y_values) + 1)

    def query(index: int) -> int:
        result = 0
        while index > 0:
            result += fenwick[index]
            index -= index & -index
        return result

    def add(index: int) -> None:
        while index < len(fenwick):
            fenwick[index] += 1
            index += index & -index

    numerator = 0
    processed = 0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and x[order[end]] == x[order[start]]:
            end += 1
        for item in order[start:end]:
            rank = y_rank[y[item]]
            less = query(rank - 1)
            greater = processed - query(rank)
            numerator += less - greater
        for item in order[start:end]:
            add(y_rank[y[item]])
            processed += 1
        start = end

    pairs = len(x) * (len(x) - 1) // 2
    tied_x = sum(count * (count - 1) // 2 for count in Counter(x).values())
    tied_y = sum(count * (count - 1) // 2 for count in Counter(y).values())
    denominator = math.sqrt((pairs - tied_x) * (pairs - tied_y))
    return numerator / denominator if denominator else 0.0


def _edge_set(parent: Tensor) -> set[frozenset[int]]:
    return {frozenset((child, int(value))) for child, value in enumerate(parent.tolist()) if value >= 0}


def _tree_scores(target: Tensor, parent: Tensor, oracle_parent: Tensor) -> tuple[float, float]:
    oracle_edges = _edge_set(oracle_parent)
    edges = _edge_set(parent)
    overlap = len(edges & oracle_edges) / max(len(oracle_edges), 1)
    oracle_weight = tree_weight(target, oracle_parent).clamp_min(1e-8)
    capture = tree_weight(target, parent) / oracle_weight
    return overlap, float(capture.item())


@torch.inference_mode()
def evaluate_dependency_head(
    model: DependencyHead, loader: DataLoader[dict[str, Tensor]], device: torch.device
) -> dict[str, Any]:
    model.eval()
    absolute_errors: list[Tensor] = []
    raw_symmetry_errors: list[Tensor] = []
    spearman: list[float] = []
    kendall: list[float] = []
    tree_values: dict[str, dict[str, list[float]]] = {
        name: {"mst_edge_overlap": [], "oracle_weight_capture": []}
        for name in ("learned", "local", "hidden_cosine", "random", "oracle")
    }
    for batch in loader:
        hidden = batch["hidden"].to(device)
        positions = batch["positions"].to(device)
        timestep = batch["timestep"].to(device)
        target = batch["target"].to(device)
        prediction, raw_scores = model(hidden, positions, timestep, return_raw_scores=True)
        nodes = prediction.shape[-1]
        upper = torch.triu(torch.ones((nodes, nodes), dtype=torch.bool, device=device), 1)
        absolute_errors.append((prediction[:, upper] - target[:, upper]).abs().reshape(-1).cpu())
        raw_symmetry_errors.append(
            (raw_scores[:, upper] - raw_scores.transpose(-1, -2)[:, upper]).abs().reshape(-1).cpu()
        )
        for row in range(prediction.shape[0]):
            predicted_edges = prediction[row, upper]
            target_edges = target[row, upper]
            spearman.append(float(_spearman(predicted_edges, target_edges).item()))
            kendall.append(_kendall_tau_b(predicted_edges, target_edges))
            predicted_parent, _ = maximum_spanning_tree(prediction[row])
            oracle_parent, _ = maximum_spanning_tree(target[row])
            distance = (positions[row, :, None] - positions[row, None, :]).abs().float()
            local_weights = 1.0 / (distance + 1.0)
            local_weights.fill_diagonal_(0.0)
            local_parent, _ = maximum_spanning_tree(local_weights)
            normalized_hidden = torch.nn.functional.normalize(hidden[row].float(), dim=-1)
            hidden_cosine = normalized_hidden @ normalized_hidden.transpose(-1, -2)
            hidden_cosine.fill_diagonal_(0.0)
            hidden_parent, _ = maximum_spanning_tree(hidden_cosine)
            record_key = int(batch.get("record_key", torch.tensor([row]))[row].item())
            generator = torch.Generator(device="cpu").manual_seed(3030 + record_key)
            random_upper = torch.rand((nodes, nodes), generator=generator)
            random_weights = torch.triu(random_upper, diagonal=1)
            random_weights = (random_weights + random_weights.T).to(device)
            random_parent, _ = maximum_spanning_tree(random_weights)
            parents = {
                "learned": predicted_parent,
                "local": local_parent,
                "hidden_cosine": hidden_parent,
                "random": random_parent,
                "oracle": oracle_parent,
            }
            for name, parent in parents.items():
                overlap, capture = _tree_scores(target[row], parent, oracle_parent)
                tree_values[name]["mst_edge_overlap"].append(overlap)
                tree_values[name]["oracle_weight_capture"].append(capture)
    errors = torch.cat(absolute_errors)
    symmetry_errors = torch.cat(raw_symmetry_errors)
    trees = {
        name: {metric: float(sum(values) / len(values)) for metric, values in metrics.items()}
        for name, metrics in tree_values.items()
    }
    return {
        "mae": float(errors.mean().item()),
        "huber": float(torch.nn.functional.huber_loss(errors, torch.zeros_like(errors)).item()),
        "spearman": float(sum(spearman) / len(spearman)),
        "kendall_tau_b": float(sum(kendall) / len(kendall)),
        "tree_edge_overlap": trees["learned"]["mst_edge_overlap"],
        "oracle_weight_capture": trees["learned"]["oracle_weight_capture"],
        "local_weight_capture": trees["local"]["oracle_weight_capture"],
        "hidden_cosine_weight_capture": trees["hidden_cosine"]["oracle_weight_capture"],
        "random_weight_capture": trees["random"]["oracle_weight_capture"],
        "raw_symmetry_mae": float(symmetry_errors.mean().item()),
        "trees": trees,
    }


def train_dependency_head(
    model: DependencyHead,
    train_dataset: Dataset[dict[str, Tensor]],
    validation_dataset: Dataset[dict[str, Tensor]],
    *,
    device: torch.device,
    output: str | Path,
    epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    ranking_weight: float = 0.25,
    symmetry_weight: float = 0.01,
    seed: int = 3030,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_capture = -math.inf
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            prediction, raw_scores = model(
                batch["hidden"].to(device),
                batch["positions"].to(device),
                batch["timestep"].to(device),
                return_raw_scores=True,
            )
            loss, _ = dependency_loss(
                prediction,
                raw_scores,
                batch["target"].to(device),
                ranking_weight=ranking_weight,
                symmetry_weight=symmetry_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            metrics = evaluate_dependency_head(model, validation_loader, device)
            row = {"epoch": float(epoch), "train_loss": total_loss / batches, **metrics}
            history.append(row)
            print(
                f"dependency epoch={epoch}/{epochs} loss={row['train_loss']:.5f} "
                f"capture={metrics['oracle_weight_capture']:.4f} "
                f"spearman={metrics['spearman']:.4f}",
                flush=True,
            )
            if metrics["oracle_weight_capture"] > best_capture:
                best_capture = metrics["oracle_weight_capture"]
                best_metrics = metrics
                torch.save(
                    {
                        "schema": "treepc.dependency_head.v1",
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
