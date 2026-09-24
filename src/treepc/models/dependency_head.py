from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def timestep_embedding(timestep: Tensor, dimension: int = 16) -> Tensor:
    timestep = timestep.float().reshape(-1, 1)
    half = dimension // 2
    frequencies = torch.exp(
        torch.arange(half, device=timestep.device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    values = timestep * frequencies.reshape(1, -1)
    result = torch.cat([values.sin(), values.cos()], dim=-1)
    return result[:, :dimension]


class DependencyHead(nn.Module):
    """Hidden-state pair scorer used by learned TreePC.

    Pair features are materialized only in configurable chunks.  The full
    ``[m,m,d_model]`` tensor forbidden by the design guide is never created.
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        projection_size: int = 128,
        timestep_size: int = 16,
        relative_position_size: int = 32,
        max_relative_position: int = 512,
        pair_hidden_size: int = 256,
        pair_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.timestep_size = timestep_size
        if relative_position_size <= 0:
            raise ValueError("relative_position_size must be positive")
        if max_relative_position <= 0:
            raise ValueError("max_relative_position must be positive")
        self.relative_position_size = relative_position_size
        self.max_relative_position = max_relative_position
        self.pair_chunk_size = pair_chunk_size
        self.node_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection_size),
            nn.GELU(),
        )
        self.relative_position_embedding = nn.Embedding(
            2 * max_relative_position + 1, relative_position_size
        )
        pair_size = 4 * projection_size + relative_position_size + timestep_size
        self.edge_mlp = nn.Sequential(
            nn.Linear(pair_size, pair_hidden_size),
            nn.GELU(),
            nn.Linear(pair_hidden_size, 1),
        )
        nn.init.normal_(self.relative_position_embedding.weight, std=0.02)

    def config_dict(self) -> dict[str, int]:
        return {
            "hidden_size": self.hidden_size,
            "projection_size": self.projection_size,
            "timestep_size": self.timestep_size,
            "relative_position_size": self.relative_position_size,
            "max_relative_position": self.max_relative_position,
            "pair_hidden_size": self.edge_mlp[0].out_features,
            "pair_chunk_size": self.pair_chunk_size,
        }

    def load_compatible_state_dict(self, state_dict: dict[str, Tensor]) -> bool:
        """Load current weights or migrate the former scalar-distance input layer."""
        state_dict = dict(state_dict)
        embedding_key = "relative_position_embedding.weight"
        input_weight_key = "edge_mlp.0.weight"
        legacy = embedding_key not in state_dict
        if legacy:
            legacy_input = state_dict[input_weight_key]
            base_size = 4 * self.projection_size
            expected_legacy_size = base_size + 1 + self.timestep_size
            if legacy_input.shape[1] != expected_legacy_size:
                raise RuntimeError("Incompatible legacy Dependency Head input dimension")
            migrated_input = torch.zeros_like(self.edge_mlp[0].weight)
            migrated_input[:, :base_size] = legacy_input[:, :base_size]
            migrated_input[:, -self.timestep_size :] = legacy_input[:, -self.timestep_size :]
            state_dict[input_weight_key] = migrated_input
            state_dict[embedding_key] = self.relative_position_embedding.weight.detach().clone()
        incompatible = super().load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Incompatible dependency checkpoint: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        return legacy

    def _relative_position_indices(self, left: Tensor, right: Tensor) -> Tensor:
        relative = (left - right).clamp(
            -self.max_relative_position, self.max_relative_position
        )
        return relative.long() + self.max_relative_position

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        timestep: Tensor,
        valid_mask: Tensor | None = None,
        *,
        normalized: bool = False,
        return_raw_scores: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        squeeze = hidden.ndim == 2
        if squeeze:
            hidden = hidden.unsqueeze(0)
            positions = positions.unsqueeze(0)
        batch, nodes, _ = hidden.shape
        if valid_mask is None:
            valid_mask = torch.ones((batch, nodes), dtype=torch.bool, device=hidden.device)
        elif valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)
        with torch.autocast(
            device_type=hidden.device.type, dtype=torch.bfloat16, enabled=hidden.is_cuda
        ):
            projected = self.node_proj(hidden.float())
            row = torch.arange(nodes, device=hidden.device).repeat_interleave(nodes)
            column = torch.arange(nodes, device=hidden.device).repeat(nodes)
            time = timestep_embedding(
                torch.as_tensor(timestep, device=hidden.device), self.timestep_size
            )
            if time.shape[0] == 1 and batch > 1:
                time = time.expand(batch, -1)
            chunks: list[Tensor] = []
            total_pairs = nodes * nodes
            for start in range(0, total_pairs, self.pair_chunk_size):
                end = min(start + self.pair_chunk_size, total_pairs)
                left_index = row[start:end]
                right_index = column[start:end]
                left = projected[:, left_index]
                right = projected[:, right_index]
                relative_indices = self._relative_position_indices(
                    positions[:, left_index], positions[:, right_index]
                )
                relative = self.relative_position_embedding(relative_indices)
                pair_time = time[:, None, :].expand(-1, end - start, -1)
                features = torch.cat(
                    [left, right, (left - right).abs(), left * right, relative, pair_time],
                    dim=-1,
                )
                chunks.append(F.softplus(self.edge_mlp(features)).squeeze(-1))
        raw_scores = torch.cat(chunks, dim=1).reshape(batch, nodes, nodes)
        diagonal = torch.eye(nodes, dtype=torch.bool, device=hidden.device).unsqueeze(0)
        valid_pairs = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2) & ~diagonal
        raw_scores = raw_scores.masked_fill(~valid_pairs, 0.0)
        weights = 0.5 * (raw_scores + raw_scores.transpose(-1, -2))
        weights = weights.masked_fill(~valid_pairs, 0.0)
        if normalized:
            denominator = (
                weights.sum(dim=(-1, -2), keepdim=True)
                / valid_pairs.sum(dim=(-1, -2), keepdim=True).clamp_min(1)
            ).clamp_min(1e-8)
            weights = weights / denominator
        if squeeze:
            weights = weights[0]
            raw_scores = raw_scores[0]
        return (weights, raw_scores) if return_raw_scores else weights
