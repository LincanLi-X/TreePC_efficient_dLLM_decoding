from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from treepc.models.dependency_head import timestep_embedding


class ConditionalCorrectionHead(nn.Module):
    """Top-K gated low-rank conditional posterior correction."""

    def __init__(
        self,
        input_embedding: Tensor,
        output_embedding: Tensor,
        *,
        hidden_size: int = 3584,
        rank: int = 32,
        feature_size: int = 32,
        global_scale_init: float = 0.05,
    ) -> None:
        super().__init__()
        if input_embedding.shape[1] != hidden_size or output_embedding.shape[1] != hidden_size:
            raise ValueError("Dream embedding dimensions do not match correction hidden size")
        if not 0.0 < global_scale_init < 1.0:
            raise ValueError("global_scale_init must be strictly between 0 and 1")
        self.hidden_size = hidden_size
        self.rank = rank
        self.feature_size = feature_size
        self.global_scale_init = float(global_scale_init)
        initial_logit = math.log(global_scale_init / (1.0 - global_scale_init))
        self.global_scale_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        self.register_buffer("input_embedding", input_embedding, persistent=False)
        self.register_buffer("output_embedding", output_embedding, persistent=False)
        feature_input = 3 * hidden_size + 2 * feature_size
        self.down = nn.Linear(feature_input, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.gate_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)

    @property
    def global_scale(self) -> Tensor:
        """Shared correction scale s=sigmoid(a), constrained to (0, 1)."""
        return torch.sigmoid(self.global_scale_logit)

    def global_scale_value(self) -> float:
        return float(self.global_scale.detach().float().cpu().item())

    def load_compatible_state_dict(self, state_dict: dict[str, Tensor]) -> bool:
        """Load current or pre-global-scale weights.

        Legacy checkpoints omit ``global_scale_logit``. They remain usable and
        receive the configured initialization (0.05 by default). The return
        value reports whether that legacy fallback was used.
        """
        incompatible = super().load_state_dict(state_dict, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        legacy = missing == {"global_scale_logit"}
        if unexpected or (missing and not legacy):
            raise RuntimeError(
                "Incompatible correction checkpoint: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        return legacy

    def config_dict(self) -> dict[str, int | float]:
        return {
            "hidden_size": self.hidden_size,
            "rank": self.rank,
            "feature_size": self.feature_size,
            "global_scale_init": self.global_scale_init,
        }

    def _relative_features(self, relative_position: Tensor) -> Tensor:
        half = self.feature_size // 2
        frequencies = torch.exp(
            torch.arange(half, device=relative_position.device, dtype=torch.float32)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        value = relative_position.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        return torch.cat([value.sin(), value.cos()], dim=-1)[:, : self.feature_size]

    def forward(
        self,
        *,
        child_hidden: Tensor,
        parent_hidden: Tensor,
        parent_tokens: Tensor,
        relative_position: Tensor,
        sequence_length: Tensor,
        timestep: Tensor,
        dependency_weight: Tensor,
        token_ids: Tensor,
        base_logits: Tensor,
        force_gate_zero: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.autocast(
            device_type=child_hidden.device.type,
            dtype=torch.bfloat16,
            enabled=child_hidden.is_cuda,
        ):
            child = child_hidden.float()
            parent = parent_hidden.float()
            token = F.embedding(parent_tokens, self.input_embedding).float()
            relative = self._relative_features(relative_position)
            time = timestep_embedding(timestep, self.feature_size)
            features = torch.cat([child, parent, token, relative, time], dim=-1)
            delta_hidden = self.up(F.gelu(self.down(features)))
            cosine = F.cosine_similarity(child, parent, dim=-1)
            lengths = sequence_length.float().reshape(-1)
            if lengths.numel() == 1 and relative_position.numel() > 1:
                lengths = lengths.expand_as(relative_position)
            if lengths.numel() != relative_position.numel() or torch.any(lengths <= 0):
                raise ValueError("sequence_length must provide one positive value per pair")
            normalized_distance = relative_position.float().abs() / lengths
            gate_features = torch.stack(
                [
                    dependency_weight.float(),
                    cosine,
                    normalized_distance,
                    timestep.float(),
                ],
                dim=-1,
            )
            gate = torch.sigmoid(self.gate_mlp(gate_features)).squeeze(-1)
            if force_gate_zero:
                gate = torch.zeros_like(gate)
            candidate_embeddings = F.embedding(token_ids, self.output_embedding).float()
            residual = torch.einsum("nkd,nd->nk", candidate_embeddings, delta_hidden)
            residual = residual / math.sqrt(self.hidden_size)
            effective_gate = self.global_scale.float() * gate
            effective_correction = effective_gate.unsqueeze(-1) * residual
            corrected = base_logits.float() + effective_correction
        return corrected, gate, effective_correction
