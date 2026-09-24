from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class DreamState:
    input_ids: Tensor
    attention_mask: Tensor | str
    tok_idx: Tensor | None
    prompt_length: int
    generation_mask: Tensor
    step_index: int
    timestep_t: Tensor
    timestep_s: Tensor


@dataclass
class DreamStepOutput:
    aligned_logits: Tensor
    aligned_hidden: Tensor | None
    masked_positions: Tensor
    base_sampled_tokens: Tensor
    confidence: Tensor


@dataclass
class StepTrace:
    step: int
    pre_state: Tensor
    post_state: Tensor
    commit_positions: Tensor
    committed_tokens: Tensor
    proposal_tokens: Tensor
    logits_digest: list[float]


@dataclass
class GenerationResult:
    sequences: Tensor
    texts: list[str]
    traces: list[StepTrace] = field(default_factory=list)
    latency_seconds: float = 0.0
    effective_nfe: int = 0


@dataclass
class TreeBatch:
    parent: Tensor
    edge_weight: Tensor
    root: Tensor
    depth: Tensor
    valid_mask: Tensor


def clone_cpu(tensor: Tensor) -> Tensor:
    return tensor.detach().to(device="cpu", non_blocking=False).clone()
