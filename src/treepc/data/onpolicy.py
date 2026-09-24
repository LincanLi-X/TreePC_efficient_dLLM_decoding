from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from treepc.data.counterfactual import state_from_record
from treepc.dream.adapter import DreamAdapter, max_token_probability
from treepc.posterior.divergences import categorical_kl_from_logits
from treepc.types import clone_cpu
from treepc.utils.seed import seed_everything


@torch.inference_mode()
def collect_pc_mid_state(
    adapter: DreamAdapter,
    prompt: str,
    sample_id: str,
    dataset_index: int,
    *,
    steps: int,
    max_new_tokens: int,
    seed: int,
    calibration_fold: str,
    candidate_size: int = 8,
) -> dict[str, Any]:
    encoded = adapter.encode_prompt(prompt)
    state = adapter.make_initial_state(encoded, max_new_tokens, steps)
    timesteps = torch.linspace(1, 1e-3, steps + 1, device=adapter.device)
    capture_step = max(0, steps // 2 - 1)
    captured: dict[str, Any] | None = None
    for step in range(steps):
        state.step_index = step
        state.timestep_t = timesteps[step]
        state.timestep_s = timesteps[step + 1]
        output = adapter.forward_state(state, need_hidden=step == capture_step)
        scheduler_confidence, proposals = adapter.propose_tokens_and_confidence(
            output, "entropy", 0.0, None, None
        )
        mask = output.masked_positions
        budget = adapter.compute_commit_budget(
            mask, timesteps[step], timesteps[step + 1], step == steps - 1
        )
        full_confidence = torch.full_like(
            state.input_ids, -torch.inf, dtype=output.aligned_logits.dtype
        )
        full_confidence[mask] = scheduler_confidence
        proposed_full = torch.full_like(state.input_ids, adapter.mask_token_id)
        proposed_full[mask] = proposals
        if step == capture_step:
            count = min(candidate_size, int(mask.sum().item()))
            candidates = torch.topk(full_confidence, count, dim=-1).indices[0]
            candidates = torch.sort(candidates).values
            captured = {
                "sample_id": sample_id,
                "dataset_index": dataset_index,
                "steps": steps,
                "calibration_fold": calibration_fold,
                "state_token_ids": clone_cpu(state.input_ids[0]),
                "generation_mask": clone_cpu(state.generation_mask[0]),
                "prompt_length": state.prompt_length,
                "step_index": step,
                "timestep": float(timesteps[step].item()),
                "candidate_positions": clone_cpu(candidates),
                "candidate_confidence": clone_cpu(
                    max_token_probability(output.aligned_logits[0, candidates])
                ),
                "candidate_confidence_type": "max_token_probability",
                "aligned_hidden": clone_cpu(output.aligned_hidden[0, candidates]),
                "pc_base_logits": clone_cpu(output.aligned_logits[0, candidates]),
                "pc_base_tokens": clone_cpu(proposed_full[0, candidates]),
                "seed": seed,
            }
        if budget:
            positions = torch.topk(full_confidence, budget, dim=-1).indices
            rows = torch.zeros_like(positions)
            state.input_ids[rows, positions] = proposed_full[rows, positions]
    if captured is None:
        raise RuntimeError("PC on-policy capture state was not created")
    captured["final_pc_tokens"] = clone_cpu(state.input_ids[0])
    return captured


def _support_from_logits(
    base_logits: torch.Tensor,
    target_logits: torch.Tensor,
    final_token: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_ids = torch.topk(base_logits, top_k).indices
    target_ids = torch.topk(target_logits, top_k).indices
    return _support_from_log_probs(
        torch.log_softmax(base_logits.float(), -1),
        torch.log_softmax(target_logits.float(), -1),
        base_ids,
        target_ids,
        final_token,
        top_k,
    )


def _support_from_log_probs(
    base_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    base_ids: torch.Tensor,
    target_ids: torch.Tensor,
    final_token: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    union = torch.unique(torch.cat([base_ids, target_ids, final_token.view(1)]), sorted=True)
    width = 2 * top_k + 1
    ids = torch.zeros(width, dtype=torch.long)
    mask = torch.zeros(width, dtype=torch.bool)
    base_values = torch.full((width,), -torch.inf, dtype=torch.float32)
    target_values = torch.full((width,), -torch.inf, dtype=torch.float32)
    count = int(union.numel())
    ids[:count] = union.cpu()
    mask[:count] = True
    base_values[:count] = base_log_probs[union].cpu()
    target_values[:count] = target_log_probs[union].cpu()
    base_other = (1 - base_values[:count].exp().sum()).clamp_min(1e-12).log()
    target_other = (1 - target_values[:count].exp().sum()).clamp_min(1e-12).log()
    return ids, mask, base_values, target_values, base_other, target_other


def _support_matrix_from_log_probs(
    base_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    base_ids: torch.Tensor,
    target_ids: torch.Tensor,
    final_tokens: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes = base_log_probs.shape[0]
    width = 2 * top_k + 1
    combined = torch.cat([base_ids, target_ids, final_tokens[:, None]], dim=-1).cpu()
    ids_cpu = torch.zeros((nodes, width), dtype=torch.long)
    mask_cpu = torch.zeros((nodes, width), dtype=torch.bool)
    for child in range(nodes):
        union = torch.unique(combined[child], sorted=True)
        count = int(union.numel())
        ids_cpu[child, :count] = union
        mask_cpu[child, :count] = True
    ids = ids_cpu.to(base_log_probs.device)
    mask = mask_cpu.to(base_log_probs.device)
    base_values = base_log_probs.gather(-1, ids).masked_fill(~mask, -torch.inf)
    target_values = target_log_probs.gather(-1, ids).masked_fill(~mask, -torch.inf)
    base_other = (
        1 - (base_values.exp() * mask).sum(dim=-1)
    ).clamp_min(1e-12).log()
    target_other = (
        1 - (target_values.exp() * mask).sum(dim=-1)
    ).clamp_min(1e-12).log()
    return (
        ids_cpu,
        mask_cpu,
        clone_cpu(base_values),
        clone_cpu(target_values),
        clone_cpu(base_other),
        clone_cpu(target_other),
    )


@torch.inference_mode()
def label_pc_onpolicy_record(
    teacher: DreamAdapter,
    record: dict[str, Any],
    *,
    top_k: int = 16,
    parent_samples: int = 3,
) -> dict[str, Any]:
    if parent_samples <= 0:
        raise ValueError("parent_samples must be positive")
    state = state_from_record(teacher, record)
    candidates = record["candidate_positions"].to(teacher.device)
    nodes = int(candidates.numel())
    teacher_base = teacher.forward_state(state, need_hidden=True)
    teacher_base_logits = teacher_base.aligned_logits[0, candidates]
    pc_base_logits = record["pc_base_logits"].to(teacher.device)
    pc_base_log_probs = torch.log_softmax(pc_base_logits.float(), -1)
    pc_base_top_ids = torch.topk(pc_base_logits, top_k, dim=-1).indices
    hidden_cosine = F.cosine_similarity(
        record["aligned_hidden"].to(teacher.device).float(),
        teacher_base.aligned_hidden[0, candidates].float(),
        dim=-1,
    )
    teacher_to_pc_kl = categorical_kl_from_logits(
        teacher_base_logits, pc_base_logits
    )
    teacher_parent_probabilities = torch.softmax(teacher_base_logits.float(), dim=-1)
    sampled_parent_tokens = torch.empty(
        (nodes, parent_samples), dtype=torch.long, device="cpu"
    )
    parent_probabilities = torch.empty((nodes, parent_samples), dtype=torch.float32)
    directed_samples = torch.zeros(
        (nodes, parent_samples, nodes), dtype=torch.float32, device=teacher.device
    )
    width = 2 * top_k + 1
    support_ids = torch.zeros((nodes, parent_samples, nodes, width), dtype=torch.long)
    support_mask = torch.zeros((nodes, parent_samples, nodes, width), dtype=torch.bool)
    base_support = torch.full((nodes, parent_samples, nodes, width), -torch.inf)
    target_support = torch.full_like(base_support, -torch.inf)
    base_other = torch.empty((nodes, parent_samples, nodes))
    target_other = torch.empty_like(base_other)
    final_tokens = record["final_pc_tokens"].to(teacher.device)[candidates]
    for parent in range(nodes):
        seed_everything(int(record["seed"]) + parent)
        parent_tokens = torch.multinomial(
            teacher_parent_probabilities[parent],
            parent_samples,
            replacement=True,
        )
        sampled_parent_tokens[parent] = parent_tokens.cpu()
        parent_probabilities[parent] = teacher_parent_probabilities[parent, parent_tokens].cpu()
        for sample_index, parent_token in enumerate(parent_tokens):
            counterfactual = state_from_record(teacher, record)
            counterfactual.input_ids[0, candidates[parent]] = parent_token
            conditional = teacher.forward_state(
                counterfactual, need_hidden=False
            ).aligned_logits[0, candidates]
            conditional_log_probs = torch.log_softmax(conditional.float(), -1)
            conditional_top_ids = torch.topk(conditional, top_k, dim=-1).indices
            dependency = categorical_kl_from_logits(conditional, teacher_base_logits)
            dependency[parent] = 0
            directed_samples[parent, sample_index] = dependency
            values = _support_matrix_from_log_probs(
                pc_base_log_probs,
                conditional_log_probs,
                pc_base_top_ids,
                conditional_top_ids,
                final_tokens,
                top_k,
            )
            support_ids[parent, sample_index] = values[0]
            support_mask[parent, sample_index] = values[1]
            base_support[parent, sample_index] = values[2]
            target_support[parent, sample_index] = values[3]
            base_other[parent, sample_index] = values[4]
            target_other[parent, sample_index] = values[5]
    directed = directed_samples.mean(dim=1)
    result = {key: value for key, value in record.items() if key != "pc_base_logits"}
    result.update(
        {
            "target_type": f"pc_onpolicy_mc_teacher_counterfactual_s{parent_samples}",
            "parent_sampling_distribution": "teacher_posterior",
            "parent_samples": parent_samples,
            "aligned_hidden": record["aligned_hidden"].float(),
            "teacher_aligned_hidden": clone_cpu(
                teacher_base.aligned_hidden[0, candidates]
            ),
            "pc_teacher_hidden_cosine": clone_cpu(hidden_cosine),
            "teacher_to_pc_posterior_kl": clone_cpu(teacher_to_pc_kl),
            "candidate_confidence": clone_cpu(max_token_probability(pc_base_logits)),
            "candidate_confidence_type": "max_token_probability",
            "sampled_parent_tokens": sampled_parent_tokens,
            "parent_probabilities": parent_probabilities,
            "directed_dependency_samples": clone_cpu(directed_samples),
            "directed_dependency": clone_cpu(directed),
            "symmetric_dependency": clone_cpu(0.5 * (directed + directed.T)),
            "support_ids": support_ids,
            "support_mask": support_mask,
            "base_support_log_probs": base_support,
            "conditional_support_log_probs": target_support,
            "base_other_log_probs": base_other,
            "conditional_other_log_probs": target_other,
        }
    )
    return result
