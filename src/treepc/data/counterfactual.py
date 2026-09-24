from __future__ import annotations

from typing import Any

import torch

from treepc.dream.adapter import DreamAdapter
from treepc.posterior.divergences import categorical_kl_from_logits, categorical_nll_from_logits
from treepc.posterior.topk_support import topk_log_probs_with_tail
from treepc.types import DreamState, clone_cpu
from treepc.utils.seed import seed_everything


def state_from_record(adapter: DreamAdapter, record: dict[str, Any]) -> DreamState:
    x = record["state_token_ids"].to(adapter.device).unsqueeze(0)
    generation_mask = record["generation_mask"].to(adapter.device).unsqueeze(0)
    timestep = torch.tensor(record["timestep"], device=adapter.device)
    return DreamState(
        input_ids=x,
        attention_mask="full",
        tok_idx=None,
        prompt_length=int(record["prompt_length"]),
        generation_mask=generation_mask,
        step_index=int(record["step_index"]),
        timestep_t=timestep,
        timestep_s=timestep,
    )


@torch.inference_mode()
def build_counterfactual_record(
    adapter: DreamAdapter,
    record: dict[str, Any],
    *,
    top_k: int = 16,
    seed: int = 2026,
    parent_samples: int = 3,
) -> dict[str, Any]:
    if parent_samples <= 0:
        raise ValueError("parent_samples must be positive")
    state = state_from_record(adapter, record)
    candidates = record["candidate_positions"].to(adapter.device)
    nodes = candidates.numel()
    base_output = adapter.forward_state(state, need_hidden=False)
    base_logits = base_output.aligned_logits[0, candidates]
    final_targets = record["final_teacher_tokens"].to(adapter.device)[candidates]
    base_top_ids, base_top_log_probs, base_tail = topk_log_probs_with_tail(base_logits, top_k)
    base_target_nll = categorical_nll_from_logits(base_logits, final_targets)
    directed_samples = torch.zeros(
        (nodes, parent_samples, nodes), dtype=torch.float32, device=adapter.device
    )
    conditional_top_ids = torch.empty(
        (nodes, parent_samples, nodes, top_k), dtype=torch.long, device="cpu"
    )
    conditional_top_log_probs = torch.empty_like(
        conditional_top_ids, dtype=torch.float32
    )
    conditional_tail = torch.empty(
        (nodes, parent_samples, nodes), dtype=torch.float32, device="cpu"
    )
    conditional_target_nll = torch.empty_like(conditional_tail)
    sampled_parent_tokens = torch.empty(
        (nodes, parent_samples), dtype=torch.long, device="cpu"
    )
    parent_probabilities = torch.empty_like(sampled_parent_tokens, dtype=torch.float32)
    top1_in_base_topk = torch.empty(
        (nodes, parent_samples, nodes), dtype=torch.bool, device="cpu"
    )
    support_width = 2 * top_k + 1
    support_ids = torch.zeros(
        (nodes, parent_samples, nodes, support_width), dtype=torch.long, device="cpu"
    )
    support_mask = torch.zeros_like(support_ids, dtype=torch.bool)
    base_support_log_probs = torch.full(
        (nodes, parent_samples, nodes, support_width),
        -torch.inf,
        dtype=torch.float32,
        device="cpu",
    )
    conditional_support_log_probs = torch.full_like(base_support_log_probs, -torch.inf)
    base_other_log_probs = torch.empty(
        (nodes, parent_samples, nodes), dtype=torch.float32, device="cpu"
    )
    conditional_other_log_probs = torch.empty_like(base_other_log_probs)
    base_full_log_probs = torch.log_softmax(base_logits.float(), dim=-1)
    for parent_index in range(nodes):
        parent_seed = (
            seed
            + int(record["dataset_index"]) * 1009
            + int(record["step_index"]) * 31
            + parent_index
        )
        seed_everything(parent_seed)
        probabilities = torch.softmax(base_logits[parent_index].float(), dim=-1)
        parent_tokens = torch.multinomial(
            probabilities, parent_samples, replacement=True
        )
        sampled_parent_tokens[parent_index] = parent_tokens.cpu()
        parent_probabilities[parent_index] = probabilities[parent_tokens].cpu()
        for sample_index, sampled_token in enumerate(parent_tokens):
            counterfactual = state_from_record(adapter, record)
            counterfactual.input_ids[0, candidates[parent_index]] = sampled_token
            conditional_output = adapter.forward_state(counterfactual, need_hidden=False)
            conditional_logits = conditional_output.aligned_logits[0, candidates]
            conditional_full_log_probs = torch.log_softmax(conditional_logits.float(), dim=-1)
            dependency = categorical_kl_from_logits(conditional_logits, base_logits)
            dependency[parent_index] = 0
            directed_samples[parent_index, sample_index] = dependency
            ids, log_probs, tail = topk_log_probs_with_tail(conditional_logits, top_k)
            conditional_top_ids[parent_index, sample_index] = ids.cpu()
            conditional_top_log_probs[parent_index, sample_index] = log_probs.cpu()
            conditional_tail[parent_index, sample_index] = tail.cpu()
            conditional_target_nll[parent_index, sample_index] = categorical_nll_from_logits(
                conditional_logits, final_targets
            ).cpu()
            top1_in_base_topk[parent_index, sample_index] = (
                ids[:, :1] == base_top_ids
            ).any(dim=-1).cpu()
            for child_index in range(nodes):
                union = torch.unique(
                    torch.cat(
                        [
                            base_top_ids[child_index],
                            ids[child_index],
                            final_targets[child_index].view(1),
                        ]
                    ),
                    sorted=True,
                )
                width = int(union.numel())
                cache_index = (parent_index, sample_index, child_index)
                support_ids[cache_index][:width] = union.cpu()
                support_mask[cache_index][:width] = True
                base_values = base_full_log_probs[child_index, union]
                conditional_values = conditional_full_log_probs[child_index, union]
                base_support_log_probs[cache_index][:width] = base_values.cpu()
                conditional_support_log_probs[cache_index][:width] = conditional_values.cpu()
                base_remaining = (1.0 - base_values.exp().sum()).clamp_min(1e-12)
                conditional_remaining = (1.0 - conditional_values.exp().sum()).clamp_min(1e-12)
                base_other_log_probs[cache_index] = base_remaining.log().cpu()
                conditional_other_log_probs[cache_index] = conditional_remaining.log().cpu()
    directed = directed_samples.mean(dim=1)
    symmetric = 0.5 * (directed + directed.transpose(0, 1))
    result = {
        key: value
        for key, value in record.items()
        if key not in {"aligned_hidden", "base_topk_logits"}
    }
    result.update(
        {
            "target_type": f"mc_kl_q_conditional_to_q_base_s{parent_samples}",
            "parent_sampling_distribution": "teacher_posterior",
            "parent_samples": parent_samples,
            "base_topk_ids": clone_cpu(base_top_ids),
            "base_topk_log_probs": clone_cpu(base_top_log_probs),
            "base_tail_mass": clone_cpu(base_tail),
            "candidate_confidence": clone_cpu(base_top_log_probs[:, 0].exp()),
            "candidate_confidence_type": "max_token_probability",
            "base_target_nll": clone_cpu(base_target_nll),
            "sampled_parent_tokens": sampled_parent_tokens,
            "parent_probabilities": parent_probabilities,
            "directed_dependency_samples": clone_cpu(directed_samples),
            "conditional_topk_ids": conditional_top_ids,
            "conditional_topk_log_probs": conditional_top_log_probs,
            "conditional_tail_mass": conditional_tail,
            "conditional_target_nll": conditional_target_nll,
            "conditional_top1_in_base_topk": top1_in_base_topk,
            "support_ids": support_ids,
            "support_mask": support_mask,
            "base_support_log_probs": base_support_log_probs,
            "conditional_support_log_probs": conditional_support_log_probs,
            "base_other_log_probs": base_other_log_probs,
            "conditional_other_log_probs": conditional_other_log_probs,
            "directed_dependency": clone_cpu(directed),
            "symmetric_dependency": clone_cpu(symmetric),
        }
    )
    return result
