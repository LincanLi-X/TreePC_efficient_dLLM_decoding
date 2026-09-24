from __future__ import annotations

from typing import Any

import torch

from treepc.dream.adapter import DreamAdapter, max_token_probability
from treepc.posterior.topk_support import topk_log_probs_with_tail
from treepc.types import clone_cpu


def stratified_step_indices(total_steps: int, states_per_example: int) -> list[int]:
    if states_per_example <= 0 or states_per_example > total_steps:
        raise ValueError("states_per_example must be in [1, total_steps]")
    return [
        int((2 * index + 1) * total_steps / (2 * states_per_example))
        for index in range(states_per_example)
    ]


@torch.inference_mode()
def collect_teacher_trajectory(
    adapter: DreamAdapter,
    prompt: str,
    sample_id: str,
    dataset_index: int,
    *,
    teacher_steps: int = 256,
    max_new_tokens: int,
    states_per_example: int = 4,
    candidate_size: int = 8,
    top_k: int = 16,
    seed: int = 2026,
) -> tuple[list[dict[str, Any]], torch.Tensor, str]:
    encoded = adapter.encode_prompt(prompt)
    state = adapter.make_initial_state(encoded, max_new_tokens, teacher_steps)
    timesteps = torch.linspace(1, 1e-3, teacher_steps + 1, device=adapter.device)
    selected_steps = set(stratified_step_indices(teacher_steps, states_per_example))
    records: list[dict[str, Any]] = []
    for step in range(teacher_steps):
        state.step_index = step
        state.timestep_t = timesteps[step]
        state.timestep_s = timesteps[step + 1]
        output = adapter.forward_state(state, need_hidden=step in selected_steps)
        scheduler_confidence, proposals = adapter.propose_tokens_and_confidence(
            output, "entropy", 0.0, None, None
        )
        mask = output.masked_positions
        full_confidence = torch.full_like(state.input_ids, -torch.inf, dtype=output.aligned_logits.dtype)
        full_confidence[mask] = scheduler_confidence
        proposed_full = torch.full_like(state.input_ids, adapter.mask_token_id)
        proposed_full[mask] = proposals
        remaining = int(mask.sum().item())
        is_last = step == teacher_steps - 1
        budget = adapter.compute_commit_budget(mask, timesteps[step], timesteps[step + 1], is_last)
        commit_positions = torch.topk(full_confidence, budget, dim=-1).indices if budget else torch.empty(
            (1, 0), dtype=torch.long, device=adapter.device
        )
        if step in selected_steps:
            marginal_positions = mask[0].nonzero(as_tuple=False).flatten()
            marginal_logits = output.aligned_logits[0, marginal_positions]
            marginal_top_ids, marginal_top_log_probs, marginal_tail_mass = (
                topk_log_probs_with_tail(marginal_logits, top_k)
            )
            count = min(candidate_size, remaining)
            candidate_positions = torch.topk(full_confidence, count, dim=-1).indices[0]
            candidate_positions = torch.sort(candidate_positions).values
            candidate_logits = output.aligned_logits[0, candidate_positions]
            top_ids, top_log_probs, tail_mass = topk_log_probs_with_tail(candidate_logits, top_k)
            records.append(
                {
                    "sample_id": sample_id,
                    "dataset_index": dataset_index,
                    "seed": seed,
                    "prompt_token_ids": clone_cpu(encoded["input_ids"][0]),
                    "state_token_ids": clone_cpu(state.input_ids[0]),
                    "generation_mask": clone_cpu(state.generation_mask[0]),
                    "prompt_length": state.prompt_length,
                    "step_index": step,
                    "total_teacher_steps": teacher_steps,
                    "timestep": float(timesteps[step].item()),
                    "mask_ratio": remaining / max_new_tokens,
                    "masked_positions": clone_cpu(mask[0].nonzero(as_tuple=False).flatten()),
                    # RQ1 uses all still-masked positions rather than only the
                    # small dependency-candidate subset.  Keeping Top-K plus
                    # an exact OTHER mass avoids caching full-vocabulary logits.
                    "marginal_positions": clone_cpu(marginal_positions),
                    "marginal_topk_ids": clone_cpu(marginal_top_ids),
                    "marginal_topk_log_probs": clone_cpu(marginal_top_log_probs),
                    "marginal_tail_mass": clone_cpu(marginal_tail_mass),
                    "candidate_positions": clone_cpu(candidate_positions),
                    "candidate_confidence": clone_cpu(
                        max_token_probability(candidate_logits)
                    ),
                    "candidate_confidence_type": "max_token_probability",
                    "base_sampled_tokens": clone_cpu(proposed_full[0, candidate_positions]),
                    "base_topk_ids": clone_cpu(top_ids),
                    "base_topk_log_probs": clone_cpu(top_log_probs),
                    "base_tail_mass": clone_cpu(tail_mass),
                    "aligned_hidden": clone_cpu(output.aligned_hidden[0, candidate_positions]),
                    "teacher_next_commit_positions": clone_cpu(commit_positions[0]),
                    "teacher_next_tokens": clone_cpu(proposed_full[0, commit_positions[0]]),
                    "dtype": str(adapter.dtype),
                }
            )
        if budget:
            rows = torch.zeros_like(commit_positions)
            state.input_ids[rows, commit_positions] = proposed_full[rows, commit_positions]
    final_tokens = clone_cpu(state.input_ids[0])
    for record in records:
        record["final_teacher_tokens"] = final_tokens
    final_text = adapter.decode(state.input_ids, state.prompt_length)[0]
    return records, final_tokens, final_text
