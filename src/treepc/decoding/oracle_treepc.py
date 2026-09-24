from __future__ import annotations

import time
from typing import Any

import torch

from treepc.data.counterfactual import state_from_record
from treepc.dream.adapter import DreamAdapter, max_token_probability
from treepc.graph.chow_liu import maximum_spanning_tree
from treepc.graph.orientation import orient_tree
from treepc.posterior.divergences import categorical_kl_from_logits
from treepc.types import GenerationResult
from treepc.utils.seed import seed_everything


def _counterfactual_logits(
    adapter: DreamAdapter,
    state_record: dict[str, Any],
    parent_position: torch.Tensor,
    parent_token: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    counterfactual = state_from_record(adapter, state_record)
    counterfactual.input_ids[0, parent_position] = parent_token
    output = adapter.forward_state(counterfactual, need_hidden=False)
    return output.aligned_logits[0, candidates]


@torch.inference_mode()
def online_oracle_generate(
    adapter: DreamAdapter,
    prompt: str,
    *,
    steps: int,
    max_new_tokens: int,
    seed: int = 2026,
) -> tuple[GenerationResult, list[dict[str, Any]], int]:
    """Diagnostic online Oracle. Extra Teacher forwards are explicitly counted."""
    encoded = adapter.encode_prompt(prompt)
    state = adapter.make_initial_state(encoded, max_new_tokens, steps)
    timesteps = torch.linspace(1, 1e-3, steps + 1, device=adapter.device)
    trace: list[dict[str, Any]] = []
    extra_teacher_forwards = 0
    torch.cuda.synchronize(adapter.device)
    started = time.perf_counter()
    for step in range(steps):
        state.step_index = step
        state.timestep_t = timesteps[step]
        state.timestep_s = timesteps[step + 1]
        base = adapter.forward_state(state, need_hidden=False)
        confidence, proposals = adapter.propose_tokens_and_confidence(
            base, "entropy", 0.0, None, None
        )
        mask = base.masked_positions
        budget = adapter.compute_commit_budget(
            mask, timesteps[step], timesteps[step + 1], step == steps - 1
        )
        full_confidence = torch.full_like(state.input_ids, -torch.inf, dtype=base.aligned_logits.dtype)
        full_confidence[mask] = confidence
        proposed_full = torch.full_like(state.input_ids, adapter.mask_token_id)
        proposed_full[mask] = proposals
        candidates = torch.topk(full_confidence, budget, dim=-1).indices[0]
        candidates = torch.sort(candidates).values
        base_logits = base.aligned_logits[0, candidates]
        base_tokens = proposed_full[0, candidates]
        node_confidence = max_token_probability(base_logits)
        nodes = candidates.numel()
        state_record = {
            "state_token_ids": state.input_ids[0].detach().cpu(),
            "generation_mask": state.generation_mask[0].detach().cpu(),
            "prompt_length": state.prompt_length,
            "step_index": step,
            "timestep": float(timesteps[step].item()),
        }
        directed = torch.zeros((nodes, nodes), dtype=torch.float32, device=adapter.device)
        sampled_tokens = torch.empty(nodes, dtype=torch.long, device=adapter.device)
        probe_logits: list[torch.Tensor] = []
        for parent_index in range(nodes):
            seed_everything(seed + step * 1009 + parent_index)
            parent_probs = torch.softmax(base_logits[parent_index].float(), dim=-1)
            sampled_tokens[parent_index] = torch.multinomial(parent_probs, 1).squeeze(0)
            conditional = _counterfactual_logits(
                adapter,
                state_record,
                candidates[parent_index],
                sampled_tokens[parent_index],
                candidates,
            )
            extra_teacher_forwards += 1
            dependency = categorical_kl_from_logits(conditional, base_logits)
            dependency[parent_index] = 0
            directed[parent_index] = dependency
            probe_logits.append(conditional)
        symmetric = 0.5 * (directed + directed.transpose(0, 1))
        prim_parent, edge_weight = maximum_spanning_tree(symmetric)
        root = int(node_confidence.argmax().item())
        parent, depth = orient_tree(prim_parent, root)
        chosen = torch.full((nodes,), adapter.mask_token_id, dtype=torch.long, device=adapter.device)
        chosen[root] = base_tokens[root]
        corrected_flips = 0
        for level in range(1, int(depth.max().item()) + 1):
            children_at_level = (depth == level).nonzero(as_tuple=False).flatten()
            for parent_index in parent[children_at_level].unique().tolist():
                children = children_at_level[parent[children_at_level] == parent_index]
                if chosen[parent_index].equal(sampled_tokens[parent_index]):
                    conditional = probe_logits[parent_index]
                else:
                    conditional = _counterfactual_logits(
                        adapter,
                        state_record,
                        candidates[parent_index],
                        chosen[parent_index],
                        candidates,
                    )
                    extra_teacher_forwards += 1
                corrected = conditional[children].argmax(dim=-1)
                corrected_flips += int(corrected.ne(base_tokens[children]).sum().item())
                chosen[children] = corrected
        state.input_ids[0, candidates] = chosen
        trace.append(
            {
                "step": step,
                "candidate_size": nodes,
                "tree_depth": int(depth.max().item()),
                "tree_dependency_sum": float(edge_weight.sum().item()),
                "corrected_token_flips": corrected_flips,
                "root_candidate_index": root,
                "root_position": int(candidates[root].item()),
                "root_probability": float(node_confidence[root].item()),
                "cumulative_extra_teacher_forwards": extra_teacher_forwards,
            }
        )
    torch.cuda.synchronize(adapter.device)
    latency = time.perf_counter() - started
    result = GenerationResult(
        sequences=state.input_ids,
        texts=adapter.decode(state.input_ids, state.prompt_length),
        latency_seconds=latency,
        effective_nfe=steps,
    )
    return result, trace, extra_teacher_forwards
