from __future__ import annotations

import time
from typing import Any

import torch

from treepc.dream.adapter import DreamAdapter, max_token_probability, sample_tokens
from treepc.dream.generation import custom_independent_generate
from treepc.graph.chow_liu import maximum_spanning_tree
from treepc.graph.orientation import orient_tree
from treepc.models.treepc_bundle import TreePCBundle
from treepc.types import GenerationResult


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


@torch.inference_mode()
def learned_treepc_generate(
    adapter: DreamAdapter,
    bundle: TreePCBundle,
    prompt: str,
    *,
    steps: int,
    max_new_tokens: int,
    alg: str = "entropy",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    alg_temp: float | None = 0.0,
    correction_top_k: int = 16,
    min_tree_nodes: int = 2,
    tree_gate: bool = True,
    edge_source: str = "learned",
) -> tuple[GenerationResult, list[dict[str, Any]], dict[str, float]]:
    if not tree_gate:
        result = custom_independent_generate(
            adapter,
            prompt,
            steps=steps,
            max_new_tokens=max_new_tokens,
            alg=alg,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            alg_temp=alg_temp,
        )
        return result, [], {
            "dependency_head_s": 0.0,
            "mst_s": 0.0,
            "correction_head_s": 0.0,
            "dream_forwards": float(steps),
        }
    encoded = adapter.encode_prompt(prompt)
    state = adapter.make_initial_state(encoded, max_new_tokens, steps)
    timesteps = torch.linspace(1, 1e-3, steps + 1, device=adapter.device)
    trace: list[dict[str, Any]] = []
    timings = {"dependency_head_s": 0.0, "mst_s": 0.0, "correction_head_s": 0.0}
    _synchronize(adapter.device)
    started = time.perf_counter()
    for step in range(steps):
        state.step_index = step
        state.timestep_t = timesteps[step]
        state.timestep_s = timesteps[step + 1]
        output = adapter.forward_state(state, need_hidden=True)
        confidence, proposals = adapter.propose_tokens_and_confidence(
            output, alg, temperature, top_p, top_k
        )
        mask = output.masked_positions
        budget = adapter.compute_commit_budget(
            mask, timesteps[step], timesteps[step + 1], step == steps - 1
        )
        if budget == 0:
            trace.append(
                {
                    "step": step,
                    "candidate_size": 0,
                    "tree_used": False,
                    "tree_depth": 0,
                    "dependency_sum": 0.0,
                    "corrected_token_flips": 0,
                }
            )
            continue
        full_confidence = torch.full_like(
            state.input_ids, -torch.inf, dtype=output.aligned_logits.dtype
        )
        full_confidence[mask] = confidence
        if alg_temp is None or alg_temp == 0:
            candidates = torch.topk(full_confidence, budget, dim=-1).indices[0]
        else:
            probabilities = torch.softmax(full_confidence / alg_temp, dim=-1)
            candidates = torch.multinomial(probabilities, num_samples=budget)[0]
        candidates = torch.sort(candidates).values
        proposed_full = torch.full_like(state.input_ids, adapter.mask_token_id)
        proposed_full[mask] = proposals
        base_tokens = proposed_full[0, candidates]
        base_logits = output.aligned_logits[0, candidates]
        hidden = output.aligned_hidden[0, candidates]
        # Root confidence is defined by the PC posterior, independently of the
        # Dream confidence heuristic used to schedule candidate commits.
        node_confidence = max_token_probability(base_logits)
        nodes = int(candidates.numel())

        _synchronize(adapter.device)
        section = time.perf_counter()
        if edge_source == "learned":
            weights = bundle.dependency_head(hidden, candidates, timesteps[step])
        elif edge_source == "local":
            distance = (candidates[:, None] - candidates[None, :]).abs().float()
            weights = 1.0 / (distance + 1.0)
            weights.fill_diagonal_(0.0)
        else:
            raise ValueError(f"Unsupported edge source: {edge_source}")
        _synchronize(adapter.device)
        timings["dependency_head_s"] += time.perf_counter() - section

        section = time.perf_counter()
        prim_parent, edge_weights = maximum_spanning_tree(weights)
        root = int(node_confidence.argmax().item())
        parent, depth = orient_tree(prim_parent, root)
        dependency_sum = float(edge_weights.sum().item())
        _synchronize(adapter.device)
        timings["mst_s"] += time.perf_counter() - section

        use_tree = nodes >= min_tree_nodes and dependency_sum >= bundle.dependency_threshold
        if not use_tree:
            chosen = base_tokens.clone()
            corrected_flips = 0
            mean_gate = 0.0
        else:
            chosen = torch.full_like(base_tokens, adapter.mask_token_id)
            chosen[root] = base_tokens[root]
            corrected_flips = 0
            gate_values: list[torch.Tensor] = []
            correction_norms: list[torch.Tensor] = []
            section = time.perf_counter()
            for level in range(1, int(depth.max().item()) + 1):
                children = (depth == level).nonzero(as_tuple=False).flatten()
                parents = parent[children]
                child_logits = base_logits[children]
                values, ids = torch.topk(
                    child_logits, min(correction_top_k, child_logits.shape[-1]), dim=-1
                )
                corrected, gate, effective_correction = bundle.correction_head(
                    child_hidden=hidden[children],
                    parent_hidden=hidden[parents],
                    parent_tokens=chosen[parents],
                    relative_position=(candidates[children] - candidates[parents]).float(),
                    sequence_length=torch.full(
                        (children.numel(),),
                        state.input_ids.shape[-1],
                        dtype=torch.float32,
                        device=adapter.device,
                    ),
                    timestep=timesteps[step].expand(children.numel()),
                    dependency_weight=weights[parents, children],
                    token_ids=ids,
                    base_logits=values,
                )
                _, selected_index = sample_tokens(corrected, temperature, None, None, "origin")
                selected = ids.gather(-1, selected_index.unsqueeze(-1)).squeeze(-1)
                chosen[children] = selected
                corrected_flips += int(selected.ne(base_tokens[children]).sum().item())
                gate_values.append(gate)
                correction_norms.append(effective_correction.norm(dim=-1))
            _synchronize(adapter.device)
            timings["correction_head_s"] += time.perf_counter() - section
            mean_gate = float(torch.cat(gate_values).mean().item()) if gate_values else 0.0
            mean_correction_l2 = (
                float(torch.cat(correction_norms).mean().item()) if correction_norms else 0.0
            )
        global_scale = bundle.correction_head.global_scale_value()
        state.input_ids[0, candidates] = chosen
        trace.append(
            {
                "step": step,
                "candidate_size": nodes,
                "tree_used": use_tree,
                "tree_depth": int(depth.max().item()),
                "dependency_sum": dependency_sum,
                "corrected_token_flips": corrected_flips,
                "mean_gate": mean_gate,
                "global_scale": global_scale,
                "mean_effective_gate": global_scale * mean_gate,
                "mean_effective_correction_l2": mean_correction_l2 if use_tree else 0.0,
                "root_candidate_index": root,
                "root_position": int(candidates[root].item()),
                "root_probability": float(node_confidence[root].item()),
                "edge_source": edge_source,
            }
        )
    _synchronize(adapter.device)
    latency = time.perf_counter() - started
    timings["dream_forwards"] = float(steps)
    result = GenerationResult(
        sequences=state.input_ids,
        texts=adapter.decode(state.input_ids, state.prompt_length),
        latency_seconds=latency,
        effective_nfe=steps,
    )
    return result, trace, timings
