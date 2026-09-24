from __future__ import annotations

from typing import Any


def validate_trajectory_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("schema") != "treepc.teacher_trajectory.v1":
        raise ValueError("Unsupported teacher trajectory schema")
    required = {
        "sample_id",
        "state_token_ids",
        "candidate_positions",
        "candidate_confidence",
        "candidate_confidence_type",
        "base_topk_ids",
        "aligned_hidden",
        "teacher_next_commit_positions",
        "final_teacher_tokens",
    }
    for record in bundle.get("records", []):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Trajectory record missing {sorted(missing)}")
        if record["candidate_confidence_type"] != "max_token_probability":
            raise ValueError("Tree root confidence must be max_token_probability")


def validate_counterfactual_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("schema") != "treepc.counterfactual.v1":
        raise ValueError("Unsupported counterfactual schema")
    for record in bundle.get("records", []):
        if record.get("candidate_confidence_type") != "max_token_probability":
            raise ValueError("Tree root confidence must be max_token_probability")
        nodes = int(record["candidate_positions"].numel())
        if tuple(record["directed_dependency"].shape) != (nodes, nodes):
            raise ValueError("Invalid directed dependency shape")
        if tuple(record["symmetric_dependency"].shape) != (nodes, nodes):
            raise ValueError("Invalid symmetric dependency shape")
        if "support_ids" in record:
            parent_tokens = record.get("sampled_parent_tokens")
            multi_sample = parent_tokens is not None and parent_tokens.ndim == 2
            samples = int(parent_tokens.shape[1]) if multi_sample else 1
            expected_prefix = (nodes, samples, nodes) if multi_sample else (nodes, nodes)
            if tuple(record["support_ids"].shape[: len(expected_prefix)]) != expected_prefix:
                raise ValueError("Invalid conditional support shape")
            for key in (
                "support_mask",
                "base_support_log_probs",
                "conditional_support_log_probs",
            ):
                if tuple(record[key].shape) != tuple(record["support_ids"].shape):
                    raise ValueError(f"Invalid {key} shape")
            expected_other = expected_prefix
            for key in ("base_other_log_probs", "conditional_other_log_probs"):
                if tuple(record[key].shape) != expected_other:
                    raise ValueError(f"Invalid {key} shape")
            if multi_sample:
                if int(record.get("parent_samples", samples)) != samples:
                    raise ValueError("Counterfactual parent sample count mismatch")
                if record.get("parent_sampling_distribution") != "teacher_posterior":
                    raise ValueError("Multi-sample parents must come from the Teacher posterior")
                parent_probabilities = record.get("parent_probabilities")
                if parent_probabilities is None or tuple(parent_probabilities.shape) != (
                    nodes,
                    samples,
                ):
                    raise ValueError("Invalid sampled parent probability shape")
                dependency_samples = record.get("directed_dependency_samples")
                if dependency_samples is None or tuple(dependency_samples.shape) != expected_prefix:
                    raise ValueError("Invalid directed dependency sample shape")
