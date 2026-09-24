from __future__ import annotations

from pathlib import Path

import torch

from treepc.dream.adapter import DreamAdapter
from treepc.models.correction_head import ConditionalCorrectionHead
from treepc.models.dependency_head import DependencyHead


class TreePCBundle:
    def __init__(
        self,
        dependency_head: DependencyHead,
        correction_head: ConditionalCorrectionHead,
        dependency_threshold: float | None = None,
    ) -> None:
        self.dependency_head = dependency_head
        self.correction_head = correction_head
        self.dependency_threshold = float(dependency_threshold or 0.0)

    @classmethod
    def from_checkpoints(
        cls,
        adapter: DreamAdapter,
        dependency_checkpoint: str | Path,
        correction_checkpoint: str | Path,
        *,
        dependency_threshold: float | None = None,
    ) -> "TreePCBundle":
        dependency_value = torch.load(dependency_checkpoint, map_location="cpu", weights_only=False)
        correction_value = torch.load(correction_checkpoint, map_location="cpu", weights_only=False)
        if dependency_value.get("schema") != "treepc.dependency_head.v1":
            raise ValueError("Unsupported dependency checkpoint")
        if correction_value.get("schema") != "treepc.correction_head.v1":
            raise ValueError("Unsupported correction checkpoint")
        dependency = DependencyHead(**dependency_value["config"])
        dependency.load_compatible_state_dict(dependency_value["state_dict"])
        correction = ConditionalCorrectionHead(
            adapter.model.get_input_embeddings().weight,
            adapter.model.get_output_embeddings().weight,
            **correction_value["config"],
        )
        correction.load_compatible_state_dict(correction_value["state_dict"])
        dependency.to(adapter.device).eval()
        correction.to(adapter.device).eval()
        if dependency_threshold is None:
            dependency_threshold = correction_value.get("tree_gate_calibration", {}).get(
                "dependency_threshold", 0.0
            )
        return cls(dependency, correction, dependency_threshold)
