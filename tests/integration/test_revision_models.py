"""Small real-model training/decoding test; all outputs go to pytest's external tmpdir."""

import os
from pathlib import Path

import pytest
import torch

from treepc.backbones.base import load_adapter
from treepc.data.cache_dataset import CorrectionCacheDataset, DependencyCacheDataset
from treepc.data.onpolicy import collect_pc_mid_state, label_pc_onpolicy_record
from treepc.data.trajectory import collect_teacher_trajectory
from treepc.decoding.learned_treepc import learned_treepc_generate
from treepc.dream.generation import custom_independent_generate
from treepc.models.correction_head import ConditionalCorrectionHead
from treepc.models.dependency_head import DependencyHead
from treepc.models.pc_lora import create_pc_lora_student
from treepc.models.treepc_bundle import TreePCBundle
from treepc.training.correction_trainer import attach_predicted_weights, train_correction_head
from treepc.training.dependency_trainer import train_dependency_head
from treepc.training.onpolicy_pc import RolloutBuffer, train_mixed_budget_pc
from treepc.utils.io import write_json


@pytest.mark.model
@pytest.mark.cuda
@pytest.mark.parametrize("backbone,directory", [("dream", "DREAM-7B"), ("llada", "LLaDA-8B")])
def test_real_revision_pipeline(backbone, directory, monkeypatch, tmp_path):
    root = os.environ.get("TREEPC_TEST_MODEL_ROOT")
    if root is None:
        pytest.skip("Set TREEPC_TEST_MODEL_ROOT to the external checkpoint parent")
    monkeypatch.setenv("TREEPC_BACKBONE", backbone)
    monkeypatch.setenv("TREEPC_MODEL_DIR", str(Path(root) / directory))
    adapter = load_adapter("cuda:0")
    try:
        adapter.model = create_pc_lora_student(adapter.model, rank=2, alpha=4, dropout=0)
        config = {
            "student_budgets": [2, 4],
            "rollout_buffer_size": 2,
            "rollout_refresh_interval": 1,
            "max_anchors": 4,
            "top_k": 4,
            "total_updates": 3,
            "warmup_fraction": 0.2,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "seed": 7,
            "validation_interval": 3,
        }
        buffer = RolloutBuffer(adapter, [("What is 2+2?", 16)], config, 7)
        buffer.refresh()
        with adapter.model.disable_adapter():
            teacher_states, _, _ = collect_teacher_trajectory(
                adapter,
                "What is 2+2?",
                "train/0",
                0,
                teacher_steps=4,
                max_new_tokens=16,
                states_per_example=2,
                candidate_size=4,
                top_k=4,
            )
        coarse, fine = teacher_states
        warmup = [
            {
                "input_ids": coarse["state_token_ids"],
                "anchor_positions": fine["candidate_positions"],
                "teacher_topk_ids": fine["base_topk_ids"],
                "teacher_topk_log_probs": fine["base_topk_log_probs"],
                "teacher_tail_mass": fine["base_tail_mass"],
            }
        ]
        validation = RolloutBuffer(adapter, [("What is 3+3?", 16)], config, 8)
        validation.refresh()
        for row in validation.records:
            row["teacher_final_tokens"] = row["teacher_topk_ids"][:, 0]
        report = train_mixed_budget_pc(
            adapter, warmup, validation.records, [("What is 2+2?", 16)], config, tmp_path / "pc_lora"
        )
        assert report["onpolicy_updates"] == 2 and report["rollout_refreshes"] == 2
        assert report["history"][0]["loss"] > 0
        assert any(torch.count_nonzero(p) for n, p in adapter.model.named_parameters() if "lora_B" in n)
        adapter.model.eval()
        record = collect_pc_mid_state(
            adapter,
            "What is 2+2?",
            "test/0",
            0,
            steps=4,
            max_new_tokens=16,
            seed=7,
            calibration_fold="train",
            candidate_size=2,
        )
        record["dataset"] = "gsm8k"
        with adapter.model.disable_adapter():
            labeled = label_pc_onpolicy_record(adapter, record, top_k=4, parent_samples=3)
        hidden = record["aligned_hidden"].shape[-1]
        dep = DependencyHead(
            hidden_size=hidden,
            projection_size=8,
            timestep_size=4,
            relative_position_size=4,
            pair_hidden_size=8,
        )
        dep_report = train_dependency_head(
            dep,
            DependencyCacheDataset([labeled]),
            DependencyCacheDataset([labeled]),
            device=adapter.device,
            output=tmp_path / "dep.pt",
            epochs=1,
        )
        dep.to(adapter.device).eval()
        attach_predicted_weights(dep, [labeled], adapter.device)
        corr = ConditionalCorrectionHead(
            adapter.model.get_input_embeddings().weight,
            adapter.model.get_output_embeddings().weight,
            hidden_size=hidden,
            rank=2,
            feature_size=4,
        )
        corr_dataset = CorrectionCacheDataset([labeled], identity_fraction=0)
        corr_report = train_correction_head(
            corr,
            corr_dataset,
            corr_dataset,
            device=adapter.device,
            output=tmp_path / "corr.pt",
            epochs=1,
            batch_size=2,
        )
        corr.to(adapter.device).eval()
        bundle = TreePCBundle(dep, corr, dependency_threshold=0)
        count = adapter.forward_count
        result, trace, _ = learned_treepc_generate(
            adapter, bundle, "What is 2+2?", steps=4, max_new_tokens=16
        )
        assert adapter.forward_count - count == 4
        assert result.effective_nfe == 4 and len(trace) == 4
        independent = custom_independent_generate(adapter, "What is 2+2?", steps=4, max_new_tokens=16)
        fallback, _, _ = learned_treepc_generate(
            adapter, bundle, "What is 2+2?", steps=4, max_new_tokens=16, tree_gate=False
        )
        assert torch.equal(independent.sequences, fallback.sequences)
        assert not result.sequences[0, -16:].eq(adapter.mask_token_id).any()
        write_json(
            tmp_path / "validation.json",
            {
                "backbone": backbone,
                "pc": report,
                "dependency": dep_report,
                "correction": corr_report,
                "gpu": torch.cuda.get_device_name(),
                "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
            },
        )
    finally:
        adapter.close()
