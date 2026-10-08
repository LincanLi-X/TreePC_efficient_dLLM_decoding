#!/usr/bin/env bash
set -euo pipefail

# Canonical large-v2 entry point. The implementation is kept separate so the
# smoke profile can reuse the same orchestration without duplicating it.
treepc_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TREEPC_PIPELINE_CONFIG="${TREEPC_PIPELINE_CONFIG:-configs/large_v2/pipeline.yaml}"
export TREEPC_PC_CONFIG="${TREEPC_PC_CONFIG:-configs/large_v2/pc_lora.yaml}"
export TREEPC_DEPENDENCY_CONFIG="${TREEPC_DEPENDENCY_CONFIG:-configs/large_v2/dependency_head.yaml}"
export TREEPC_CORRECTION_CONFIG="${TREEPC_CORRECTION_CONFIG:-configs/large_v2/correction_head.yaml}"
export TREEPC_TEACHER_STEPS="${TREEPC_TEACHER_STEPS:-256}"
export TREEPC_HEAD_STEPS="${TREEPC_HEAD_STEPS:-8,16,32,64}"
export TREEPC_EVAL_STEPS="${TREEPC_EVAL_STEPS:-8,16,32,64}"
export TREEPC_RQ1_RECALL_K="${TREEPC_RQ1_RECALL_K:-16}"
export TREEPC_RQ1_ECE_BINS="${TREEPC_RQ1_ECE_BINS:-15}"
exec bash "${treepc_project_dir}/scripts/run_large_v2_impl.sh" "$@"
