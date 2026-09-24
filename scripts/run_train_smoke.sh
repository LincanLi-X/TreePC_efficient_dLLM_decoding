#!/usr/bin/env bash
set -euo pipefail

: "${TREEPC_MODEL_DIR:?Set TREEPC_MODEL_DIR to the complete Dream-7B checkpoint directory.}"
: "${TREEPC_DATA_ROOT:?Set TREEPC_DATA_ROOT to the directory containing gsm8k/ and humaneval/.}"
: "${TREEPC_RUN_DIR:?Set TREEPC_RUN_DIR to a new output directory on /blue.}"

treepc_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
treepc_python="${TREEPC_PYTHON:-${treepc_project_dir}/.venv/bin/python}"
treepc_steps_csv="${TREEPC_EVAL_STEPS:-4,16}"
IFS=',' read -r -a treepc_steps <<< "${treepc_steps_csv}"

if [[ ! -x "${treepc_python}" ]]; then
  echo "Python environment not found: ${treepc_python}" >&2
  exit 2
fi

export TREEPC_MODEL_DIR TREEPC_DATA_ROOT TREEPC_RUN_DIR
export TREEPC_PYTHON="${treepc_python}"
export TREEPC_GPU_IDS="${TREEPC_GPU_IDS:-0}"
export TREEPC_PIPELINE_CONFIG="configs/train_smoke/pipeline.yaml"
export TREEPC_PC_CONFIG="configs/train_smoke/pc_lora.yaml"
export TREEPC_DEPENDENCY_CONFIG="configs/train_smoke/dependency_head.yaml"
export TREEPC_CORRECTION_CONFIG="configs/train_smoke/correction_head.yaml"
export TREEPC_TEACHER_STEPS=128
export TREEPC_TEACHER_STATES_PER_EXAMPLE=4
export TREEPC_TEACHER_CANDIDATE_SIZE=8
export TREEPC_TOP_K=16
export TREEPC_PARENT_SAMPLES=3
export TREEPC_HEAD_STEPS="${TREEPC_HEAD_STEPS:-4,16}"
export TREEPC_EVAL_STEPS="${treepc_steps_csv}"
export TREEPC_MIN_FREE_MEMORY_GIB="${TREEPC_MIN_FREE_MEMORY_GIB:-40}"
export PYTHONPATH="${treepc_project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${TREEPC_HF_HOME:-${treepc_project_dir}/.hf_cache}"

mkdir -p "${TREEPC_RUN_DIR}" "${HF_HOME}"
cd "${treepc_project_dir}"

if [[ -f "${TREEPC_RUN_DIR}/COMPLETE.json" ]]; then
  echo "TreePC from-scratch smoke run already complete: ${TREEPC_RUN_DIR}"
  exit 0
fi

"${treepc_python}" scripts/23_check_assets.py \
  --minimum-dataset-rows 12 \
  --minimum-free-memory-gib "${TREEPC_MIN_FREE_MEMORY_GIB}" \
  --output "${TREEPC_RUN_DIR}/preflight.json"

if [[ ! -f "${TREEPC_RUN_DIR}/dream_smoke.json" ]]; then
  "${treepc_python}" scripts/01_smoke_dream.py \
    --device cuda:0 --max-examples 1 --output "${TREEPC_RUN_DIR}/dream_smoke.json"
fi

bash scripts/run_large_v2_impl.sh all

"${treepc_python}" scripts/24_validate_train_run.py \
  --run-dir "${TREEPC_RUN_DIR}" \
  --steps "${treepc_steps[@]}" \
  --output "${TREEPC_RUN_DIR}/COMPLETE.json"

echo "TreePC training and held-out evaluation are complete."
echo "Summary: ${TREEPC_RUN_DIR}/reports/head_test/summary.csv"
echo "Checkpoints: ${TREEPC_RUN_DIR}/checkpoints"
