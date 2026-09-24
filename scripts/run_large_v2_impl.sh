#!/usr/bin/env bash
set -euo pipefail

# Shared multi-GPU pipeline implementation. For a formal large-v2 run, prefer
# scripts/run_large_v2_pipeline.sh. This path remains for smoke compatibility.
# Usage:
#   TREEPC_DATA_ROOT=/path/to/track1_general \
#   TREEPC_MODEL_DIR=/path/to/DREAM-7B \
#   TREEPC_GPU_IDS=0,1,2,3 \
#   bash scripts/run_large_v2_pipeline.sh all

treepc_action="${1:-all}"
: "${TREEPC_DATA_ROOT:?Set TREEPC_DATA_ROOT to the directory containing gsm8k/ and humaneval/.}"
: "${TREEPC_MODEL_DIR:?Set TREEPC_MODEL_DIR to the local Dream-7B-Instruct checkpoint.}"

treepc_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
treepc_run_dir="${TREEPC_RUN_DIR:-${treepc_project_dir}/results/large_v2}"
treepc_python="${TREEPC_PYTHON:-python}"
treepc_gpu_csv="${TREEPC_GPU_IDS:-0}"
IFS=',' read -r -a treepc_gpus <<< "${treepc_gpu_csv}"
treepc_shards="${#treepc_gpus[@]}"
treepc_manifest="${treepc_run_dir}/manifest.json"
treepc_pc_adapter="${treepc_run_dir}/checkpoints/pc_lora"
treepc_dependency="${treepc_run_dir}/checkpoints/dependency_head.pt"
treepc_correction="${treepc_run_dir}/checkpoints/correction_head.pt"
treepc_pipeline_config="${TREEPC_PIPELINE_CONFIG:-configs/large_v2/pipeline.yaml}"
treepc_pc_config="${TREEPC_PC_CONFIG:-configs/large_v2/pc_lora.yaml}"
treepc_dependency_config="${TREEPC_DEPENDENCY_CONFIG:-configs/large_v2/dependency_head.yaml}"
treepc_correction_config="${TREEPC_CORRECTION_CONFIG:-configs/large_v2/correction_head.yaml}"
treepc_teacher_steps="${TREEPC_TEACHER_STEPS:-256}"
treepc_teacher_states="${TREEPC_TEACHER_STATES_PER_EXAMPLE:-4}"
treepc_teacher_candidates="${TREEPC_TEACHER_CANDIDATE_SIZE:-8}"
treepc_top_k="${TREEPC_TOP_K:-16}"
treepc_parent_samples="${TREEPC_PARENT_SAMPLES:-3}"
treepc_rq1_recall_k="${TREEPC_RQ1_RECALL_K:-16}"
treepc_rq1_ece_bins="${TREEPC_RQ1_ECE_BINS:-15}"
treepc_head_steps_csv="${TREEPC_HEAD_STEPS:-8,16,32,64}"
treepc_eval_steps_csv="${TREEPC_EVAL_STEPS:-8,16,32,64}"
treepc_min_free_gib="${TREEPC_MIN_FREE_MEMORY_GIB:-20}"
IFS=',' read -r -a treepc_head_steps <<< "${treepc_head_steps_csv}"
IFS=',' read -r -a treepc_eval_steps <<< "${treepc_eval_steps_csv}"

if [[ ! -x "$(command -v "${treepc_python}" 2>/dev/null)" && ! -x "${treepc_python}" ]]; then
  echo "Python interpreter is not executable: ${treepc_python}" >&2
  exit 2
fi

export TREEPC_DATA_ROOT TREEPC_MODEL_DIR
export PYTHONPATH="${treepc_project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${treepc_run_dir}"/{cache/teacher,cache/pc,cache/heads,checkpoints,reports,logs,eval}
cd "${treepc_project_dir}"

treepc_wait_jobs() {
  local treepc_failure=0
  local treepc_pid
  for treepc_pid in "$@"; do
    if ! wait "${treepc_pid}"; then
      treepc_failure=1
    fi
  done
  if [[ "${treepc_failure}" -ne 0 ]]; then
    echo "At least one GPU shard failed. Inspect ${treepc_run_dir}/logs/." >&2
    return 1
  fi
}

treepc_prepare() {
  CUDA_VISIBLE_DEVICES="${treepc_gpu_csv}" "${treepc_python}" scripts/00_check_environment.py \
    --min-gpus "${treepc_shards}" --min-free-memory-gib "${treepc_min_free_gib}" \
    --output "${treepc_run_dir}/environment.json"
  if [[ ! -f "${treepc_manifest}" ]]; then
    "${treepc_python}" scripts/20_prepare_large_scale.py \
      --config "${treepc_pipeline_config}" --output "${treepc_manifest}"
  else
    "${treepc_python}" scripts/20_prepare_large_scale.py \
      --config "${treepc_pipeline_config}" --output "${treepc_manifest}" --validate-existing
  fi
}

treepc_collect_teacher_partition() {
  local treepc_dataset="$1"
  local treepc_split="$2"
  local treepc_merged="${treepc_run_dir}/cache/teacher/${treepc_dataset}_${treepc_split}.pt"
  local treepc_pc_cache="${treepc_run_dir}/cache/pc/${treepc_dataset}_${treepc_split}.pt"
  if [[ -f "${treepc_merged}" && -f "${treepc_pc_cache}" ]]; then
    echo "teacher/PC cache already complete: ${treepc_dataset} ${treepc_split}"
    return
  fi
  local -a treepc_pids=()
  local treepc_shard treepc_gpu
  for ((treepc_shard=0; treepc_shard<treepc_shards; treepc_shard++)); do
    treepc_gpu="${treepc_gpus[treepc_shard]}"
    CUDA_VISIBLE_DEVICES="${treepc_gpu}" "${treepc_python}" scripts/03_collect_teacher_trajectories.py collect \
      --dataset "${treepc_dataset}" --manifest "${treepc_manifest}" \
      --split "${treepc_split}" --teacher-steps "${treepc_teacher_steps}" \
      --states-per-example "${treepc_teacher_states}" \
      --candidate-size "${treepc_teacher_candidates}" --top-k "${treepc_top_k}" \
      --device cuda:0 --resume \
      --shard-index "${treepc_shard}" --num-shards "${treepc_shards}" \
      --output "${treepc_run_dir}/cache/teacher/${treepc_dataset}_${treepc_split}_${treepc_shard}.pt" \
      --summary "${treepc_run_dir}/cache/teacher/${treepc_dataset}_${treepc_split}_${treepc_shard}.json" \
      >"${treepc_run_dir}/logs/teacher_${treepc_dataset}_${treepc_split}_${treepc_shard}.log" 2>&1 &
    treepc_pids+=("$!")
  done
  treepc_wait_jobs "${treepc_pids[@]}"

  if [[ ! -f "${treepc_merged}" ]]; then
    local -a treepc_inputs=() treepc_summaries=()
    for ((treepc_shard=0; treepc_shard<treepc_shards; treepc_shard++)); do
      treepc_inputs+=("${treepc_run_dir}/cache/teacher/${treepc_dataset}_${treepc_split}_${treepc_shard}.pt")
      treepc_summaries+=("${treepc_run_dir}/cache/teacher/${treepc_dataset}_${treepc_split}_${treepc_shard}.json")
    done
    "${treepc_python}" scripts/03_collect_teacher_trajectories.py merge \
      --inputs "${treepc_inputs[@]}" --summaries "${treepc_summaries[@]}" \
      --manifest "${treepc_manifest}" --dataset "${treepc_dataset}" \
      --output "${treepc_merged}" \
      --summary "${treepc_run_dir}/cache/teacher/${treepc_dataset}_${treepc_split}.json"
  fi
  if [[ ! -f "${treepc_pc_cache}" ]]; then
    "${treepc_python}" scripts/09_train_pc_lora.py build-cache \
      --trajectory "${treepc_merged}" \
      --output "${treepc_pc_cache}" \
      --summary "${treepc_run_dir}/cache/pc/${treepc_dataset}_${treepc_split}.json"
  fi
}

treepc_pc_cache() {
  treepc_collect_teacher_partition gsm8k train
  treepc_collect_teacher_partition gsm8k validation
  treepc_collect_teacher_partition gsm8k test
  treepc_collect_teacher_partition humaneval train
  treepc_collect_teacher_partition humaneval validation
  treepc_collect_teacher_partition humaneval test
}

treepc_train_pc() {
  if [[ ! -f "${treepc_run_dir}/reports/pc_lora.json" \
        || ! -f "${treepc_pc_adapter}/adapter_config.json" \
        || ! -f "${treepc_pc_adapter}/adapter_model.safetensors" ]]; then
    CUDA_VISIBLE_DEVICES="${treepc_gpus[0]}" "${treepc_python}" scripts/09_train_pc_lora.py train \
      --train-caches "${treepc_run_dir}/cache/pc/gsm8k_train.pt" "${treepc_run_dir}/cache/pc/humaneval_train.pt" \
      --validation-caches "${treepc_run_dir}/cache/pc/gsm8k_validation.pt" "${treepc_run_dir}/cache/pc/humaneval_validation.pt" \
      --test-caches "${treepc_run_dir}/cache/pc/gsm8k_test.pt" "${treepc_run_dir}/cache/pc/humaneval_test.pt" \
      --config "${treepc_pc_config}" --device cuda:0 \
      --adapter-dir "${treepc_pc_adapter}" --report "${treepc_run_dir}/reports/pc_lora.json" \
      >"${treepc_run_dir}/logs/train_pc_lora.log" 2>&1
  fi
}

treepc_head_partition() {
  local treepc_dataset="$1"
  local treepc_split="$2"
  local -a treepc_pids=()
  local treepc_shard treepc_gpu
  for ((treepc_shard=0; treepc_shard<treepc_shards; treepc_shard++)); do
    treepc_gpu="${treepc_gpus[treepc_shard]}"
    CUDA_VISIBLE_DEVICES="${treepc_gpu}" "${treepc_python}" scripts/10_recalibrate_heads.py collect \
      --pc-lora "${treepc_pc_adapter}" --manifest "${treepc_manifest}" \
      --dataset "${treepc_dataset}" --split "${treepc_split}" \
      --steps "${treepc_head_steps[@]}" --device cuda:0 --resume \
      --shard-index "${treepc_shard}" --num-shards "${treepc_shards}" \
      --output "${treepc_run_dir}/cache/heads/${treepc_dataset}_${treepc_split}_${treepc_shard}.states.pt" \
      --summary "${treepc_run_dir}/cache/heads/${treepc_dataset}_${treepc_split}_${treepc_shard}.states.json" \
      >"${treepc_run_dir}/logs/head_states_${treepc_dataset}_${treepc_split}_${treepc_shard}.log" 2>&1 &
    treepc_pids+=("$!")
  done
  treepc_wait_jobs "${treepc_pids[@]}"

  treepc_pids=()
  for ((treepc_shard=0; treepc_shard<treepc_shards; treepc_shard++)); do
    treepc_gpu="${treepc_gpus[treepc_shard]}"
    CUDA_VISIBLE_DEVICES="${treepc_gpu}" "${treepc_python}" scripts/10_recalibrate_heads.py label \
      --input "${treepc_run_dir}/cache/heads/${treepc_dataset}_${treepc_split}_${treepc_shard}.states.pt" \
      --output "${treepc_run_dir}/cache/heads/${treepc_dataset}_${treepc_split}_${treepc_shard}.labels.pt" \
      --summary "${treepc_run_dir}/cache/heads/${treepc_dataset}_${treepc_split}_${treepc_shard}.labels.json" \
      --top-k "${treepc_top_k}" --parent-samples "${treepc_parent_samples}" \
      --device cuda:0 --resume \
      >"${treepc_run_dir}/logs/head_labels_${treepc_dataset}_${treepc_split}_${treepc_shard}.log" 2>&1 &
    treepc_pids+=("$!")
  done
  treepc_wait_jobs "${treepc_pids[@]}"
}

treepc_head_cache() {
  local treepc_split treepc_dataset
  for treepc_split in train validation test; do
    for treepc_dataset in gsm8k humaneval; do
      treepc_head_partition "${treepc_dataset}" "${treepc_split}"
    done
  done
}

treepc_rq1_partition() {
  local treepc_split="$1"
  local treepc_report="${treepc_run_dir}/reports/rq1_${treepc_split}.json"
  local treepc_csv="${treepc_run_dir}/reports/rq1_${treepc_split}.csv"
  if [[ -f "${treepc_report}" && -f "${treepc_csv}" ]]; then
    echo "RQ1 report already complete: ${treepc_split}"
    return
  fi
  local -a treepc_labels=()
  mapfile -t treepc_labels < <(treepc_label_paths "${treepc_split}")
  CUDA_VISIBLE_DEVICES="${treepc_gpus[0]}" "${treepc_python}" scripts/25_evaluate_rq1.py \
    --student-state-caches "${treepc_labels[@]}" \
    --teacher-trajectories \
      "${treepc_run_dir}/cache/teacher/gsm8k_${treepc_split}.pt" \
      "${treepc_run_dir}/cache/teacher/humaneval_${treepc_split}.pt" \
    --pc-lora "${treepc_pc_adapter}" --partition "head_${treepc_split}" \
    --steps "${treepc_head_steps[@]}" --teacher-steps "${treepc_teacher_steps}" \
    --recall-k "${treepc_rq1_recall_k}" --ece-bins "${treepc_rq1_ece_bins}" \
    --device cuda:0 --report "${treepc_report}" --csv "${treepc_csv}" \
    >"${treepc_run_dir}/logs/evaluate_rq1_${treepc_split}.log" 2>&1
}

treepc_rq1() {
  treepc_rq1_partition validation
  treepc_rq1_partition test
}

treepc_label_paths() {
  local treepc_split="$1"
  local treepc_dataset treepc_shard
  for treepc_dataset in gsm8k humaneval; do
    for ((treepc_shard=0; treepc_shard<treepc_shards; treepc_shard++)); do
      printf '%s\n' "${treepc_run_dir}/cache/heads/${treepc_dataset}_${treepc_split}_${treepc_shard}.labels.pt"
    done
  done
}

treepc_train_heads() {
  local -a treepc_train_labels=() treepc_validation_labels=() treepc_test_labels=()
  mapfile -t treepc_train_labels < <(treepc_label_paths train)
  mapfile -t treepc_validation_labels < <(treepc_label_paths validation)
  mapfile -t treepc_test_labels < <(treepc_label_paths test)
  if [[ ! -f "${treepc_dependency}" || ! -f "${treepc_run_dir}/reports/dependency_head.json" ]]; then
    CUDA_VISIBLE_DEVICES="${treepc_gpus[0]}" "${treepc_python}" scripts/06_train_dependency_head.py \
      --train-labeled "${treepc_train_labels[@]}" \
      --validation-labeled "${treepc_validation_labels[@]}" \
      --config "${treepc_dependency_config}" --device cuda:0 \
      --checkpoint "${treepc_dependency}" --report "${treepc_run_dir}/reports/dependency_head.json" \
      >"${treepc_run_dir}/logs/train_dependency_head.log" 2>&1
  fi
  if [[ ! -f "${treepc_correction}" || ! -f "${treepc_run_dir}/reports/correction_head.json" ]]; then
    CUDA_VISIBLE_DEVICES="${treepc_gpus[0]}" "${treepc_python}" scripts/07_train_correction_head.py \
      --train-labeled "${treepc_train_labels[@]}" \
      --validation-labeled "${treepc_validation_labels[@]}" \
      --dependency-checkpoint "${treepc_dependency}" \
      --config "${treepc_correction_config}" --device cuda:0 \
      --checkpoint "${treepc_correction}" --report "${treepc_run_dir}/reports/correction_head.json" \
      >"${treepc_run_dir}/logs/train_correction_head.log" 2>&1
  fi
  if [[ ! -f "${treepc_run_dir}/reports/heldout_heads.json" \
        || ! -f "${treepc_run_dir}/reports/rq2_test.json" \
        || ! -f "${treepc_run_dir}/reports/rq2_test.csv" ]]; then
    CUDA_VISIBLE_DEVICES="${treepc_gpus[0]}" "${treepc_python}" scripts/22_evaluate_heads.py \
      --test-labeled "${treepc_test_labels[@]}" \
      --dependency-checkpoint "${treepc_dependency}" --correction-checkpoint "${treepc_correction}" \
      --correction-config "${treepc_correction_config}" \
      --steps "${treepc_head_steps[@]}" --device cuda:0 \
      --report "${treepc_run_dir}/reports/heldout_heads.json" \
      --rq2-report "${treepc_run_dir}/reports/rq2_test.json" \
      --rq2-csv "${treepc_run_dir}/reports/rq2_test.csv" \
      >"${treepc_run_dir}/logs/evaluate_heads.log" 2>&1
  fi
}

treepc_eval_partition() {
  local treepc_split="$1"
  shift
  local -a treepc_methods=("$@") treepc_pids=()
  local treepc_shard treepc_gpu treepc_output
  for ((treepc_shard=0; treepc_shard<treepc_shards; treepc_shard++)); do
    treepc_gpu="${treepc_gpus[treepc_shard]}"
    treepc_output="${treepc_run_dir}/eval/${treepc_split}/shard_${treepc_shard}"
    CUDA_VISIBLE_DEVICES="${treepc_gpu}" "${treepc_python}" scripts/12_run_main_experiments.py \
      --pc-lora "${treepc_pc_adapter}" \
      --dependency-checkpoint "${treepc_dependency}" --correction-checkpoint "${treepc_correction}" \
      --manifest "${treepc_manifest}" --evaluation-split "${treepc_split}" \
      --steps "${treepc_eval_steps[@]}" \
      --num-shards "${treepc_shards}" --shard-index "${treepc_shard}" \
      --phase both --methods "${treepc_methods[@]}" --device cuda:0 --resume \
      --output-dir "${treepc_output}" \
      >"${treepc_run_dir}/logs/eval_${treepc_split}_${treepc_shard}.log" 2>&1 &
    treepc_pids+=("$!")
  done
  if [[ "${#treepc_pids[@]}" -gt 0 ]]; then
    treepc_wait_jobs "${treepc_pids[@]}"
  fi
  local -a treepc_rows=()
  for ((treepc_shard=0; treepc_shard<treepc_shards; treepc_shard++)); do
    treepc_rows+=("${treepc_run_dir}/eval/${treepc_split}/shard_${treepc_shard}/rows.jsonl")
  done
  "${treepc_python}" scripts/16_aggregate_results.py --inputs "${treepc_rows[@]}" \
    --output-dir "${treepc_run_dir}/reports/${treepc_split}"
}

treepc_test() {
  treepc_eval_partition head_test dream_baseline pc_only pc_local_treepc pc_learned_treepc
}

case "${treepc_action}" in
  prepare) treepc_prepare ;;
  pc-cache) treepc_prepare; treepc_pc_cache ;;
  pc-train) treepc_train_pc ;;
  head-cache) treepc_head_cache ;;
  rq1) treepc_rq1 ;;
  head-train) treepc_train_heads ;;
  test) treepc_test ;;
  all)
    treepc_prepare
    treepc_pc_cache
    treepc_train_pc
    treepc_head_cache
    treepc_rq1
    treepc_train_heads
    treepc_test
    ;;
  *) echo "Unknown stage: ${treepc_action}" >&2; exit 2 ;;
esac
