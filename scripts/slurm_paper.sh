#!/usr/bin/env bash
#SBATCH --job-name=treepc-paper
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=treepc-paper-%j.out
set -euo pipefail
: "${TREEPC_RUN_ROOT:?Set an external /blue/group/user run root}"
: "${TREEPC_DATA_ROOT:?Set the normalized four-task data root}"
treepc_python="${TREEPC_PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# CUDA IDs below are allocation-local, not physical node IDs.
"$treepc_python" scripts/run_paper_experiments.py \
  --run-root "$TREEPC_RUN_ROOT" --data-root "$TREEPC_DATA_ROOT" \
  --config "${TREEPC_PAPER_CONFIG:-configs/paper/experiments.yaml}" \
  --gpus "${TREEPC_GPUS:-0,1,2,3}" --stage "${TREEPC_STAGE:-all}"
