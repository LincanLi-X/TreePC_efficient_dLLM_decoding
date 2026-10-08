#!/usr/bin/env bash
#SBATCH --job-name=treepc-large-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=treepc-large-v2-%j.out

set -euo pipefail

# Supply site-specific paths with sbatch --export, or edit these two variables.
: "${TREEPC_DATA_ROOT:?Pass TREEPC_DATA_ROOT with sbatch --export.}"
: "${TREEPC_MODEL_DIR:?Pass TREEPC_MODEL_DIR with sbatch --export.}"
export TREEPC_GPU_IDS="${TREEPC_GPU_IDS:-0,1,2,3}"

cd "${SLURM_SUBMIT_DIR}"
bash scripts/run_large_v2_pipeline.sh all
