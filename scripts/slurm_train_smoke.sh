#!/usr/bin/env bash
#SBATCH --job-name=treepc-train-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=treepc-train-smoke-%j.out

set -euo pipefail

treepc_project_dir="${TREEPC_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${treepc_project_dir}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "project=${treepc_project_dir}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

bash scripts/run_train_smoke.sh
