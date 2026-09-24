# TreePC

This repository is the anonymous review artifact for TreePC, a posterior-consistency decoding
method for diffusion language models. It contains the implementation, experiment configurations,
multi-GPU orchestration, evaluation code, and tests. Model weights, benchmark records, generated
caches, checkpoints, and result directories are deliberately excluded.

The canonical checked-in experiment is `large_v2`: Dream-v0-Instruct-7B as the backbone, a
256-step teacher, student NFEs in `{8, 16, 32, 64}`, and fixed prompt-level splits from GSM8K and
HumanEval. The resource manifest also links LLaDA-8B-Instruct, MATH-500, and MBPP for the broader
benchmark suite. This code snapshot does not yet include the LLaDA or MATH-500/MBPP adapters.

## Repository layout

```text
TreePC_Code/
├── README.md                         # Setup, resources, execution, and repository map
├── pyproject.toml                    # Package metadata, dependencies, pytest, and Ruff settings
├── requirements.txt                 # Exact runtime versions used for the released runs
├── requirements-dev.txt             # Runtime plus pytest/Ruff/networkx
├── .gitignore                       # Excludes weights, datasets, caches, results, and environments
├── models/
│   └── README.md                    # Official model links and expected local names
├── data/
│   └── README.md                    # Official dataset links and expected JSONL layout
├── configs/
│   ├── model/
│   │   └── dream_7b_instruct.yaml   # Dream loading and memory settings
│   ├── eval/
│   │   └── quality.yaml             # Evaluation defaults
│   ├── train_smoke/                 # Small end-to-end PC-LoRA/head training profile
│   ├── large_v2/                    # Canonical large-v2 pipeline and training profiles
│   ├── stage1_baseline/             # Independent Dream baselines at 4/8/16/32 NFE
│   ├── stage2_oracle/               # Teacher trajectory and oracle diagnostics
│   ├── stage3_frozen/               # Frozen-backbone head training/decoding profiles
│   └── stage4_pc/                   # PC-LoRA and head-recalibration profiles
├── scripts/
│   ├── prepare_data.py              # Download/normalize official GSM8K and HumanEval files
│   ├── 00_check_environment.py      # Record software/GPU environment and enforce resource checks
│   ├── 01_smoke_dream.py            # Minimal Dream loading and generation smoke test
│   ├── 02_reproduce_dream_baseline.py # Prepare/run/aggregate independent Dream baselines
│   ├── 03_collect_teacher_trajectories.py # Build manifests and collect/merge teacher states
│   ├── 04_build_counterfactual_cache.py # Create counterfactual posterior labels
│   ├── 05_run_oracle_treepc.py      # Run cached/online oracle TreePC diagnostics
│   ├── 06_train_dependency_head.py  # Train the dependency head
│   ├── 07_train_correction_head.py  # Train the correction head
│   ├── 08_decode_frozen_treepc.py   # Evaluate frozen-backbone learned TreePC
│   ├── 09_train_pc_lora.py          # Build PC caches and train PC-LoRA
│   ├── 10_recalibrate_heads.py      # Collect on-policy states and relabel/recalibrate heads
│   ├── 12_run_main_experiments.py   # Run Independent, PC-only, Local, and Learned TreePC
│   ├── 16_aggregate_results.py      # Aggregate sharded JSONL results into reports
│   ├── 20_prepare_large_scale.py    # Create or verify the deterministic large-v2 manifest
│   ├── 22_evaluate_heads.py         # Evaluate heads and produce the RQ2 report
│   ├── 23_check_assets.py           # Validate model/data/software/GPU assets before a run
│   ├── 24_validate_train_run.py     # Validate a completed smoke run and its artifacts
│   ├── 25_evaluate_rq1.py           # Produce RQ1 posterior-consistency metrics
│   ├── run_train_smoke.sh           # One-GPU end-to-end smoke pipeline
│   ├── run_large_v2_pipeline.sh     # Public entry point for the canonical experiment
│   ├── run_large_v2_impl.sh         # Resumable, sample-sharded multi-GPU implementation
│   ├── slurm_train_smoke.sh         # Generic one-GPU Slurm job template
│   └── slurm_large_v2.sh            # Generic four-GPU Slurm job template
├── src/treepc/
│   ├── types.py                     # Shared dataclasses and trace/result types
│   ├── data/
│   │   ├── datasets.py              # Dataset loading and deterministic split manifests
│   │   ├── trajectory.py            # Teacher trajectory cache representation
│   │   ├── onpolicy.py              # Student on-policy state representation
│   │   ├── counterfactual.py        # Counterfactual label construction
│   │   ├── pc_cache.py              # PC-LoRA supervision cache construction
│   │   ├── cache_schema.py          # Cache schema validation
│   │   └── cache_dataset.py         # PyTorch datasets over cached supervision
│   ├── dream/
│   │   ├── loader.py                # Local Dream checkpoint loading
│   │   ├── adapter.py               # Unified Dream forward/tokenization adapter
│   │   ├── generation.py            # Independent diffusion decoding and trace capture
│   │   └── alignment.py             # Teacher/student state alignment
│   ├── graph/
│   │   ├── chow_liu.py              # Maximum-spanning dependency tree construction
│   │   └── orientation.py           # Tree rooting and parent/child orientation
│   ├── posterior/
│   │   ├── consistency.py           # Tree posterior-consistency updates
│   │   ├── divergences.py           # Distribution divergence utilities
│   │   └── topk_support.py          # Top-k posterior support operations
│   ├── models/
│   │   ├── pc_lora.py               # LoRA attachment and PC model helpers
│   │   ├── dependency_head.py       # Pairwise dependency scoring head
│   │   ├── correction_head.py       # Conditional token correction head
│   │   └── treepc_bundle.py         # Checkpoint bundle loading
│   ├── decoding/
│   │   ├── learned_treepc.py        # Learned TreePC decoder
│   │   └── oracle_treepc.py         # Oracle/diagnostic TreePC decoder
│   ├── training/
│   │   ├── pc_trainer.py            # PC-LoRA training loop
│   │   ├── dependency_trainer.py    # Dependency-head training loop
│   │   ├── correction_trainer.py    # Correction-head training loop
│   │   └── losses.py                # Training objectives
│   ├── evaluation/
│   │   ├── task_metrics.py          # GSM8K and HumanEval task metrics
│   │   ├── humaneval.py             # Isolated HumanEval execution helper
│   │   ├── oracle_metrics.py        # Oracle diagnostic metrics
│   │   ├── rq1.py                   # RQ1 calibration/recall metrics
│   │   └── statistics.py            # Aggregation and confidence intervals
│   └── utils/
│       ├── environment.py           # Runtime/environment capture
│       ├── io.py                    # Atomic JSON/JSONL and hashing helpers
│       └── seed.py                  # Reproducible RNG initialization
└── tests/
    ├── unit/                        # CPU tests for graph, posterior, heads, splits, and metrics
    └── integration/                 # Opt-in CUDA/model generation tests
```

Empty `__init__.py` files establish the Python packages and are omitted from the annotations above.

## External assets

### Model checkpoints

Install the Hugging Face CLI and download weights into the expected local directories:

```bash
python -m pip install -U huggingface_hub

hf download Dream-org/Dream-v0-Instruct-7B \
  --local-dir models/DREAM-7B

hf download GSAI-ML/LLaDA-8B-Instruct \
  --local-dir models/LLaDA-8B
```

Official pages:

- [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)
- [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)

`TREEPC_MODEL_DIR` overrides the repository default (`models/DREAM-7B`). The canonical runner in
this snapshot uses Dream; merely placing LLaDA weights in `models/LLaDA-8B` does not switch the
backbone.

### Datasets

The two datasets used by `large_v2` can be downloaded and normalized directly from their official
repositories:

```bash
python scripts/prepare_data.py \
  --output-root data/processed/track1_general
```

This creates:

```text
data/processed/track1_general/
├── gsm8k/samples.jsonl
└── humaneval/samples.jsonl
```

Official benchmark sources:

- [GSM8K](https://github.com/openai/grade-school-math)
- [HumanEval](https://github.com/openai/human-eval)
- [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)

Set `TREEPC_DATA_ROOT` if the processed files live elsewhere. MATH-500 and MBPP are not consumed by
the checked-in Dream/GSM8K/HumanEval runner and require their corresponding task adapters.

## Environment and dependencies

The released run used Python 3.11, PyTorch 2.11.0 with CUDA 12.8, Transformers 4.48.0,
Accelerate 1.9.0, PEFT 0.14.0, Safetensors 0.5.2, and PyYAML 6.0.3. A CUDA GPU with BF16 support is
required for model experiments. CPU-only unit tests do not require model weights or benchmark data.

Example setup for CUDA 12.8:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
```

For a different CUDA stack, install the matching PyTorch build first, then install the remaining
requirements. The preflight checker intentionally reports a version mismatch when the experiment
environment differs from the released one.

## Basic verification

Run the CPU-safe checks from the repository root:

```bash
ruff check .
pytest
bash -n scripts/*.sh
```

The default pytest configuration excludes tests marked `model`. After downloading Dream and the
datasets on a CUDA machine, run the integration checks explicitly:

```bash
export TREEPC_MODEL_DIR="$PWD/models/DREAM-7B"
export TREEPC_DATA_ROOT="$PWD/data/processed/track1_general"
pytest -m model -o addopts=-ra
```

## Running TreePC

All commands below are issued from the repository root. Output paths are deliberately externalized
through environment variables, so weights and generated artifacts never need to be committed.

### One-GPU smoke run

```bash
export TREEPC_MODEL_DIR="$PWD/models/DREAM-7B"
export TREEPC_DATA_ROOT="$PWD/data/processed/track1_general"
export TREEPC_RUN_DIR="$PWD/results/train_smoke_$(date +%Y%m%d_%H%M%S)"
export TREEPC_PYTHON="$PWD/.venv/bin/python"
export TREEPC_GPU_IDS=0

bash scripts/run_train_smoke.sh
```

This performs asset validation, a Dream generation check, PC-LoRA training, dependency/correction
head training, held-out evaluation, and final artifact validation.

### Canonical four-GPU `large_v2` run

```bash
export TREEPC_MODEL_DIR="$PWD/models/DREAM-7B"
export TREEPC_DATA_ROOT="$PWD/data/processed/track1_general"
export TREEPC_RUN_DIR="$PWD/results/large_v2_$(date +%Y%m%d_%H%M%S)"
export TREEPC_PYTHON="$PWD/.venv/bin/python"
export TREEPC_GPU_IDS=0,1,2,3

bash scripts/run_large_v2_pipeline.sh all
```

The orchestration is resumable: completed caches/checkpoints/reports are detected and reused.
Teacher collection, on-policy state collection, counterfactual labeling, and final evaluation are
sample-sharded over visible GPUs; PC-LoRA and both lightweight heads train on the first visible GPU.

Individual stages are also available:

```bash
bash scripts/run_large_v2_pipeline.sh prepare
bash scripts/run_large_v2_pipeline.sh pc-cache
bash scripts/run_large_v2_pipeline.sh pc-train
bash scripts/run_large_v2_pipeline.sh head-cache
bash scripts/run_large_v2_pipeline.sh rq1
bash scripts/run_large_v2_pipeline.sh head-train
bash scripts/run_large_v2_pipeline.sh test
```

### Slurm

The supplied scripts are cluster-neutral templates. Provide the partition/account/QoS required by
your site and export only local paths:

```bash
RUN_DIR=/path/to/project-storage/runs/large_v2_$(date +%Y%m%d_%H%M%S)

sbatch \
  --partition=<gpu-partition> \
  --account=<account> \
  --qos=<qos> \
  --export=ALL,TREEPC_MODEL_DIR="$PWD/models/DREAM-7B",TREEPC_DATA_ROOT="$PWD/data/processed/track1_general",TREEPC_RUN_DIR="$RUN_DIR",TREEPC_PYTHON="$PWD/.venv/bin/python" \
  scripts/slurm_large_v2.sh
```

Adjust `#SBATCH --gres`, memory, and time in the template or override them on the `sbatch` command
line. The canonical setup requests four GPUs, 16 CPUs, 128 GB host memory, and 48 hours.

## Outputs

Each run directory contains:

```text
<run-dir>/
├── manifest.json                    # Exact prompt-level split and source hashes
├── environment.json                 # Software and accelerator metadata
├── cache/                           # Regenerable teacher/PC/head caches (large)
├── checkpoints/                     # PC-LoRA and two learned heads
├── logs/                            # Stage-specific logs
├── eval/                            # Per-example sharded evaluation rows
└── reports/
    ├── rq1_test.{json,csv}
    ├── rq2_test.{json,csv}
    ├── heldout_heads.json
    └── head_test/summary.{json,csv,md}
```

The `large_v2` manifest uses seed 5050. PC-LoRA splits contain 800/160/200 GSM8K and 96/24/32
HumanEval prompts for train/validation/test. Head splits are nested within the matching PC splits
and contain 500/100/200 and 64/16/32 prompts. These source files are public benchmark test rows
repurposed into mutually disjoint internal splits; results from this profile must not be described
as standard leaderboard test-set results.

HumanEval evaluation executes generated Python. Run it only in an isolated environment with no
secrets or privileged filesystem access.
