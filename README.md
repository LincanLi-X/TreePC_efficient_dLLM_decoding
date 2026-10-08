# TreePC: Posterior-Consistent and Dependency Aware Decoding for Few-Step Diffusion Language Models

> Official code repository for TreePC: Posterior-Consistent and Dependency Aware Decoding for Few-Step Diffusion Language Models


## Method and experiment coverage

- Dream-v0-Instruct-7B and LLaDA-8B-Instruct, retaining their native schedulers.
- PC-LoRA: coarse/fine offline warm-up (20% of updates), followed by mixed-budget
  current-student rollouts and periodic in-memory buffer refresh; frozen backbone.
- Final-PC on-policy teacher counterfactuals with three sampled parent tokens.
- Dependency regression/ranking/symmetry, maximum spanning tree and confidence-rooted BFS.
- Oracle-tree correction warm-up followed by learned-tree adaptation. Both stages and
  held-out evaluation use **predicted dependency strengths** as gate features.
- Low-rank correction with local gate and learned global scale. Online corrections
  cover the **full vocabulary** and retain the backbone's sampling controls.
- One backbone forward per fixed-budget step, without online teacher calls.
- Four task adapters and RQ1–RQ4 experiment/report drivers.

The paper configuration covers two backbones, four tasks, teachers `{256,512}`,
students `{4,8,16,32,48,64,80,96,112}`, and generation length 512 for all methods.
The common length prevents empty late LLaDA teacher states when steps exceed length.
Four GPUs shard examples; PC-LoRA and heads train on one GPU.
Dependency batches group states with equal candidate counts, without padded graph edges.
Historical smoke/large_v2 profiles remain available, but the new paper runner is recommended.
Historical shell launchers also require an explicit external `TREEPC_RUN_DIR`.

Fast-dLLM v1 bridges use its official Dream/LLaDA no-cache parallel implementation.
The checked official CD4LM implementation provides LLaDA CAD, not a verified Dream port.
That unsupported matrix cell is explicitly skipped, never substituted. Configure CD4LM's
separate DSCD checkpoint; vanilla-checkpoint CAD must be labeled zero-shot CAD, not distilled CD4LM.
These threshold-based external decoders use adaptive NFE: their actual counted forwards,
not the nominal `steps` field, must be used for same-NFE claims. External revisions and
source-file hashes are recorded in evaluation rows.

**Code coverage is not proof of numerical reproduction.** See [VALIDATION.md](VALIDATION.md).

## Repository map

```text
TreePC_Code/
├── README.md                         # Resources, setup, workflow and protocol
├── VALIDATION.md                     # Checks actually performed and their boundaries
├── pyproject.toml                    # Dependencies, package and test/lint settings
├── requirements*.txt                # Runtime, developer and evaluation dependencies
├── models/README.md                  # Official checkpoint links and local names
├── data/README.md                    # Official benchmark links and JSONL schemas
├── configs/
│   ├── paper/experiments.yaml        # Full backbone/task/teacher/student matrix
│   ├── model/                       # Dream/LLaDA resource profiles
│   ├── large_v2/                    # PC and head training hyperparameters
│   ├── train_smoke/                 # Small diagnostic profile
│   └── stage*/                      # Historical baseline/oracle/frozen/PC profiles
├── scripts/
│   ├── prepare_data.py              # Optional official evaluation-data normalization
│   ├── 00_check_environment.py       # Environment/GPU preflight
│   ├── 01_smoke_dream.py             # Historical Dream loading/generation check
│   ├── 02_reproduce_dream_baseline.py # Official/custom Dream parity
│   ├── 03_collect_teacher_trajectories.py # Teacher state collection/shard merge
│   ├── 04_build_counterfactual_cache.py # Frozen-backbone diagnostic labels
│   ├── 05_run_oracle_treepc.py        # Oracle decoding diagnostics
│   ├── 06_train_dependency_head.py   # Dependency training from external caches
│   ├── 07_train_correction_head.py   # Oracle-tree then learned-tree training
│   ├── 08_decode_frozen_treepc.py    # Frozen-backbone diagnostics
│   ├── 09_train_pc_lora.py           # Warm-up cache loading + mixed-budget PC training
│   ├── 10_recalibrate_heads.py       # Optional final-PC states and teacher labels
│   ├── 12_run_main_experiments.py    # Historical small comparison
│   ├── 16_aggregate_results.py       # Historical sharded reports
│   ├── 20_prepare_large_scale.py     # Deterministic split manifest
│   ├── 22_evaluate_heads.py          # RQ2 and correction diagnostics
│   ├── 23_check_assets.py            # Historical Dream asset checks
│   ├── 24_validate_train_run.py      # Historical smoke artifact checks
│   ├── 25_evaluate_rq1.py            # KL/agreement/recall/calibration
│   ├── 26_evaluate_efficiency.py     # Isolated method runs and counted actual NFE
│   ├── 27_build_paper_reports.py     # RQ3 curves, matched-quality RQ4 and proxies
│   ├── run_paper_experiments.py      # Unified stages and per-GPU example sharding
│   ├── slurm_paper.sh               # Generic four-GPU HiPerGator launcher
│   └── run_* / slurm_*              # Historical smoke/large_v2 launchers
├── src/treepc/
│   ├── backbones/                   # Common interface, factory and LLaDA semantics
│   ├── dream/                       # Dream loading, shifted alignment and generation
│   ├── data/                        # Datasets, manifests and cache schemas/loaders
│   ├── models/                      # PC-LoRA, dependency/correction heads and bundles
│   ├── training/                    # Offline/on-policy PC and head objectives
│   ├── graph/                       # Maximum spanning tree and BFS orientation
│   ├── posterior/                   # KL and support utilities
│   ├── decoding/                    # Learned TreePC and diagnostic oracle
│   ├── evaluation/                  # Four-task metrics and official baseline bridges
│   ├── utils/                       # IO, provenance, seeds and environment
│   └── types.py                     # State, trace and result types
└── tests/{unit,integration}/        # CPU invariants and opt-in GPU model tests
```

## Setup

Python 3.11 and a BF16-capable CUDA GPU. Tested core versions are PyTorch 2.11.0/cu128,
Transformers 4.48.0 and PEFT 0.14.0. Install a PyTorch build compatible with your driver.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-dev.txt -r requirements-evaluation.txt
python -m pip install --no-deps -e .
```

MATH grading uses [Math-Verify](https://github.com/huggingface/Math-Verify).
HumanEval/MBPP execute generated Python: use an externally isolated worker/container
without credentials or privileged mounts. A timeout subprocess is **not** a security sandbox.

## Official resources and local names

| Resource | Official source | Local path |
|---|---|---|
| Dream | [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) | `models/DREAM-7B/` |
| LLaDA | [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | `models/LLaDA-8B/` |
| GSM8K | [OpenAI](https://github.com/openai/grade-school-math) | `data/processed/track1_general/gsm8k/samples.jsonl` |
| HumanEval | [OpenAI](https://github.com/openai/human-eval) | `data/processed/track1_general/humaneval/samples.jsonl` |
| MATH-500 | [HuggingFaceH4](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | `data/processed/track1_general/math500/samples.jsonl` |
| MBPP | [Google Research](https://github.com/google-research/google-research/tree/master/mbpp) | `data/processed/track1_general/mbpp/samples.jsonl` |
| Fast-dLLM | [NVlabs](https://github.com/NVlabs/Fast-dLLM) | `external/Fast-dLLM/` |
| CD4LM | [yihao-liang/CDLM](https://github.com/yihao-liang/CDLM) | `external/CDLM/` |

Resources can live outside the repository: edit YAML model/baseline paths and `--data-root`.
Single-stage scripts honor `TREEPC_BACKBONE=dream|llada`, `TREEPC_MODEL_DIR`,
and `TREEPC_DATA_ROOT`. Review the checkpoint's local remote-code files before loading.
Record the external code revision; never confuse CD4LM DSCD assets with TreePC adapters.

Optional evaluation-data preparation (not run to create this release):

```bash
python scripts/prepare_data.py --datasets gsm8k humaneval math500 mbpp \
  --output-root /external/data/track1_general
```

## Evaluation protocol

The default configuration is **internal held-out evaluation**, not a leaderboard claim:
it repurposes mutually disjoint subsets of public evaluation records for training.
For standard evaluation, set `protocol: standard` and supply genuinely separate training
sources tagged `source_split: train` plus official test rows tagged `source_split: test`.
Validation is drawn only from training sources. Duplicate prompt text across splits is rejected.
HumanEval/MATH-500 are evaluation sets: readers must separately provide training sources,
not manufacture training rows from those test prompts. IDs must be unique within each dataset.
Official test subsets are labeled as subsets, never full benchmark scores.

All backbone/teacher profiles reuse the same source hashes, seed and prompt splits.
RQ1 KL is a Top-K-plus-OTHER coarse-grained KL, not exact vocabulary KL. Comparison uses
the same PC-visited state and comparable masked positions. Head objectives also use cached
support plus exact OTHER mass; online decoding nevertheless corrects full-vocabulary logits.
Specifically, correction training/held-out cache metrics project onto the cached support
and hold the base OTHER log mass fixed before renormalization. This is a surrogate KL,
not the exact KL of the full-vocabulary online corrected posterior; report it as such.

## External intermediate inputs

All runtime artifacts belong **outside TreePC_Code**, e.g. `/blue/<group>/<user>/treepc-paper`.
No intermediate inputs are supplied. Stages assume the documented artifacts already exist
and fail explicitly when inputs are missing. Per backbone/teacher profile:

```text
<artifact-root>/<dream|llada>/teacher_<256|512>/
├── manifest.json
├── cache/
│   ├── teacher/<dataset>_test.pt     # Four teacher states; trajectory.v1
│   ├── pc/<dataset>_{train,validation}.pt # Coarse/fine warm-up; pc_nested.v1
│   └── heads/<dataset>_{train,validation,test}_<shard>.pt # counterfactual.v1
├── checkpoints/
│   ├── pc_lora/                     # PEFT adapter_config + adapter_model.safetensors
│   ├── dependency_head.pt
│   └── correction_head.pt
├── logs/
├── eval/                            # Per-example task/timing/NFE rows
└── reports/                         # RQ1–RQ4 summaries
```

Readers can prepare warm-up caches with script 03 and script 09's `build-cache` command,
or supply schema-compatible existing inputs. Script 10 optionally creates final-PC head
states/labels outside the repository. Head caches must contain the **current PC adapter
SHA256, backbone, manifest fingerprint and configured split/budget coverage**;
stale/pre-revision caches are rejected. Their shard count matches
`--gpus`. Preserve split metadata and manifest fingerprints. PC rollout buffers stay in RAM.

## Running

Run from this directory, after configuring YAML and external paths:

```bash
export TREEPC_DATA_ROOT=/external/data/track1_general
export TREEPC_RUN_ROOT=/external/treepc-paper

# Read-only plan: no model loading or artifact generation.
python scripts/run_paper_experiments.py --run-root "$TREEPC_RUN_ROOT" \
  --data-root "$TREEPC_DATA_ROOT" --gpus 0,1,2,3 --dry-run

# Manifest and PC training from existing warm-up caches.
for stage in manifest pc; do
  python scripts/run_paper_experiments.py --run-root "$TREEPC_RUN_ROOT" \
    --data-root "$TREEPC_DATA_ROOT" --gpus 0,1,2,3 --stage "$stage"
done

# Supply final-PC head caches, OR opt into their runtime external generation.
python scripts/run_paper_experiments.py --run-root "$TREEPC_RUN_ROOT" \
  --data-root "$TREEPC_DATA_ROOT" --gpus 0,1,2,3 --stage collect-heads

for stage in heads rq1 rq2 eval efficiency external reports; do
  python scripts/run_paper_experiments.py --run-root "$TREEPC_RUN_ROOT" \
    --data-root "$TREEPC_DATA_ROOT" --gpus 0,1,2,3 --stage "$stage"
done
```

`--stage all` assumes intermediate inputs. An existing PC adapter is not implicitly
overwritten; `--prepare-heads` explicitly opts into runtime external regeneration after PC.
Use a fresh artifact root for changed code/configuration. Existing evaluation outputs are
not silently overwritten. Missing resources cause errors, never placeholder results.

RQ4 uses isolated method processes, warm-up, repeated seeded timing and counted model calls.
Auxiliary profiling is separate and excluded from unprofiled end-to-end speedups.
Peak GPU memory is PyTorch's process-local `max_memory_allocated` during generation,
including resident model parameters; it is not total node VRAM usage or training peak memory.
Budget selection uses validation quality, followed by held-out test reporting. Missing
matched-quality candidates are reported, not filled with invented speedups.
Optional `27_build_paper_reports.py --proxy-spec <json>` generates explicitly labeled
linear estimates with their measured anchors, never target-task measurement claims.

## HiPerGator and GPU profiles

Example memory profiles: L40 (48 GB), RTX PRO 6000 (96 GB), B200 (180 GB).
These are configuration examples, not guaranteed current cluster availability.
L4 is **not** L40; query partitions/GRES with `sinfo -o '%P %G'`.
Whether full training fits depends on sequence length and candidate count; a smoke peak is not a bound.

```bash
sbatch --account=<group> --qos=<group> --partition=<available-gpu-partition> \
  --gres=gpu:4 --time=48:00:00 \
  --export=ALL,TREEPC_RUN_ROOT="$TREEPC_RUN_ROOT",TREEPC_DATA_ROOT="$TREEPC_DATA_ROOT",TREEPC_PYTHON="$PWD/.venv/bin/python" \
  scripts/slurm_paper.sh
```

GPU IDs are allocation-local; the driver maps them to scheduler-visible devices.
Jobs survive SSH disconnection; project-storage artifacts persist after GPU allocations end.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -p no:cacheprovider
ruff check .
for script in scripts/*.sh; do bash -n "$script"; done

CUDA_VISIBLE_DEVICES=<idle-lab-gpu> TREEPC_TEST_MODEL_ROOT=/external/checkpoint/parent \
  PYTHONPATH=src pytest tests/integration/test_revision_models.py -m model -v
```

Real-model tests write only to pytest's external temporary directory. Full-scale training
and paper-score reproduction are not established by these small tests.


## Citation

```
@inproceedings{anonymous2026treepc,
  title     = {TreePC: Posterior-Consistent and Dependency Aware Decoding for Few-Step Diffusion Language Models},
  author    = {Anonymous},
  booktitle = {Under Review},
  year      = {2026}
}
```
