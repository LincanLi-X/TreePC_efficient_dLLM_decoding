# Validation and limitations

Validation date: 2026-10-08. Checks were performed on laboratory GPUs, not a new
HiPerGator allocation. Runtime: Python 3.11, PyTorch 2.11.0+cu128,
Transformers 4.48.0, PEFT 0.14.0; MATH grading used Math-Verify 0.8.0.

## Checks performed

- CPU suite: **49 passed**; 9 opt-in model cases deselected in that command.
- Across separate GPU commands, **8 opt-in model cases passed**: two backbone
  revision cases, three external-API cases, Dream parity, oracle termination,
  and the four-task driver. The ninth case is the asset-presence check noted below.
- Ruff lint and shell syntax checks passed. New training/evaluation CLI help and
  the full paper-matrix dry-run succeeded without loading model/data assets.
- Real Dream and LLaDA small GPU tests exercise nonzero coarse/fine PC warm-up,
  mixed-budget optimization, two buffer refreshes, nonzero LoRA updates,
  three-parent counterfactual labeling, dependency/correction training and decoding.
  They verify one backbone call per step and the PC-only tree-disabled fallback.
- Official Dream/custom generation parity and online oracle termination were checked.
- Official Fast-dLLM Dream/LLaDA bridges and the CD4LM LLaDA CAD interface were
  exercised using local pretrained checkpoints. The LLaDA fixed-budget decoder is
  also compared with the official reference's non-threshold decoding path.
- The four-task integration test uses Dream, synthetic records (one prompt per
  split/task), 16 generated tokens, teacher budget 4 and student budgets 2/4.
  It exercises PC training, external head-cache generation, both head training
  stages, RQ1/RQ2, validation/test task evaluation, separate auxiliary profiling
  and the RQ3/RQ4 report chain. These are plumbing checks, not benchmark results.

GPU model tests used NVIDIA RTX 6000 Ada Generation (48 GB); the four-task driver
used NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB). Each pipeline used one GPU;
four-GPU allocation-local sharding was checked through the generated execution plan,
not through a concurrent four-GPU HiPerGator run.

All test datasets, caches, adapters and reports were written to external temporary
directories. No model weights, dataset rows or intermediate training artifacts are
included in this source release. The local-asset presence test is intentionally not
run against the asset-free release.

## External source revisions

The interface checks used:

- [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM), v1 source at
  `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b`.
- [yihao-liang/CDLM](https://github.com/yihao-liang/CDLM) at
  `bb8db4543bed96de2206f254e6de9f4cdbb9af9e`.

Pin these revisions for the checked APIs. The evaluation records capture the actual
external revision and Python source hashes. No external source is vendored here.
The CD4LM test validates **zero-shot CAD on a vanilla LLaDA checkpoint**; it does
not validate a separately trained DSCD checkpoint or its reported quality.
An official Dream CD4LM port is not implemented and is explicitly skipped.

## What these checks do not establish

- No full canonical training with 256/512-step teachers, all nine student budgets
  and the complete two-backbone/four-task matrix has been run for this revision.
- Small regression tests do not reproduce manuscript scores or establish useful
  speedups, memory bounds, convergence, or multi-GPU scaling.
- RQ1 uses grouped Top-K-plus-OTHER KL. Correction-cache training/evaluation uses
  a cached-support surrogate with fixed base OTHER mass; it is not exact
  full-vocabulary corrected KL. Online decoding itself uses full-vocabulary logits.
- The default split protocol is internal held-out evaluation. Standard benchmark
  claims require separately sourced training prompts and official held-out test
  rows; see README. An official test subset is not a full leaderboard evaluation.
- Generated-code subprocesses have timeouts but are **not security sandboxes**.
  Use an externally isolated execution environment for untrusted generated code.

Do not relabel historical checkpoints/results as outcomes of this corrected method.
Use a fresh external run root and regenerate stale head supervision from the final
PC adapter; the runner checks adapter hashes, manifests and split/budget coverage.
