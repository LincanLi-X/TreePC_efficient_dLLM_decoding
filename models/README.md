# Model assets

Model weights are intentionally excluded from this repository. Download them from the official
model pages and keep the local directory names shown below.

| Model | Official source | Expected local directory |
|---|---|---|
| Dream-v0-Instruct-7B | [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) | `models/DREAM-7B/` |
| LLaDA-8B-Instruct | [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | `models/LLaDA-8B/` |

The paper runner supports both backbones. Single-stage scripts use `TREEPC_BACKBONE=dream|llada`
and `TREEPC_MODEL_DIR`; the paper YAML configures model directories per backbone.
Dream uses shifted alignment; LLaDA uses unshifted logits, full-block linear transfer quotas,
low-confidence remasking and CFG=0. CD4LM DSCD checkpoints are separate reader-provided assets.
