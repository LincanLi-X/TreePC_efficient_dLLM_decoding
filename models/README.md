# Model assets

Model weights are intentionally excluded from this repository. Download them from the official
model pages and keep the local directory names shown below.

| Model | Official source | Expected local directory |
|---|---|---|
| Dream-v0-Instruct-7B | [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) | `models/DREAM-7B/` |
| LLaDA-8B-Instruct | [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | `models/LLaDA-8B/` |

The checked-in `large_v2` runner uses Dream. Set `TREEPC_MODEL_DIR` to an absolute checkpoint
path if the weights are stored outside this repository.
