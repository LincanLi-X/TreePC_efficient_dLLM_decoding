"""Invoke readers' external official repositories; no code/weights are vendored."""

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import torch

from treepc.utils.io import sha256_file


def import_file(path, name):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configure the official baseline checkout: {path}")
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if path.name == "__init__.py" else None
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_llada_entry(checkout):
    """The official script imports an absolute `model` package; isolate that import."""
    previous = {
        key: value for key, value in sys.modules.items() if key == "model" or key.startswith("model.")
    }
    for key in previous:
        del sys.modules[key]
    search_path = str(checkout / "v1/llada")
    sys.path.insert(0, search_path)
    try:
        return import_file(checkout / "v1/llada/generate.py", "treepc_fast_llada")
    finally:
        sys.path.remove(search_path)
        for key in list(sys.modules):
            if key == "model" or key.startswith("model."):
                del sys.modules[key]
        sys.modules.update(previous)


class ExternalDecoder:
    def __init__(self, adapter, method, checkout, threshold=0.9):
        self.adapter, self.method, self.threshold = adapter, method, threshold
        self.backbone = adapter.metadata["backbone"]
        checkout = Path(checkout)
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        self.metadata = {
            "method": method,
            "threshold": threshold,
            "checkout": str(checkout.resolve()),
            "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "cache_policy": "no_kv_cache",
            "source_sha256": {
                str(path.relative_to(checkout)): sha256_file(path) for path in checkout.rglob("*.py")
            },
        }
        if method == "cd4lm":
            if self.backbone != "llada":
                raise ValueError(
                    "Official CD4LM currently supplies LLaDA; Dream needs a separately validated port"
                )
            self.generate = import_file(
                checkout / "scripts/LLaDA_generate_dynamic.py", "treepc_cd4lm"
            ).generate
        elif method == "fast_dllm" and self.backbone == "llada":
            self.generate = import_llada_entry(checkout).generate
        elif method == "fast_dllm" and self.backbone == "dream":
            package = import_file(checkout / "v1/dream/model/__init__.py", "treepc_fast_dream")
            mixin = import_file(
                checkout / "v1/dream/model/generation_utils.py", "treepc_fast_dream.generation_utils"
            ).DreamGenerationMixin
            # Official no-cache parallel decoder, not a locally reimplemented threshold heuristic.
            old = adapter.model
            adapter.model = None
            del old
            torch.cuda.empty_cache()
            adapter.model = (
                package.DreamModel.from_pretrained(
                    adapter.metadata["local_dir"], local_files_only=True, torch_dtype=torch.bfloat16
                )
                .to(adapter.device)
                .eval()
            )
            adapter.model.diffusion_generate = types.MethodType(mixin.diffusion_generate, adapter.model)
            adapter.model._sample = types.MethodType(mixin._sample, adapter.model)
        else:
            raise ValueError(f"Unknown baseline: {method}")

    def __call__(self, prompt, steps, length):
        adapter = self.adapter
        encoded = adapter.encode_prompt(prompt)
        if self.method == "fast_dllm" and self.backbone == "dream":
            out = adapter.model.diffusion_generate(
                encoded["input_ids"],
                max_new_tokens=length,
                steps=steps,
                alg="confidence_threshold",
                threshold=self.threshold,
                use_cache=False,
                return_dict_in_generate=True,
                temperature=0.0,
                alg_temp=0.0,
            )
            return out.sequences
        if self.method == "fast_dllm":
            out, _ = self.generate(
                adapter.model,
                encoded["input_ids"],
                steps=steps,
                gen_length=length,
                block_length=length,
                temperature=0.0,
                threshold=self.threshold,
            )
        else:
            out, _ = self.generate(
                adapter.model,
                encoded["input_ids"],
                gen_length=length,
                confidence_threshold=self.threshold,
                max_steps=length,
                temperature=0.0,
                cfg_scale=0.0,
                max_tokens_per_step=length,
                verbose=False,
                tokenizer=adapter.tokenizer,
            )
        return out
