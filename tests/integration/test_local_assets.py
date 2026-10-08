from pathlib import Path

import pytest

from treepc.data.datasets import DATASETS
from treepc.dream.loader import load_model_config, resolve_model_dir


@pytest.mark.model
def test_local_dream_and_datasets_exist() -> None:
    model_dir = Path(resolve_model_dir(load_model_config()))
    assert (model_dir / "config.json").is_file()
    assert len(list(model_dir.glob("model-*.safetensors"))) == 4
    assert all(path.is_file() for path in DATASETS.values())
