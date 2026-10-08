"""Dream compatibility exports; logits/hidden alignment stays Dream-specific."""

from treepc.dream.adapter import DreamAdapter
from treepc.dream.loader import load_dream

__all__ = ["DreamAdapter", "load_dream"]
