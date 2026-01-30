"""ArrayDict: A lightweight, JAX-backed container for arrays sharing batch dimensions."""

from arraydict.core import ArrayDict, KeyType, ValueType
from arraydict.ops import stack, concat

__all__ = ["ArrayDict", "KeyType", "ValueType", "stack", "concat"]
