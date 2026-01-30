from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, SupportsIndex, Tuple, Union, cast, overload

import jax.numpy as jnp
import numpy as np

KeyType = Union[str, Tuple[Any, ...]]
ValueType = Any


ArrayLike = Union[jnp.ndarray, np.ndarray]

# Batch indexing types
BatchIndex = Union[
    int,
    SupportsIndex,
    slice,
    jnp.ndarray,
    np.ndarray,
    None,
    type(Ellipsis),
    Tuple[Union[int, SupportsIndex, slice, jnp.ndarray, np.ndarray, None, type(Ellipsis)], ...],
]


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _is_array(value: Any) -> bool:
    return isinstance(value, (jnp.ndarray, np.ndarray))


def _normalize_key(key: Any) -> Tuple[Any, ...]:
    if isinstance(key, tuple):
        return key
    return (key,)


def _starts_with(full: Tuple[Any, ...], prefix: Tuple[Any, ...]) -> bool:
    if len(prefix) > len(full):
        return False
    return full[: len(prefix)] == prefix


def _common_prefix(shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:
    if not shapes:
        return ()
    prefix = shapes[0]
    for shape in shapes[1:]:
        limit = min(len(prefix), len(shape))
        new_prefix: List[int] = []
        for i in range(limit):
            if prefix[i] != shape[i]:
                break
            new_prefix.append(prefix[i])
        prefix = tuple(new_prefix)
        if not prefix:
            break
    return prefix


def _ensure_object_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.array(value, dtype=object)


def _reshape_with_batch(value: Any, new_batch: Tuple[int, ...], old_batch: Tuple[int, ...]) -> Any:
    if _is_array(value):
        tail = value.shape[len(old_batch) :]
        return value.reshape(new_batch + tail)
    if isinstance(value, np.ndarray):
        tail = value.shape[len(old_batch) :]
        return value.reshape(new_batch + tail)
    return value


def _normalize_index(index: Any) -> Any:
    """Normalize index to be compatible with JAX/numpy indexing.
    
    Only converts SupportsIndex objects (non-built-in) to int when necessary.
    Recursively processes tuples.
    """
    # Built-in types pass through
    if isinstance(index, (int, slice, type(None), type(Ellipsis))):
        return index
    
    # Arrays pass through
    if _is_array(index):
        return index
    
    # Handle SupportsIndex protocol (non-built-in types)
    if hasattr(index, '__index__'):
        try:
            return int(index.__index__())
        except (TypeError, AttributeError):
            pass
    
    # Handle tuples recursively
    if isinstance(index, tuple):
        return tuple(_normalize_index(item) for item in index)
    
    # Everything else passes through
    return index


def _apply_index(value: Any, index: Any) -> Any:
    if _is_array(value) or isinstance(value, np.ndarray):
        return value[index]
    return value


def _apply_take(value: Any, indices: Any, axis: int) -> Any:
    if isinstance(value, jnp.ndarray):
        return jnp.take(value, indices, axis=axis)
    if isinstance(value, np.ndarray):
        return np.take(value, indices, axis=axis)
    return value


def _apply_split(value: Any, num: int, axis: int) -> List[Any]:
    if isinstance(value, jnp.ndarray):
        return list(jnp.split(value, num, axis=axis))
    if isinstance(value, np.ndarray):
        return list(np.split(value, num, axis=axis))
    return [value for _ in range(num)]


def _apply_stack(values: List[Any], axis: int) -> Any:
    if isinstance(values[0], jnp.ndarray):
        return jnp.stack(values, axis=axis)
    if isinstance(values[0], np.ndarray):
        return np.stack(values, axis=axis)
    return values


def _apply_concat(values: List[Any], axis: int) -> Any:
    if isinstance(values[0], jnp.ndarray):
        return jnp.concatenate(values, axis=axis)
    if isinstance(values[0], np.ndarray):
        return np.concatenate(values, axis=axis)
    return values


def _resolve_batch_size(batch_size: Optional[Union[int, Sequence[int]]], shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:
    if batch_size is None:
        inferred = _common_prefix(shapes)
        if not inferred:
            raise ValueError("Unable to infer batch_size from provided values.")
        return inferred
    if isinstance(batch_size, int):
        return (batch_size,)
    return tuple(batch_size)


def _validate_batch_size(values: Dict[Tuple[Any, ...], Any], batch_size: Tuple[int, ...]) -> None:
    for key, value in values.items():
        if _is_array(value) or isinstance(value, np.ndarray):
            if value.shape[: len(batch_size)] != batch_size:
                raise ValueError(f"Value for key {key} does not match batch_size {batch_size}.")


def _flatten_mapping(mapping: Mapping[Any, Any], parent: Tuple[Any, ...], flat: Dict[Tuple[Any, ...], Any]) -> None:
    for key, value in mapping.items():
        key_tuple = _normalize_key(key)
        full_key = parent + key_tuple
        if isinstance(value, ArrayDict):
            for sub_key, sub_value in value._data.items():
                flat[full_key + sub_key] = sub_value
            continue
        if _is_mapping(value):
            _flatten_mapping(value, full_key, flat)
            continue
        if _is_sequence(value) and not _is_array(value):
            flat[full_key] = _ensure_object_array(value)
        else:
            flat[full_key] = value


def _nest_from_flat(flat: Dict[Tuple[Any, ...], Any]) -> Dict[Any, Any]:
    nested: Dict[Any, Any] = {}
    for key, value in flat.items():
        current: Dict[Any, Any] = nested
        for part in key[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = cast(Dict[Any, Any], current[part])
        current[key[-1]] = value
    return nested


def _rows_from_flat(flat: Dict[Tuple[Any, ...], Any], batch_size: Tuple[int, ...]) -> List[Dict[Any, Any]]:
    if not batch_size:
        return [_nest_from_flat(flat)]
    dummy = jnp.empty(batch_size)
    rows: List[Dict[Any, Any]] = []
    for i in range(dummy.shape[0]):
        row_flat: Dict[Tuple[Any, ...], Any] = {}
        for key, value in flat.items():
            if _is_array(value) or isinstance(value, np.ndarray):
                row_flat[key] = value[i]
            else:
                row_flat[key] = value
        rows.append(_nest_from_flat(row_flat))
    return rows


@dataclass(frozen=True)
class ArrayDict:
    _data: Dict[Tuple[Any, ...], Any]
    batch_size: Tuple[int, ...]

    def __init__(self, source: Mapping[Any, Any], batch_size: Optional[Union[int, Sequence[int]]] = None) -> None:
        flat: Dict[Tuple[Any, ...], Any] = {}
        _flatten_mapping(source, (), flat)
        shapes: List[Tuple[int, ...]] = []
        for value in flat.values():
            if _is_array(value) or isinstance(value, np.ndarray):
                shapes.append(value.shape)
        resolved = _resolve_batch_size(batch_size, shapes)
        _validate_batch_size(flat, resolved)
        object.__setattr__(self, "_data", flat)
        object.__setattr__(self, "batch_size", resolved)

    def keys(self) -> Iterable[Tuple[Any, ...]]:
        return self._data.keys()

    def values(self) -> Iterable[Any]:
        return self._data.values()

    def items(self) -> Iterable[Tuple[Tuple[Any, ...], Any]]:
        return self._data.items()

    def __len__(self) -> int:
        if not self.batch_size:
            return 0
        return self.batch_size[0]

    def __repr__(self) -> str:
        return f"ArrayDict(batch_size={self.batch_size}, keys={list(self._data.keys())})"

    def _is_column_key(self, key: Union[str, Tuple[Any, ...], Any]) -> bool:
        """Check if key is a column key (str or tuple of strings) rather than a batch index.
        
        Column keys:
        - Single string: "key"
        - Tuple of strings: ("nested", "key")
        
        Batch indices (everything else):
        - int, SupportsIndex: 0, 1, CustomIndex(2)
        - slice: slice(0, 5)
        - array: jnp.array([0, 1])
        - tuple with ints: (1,), (0, 1), (slice(None), 2)
        - None, Ellipsis
        """
        # Single string is a column key
        if isinstance(key, str):
            return (key,) in self._data or any(_starts_with(k, (key,)) for k in self._data)
        
        # Tuple: column key only if all elements are strings
        if isinstance(key, tuple):
            if all(isinstance(k, str) for k in key):
                return key in self._data or any(_starts_with(k, key) for k in self._data)
            # Tuple with non-strings is a batch index
            return False
        
        # Everything else is a batch index
        return False

    def _select_column(self, key: Union[str, Tuple[Any, ...]]) -> Union[jnp.ndarray, np.ndarray, "ArrayDict"]:
        """Select column(s) by key, returning array or sub-ArrayDict."""
        key_tuple = _normalize_key(key)
        if key_tuple in self._data:
            return self._data[key_tuple]
        sub: Dict[Tuple[Any, ...], Any] = {}
        for full_key, value in self._data.items():
            if _starts_with(full_key, key_tuple):
                sub_key = full_key[len(key_tuple) :]
                if not sub_key:
                    sub_key = ()
                sub[sub_key] = value
        if not sub:
            raise KeyError(key)
        if () in sub:
            return sub[()]
        return ArrayDict(_nest_from_flat(sub), batch_size=self.batch_size)

    @overload
    def __getitem__(self, key: str) -> Union[jnp.ndarray, np.ndarray, "ArrayDict"]: ...
    
    @overload
    def __getitem__(self, key: Tuple[Any, ...]) -> Union[jnp.ndarray, np.ndarray, "ArrayDict"]: ...
    
    @overload
    def __getitem__(self, key: BatchIndex) -> "ArrayDict": ...
    
    def __getitem__(self, key: Union[str, Tuple[Any, ...], BatchIndex]) -> Union[jnp.ndarray, np.ndarray, "ArrayDict"]:
        """Get item by column key or batch index.
        
        Column keys (str or tuple without batch indices) return arrays or sub-ArrayDict.
        Batch indices (int, slice, array, etc.) return new ArrayDict with indexed batch.
        
        Args:
            key: Column key (str or tuple) or batch index (int, slice, array, etc.)
            
        Returns:
            Array/ArrayDict for column selection, ArrayDict for batch indexing.
        """
        if self._is_column_key(key):
            return self._select_column(key)  # type: ignore
        return self._apply_batch_index(key)

    def _apply_batch_index(self, index: BatchIndex) -> "ArrayDict":
        """Apply batch indexing to all arrays, returning new ArrayDict."""
        # Normalize index only when necessary (for SupportsIndex objects)
        normalized_index = _normalize_index(index)
        
        # Apply index to all arrays
        new_data = {k: _apply_index(v, normalized_index) for k, v in self._data.items()}
        
        # Infer new batch size from applying index to batch shape
        dummy = jnp.empty(self.batch_size)
        new_batch = tuple(dummy[normalized_index].shape)
        
        return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)

    def reshape(self, new_shape: Sequence[int]) -> "ArrayDict":
        new_batch = self._resolve_new_batch(tuple(new_shape))
        new_data = {k: _reshape_with_batch(v, new_batch, self.batch_size) for k, v in self._data.items()}
        return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)

    def split(self, num: int, axis: int = 0) -> List["ArrayDict"]:
        split_values: Dict[Tuple[Any, ...], List[Any]] = {}
        for key, value in self._data.items():
            split_values[key] = _apply_split(value, num, axis)
        result: List[ArrayDict] = []
        for i in range(num):
            chunk = {k: v[i] for k, v in split_values.items()}
            batch = list(self.batch_size)
            if batch:
                batch[axis] = chunk[next(iter(chunk))].shape[axis]
            result.append(ArrayDict(_nest_from_flat(chunk), batch_size=tuple(batch)))
        return result

    def gather(self, indices: Any, axis: int = 0) -> "ArrayDict":
        new_data = {k: _apply_take(v, indices, axis) for k, v in self._data.items()}
        dummy = jnp.empty(self.batch_size)
        new_batch = tuple(jnp.take(dummy, indices, axis=axis).shape)
        return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)

    def to_nested_dict(self) -> Dict[Any, Any]:
        return _nest_from_flat(self._data)

    def to_rows(self) -> List[Dict[Any, Any]]:
        return _rows_from_flat(self._data, self.batch_size)

    def _resolve_new_batch(self, new_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if -1 not in new_shape:
            return new_shape
        old_size = int(np.prod(self.batch_size)) if self.batch_size else 0
        known = 1
        missing_index = None
        for i, dim in enumerate(new_shape):
            if dim == -1:
                if missing_index is not None:
                    raise ValueError("Only one -1 is allowed in reshape.")
                missing_index = i
            else:
                known *= dim
        if missing_index is None:
            return new_shape
        if known == 0:
            raise ValueError("Cannot infer shape with zero known size.")
        inferred = old_size // known
        resolved = list(new_shape)
        resolved[missing_index] = inferred
        return tuple(resolved)


def stack(items: Sequence[ArrayDict], axis: int = 0) -> ArrayDict:
    if not items:
        raise ValueError("stack requires at least one ArrayDict.")
    keys = list(items[0].keys())
    for item in items[1:]:
        if list(item.keys()) != keys:
            raise ValueError("All ArrayDict instances must have matching keys to stack.")
    new_data: Dict[Tuple[Any, ...], Any] = {}
    for key in keys:
        values = [item._data[key] for item in items]
        new_data[key] = _apply_stack(values, axis)
    new_batch = (len(items),) + items[0].batch_size
    return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)


def concat(items: Sequence[ArrayDict], axis: int = 0) -> ArrayDict:
    if not items:
        raise ValueError("concat requires at least one ArrayDict.")
    keys = list(items[0].keys())
    for item in items[1:]:
        if list(item.keys()) != keys:
            raise ValueError("All ArrayDict instances must have matching keys to concat.")
    new_data: Dict[Tuple[Any, ...], Any] = {}
    for key in keys:
        values = [item._data[key] for item in items]
        new_data[key] = _apply_concat(values, axis)
    batch = list(items[0].batch_size)
    if batch:
        batch[axis] = new_data[keys[0]].shape[axis]
    return ArrayDict(_nest_from_flat(new_data), batch_size=tuple(batch))
