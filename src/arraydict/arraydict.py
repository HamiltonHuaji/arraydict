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


def _is_pure_numpy(value: Any) -> bool:
    """Check if value is a pure NumPy array (not JAX-wrapped)."""
    return isinstance(value, np.ndarray) and not isinstance(value, jnp.ndarray)


def _is_jax_array(value: Any) -> bool:
    """Check if value is a JAX array."""
    return isinstance(value, jnp.ndarray)


def _is_object_array(value: Any) -> bool:
    """Check if value is an array with object dtype (non-numeric data)."""
    return isinstance(value, np.ndarray) and value.dtype == object


def _dispatch_array_op(
    value: Any,
    jax_op: callable,
    numpy_op: callable,
    *args,
    **kwargs
) -> Any:
    """
    Dispatch array operation based on value type.
    
    Invariants enforced:
    - Pure numpy arrays (object dtype) use numpy_op
    - JAX arrays use jax_op
    - Non-arrays return unchanged
    
    Args:
        value: Array to operate on
        jax_op: Operation for JAX arrays (e.g., jnp.squeeze)
        numpy_op: Operation for NumPy arrays (e.g., np.squeeze)
        *args, **kwargs: Arguments passed to the operation
        
    Returns:
        Result of operation, maintaining array type
    """
    if _is_pure_numpy(value):
        return numpy_op(value, *args, **kwargs)
    elif _is_jax_array(value):
        return jax_op(value, *args, **kwargs)
    else:
        return value


def _validate_dimension(dim: int, batch_size: Tuple[int, ...], operation: str) -> None:
    """Validate dimension index for batch operations."""
    if dim < 0 or dim >= len(batch_size):
        raise ValueError(
            f"Dimension {dim} out of range for {operation} with batch_size {batch_size}"
        )


def _validate_squeeze_dimension(dim: int, batch_size: Tuple[int, ...]) -> None:
    """Validate that dimension can be squeezed (size must be 1)."""
    _validate_dimension(dim, batch_size, "squeeze")
    if batch_size[dim] != 1:
        raise ValueError(
            f"Cannot squeeze dimension {dim} with size {batch_size[dim]}; must be 1"
        )


def _validate_unsqueeze_dimension(dim: int, batch_size: Tuple[int, ...]) -> None:
    """Validate dimension index for unsqueeze (can be at end)."""
    if dim < 0 or dim > len(batch_size):
        raise ValueError(
            f"Dimension {dim} out of range for unsqueeze in batch_size {batch_size}"
        )


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
    """
    Reshape array preserving feature dimensions beyond batch.
    
    Invariant: Only batch dimensions change, feature dims preserved.
    """
    if not _is_array(value):
        return value
    
    tail = value.shape[len(old_batch):]
    new_shape = new_batch + tail
    
    # Use appropriate reshape based on array type
    return _dispatch_array_op(value, 
                               lambda v: v.reshape(new_shape),
                               lambda v: v.reshape(new_shape))


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
    """Apply index to value, treating scalars as object arrays for consistent behavior."""
    if _is_array(value) or isinstance(value, np.ndarray):
        result = value[index]
    else:
        # For non-array values, wrap as object array and apply index
        arr = np.array(value, dtype=object)
        result = arr[index]
    
    # If result is still a scalar/string after indexing, wrap it back as object array
    if not (_is_array(result) or isinstance(result, np.ndarray)):
        result = np.array(result, dtype=object)
    
    return result


def _apply_take(value: Any, indices: Any, axis: int) -> Any:
    """Apply take/indexing operation along an axis."""
    return _dispatch_array_op(value, jnp.take, np.take, indices, axis=axis)


def _apply_split(value: Any, num: int, axis: int) -> List[Any]:
    """Split array into num parts along axis."""
    def jax_split(arr, n, ax):
        return list(jnp.split(arr, n, axis=ax))
    
    def numpy_split(arr, n, ax):
        return list(np.split(arr, n, axis=ax))
    
    if _is_array(value):
        if _is_pure_numpy(value):
            return numpy_split(value, num, axis)
        else:
            return jax_split(value, num, axis)
    return [value for _ in range(num)]


def _apply_stack(values: List[Any], axis: int) -> Any:
    """Stack a list of arrays along a new axis."""
    if not values:
        return values
    first = values[0]
    if _is_pure_numpy(first):
        return np.stack(values, axis=axis)
    elif _is_jax_array(first):
        return jnp.stack(values, axis=axis)
    return values


def _apply_concat(values: List[Any], axis: int) -> Any:
    """Concatenate a list of arrays along an existing axis."""
    if not values:
        return values
    first = values[0]
    if _is_pure_numpy(first):
        return np.concatenate(values, axis=axis)
    elif _is_jax_array(first):
        return jnp.concatenate(values, axis=axis)
    return values


def _resolve_batch_size(batch_size: Optional[Union[int, Sequence[int]]], shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:

    if batch_size is None:
        if not shapes:
            # No array values to infer from - default to scalar
            return ()
        inferred = _common_prefix(shapes)
        # inferred can be empty tuple () which is valid (all scalars)
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


@dataclass
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
        self._data = flat
        self.batch_size = resolved

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

    @property
    def ndim(self) -> int:
        return len(self.batch_size)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get shape - returns batch_size (the public/exposed dimensions).
        
        Feature dimensions (after batch dimensions) are private/opaque.
        Only batch_size is exposed as the public "shape" of the ArrayDict.
        """
        return self.batch_size

    def __repr__(self) -> str:
        """Return a detailed, multi-line representation of the ArrayDict.
        
        Format similar to TensorDict, with nested structures:
        ArrayDict(
            fields={
                key: Tensor(shape=..., dtype=...),
                str_key: StringTensor(shape=..., content=['...', ...]),
                nested_key: ArrayDict(...),
                ...},
            batch_size=...,
            ...)
        """
        def format_value(value: Any, max_items: int = 3) -> str:
            """Format a single value with intelligent type detection."""
            # Check if it's an array (JAX or numpy)
            is_jax = _is_array(value)
            is_numpy = isinstance(value, np.ndarray)
            
            if is_jax or is_numpy:
                # Convert to numpy for inspection if needed
                arr = np.asarray(value) if is_jax else value
                
                if arr.dtype == object:
                    # Object array - check content type
                    flat = arr.flatten()
                    if len(flat) == 0:
                        return f"ObjectArray(shape={arr.shape}, dtype=object, empty=True)"
                    
                    # Sample first few non-None elements to determine type
                    sample_items = []
                    sample_types = set()
                    for item in flat[:min(20, len(flat))]:
                        if item is not None:
                            sample_items.append(item)
                            sample_types.add(type(item))
                        if len(sample_items) >= 10:
                            break
                    
                    if len(sample_types) == 0:
                        return f"ObjectArray(shape={arr.shape}, dtype=object, all_none=True)"
                    
                    if len(sample_types) == 1:
                        elem_type = next(iter(sample_types))
                        elem_type_name = elem_type.__name__
                        
                        # Special handling for strings
                        if elem_type == str:
                            # Show sample strings
                            samples = []
                            for i, item in enumerate(sample_items[:max_items]):
                                s = str(item)
                                if len(s) > 30:
                                    s = s[:27] + "..."
                                samples.append(f"'{s}'")
                            if len(flat) > max_items:
                                samples.append("...")
                            content = ", ".join(samples)
                            return f"StringArray(shape={arr.shape}, content=[{content}])"
                        elif elem_type.__name__ in ['WindowsPath', 'PosixPath', 'Path']:
                            # Show sample paths
                            samples = []
                            for i, item in enumerate(sample_items[:max_items]):
                                s = str(item)
                                if len(s) > 30:
                                    s = s[:27] + "..."
                                samples.append(f"'{s}'")
                            if len(flat) > max_items:
                                samples.append("...")
                            content = ", ".join(samples)
                            return f"PathArray(shape={arr.shape}, content=[{content}])"
                        else:
                            # Non-string uniform type
                            return f"ObjectArray(shape={arr.shape}, dtype={elem_type_name})"
                    else:
                        # Mixed types
                        type_names = sorted([t.__name__ for t in sample_types])
                        return f"ObjectArray(shape={arr.shape}, dtype=object, types={type_names})"
                else:
                    # Regular numeric array
                    dtype_str = str(arr.dtype)
                    return f"Tensor(shape={arr.shape}, dtype={dtype_str})"
            elif isinstance(value, ArrayDict):
                return f"ArrayDict(batch_size={value.batch_size})"
            else:
                return "..."
        
        def format_fields(data: Dict[Tuple[Any, ...], Any], base_indent: str = "") -> List[str]:
            """Format fields with proper nesting."""
            lines = []
            indent_str = base_indent + "        "  # 8 spaces for top-level fields
            
            # Group keys by first element
            grouped: Dict[Any, List[Tuple[Any, ...]]] = {}
            for key in sorted(data.keys()):
                if key:
                    first = key[0]
                    if first not in grouped:
                        grouped[first] = []
                    grouped[first].append(key)
            
            # Format each group
            for first_key in sorted(grouped.keys()):
                keys_with_prefix = grouped[first_key]
                
                if len(keys_with_prefix) == 1 and len(keys_with_prefix[0]) == 1:
                    # Single key, not nested
                    key = keys_with_prefix[0]
                    value = data[key]
                    value_str = format_value(value)
                    lines.append(f"{indent_str}{first_key}: {value_str},")
                else:
                    # Multiple keys with same prefix or nested keys
                    # Check if all have the same structure (single nested level)
                    all_length_2 = all(len(k) == 2 for k in keys_with_prefix)
                    
                    if all_length_2:
                        # Format as nested dict
                        lines.append(f"{indent_str}{first_key}: {{")
                        nested_indent = indent_str + "    "  # 4 more spaces for nested content
                        for key in keys_with_prefix:
                            value = data[key]
                            rest = key[1]
                            value_str = format_value(value)
                            lines.append(f"{nested_indent}{rest}: {value_str},")
                        lines.append(f"{indent_str}}},")
                    else:
                        # Mixed depths or deeper nesting - format as nested ArrayDict
                        nested_data = {}
                        for key in keys_with_prefix:
                            rest = key[1:]
                            nested_data[rest] = data[key]
                        
                        nested_ad = ArrayDict(_nest_from_flat(nested_data), batch_size=self.batch_size)
                        nested_repr = repr(nested_ad)
                        # Indent nested repr
                        nested_lines = nested_repr.split("\n")
                        lines.append(f"{indent_str}{first_key}: {nested_lines[0]}")
                        for line in nested_lines[1:]:
                            lines.append(f"{indent_str}{line}")
            
            return lines
        
        lines = ["ArrayDict("]
        lines.append("    fields={")
        
        field_lines = format_fields(self._data, base_indent="")
        lines.extend(field_lines)
        
        lines.append("    },")
        lines.append(f"    batch_size={self.batch_size})")
        
        return "\n".join(lines)

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

    @overload
    def __setitem__(self, key: str, value: Any) -> None: ...

    @overload
    def __setitem__(self, key: Tuple[Any, ...], value: Any) -> None: ...

    @overload
    def __setitem__(self, key: BatchIndex, value: Any) -> None: ...

    def __setitem__(self, key: Union[str, Tuple[Any, ...], BatchIndex], value: Any) -> None:
        """Set item by column key.
        
        Only column assignment is supported for now. Batch/row indexing assignment
        is not implemented yet.
        """
        if isinstance(key, tuple):
            key_tuple = key
        elif isinstance(key, str):
            key_tuple = (key,)
        else:
            raise NotImplementedError("ArrayDict __setitem__ currently supports only column assignment with string keys.")

        if not key_tuple or not all(isinstance(part, str) for part in key_tuple):
            raise NotImplementedError(
                "ArrayDict __setitem__ currently supports only column assignment with string keys."
            )

        new_flat: Dict[Tuple[Any, ...], Any] = {}
        if isinstance(value, ArrayDict):
            if value.batch_size != self.batch_size:
                raise ValueError(
                    f"Assigned ArrayDict batch_size {value.batch_size} does not match {self.batch_size}."
                )
            for sub_key, sub_value in value._data.items():
                new_flat[key_tuple + sub_key] = sub_value
        elif _is_mapping(value):
            _flatten_mapping(cast(Mapping[Any, Any], value), key_tuple, new_flat)
        elif _is_sequence(value) and not _is_array(value):
            new_flat[key_tuple] = _ensure_object_array(value)
        else:
            new_flat[key_tuple] = value

        _validate_batch_size(new_flat, self.batch_size)

        updated = dict(self._data)
        updated.update(new_flat)
        self._data = updated

    def set(self, key: Union[str, Tuple[Any, ...], BatchIndex], value: Any) -> "ArrayDict":
        """Return a new ArrayDict with the column assignment applied.

        This keeps the original ArrayDict unchanged.
        """
        if isinstance(key, tuple):
            key_tuple = key
        elif isinstance(key, str):
            key_tuple = (key,)
        else:
            raise NotImplementedError(
                "ArrayDict set currently supports only column assignment with string keys."
            )

        if not key_tuple or not all(isinstance(part, str) for part in key_tuple):
            raise NotImplementedError(
                "ArrayDict set currently supports only column assignment with string keys."
            )

        new_flat: Dict[Tuple[Any, ...], Any] = {}
        if isinstance(value, ArrayDict):
            if value.batch_size != self.batch_size:
                raise ValueError(
                    f"Assigned ArrayDict batch_size {value.batch_size} does not match {self.batch_size}."
                )
            for sub_key, sub_value in value._data.items():
                new_flat[key_tuple + sub_key] = sub_value
        elif _is_mapping(value):
            _flatten_mapping(cast(Mapping[Any, Any], value), key_tuple, new_flat)
        elif _is_sequence(value) and not _is_array(value):
            new_flat[key_tuple] = _ensure_object_array(value)
        else:
            new_flat[key_tuple] = value

        _validate_batch_size(new_flat, self.batch_size)

        updated = dict(self._data)
        updated.update(new_flat)
        return ArrayDict(_nest_from_flat(updated), batch_size=self.batch_size)

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

    def squeeze(self, dim: int) -> "ArrayDict":
        """Remove a batch dimension of size 1.
        
        Args:
            dim: Batch dimension index to squeeze.
            
        Raises:
            ValueError: If the dimension is not of size 1.
        """
        _validate_squeeze_dimension(dim, self.batch_size)
        
        new_batch = self.batch_size[:dim] + self.batch_size[dim + 1:]
        new_data = {k: _dispatch_array_op(v, jnp.squeeze, np.squeeze, axis=dim) 
                    for k, v in self._data.items()}
        
        return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)

    def unsqueeze(self, dim: int) -> "ArrayDict":
        """Insert a new batch dimension of size 1 at the specified position.
        
        Args:
            dim: Position to insert the new dimension (0 <= dim <= len(batch_size)).
        """
        _validate_unsqueeze_dimension(dim, self.batch_size)
        
        new_batch = self.batch_size[:dim] + (1,) + self.batch_size[dim:]
        new_data = {k: _dispatch_array_op(v, jnp.expand_dims, np.expand_dims, axis=dim)
                    for k, v in self._data.items()}
        
        return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)

    def repeat(self, repeats: Union[int, Tuple[int, ...]]) -> "ArrayDict":
        """Repeat batch dimensions efficiently without creating intermediate copies.
        
        Uses jnp.repeat to repeat along batch dimensions. Each field's feature
        dimensions are preserved unchanged.
        
        Args:
            repeats: Repeat count(s) for batch dimensions:
                - int: repeat all batch dimensions the same number of times
                - tuple: repeat count for each batch dimension
        
        Returns:
            New ArrayDict with repeated batch dimensions
        
        Example:
            >>> ad = ArrayDict({'x': np.ones((2, 3, 4))}, batch_size=(2, 3))
            >>> result = ad.repeat((2, 3))
            >>> result.batch_size
            (4, 9)
            >>> result['x'].shape
            (4, 9, 4)
        """
        # repeats is a tuple where each element is the repeat count for that batch axis
        if not repeats or all(r == 1 for r in (repeats if isinstance(repeats, tuple) else (repeats,))):
            return ArrayDict(_nest_from_flat(self._data), batch_size=self.batch_size)
        
        # Ensure repeats is a tuple
        if isinstance(repeats, int):
            repeats = (repeats,)
        else:
            repeats = tuple(repeats)
        
        # Pad repeats with 1s for batch dimensions not specified
        batch_ndim = len(self.batch_size)
        while len(repeats) < batch_ndim:
            repeats = repeats + (1,)
        
        new_data = {}
        for key, value in self._data.items():
            if _is_array(value):
                # Apply repeat to each batch dimension
                result = value
                # Process from last batch dim to first to maintain correct axis positions
                for axis in range(batch_ndim - 1, -1, -1):
                    if repeats[axis] > 1:
                        result = jnp.repeat(result, repeats[axis], axis=axis)
                new_data[key] = result
            else:
                new_data[key] = value
        
        # Compute new batch size
        new_batch = tuple(self.batch_size[i] * repeats[i] for i in range(batch_ndim))
        
        return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)

    def to_nested_dict(self) -> Dict[Any, Any]:
        return _nest_from_flat(self._data)

    def to_rows(self) -> List[Dict[Any, Any]]:
        return _rows_from_flat(self._data, self.batch_size)

    def _resolve_new_batch(self, new_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if -1 not in new_shape:
            return new_shape
        old_size = int(np.prod(self.batch_size))  # prod([]) = 1, so this always works
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
    
    # Compute new batch_size using dummy arrays to handle all axis cases correctly
    dummies = [jnp.empty(item.batch_size) for item in items]
    new_batch = tuple(jnp.stack(dummies, axis=axis).shape)
    
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
    
    # Compute new batch_size using dummy arrays to handle all axis cases correctly
    dummies = [jnp.empty(item.batch_size) for item in items]
    new_batch = tuple(jnp.concat(dummies, axis=axis).shape)
    
    return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)
