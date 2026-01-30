"""Core ArrayDict implementation."""

from typing import Union, Tuple, List, Dict, Any, Mapping, Sequence, Iterator, Optional
import jax.numpy as jnp
import numpy as np

# Type definitions
KeyType = Union[str, Tuple[str, ...]]
ValueType = Union[
    Any,  # JAX arrays, torch tensors, numpy arrays, etc.
    "ArrayDict",  # Forward reference to ArrayDict
    Mapping["KeyType", "ValueType"],  # Nested mappings
    Sequence["ValueType"],  # Sequences
]


class ArrayDict:
    """A lightweight, JAX-backed container for arrays sharing leading batch dimensions."""

    def __init__(
        self,
        data: Mapping[KeyType, ValueType],
        batch_size: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
    ):
        """
        Initialize ArrayDict.

        Args:
            data: A mapping of keys to arrays/ArrayDict/mappings/sequences.
            batch_size: Expected batch size as int, list of ints, or tuple of ints.
                If int, treated as a single batch dimension.
                If None, inferred from data.
        """
        self._data: Dict[KeyType, Any] = {}
        self._batch_size: Tuple[int, ...] = ()

        # Flatten nested structures
        flattened = self._flatten_input(data)

        if batch_size is not None:
            if isinstance(batch_size, list):
                batch_size = tuple(batch_size)
            elif isinstance(batch_size, int):
                batch_size = (batch_size,)
            self._batch_size = batch_size
        else:
            # Infer batch size from data
            self._batch_size = self._infer_batch_size(flattened)

        # Store data and validate batch dimensions
        for key, value in flattened.items():
            self._validate_and_store(key, value)

    @staticmethod
    def _flatten_input(
        data: Mapping[KeyType, ValueType], prefix: Tuple[str, ...] = ()
    ) -> Dict[KeyType, Any]:
        """Flatten nested structures (dicts and ArrayDicts) into a flat dictionary."""
        flattened = {}
        for key, value in data.items():
            # Construct new key
            if isinstance(key, tuple):
                new_key = prefix + key
            else:
                new_key = prefix + (key,)

            if isinstance(value, ArrayDict):
                # Flatten nested ArrayDict
                nested_flat = ArrayDict._flatten_input(
                    ArrayDict._arraydict_to_dict(value), new_key
                )
                flattened.update(nested_flat)
            elif isinstance(value, dict):
                # Flatten nested dict
                nested_flat = ArrayDict._flatten_input(value, new_key)
                flattened.update(nested_flat)
            else:
                # Keep tuples as-is, convert single-string keys to strings
                if len(new_key) == 1:
                    final_key = new_key[0]
                else:
                    final_key = new_key
                flattened[final_key] = value

        return flattened

    @staticmethod
    def _arraydict_to_dict(ad: "ArrayDict") -> Dict[KeyType, Any]:
        """Convert ArrayDict to nested dict for flattening."""
        result = {}
        for key, value in ad._data.items():
            if isinstance(key, tuple):
                # Navigate nested structure
                current = result
                for k in key[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[key[-1]] = value
            else:
                result[key] = value
        return result

    def _infer_batch_size(self, flattened: Dict[KeyType, Any]) -> Tuple[int, ...]:
        """Infer batch size from data."""
        if not flattened:
            return ()

        inferred_size = None
        for key, value in flattened.items():
            # Get leading dimension
            if hasattr(value, "shape"):
                # Array-like (JAX, numpy, torch)
                shape = value.shape
                if len(shape) > 0:
                    leading_dim = shape[0]
                    if inferred_size is None:
                        inferred_size = (leading_dim,)
                    elif inferred_size != (leading_dim,):
                        raise ValueError(
                            f"Inconsistent batch sizes: {inferred_size} vs {(leading_dim,)}"
                        )
            elif isinstance(value, (list, tuple)):
                # Sequence
                seq_len = len(value)
                if inferred_size is None:
                    inferred_size = (seq_len,)
                elif inferred_size != (seq_len,):
                    raise ValueError(
                        f"Inconsistent batch sizes: {inferred_size} vs {(seq_len,)}"
                    )

        return inferred_size if inferred_size is not None else ()

    def _validate_and_store(self, key: KeyType, value: Any) -> None:
        """Validate that value matches batch size and store it."""
        expected_batch_dim = self._batch_size[0] if self._batch_size else None

        if hasattr(value, "shape"):
            # Array-like
            if expected_batch_dim is not None and value.shape[0] != expected_batch_dim:
                raise ValueError(
                    f"Value for key {key} has batch dimension {value.shape[0]}, "
                    f"expected {expected_batch_dim}"
                )
        elif isinstance(value, (list, tuple)):
            # Sequence
            if expected_batch_dim is not None and len(value) != expected_batch_dim:
                raise ValueError(
                    f"Value for key {key} has length {len(value)}, "
                    f"expected {expected_batch_dim}"
                )

        self._data[key] = value

    def __getitem__(
        self, key: Union[KeyType, int, slice, np.ndarray, jnp.ndarray]
    ) -> Union["ArrayDict", Any]:
        """Get item by key or index."""
        # Handle indexing/slicing
        if isinstance(key, int):
            return self._index_single(key)
        elif isinstance(key, slice):
            return self._index_slice(key)
        elif isinstance(key, (np.ndarray, jnp.ndarray)):
            if key.dtype == bool:
                return self._index_bool(key)
            else:
                return self._index_gather(key)
        elif isinstance(key, (list, tuple)) and key and isinstance(key[0], (int, np.integer)):
            # List/tuple of indices
            return self._index_gather(jnp.array(key))

        # Handle key access
        if isinstance(key, str):
            return self._get_key(key)
        elif isinstance(key, tuple):
            return self._get_key(key)

        raise TypeError(f"Unsupported index type: {type(key)}")

    def _index_single(self, idx: int) -> "ArrayDict":
        """Get single element by index."""
        new_data = {}
        for key, value in self._data.items():
            if hasattr(value, "shape"):
                new_data[key] = value[idx]
            elif isinstance(value, (list, tuple)):
                new_data[key] = value[idx]
            else:
                new_data[key] = value

        return ArrayDict._from_flat_data(new_data, batch_size=())

    def _index_slice(self, slc: slice) -> "ArrayDict":
        """Get slice of elements."""
        new_data = {}
        for key, value in self._data.items():
            if hasattr(value, "shape"):
                new_data[key] = value[slc]
            elif isinstance(value, (list, tuple)):
                new_data[key] = value[slc]
            else:
                new_data[key] = value

        # Calculate new batch size
        start = slc.start or 0
        stop = slc.stop or self._batch_size[0]
        new_batch_size = (stop - start,)
        return ArrayDict._from_flat_data(new_data, batch_size=new_batch_size)

    def _index_bool(self, mask: jnp.ndarray) -> "ArrayDict":
        """Get elements using boolean mask."""
        if mask.shape != (self._batch_size[0],):
            raise ValueError(
                f"Boolean mask shape {mask.shape} doesn't match batch size {self._batch_size}"
            )

        new_data = {}
        for key, value in self._data.items():
            if hasattr(value, "shape"):
                new_data[key] = value[mask]
            elif isinstance(value, (list, tuple)):
                # For lists, apply mask by converting to array
                arr = jnp.array(value)
                masked = arr[mask]
                new_data[key] = masked.tolist() if isinstance(value, list) else tuple(masked)
            else:
                new_data[key] = value

        new_batch_size = (int(jnp.sum(mask)),)
        return ArrayDict._from_flat_data(new_data, batch_size=new_batch_size)

    def _index_gather(self, indices: jnp.ndarray) -> "ArrayDict":
        """Get elements using indices array."""
        new_data = {}
        for key, value in self._data.items():
            if hasattr(value, "shape"):
                new_data[key] = value[indices]
            elif isinstance(value, (list, tuple)):
                # For lists, apply indices
                new_list = [value[int(i)] for i in indices]
                new_data[key] = new_list if isinstance(value, list) else tuple(new_list)
            else:
                new_data[key] = value

        new_batch_size = (len(indices),)
        return ArrayDict._from_flat_data(new_data, batch_size=new_batch_size)

    def _get_key(self, key: KeyType) -> Any:
        """Get value by key, reconstructing nested structures if needed."""
        if key in self._data:
            return self._data[key]

        # Check if this is a prefix of nested keys
        if isinstance(key, str):
            prefix = (key,)
        else:
            prefix = key

        nested_keys = {
            k: v for k, v in self._data.items() if isinstance(k, tuple) and k[: len(prefix)] == prefix
        }

        if nested_keys:
            # Reconstruct nested ArrayDict
            nested_data = {}
            for k, v in nested_keys.items():
                # Remove prefix and reconstruct
                suffix = k[len(prefix) :]
                if len(suffix) == 1:
                    nested_data[suffix[0]] = v
                else:
                    nested_data[suffix] = v
            return ArrayDict._from_flat_data(nested_data, batch_size=self._batch_size)

        raise KeyError(f"Key {key} not found")

    def reshape(self, new_shape: Union[Tuple[int, ...], List[int]]) -> "ArrayDict":
        """Reshape batch dimensions."""
        if isinstance(new_shape, list):
            new_shape = tuple(new_shape)

        new_data = {}
        for key, value in self._data.items():
            if hasattr(value, "shape"):
                # Reshape arrays
                orig_shape = value.shape
                new_value_shape = new_shape + orig_shape[len(self._batch_size) :]
                new_data[key] = jnp.reshape(value, new_value_shape)
            elif isinstance(value, (list, tuple)):
                # Can't reshape lists, keep as-is
                new_data[key] = value
            else:
                new_data[key] = value

        return ArrayDict._from_flat_data(new_data, batch_size=new_shape)

    def split(self, num_splits: int) -> List["ArrayDict"]:
        """Split along first batch dimension."""
        batch_size = self._batch_size[0]
        if batch_size % num_splits != 0:
            raise ValueError(
                f"Batch size {batch_size} not divisible by {num_splits}"
            )

        chunk_size = batch_size // num_splits
        result = []

        for i in range(num_splits):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            result.append(self[start:end])

        return result

    def gather(self, indices: Union[List[int], jnp.ndarray]) -> "ArrayDict":
        """Gather elements by indices."""
        if isinstance(indices, list):
            indices = jnp.array(indices)
        return self._index_gather(indices)

    def keys(self) -> Iterator[KeyType]:
        """Return iterator over keys."""
        return iter(self._data.keys())

    def values(self) -> Iterator[Any]:
        """Return iterator over values."""
        return iter(self._data.values())

    def items(self) -> Iterator[Tuple[KeyType, Any]]:
        """Return iterator over items."""
        return iter(self._data.items())

    @property
    def batch_size(self) -> Tuple[int, ...]:
        """Return batch size."""
        return self._batch_size

    @staticmethod
    def _from_flat_data(
        data: Dict[KeyType, Any], batch_size: Tuple[int, ...]
    ) -> "ArrayDict":
        """Create ArrayDict from flat data (internal use)."""
        ad = ArrayDict.__new__(ArrayDict)
        ad._data = data
        ad._batch_size = batch_size
        return ad

    def __repr__(self) -> str:
        """String representation in tensordict-style format."""
        # Get string representation of batch size
        batch_size_str = f"shape={self._batch_size}" if self._batch_size else "shape=()"
        
        # Build content lines
        lines = [f"{self.__class__.__name__}("]
        lines.append(f"    {batch_size_str},")
        
        if self._data:
            lines.append("    data={")
            
            # Organize keys into nested structure for display
            structured = {}
            
            for key in sorted(self._data.keys(), key=lambda x: str(x)):
                if isinstance(key, tuple):
                    # Build nested structure for tuple keys
                    current = structured
                    for part in key[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[key[-1]] = self._data[key]
                else:
                    structured[key] = self._data[key]
            
            # Recursively format the structured data
            self._format_repr_lines(structured, lines, base_indent=8)
            
            lines.append("    }")
        
        lines.append(")")
        return "\n".join(lines)

    def __len__(self) -> int:
        """Return length of the leading batch dimension."""
        if not self._batch_size:
            raise TypeError("ArrayDict with scalar batch has no len()")
        return int(self._batch_size[0])
    
    @staticmethod
    def _format_repr_lines(
        data: Dict[Any, Any], lines: List[str], base_indent: int = 8
    ) -> None:
        """Recursively format data structure for repr."""
        indent = base_indent
        indent_str = " " * indent
        
        for key in sorted(data.keys(), key=lambda x: str(x)):
            value = data[key]
            
            if isinstance(value, dict) and value:
                # Nested dict - show as nested structure
                lines.append(f"{indent_str}{key!r}: {{")
                ArrayDict._format_repr_lines(value, lines, base_indent=indent + 4)
                lines.append(f"{indent_str}}},")
            else:
                # Leaf value
                value_repr = ArrayDict._value_repr(value)
                lines.append(f"{indent_str}{key!r}: {value_repr},")
    
    @staticmethod
    def _value_repr(value: Any) -> str:
        """Get a concise string representation of a value."""
        if hasattr(value, "shape"):
            # Array-like (JAX, numpy, torch)
            shape = value.shape
            dtype = getattr(value, "dtype", "unknown")
            return f"Array(shape={shape}, dtype={dtype})"
        elif isinstance(value, (list, tuple)):
            length = len(value)
            if length == 0:
                return f"{type(value).__name__}(len=0)"
            
            # Check element types and decide how to display
            first_elem = value[0]
            
            # For sequences of simple types or strings, show samples
            if isinstance(first_elem, (str, int, float, bool)):
                # Show first 3 samples
                samples = value[:3] if length > 3 else value
                samples_str = ", ".join(repr(s) for s in samples)
                if length > 3:
                    return f"{type(value).__name__}([{samples_str}, ...], len={length})"
                else:
                    return f"{type(value).__name__}([{samples_str}], len={length})"
            else:
                # For complex types, find common element types
                types_set = set(type(elem).__name__ for elem in value)
                if len(types_set) == 1:
                    elem_type = list(types_set)[0]
                    return f"{type(value).__name__}(dtype={elem_type}, len={length})"
                else:
                    type_list = sorted(list(types_set)[:3])
                    return f"{type(value).__name__}(types={type_list}, len={length})"
        elif isinstance(value, dict):
            keys = list(value.keys())
            return f"dict(keys={keys})"
        else:
            return repr(value)[:50]  # Truncate long representations
