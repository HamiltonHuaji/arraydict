"""
einops backend for ArrayDict.

This module registers ArrayDict as an einops backend, allowing direct use:

    >>> import einops
    >>> from arraydict import ArrayDict
    >>> import jax.numpy as jnp
    >>>
    >>> ad = ArrayDict({'x': jnp.ones((2, 3, 4))}, batch_size=(2, 3))
    >>> 
    >>> # Use einops directly on ArrayDict!
    >>> result = einops.rearrange(ad, 'b1 b2 f -> (b1 b2) f')
    >>> print(result.batch_size)  # (6,)

Operations apply to all numeric fields while preserving non-numeric fields.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from einops._backends import AbstractBackend
    HAS_EINOPS = True
except ImportError:
    HAS_EINOPS = False
    AbstractBackend = object  # type: ignore

import jax.numpy as jnp
import numpy as np

from .arraydict import (
    ArrayDict,
    _is_array,
    _nest_from_flat,
)


def _check_einops() -> None:
    """Check if einops is available."""
    if not HAS_EINOPS:
        raise ImportError(
            "einops is required for ArrayDict backend. "
            "Install it with: pip install einops"
        )


def _infer_batch_from_shapes(shapes: list) -> Tuple[int, ...]:
    """Infer batch_size as common leading dimensions of shapes.
    
    For einops operations, we typically want batch to be just the first dimension
    if all fields have the same shape. This simplifies batch_size inference after
    transformations like transpose.
    """
    if not shapes:
        return ()
    
    # Check if all shapes are identical
    first_shape = shapes[0]
    if all(s == first_shape for s in shapes):
        # All identical - just use first dimension as batch
        if len(first_shape) > 0:
            return (first_shape[0],)
        return ()
    
    # Shapes differ - find common prefix
    min_ndim = min(len(s) for s in shapes)
    if min_ndim == 0:
        return ()
    
    batch_ndim = 0
    for dim_idx in range(min_ndim):
        dim_values = [s[dim_idx] for s in shapes]
        if len(set(dim_values)) == 1:
            batch_ndim += 1
        else:
            break
    
    return tuple(first_shape[:batch_ndim])



if HAS_EINOPS:
    class ArrayDictBackend(AbstractBackend):
        """einops backend for ArrayDict.
        
        This backend allows einops operations to work directly on ArrayDict objects.
        Operations apply ONLY to the batch_size dimensions, leaving feature dimensions
        (after batch_size) unchanged. Non-numeric fields are always preserved.
        
        Important: Only batch dimensions are reshaped/transposed. Reduce is not
        supported as we don't know how to reduce feature dimensions.
        """
        
        framework_name = "arraydict"
        
        def is_appropriate_type(self, tensor: Any) -> bool:
            """Check if tensor is an ArrayDict."""
            return isinstance(tensor, ArrayDict)
        
        def from_numpy(self, x: np.ndarray) -> Any:
            """Convert numpy array to JAX array."""
            return jnp.asarray(x)
        
        def to_numpy(self, x: Any) -> np.ndarray:
            """Convert array to numpy."""
            if isinstance(x, np.ndarray):
                return x
            return np.asarray(x)
        
        def shape(self, x: ArrayDict) -> Tuple[int, ...]:
            """Get shape - return only batch dimensions for einops.
            
            This is key: einops operates on the dimensions described in the pattern.
            For ArrayDict, the pattern describes the batch dimensions only.
            """
            return x.batch_size
        
        def _reshape_batch_dims(self, x: ArrayDict, new_batch_shape: Tuple[int, ...]) -> ArrayDict:
            """Reshape only the batch dimensions, preserving feature dimensions."""
            # Convert new_batch_shape to tuple if it's a list
            if isinstance(new_batch_shape, list):
                new_batch_shape = tuple(new_batch_shape)
            
            new_data = {}
            for key, value in x._data.items():
                if _is_array(value):
                    # Get this field's feature dimensions (everything after batch dims)
                    batch_ndim = len(x.batch_size)
                    feature_shape = value.shape[batch_ndim:]
                    # Combine new batch shape with THIS field's feature shape
                    new_shape = tuple(new_batch_shape) + feature_shape
                    new_data[key] = value.reshape(new_shape)
                else:
                    new_data[key] = value
            
            return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch_shape)
        
        def reshape(self, x: ArrayDict, shape: Tuple[int, ...]) -> ArrayDict:
            """Reshape batch dimensions only."""
            # shape passed by einops is the BATCH shape (not including features)
            return self._reshape_batch_dims(x, shape)
        
        def transpose(self, x: ArrayDict, axes: Tuple[int, ...]) -> ArrayDict:
            """Transpose batch dimensions only.
            
            axes apply only to batch dimensions. We need to adjust them
            to account for feature dimensions that come after.
            """
            batch_ndim = len(x.batch_size)
            
            new_data = {}
            for key, value in x._data.items():
                if _is_array(value):
                    # Adjust axes to include feature dimensions
                    adjusted_axes = tuple(ax if ax < batch_ndim else ax 
                                        for ax in axes)
                    # Add feature dimension indices (unchanged)
                    adjusted_axes = tuple(adjusted_axes) + tuple(range(batch_ndim, len(value.shape)))
                    new_data[key] = jnp.transpose(value, adjusted_axes)
                else:
                    new_data[key] = value
            
            # Compute new batch size by applying the permutation
            new_batch = tuple(x.batch_size[i] for i in axes)
            return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)
        
        def reduce(self, x: ArrayDict, operation: str, axes: Tuple[int, ...]) -> ArrayDict:
            """Reduce is not supported for ArrayDict."""
            raise NotImplementedError(
                "reduce() is not supported for ArrayDict einops backend, "
                "as we cannot determine how to reduce feature dimensions. "
                "Consider using ArrayDict-specific methods instead."
            )
        
        def stack_on_zeroth_dimension(self, tensors: List[ArrayDict]) -> ArrayDict:
            """Stack ArrayDicts along new axis 0 in batch dimensions."""
            if not tensors:
                raise ValueError("Need at least one ArrayDict to stack")
            
            # Check all have same keys
            keys = list(tensors[0].keys())
            for tensor in tensors[1:]:
                if list(tensor.keys()) != keys:
                    raise ValueError("All ArrayDicts must have same keys")
            
            # Get feature shape from first array
            feature_shape = ()
            for value in tensors[0]._data.values():
                if _is_array(value):
                    batch_ndim = len(tensors[0].batch_size)
                    feature_shape = value.shape[batch_ndim:]
                    break
            
            new_data = {}
            for key in keys:
                values = [t._data[key] for t in tensors]
                if _is_array(values[0]):
                    # Stack along batch dimension (axis 0)
                    stacked = jnp.stack(values, axis=0)
                    new_data[key] = stacked
                else:
                    new_data[key] = values[0]
            
            new_batch = (len(tensors),) + tensors[0].batch_size
            return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)
        
        def add_axis(self, x: ArrayDict, new_position: int) -> ArrayDict:
            """Add axis to batch dimensions only."""
            new_data = {}
            
            # Get feature shape
            feature_shape = ()
            for value in x._data.values():
                if _is_array(value):
                    batch_ndim = len(x.batch_size)
                    feature_shape = value.shape[batch_ndim:]
                    break
            
            for key, value in x._data.items():
                if _is_array(value):
                    # Expand only in batch dimension range
                    new_data[key] = jnp.expand_dims(value, new_position)
                else:
                    new_data[key] = value
            
            # Add dimension to batch_size
            batch_list = list(x.batch_size)
            batch_list.insert(new_position, 1)
            new_batch = tuple(batch_list)
            
            return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)
        
        def tile(self, x: ArrayDict, repeats: Tuple[int, ...]) -> ArrayDict:
            """Tile only batch dimensions."""
            # Get feature shape
            feature_shape = ()
            for value in x._data.values():
                if _is_array(value):
                    batch_ndim = len(x.batch_size)
                    feature_shape = value.shape[batch_ndim:]
                    break
            
            # Adjust repeats to only apply to batch dimensions
            batch_ndim = len(x.batch_size)
            adjusted_repeats = repeats + (1,) * len(feature_shape)
            
            new_data = {}
            for key, value in x._data.items():
                if _is_array(value):
                    new_data[key] = jnp.tile(value, adjusted_repeats)
                else:
                    new_data[key] = value
            
            # Compute new batch size by multiplying tiled dimensions
            new_batch = tuple(x.batch_size[i] * repeats[i] if i < len(repeats) else x.batch_size[i]
                            for i in range(len(x.batch_size)))
            
            return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)
        
        def concat(self, tensors: List[ArrayDict], axis: int) -> ArrayDict:
            """Concatenate ArrayDicts along batch axis."""
            if not tensors:
                raise ValueError("Need at least one ArrayDict to concat")
            
            keys = list(tensors[0].keys())
            for tensor in tensors[1:]:
                if list(tensor.keys()) != keys:
                    raise ValueError("All ArrayDicts must have same keys")
            
            new_data = {}
            for key in keys:
                values = [t._data[key] for t in tensors]
                if _is_array(values[0]):
                    new_data[key] = jnp.concatenate(values, axis=axis)
                else:
                    new_data[key] = values[0]
            
            # Get first field to compute new batch size
            batch_ndim = len(tensors[0].batch_size)
            first_array = None
            for value in new_data.values():
                if _is_array(value):
                    first_array = value
                    break
            
            if first_array is not None:
                # New batch_size is derived from concatenated shape up to batch_ndim
                new_batch = first_array.shape[:batch_ndim] if first_array.ndim >= batch_ndim else first_array.shape
            else:
                new_batch = tensors[0].batch_size
            
            return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)
        
        def repeat(self, x: ArrayDict, repeats: Union[int, Tuple[int, ...]]) -> ArrayDict:
            """Repeat batch dimensions.
            
            Delegates to ArrayDict.repeat() instance method for the implementation.
            Uses efficient jnp.repeat operation without creating intermediate copies.
            
            Args:
                x: ArrayDict to repeat
                repeats: Tuple of repeat counts for each batch dimension
            
            Returns:
                New ArrayDict with repeated batch dimensions
            """
            return x.repeat(repeats)
        
        def arange(self, start: int, stop: int) -> Any:
            """Create integer array."""
            return jnp.arange(start, stop)
        
        def einsum(self, pattern: str, *args: ArrayDict) -> ArrayDict:
            """Einsum is not supported for ArrayDict."""
            raise NotImplementedError(
                "einsum() is not supported for ArrayDict einops backend."
            )

# Auto-register the backend with einops when this module is imported
if HAS_EINOPS:
    import sys
    # Register our module in sys.modules so einops can find it
    sys.modules["arraydict_backend"] = sys.modules[__name__]
    
    # Also directly register in _loaded_backends
    from einops._backends import _loaded_backends
    if "arraydict" not in _loaded_backends:
        _loaded_backends["arraydict"] = ArrayDictBackend()
