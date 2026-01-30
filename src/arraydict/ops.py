"""Operations on ArrayDict instances."""

from typing import List, Union
import jax.numpy as jnp
from arraydict.core import ArrayDict, KeyType


def stack(
    arrays: List[ArrayDict], axis: int = 0
) -> ArrayDict:
    """
    Stack multiple ArrayDict instances along a new batch dimension.

    Args:
        arrays: List of ArrayDict instances to stack.
        axis: Axis along which to stack (default 0).

    Returns:
        Stacked ArrayDict with new batch dimension.
    """
    if not arrays:
        raise ValueError("Cannot stack empty list")

    # Verify all have same structure
    keys = set(arrays[0]._data.keys())
    for ad in arrays[1:]:
        if set(ad._data.keys()) != keys:
            raise ValueError("All ArrayDict instances must have the same keys")

    new_data = {}
    for key in keys:
        values = [ad._data[key] for ad in arrays]

        # Stack the values
        if hasattr(values[0], "shape"):
            # Array-like
            stacked = jnp.stack(values, axis=axis)
        elif isinstance(values[0], (list, tuple)):
            # For sequences, stack along axis 0 only
            if axis != 0:
                raise ValueError("Cannot stack sequences along non-zero axis")
            # Combine all lists/tuples
            combined = []
            for v in values:
                combined.extend(v if isinstance(v, (list, tuple)) else [v])
            stacked = combined if isinstance(values[0], list) else tuple(combined)
        else:
            # Other types
            stacked = values

        new_data[key] = stacked

    # Calculate new batch size
    if arrays[0]._batch_size:
        new_batch_size = (len(arrays),) + arrays[0]._batch_size
    else:
        new_batch_size = (len(arrays),)

    return ArrayDict._from_flat_data(new_data, batch_size=new_batch_size)


def concat(
    arrays: List[ArrayDict], axis: int = 0
) -> ArrayDict:
    """
    Concatenate multiple ArrayDict instances along an existing batch dimension.

    Args:
        arrays: List of ArrayDict instances to concatenate.
        axis: Axis along which to concatenate (default 0).

    Returns:
        Concatenated ArrayDict.
    """
    if not arrays:
        raise ValueError("Cannot concatenate empty list")

    # Verify all have same structure
    keys = set(arrays[0]._data.keys())
    for ad in arrays[1:]:
        if set(ad._data.keys()) != keys:
            raise ValueError("All ArrayDict instances must have the same keys")

    new_data = {}
    for key in keys:
        values = [ad._data[key] for ad in arrays]

        # Concatenate the values
        if hasattr(values[0], "shape"):
            # Array-like
            concatenated = jnp.concatenate(values, axis=axis)
        elif isinstance(values[0], (list, tuple)):
            # For sequences, concatenate along axis 0 only
            if axis != 0:
                raise ValueError("Cannot concatenate sequences along non-zero axis")
            # Combine all lists/tuples
            combined = []
            for v in values:
                combined.extend(v if isinstance(v, (list, tuple)) else [v])
            concatenated = combined if isinstance(values[0], list) else tuple(combined)
        else:
            # Other types
            concatenated = values

        new_data[key] = concatenated

    # Calculate new batch size
    if arrays[0]._batch_size:
        old_batch_size = arrays[0]._batch_size[axis]
        new_batch_dim = sum(ad._batch_size[axis] for ad in arrays)
        new_batch_size = (
            arrays[0]._batch_size[:axis]
            + (new_batch_dim,)
            + arrays[0]._batch_size[axis + 1 :]
        )
    else:
        new_batch_size = ()

    return ArrayDict._from_flat_data(new_data, batch_size=new_batch_size)
