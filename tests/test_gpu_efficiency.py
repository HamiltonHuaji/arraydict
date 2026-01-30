"""
Test that indexing operations preserve GPU placement when using jax.Array indices.
This ensures no unnecessary CPU-GPU transfers occur during advanced indexing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arraydict import ArrayDict


def test_jax_array_indices_stay_on_device():
    """Verify that jax.Array indices are not converted to numpy during indexing."""
    # Create ArrayDict with jax arrays
    source = {
        "x": jnp.ones((10, 5)),
        "y": jnp.zeros((10, 3)),
        "z": jnp.arange(10),
    }
    ad_dict = ArrayDict(source, batch_size=10)
    
    # Create jax.Array indices (these would typically be on GPU)
    indices = jnp.array([1, 3, 5, 7])
    
    # Test direct indexing
    result = ad_dict[indices]
    assert isinstance(result["x"], jnp.ndarray), "Result should remain as jax.Array"
    assert result["x"].shape == (4, 5)
    assert result.batch_size == (4,)
    
    # Test gather operation
    gathered = ad_dict.gather(indices, axis=0)
    assert isinstance(gathered["x"], jnp.ndarray), "Gathered result should be jax.Array"
    assert gathered["x"].shape == (4, 5)
    assert gathered.batch_size == (4,)


def test_mixed_jax_and_numpy_arrays():
    """Test that mixed jax and numpy arrays (object dtype) are handled correctly."""
    # Mixed data: jax arrays (numeric) and numpy arrays (non-numeric objects)
    source = {
        "numeric": jnp.ones((10, 5)),
        "objects": [f"item-{i}" for i in range(10)],  # Will become np.ndarray dtype=object
    }
    ad_dict = ArrayDict(source, batch_size=10)
    
    # Use jax.Array indices
    indices = jnp.array([0, 2, 4])
    
    result = ad_dict[indices]
    
    # Numeric data stays as jax.Array
    assert isinstance(result["numeric"], jnp.ndarray)
    assert result["numeric"].shape == (3, 5)
    
    # Object data becomes numpy array (necessary, as objects can't be on GPU)
    assert isinstance(result["objects"], np.ndarray)
    assert result["objects"].dtype == object
    assert len(result["objects"]) == 3
    assert result["objects"][0] == "item-0"


def test_advanced_indexing_preserves_device():
    """Test that complex indexing operations preserve GPU placement."""
    source = {
        "data": jnp.arange(20).reshape(10, 2),
    }
    ad_dict = ArrayDict(source, batch_size=10)
    
    # Boolean indexing with jax array
    mask = jnp.array([i % 2 == 0 for i in range(10)])
    result = ad_dict[mask]
    assert isinstance(result["data"], jnp.ndarray)
    assert result["data"].shape == (5, 2)
    
    # Integer array indexing
    indices = jnp.array([1, 1, 3, 3, 5])
    result2 = ad_dict[indices]
    assert isinstance(result2["data"], jnp.ndarray)
    assert result2["data"].shape == (5, 2)


def test_no_unnecessary_conversions_in_operations():
    """Verify that operations don't force unnecessary device transfers."""
    source = {
        "x": jnp.ones((10, 3)),
        "y": jnp.zeros((10, 2)),
    }
    ad_dict = ArrayDict(source, batch_size=10)
    
    # All these operations should preserve jax.Array type
    sliced = ad_dict[2:8]
    assert isinstance(sliced["x"], jnp.ndarray)
    
    reshaped = ad_dict.reshape((5, 2))
    assert isinstance(reshaped["x"], jnp.ndarray)
    
    parts = ad_dict.split(2, axis=0)
    assert all(isinstance(part["x"], jnp.ndarray) for part in parts)
    
    # Using jax indices for gather
    indices = jnp.array([0, 5, 9])
    gathered = ad_dict.gather(indices)
    assert isinstance(gathered["x"], jnp.ndarray)


def test_shape_extraction_no_conversion():
    """Verify that extracting shapes doesn't cause data conversions."""
    source = {"data": jnp.ones((10, 5, 3))}
    ad_dict = ArrayDict(source, batch_size=10)
    
    # batch_size should be a Python tuple, not an array
    assert isinstance(ad_dict.batch_size, tuple)
    assert ad_dict.batch_size == (10,)
    
    # Indexing should update batch_size without converting data
    indices = jnp.array([1, 2, 3])
    result = ad_dict[indices]
    assert isinstance(result.batch_size, tuple)
    assert result.batch_size == (3,)
    assert isinstance(result["data"], jnp.ndarray)
