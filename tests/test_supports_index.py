"""
Test SupportsIndex protocol handling in ArrayDict.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arraydict import ArrayDict


class CustomIndex:
    """Custom class implementing SupportsIndex protocol."""
    
    def __init__(self, value: int):
        self.value = value
    
    def __index__(self) -> int:
        return self.value


def test_supports_index_protocol():
    """Test that objects implementing SupportsIndex are treated as batch indices."""
    rng = jax.random.PRNGKey(1000)
    keys = jax.random.split(rng, 3)
    
    data = {
        "x": jax.random.normal(keys[0], (10, 5)),
        "y": jax.random.uniform(keys[1], (10, 3)),
    }
    
    ad = ArrayDict(data, batch_size=10)
    
    # Test with int (which implements SupportsIndex)
    result_int = ad[3]
    assert isinstance(result_int, ArrayDict)
    assert result_int.batch_size == ()
    assert result_int["x"].shape == (5,)
    
    # Test with custom SupportsIndex implementation
    custom_idx = CustomIndex(5)
    result_custom = ad[custom_idx]
    assert isinstance(result_custom, ArrayDict)
    assert result_custom.batch_size == ()
    assert result_custom["x"].shape == (5,)
    
    # Verify values match
    np.testing.assert_allclose(
        np.array(ad[5]["x"]),
        np.array(result_custom["x"]),
        rtol=1e-5
    )
    
    print("✓ SupportsIndex protocol handled correctly")


def test_int_not_column_key():
    """Test that int is treated as batch index, not column key."""
    data = {
        "x": jnp.ones((10, 3)),
        "key_int": jnp.zeros((10, 2)),  # Use string key instead
    }
    
    ad = ArrayDict(data, batch_size=10)
    
    # Using int as index should do batch indexing, not column selection
    result = ad[0]
    assert isinstance(result, ArrayDict)
    assert result.batch_size == ()
    
    # To access the column with string key, use that key
    col = ad["key_int"]
    assert isinstance(col, jnp.ndarray)
    assert col.shape == (10, 2)
    
    print("✓ int correctly treated as batch index, not column key")


def test_tuple_with_supports_index():
    """Test tuple containing SupportsIndex elements."""
    rng = jax.random.PRNGKey(2000)
    keys = jax.random.split(rng, 2)
    
    data = {
        "a": jax.random.normal(keys[0], (10, 8, 4)),
        "b": jax.random.uniform(keys[1], (10, 8, 3)),
    }
    
    ad = ArrayDict(data, batch_size=(10, 8))
    
    # Tuple with int (SupportsIndex) should be batch index
    result1 = ad[(2, 3)]
    assert isinstance(result1, ArrayDict)
    assert result1.batch_size == ()
    assert result1["a"].shape == (4,)
    
    # Tuple with CustomIndex
    custom_idx = CustomIndex(5)
    result2 = ad[(custom_idx, 4)]
    assert isinstance(result2, ArrayDict)
    assert result2.batch_size == ()
    
    # Mixed: slice and SupportsIndex
    result3 = ad[(slice(2, 5), custom_idx)]
    assert isinstance(result3, ArrayDict)
    assert result3.batch_size == (3,)
    assert result3["a"].shape == (3, 4)
    
    print("✓ Tuples with SupportsIndex handled correctly")


if __name__ == "__main__":
    test_supports_index_protocol()
    test_int_not_column_key()
    test_tuple_with_supports_index()
    
    print("\n" + "="*60)
    print("✓✓✓ All SupportsIndex tests passed ✓✓✓")
    print("="*60)
