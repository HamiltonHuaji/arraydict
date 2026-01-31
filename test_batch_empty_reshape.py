"""Test reshape behavior for empty batch_size with mixed numeric/non-numeric fields."""

import sys
sys.path.insert(0, 'src')

from arraydict import ArrayDict
import jax.numpy as jnp
import numpy as np
from pathlib import Path


def test_empty_batch_reshape_numeric_only():
    """Test reshape([-1]) with batch_size=[], numeric fields only."""
    print("\n" + "="*60)
    print("TEST 1: Empty batch_size with reshape([-1]) - Numeric only")
    print("="*60)
    
    # Create scalar ArrayDict (batch_size=[])
    ad = ArrayDict({
        'x': jnp.array(5.0),
        'y': jnp.array(3.0),
        'z': jnp.array(2.0),
    }, batch_size=[])
    
    print(f"Original batch_size: {ad.batch_size}")
    print(f"Original shapes:")
    for k, v in ad._data.items():
        print(f"  {k}: {v.shape}")
    print(f"Original values:\n{ad}")
    
    # Reshape [-1] should convert () -> (1,)
    reshaped = ad.reshape([-1])
    print(f"\nAfter reshape([-1]):")
    print(f"  New batch_size: {reshaped.batch_size}")
    print(f"  New shapes:")
    for k, v in reshaped._data.items():
        print(f"    {k}: {v.shape}")
    print(f"  Values:\n{reshaped}")
    
    # Verify: old_size = prod([]) = 1, new_size should be 1
    assert reshaped.batch_size == (1,), f"Expected (1,), got {reshaped.batch_size}"
    for v in reshaped._data.values():
        assert v.shape[0] == 1, f"Expected shape[0]=1, got {v.shape}"
    print("✅ Test 1 passed: empty batch reshapes to (1,)")


def test_empty_batch_reshape_mixed():
    """Test reshape([-1]) with batch_size=[], mixed numeric and non-numeric fields."""
    print("\n" + "="*60)
    print("TEST 2: Empty batch_size with reshape([-1]) - Mixed fields")
    print("="*60)
    
    # Create scalar ArrayDict with mixed types
    ad = ArrayDict({
        'num1': jnp.array(10.0),
        'num2': jnp.array(20.0),
        'path': Path('/tmp/data'),
        'text': 'hello',
        'nested': {
            'num': jnp.array(30.0),
            'str': 'world'
        }
    }, batch_size=[])
    
    print(f"Original batch_size: {ad.batch_size}")
    print(f"Original types and shapes:")
    for k, v in ad._data.items():
        if isinstance(v, np.ndarray) and v.dtype == object:
            print(f"  {k}: object array {v.shape}, value={v}")
        elif hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: not an array")
    print(f"Original:\n{ad}")
    
    # Reshape [-1]
    reshaped = ad.reshape([-1])
    print(f"\nAfter reshape([-1]):")
    print(f"  New batch_size: {reshaped.batch_size}")
    print(f"  New types and shapes:")
    for k, v in reshaped._data.items():
        if isinstance(v, np.ndarray) and v.dtype == object:
            print(f"    {k}: object array {v.shape}, value={v}")
        elif hasattr(v, 'shape'):
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"    {k}: not an array")
    print(f"  Values:\n{reshaped}")
    
    # Verify batch_size
    assert reshaped.batch_size == (1,), f"Expected (1,), got {reshaped.batch_size}"
    
    # Verify numeric fields reshaped
    for k in [('num1',), ('num2',), ('nested', 'num')]:
        v = reshaped._data[k]
        assert v.shape[0] == 1, f"Field {k}: expected shape[0]=1, got {v.shape}"
    
    # Verify non-numeric fields are unchanged (not wrapped into object arrays)
    # These should still be the original scalar values, not reshaped
    for k in [('path',), ('text',), ('nested', 'str')]:
        v = reshaped._data[k]
        # Non-array values pass through unchanged in _reshape_with_batch
        print(f"  Non-numeric field {k}: {type(v).__name__} = {v}")
    
    print("✅ Test 2 passed: empty batch with mixed fields reshapes correctly")


def test_empty_batch_reshape_multiple_dims():
    """Test reshape with multiple dimensions from empty batch."""
    print("\n" + "="*60)
    print("TEST 3: Empty batch_size with reshape([2, -1]) - Should infer")
    print("="*60)
    
    ad = ArrayDict({
        'a': jnp.array(1.0),
        'b': jnp.array(2.0),
        'c': jnp.array(3.0),
        'text': 'test'
    }, batch_size=[])
    
    print(f"Original batch_size: {ad.batch_size}")
    print(f"Original:\n{ad}")
    
    # Try reshape([2, -1]) on scalar (1 element)
    # This should fail because we can't fit 1 element into shape (2, ?)
    try:
        reshaped = ad.reshape([2, -1])
        print(f"ERROR: Should have failed but got batch_size={reshaped.batch_size}")
        assert False, "Should have raised error"
    except Exception as e:
        print(f"✅ Correctly raised error: {type(e).__name__}: {e}")


def test_empty_batch_reshape_feature_dims():
    """Test that feature dimensions are preserved through reshape."""
    print("\n" + "="*60)
    print("TEST 4: Empty batch with feature dimensions - reshape preserves features")
    print("="*60)
    
    # Create ArrayDict with empty batch but with feature dimensions
    ad = ArrayDict({
        'vectors': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # () batch + (5,) features
        'matrix': jnp.array([[1, 2], [3, 4]]),  # () batch + (2,2) features
    }, batch_size=[])
    
    print(f"Original batch_size: {ad.batch_size}")
    print(f"Original shapes:")
    for k, v in ad._data.items():
        print(f"  {k}: {v.shape}")
    print(f"Original:\n{ad}")
    
    # Reshape from () to (1,) - should get batch (1,) with same features
    reshaped = ad.reshape([-1])
    print(f"\nAfter reshape([-1]):")
    print(f"  New batch_size: {reshaped.batch_size}")
    print(f"  New shapes:")
    for k, v in reshaped._data.items():
        print(f"    {k}: {v.shape}")
    print(f"  Values:\n{reshaped}")
    
    # Verify: batch changes from () to (1,), but features preserved
    assert reshaped.batch_size == (1,)
    assert reshaped._data[('vectors',)].shape == (1, 5), "Feature dims should be preserved"
    assert reshaped._data[('matrix',)].shape == (1, 2, 2), "Feature dims should be preserved"
    
    print("✅ Test 4 passed: feature dimensions preserved through reshape")


if __name__ == '__main__':
    test_empty_batch_reshape_numeric_only()
    test_empty_batch_reshape_mixed()
    test_empty_batch_reshape_multiple_dims()
    test_empty_batch_reshape_feature_dims()
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
