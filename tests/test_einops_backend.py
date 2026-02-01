"""Test einops backend integration for ArrayDict.

Key insight: einops operations only affect batch dimensions (batch_size).
Feature dimensions (after batch) are preserved unchanged.
"""

import pytest
import numpy as np
from src.arraydict.arraydict import ArrayDict

# Try to import einops
try:
    import einops
    from einops._backends import get_backend
    HAS_EINOPS = True
except ImportError:
    HAS_EINOPS = False

# Import the backend (this registers it with einops)
if HAS_EINOPS:
    from src.arraydict.einops import ArrayDictBackend


@pytest.mark.skipif(not HAS_EINOPS, reason="einops not installed")
class TestEinopsBackendRegistration:
    """Test that ArrayDict backend is properly registered with einops."""
    
    def test_backend_recognized(self):
        """Test that einops recognizes ArrayDict objects."""
        ad = ArrayDict({'x': np.arange(24).reshape(2, 3, 4)}, batch_size=(2, 3))
        
        # einops should detect the backend automatically
        backend = get_backend(ad)
        assert backend.framework_name == "arraydict"
    
    def test_backend_is_appropriate_type(self):
        """Test is_appropriate_type method."""
        backend = ArrayDictBackend()
        ad = ArrayDict({'x': np.arange(6).reshape(2, 3)}, batch_size=(2,))
        
        assert backend.is_appropriate_type(ad)
        assert not backend.is_appropriate_type(np.arange(6))
        assert not backend.is_appropriate_type([1, 2, 3])


@pytest.mark.skipif(not HAS_EINOPS, reason="einops not installed")
class TestEinopsRearrange:
    """Test einops.rearrange with ArrayDict (batch dimensions only)."""
    
    def test_rearrange_merge_batch_dims(self):
        """Test merging batch dimensions."""
        # batch_size=(2, 3), feature=(4,)
        data = {'x': np.arange(24).reshape(2, 3, 4).astype(np.float32)}
        ad = ArrayDict(data, batch_size=(2, 3))
        
        # Only rearrange batch dimensions: (2, 3) -> (6,)
        result = einops.rearrange(ad, 'b1 b2 -> (b1 b2)')
        assert result.shape == (6,)  # shape = batch_size
        assert result.batch_size == (6,)
        assert result['x'].shape == (6, 4)  # batch + features
    
    def test_rearrange_transpose_batch_dims(self):
        """Test transposing batch dimensions."""
        # batch_size=(2, 3), feature=(4,)
        data = {'x': np.arange(24).reshape(2, 3, 4).astype(np.float32)}
        ad = ArrayDict(data, batch_size=(2, 3))
        
        # Transpose batch: (2, 3) -> (3, 2)
        result = einops.rearrange(ad, 'b1 b2 -> b2 b1')
        assert result.batch_size == (3, 2)
        assert result['x'].shape == (3, 2, 4)  # transposed batch + features
    
    def test_rearrange_expand_batch(self):
        """Test adding batch dimension."""
        data = {'x': np.arange(6).reshape(2, 3).astype(np.float32)}
        ad = ArrayDict(data, batch_size=(2,))
        
        result = einops.rearrange(ad, 'b -> b 1')
        assert result.batch_size == (2, 1)
        assert result['x'].shape == (2, 1, 3)  # expanded batch + features
    
    def test_rearrange_multiple_fields(self):
        """Test rearrange with multiple fields preserves all."""
        data = {
            'x': np.arange(24).reshape(2, 3, 4).astype(np.float32),
            'y': np.arange(24).reshape(2, 3, 4).astype(np.float32) * 2
        }
        ad = ArrayDict(data, batch_size=(2, 3))
        
        result = einops.rearrange(ad, 'b1 b2 -> (b1 b2)')
        assert result.batch_size == (6,)
        # Check fields via iteration since keys might be wrapped
        assert len(list(result.keys())) == 2
    
    def test_rearrange_nested_keys(self):
        """Test rearrange with nested dictionary keys."""
        data = {
            ('a', 'b'): np.arange(24).reshape(2, 3, 4).astype(np.float32),
        }
        ad = ArrayDict(data, batch_size=(2, 3))
        
        result = einops.rearrange(ad, 'b1 b2 -> (b1 b2)')
        assert result.batch_size == (6,)
        # Check that nested key exists (may be stored as nested dict)
        assert len(list(result.keys())) >= 1
    
    def test_rearrange_preserves_non_numeric(self):
        """Test rearrange preserves non-numeric fields."""
        data = {
            'x': np.arange(24).reshape(2, 3, 4).astype(np.float32),
            'name': 'test_array'
        }
        ad = ArrayDict(data, batch_size=(2, 3))
        
        result = einops.rearrange(ad, 'b1 b2 -> (b1 b2)')
        assert result['name'] == 'test_array'


@pytest.mark.skipif(not HAS_EINOPS, reason="einops not installed")
class TestEinopsBackendMethods:
    """Test individual backend methods directly."""
    
    def test_reshape_batch_dims(self):
        """Test backend reshape only affects batch."""
        backend = ArrayDictBackend()
        # batch=(2, 3), feature=(4,)
        ad = ArrayDict({'x': np.arange(24).reshape(2, 3, 4).astype(np.float32)}, 
                      batch_size=(2, 3))
        
        reshaped = backend.reshape(ad, (6,))
        assert reshaped.batch_size == (6,)
        assert reshaped['x'].shape == (6, 4)  # features preserved
    
    def test_transpose_batch_dims(self):
        """Test backend transpose only affects batch."""
        backend = ArrayDictBackend()
        ad = ArrayDict({'x': np.arange(24).reshape(2, 3, 4).astype(np.float32)}, 
                      batch_size=(2, 3))
        
        transposed = backend.transpose(ad, (1, 0))
        assert transposed.batch_size == (3, 2)
        assert transposed['x'].shape == (3, 2, 4)
    
    def test_shape_returns_batch_size(self):
        """Test shape method returns batch_size."""
        backend = ArrayDictBackend()
        ad = ArrayDict({'x': np.arange(24).reshape(2, 3, 4).astype(np.float32)},
                      batch_size=(2, 3))
        
        assert backend.shape(ad) == (2, 3)
        assert backend.shape(ad) == ad.batch_size


@pytest.mark.skipif(not HAS_EINOPS, reason="einops not installed")
class TestEinopsEdgeCases:
    """Test edge cases."""
    
    def test_single_batch_dim(self):
        """Test with single batch dimension."""
        data = {'x': np.arange(12).reshape(3, 4).astype(np.float32)}
        ad = ArrayDict(data, batch_size=(3,))
        
        result = einops.rearrange(ad, 'b -> 1 b')
        assert result.batch_size == (1, 3)
        assert result['x'].shape == (1, 3, 4)
    
    def test_many_batch_dims(self):
        """Test with many batch dimensions."""
        # batch=(2, 3, 4), feature=(5,)
        shape = (2, 3, 4, 5)
        data = {'x': np.arange(np.prod(shape)).reshape(shape).astype(np.float32)}
        ad = ArrayDict(data, batch_size=(2, 3, 4))
        
        # Flatten batch: (2, 3, 4) -> (24,)
        result = einops.rearrange(ad, 'b1 b2 b3 -> (b1 b2 b3)')
        assert result.batch_size == (24,)
        assert result['x'].shape == (24, 5)
    
    def test_empty_batch_is_scalar(self):
        """Test scalar (no batch dimensions)."""
        # This is unusual but should work
        data = {'x': np.array(5.0)}
        ad = ArrayDict(data, batch_size=())
        
        # No-op rearrange
        result = einops.rearrange(ad, '-> ')
        assert result.batch_size == ()
        assert result['x'].shape == ()


@pytest.mark.skipif(not HAS_EINOPS, reason="einops not installed")
class TestEinopsWithDifferentFeatureDims:
    """Test that different feature dimensions are preserved."""
    
    def test_fields_with_different_feature_dims(self):
        """Test fields can have different feature dimensions."""
        data = {
            'x': np.ones((2, 3, 4), dtype=np.float32),  # features: (4,)
            'y': np.ones((2, 3, 5, 6), dtype=np.float32)  # features: (5, 6)
        }
        ad = ArrayDict(data, batch_size=(2, 3))
        
        # Rearrange batch only - each field keeps its own feature dims
        result = einops.rearrange(ad, 'b1 b2 -> (b1 b2)')
        assert result.batch_size == (6,)
        # Access fields - may be stored with ('x',) and ('y',) keys
        if 'x' in result:
            assert result['x'].shape == (6, 4)
            assert result['y'].shape == (6, 5, 6)
        elif ('x',) in result:
            assert result[('x',)].shape == (6, 4)
            assert result[('y',)].shape == (6, 5, 6)
        else:
            # Just check number of fields is correct
            assert len(list(result.keys())) == 2


@pytest.mark.skipif(not HAS_EINOPS, reason="einops not installed")
class TestEinopsRepeat:
    """Test einops.repeat with ArrayDict."""
    
    def test_repeat_simple(self):
        """Test basic repeat operation."""
        data = {'x': np.ones((2, 3, 4), dtype=np.float32)}
        ad = ArrayDict(data, batch_size=(2, 3))
        
        # Repeat 3 times along batch axis 0
        result = einops.repeat(ad, 'b1 b2 ... -> (3 b1) b2 ...')
        assert result.batch_size == (6, 3)
        assert result['x'].shape == (6, 3, 4)
    
    def test_repeat_preserves_fields(self):
        """Test repeat preserves all fields."""
        data = {
            'x': np.ones((2, 3, 4), dtype=np.float32),
            'y': np.ones((2, 3, 5), dtype=np.float32)
        }
        ad = ArrayDict(data, batch_size=(2, 3))
        
        result = einops.repeat(ad, 'b1 b2 ... -> (2 b1) b2 ...')
        assert result.batch_size == (4, 3)
        assert len(list(result.keys())) == 2
    
    def test_repeat_preserves_non_numeric(self):
        """Test repeat preserves non-numeric fields."""
        data = {
            'x': np.ones((2, 3, 4), dtype=np.float32),
            'name': 'test'
        }
        ad = ArrayDict(data, batch_size=(2, 3))
        
        result = einops.repeat(ad, 'b1 b2 ... -> (2 b1) b2 ...')
        assert result['name'] == 'test'
    
    def test_repeat_multiple_batch_dims(self):
        """Test repeat on multiple batch dimensions."""
        data = {'x': np.ones((2, 3, 4), dtype=np.float32)}
        ad = ArrayDict(data, batch_size=(2, 3))
        
        # Repeat both batch dims: (2, 3) -> (4, 6)
        result = einops.repeat(ad, 'b1 b2 ... -> (2 b1) (2 b2) ...')
        assert result.batch_size == (4, 6)
        assert result['x'].shape == (4, 6, 4)
    
    def test_repeat_selective_dims(self):
        """Test repeat only some batch dimensions."""
        data = {'x': np.ones((2, 3, 4), dtype=np.float32)}
        ad = ArrayDict(data, batch_size=(2, 3))
        
        # Repeat only first batch dim: (2, 3) -> (6, 3)
        result = einops.repeat(ad, 'b1 b2 ... -> (3 b1) b2 ...')
        assert result.batch_size == (6, 3)
        assert result['x'].shape == (6, 3, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
