"""Tests for ArrayDict."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from arraydict import ArrayDict, stack, concat


def _has_torch() -> bool:
    """Check if torch is available."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class TestArrayDictBasics:
    """Test basic ArrayDict functionality."""

    def test_simple_initialization(self):
        """Test basic initialization with arrays."""
        ad = ArrayDict({
            'foo': jnp.zeros((10, 4, 3)),
            'bar': jnp.ones((10, 2)),
        }, batch_size=[10])
        
        assert ad.batch_size == (10,)
        assert ad['foo'].shape == (10, 4, 3)
        assert ad['bar'].shape == (10, 2)

    def test_nested_dict_initialization(self):
        """Test initialization with nested dictionaries."""
        ad = ArrayDict({
            'foo': jnp.zeros((10, 4, 3)),
            'bar': {
                'baz': jnp.ones((10, 2, 5)),
                'qux': jnp.full((10, 6), 7),
            },
        }, batch_size=[10])
        
        assert ad['bar']['baz'].shape == (10, 2, 5)
        assert ad['bar']['qux'].shape == (10, 6)

    def test_nested_arraydict_initialization(self):
        """Test initialization with nested ArrayDict."""
        inner = ArrayDict({
            'baz': jnp.ones((10, 2, 5)),
            'qux': jnp.full((10, 6), 7),
        }, batch_size=[10])
        
        ad = ArrayDict({
            'foo': jnp.zeros((10, 4, 3)),
            'bar': inner,
        }, batch_size=[10])
        
        assert ad['bar']['baz'].shape == (10, 2, 5)

    def test_tuple_keys(self):
        """Test tuple keys in ArrayDict."""
        ad = ArrayDict({
            ('tuple', 'key'): jax.random.normal(jax.random.PRNGKey(0), (10, 8)),
        }, batch_size=[10])
        
        assert ad[('tuple', 'key')].shape == (10, 8)

    def test_lists_and_sequences(self):
        """Test non-numeric lists."""
        ad = ArrayDict({
            'foo': jnp.zeros((10, 4)),
            'non-numeric': ['hello' for _ in range(10)],
        }, batch_size=[10])
        
        assert len(ad['non-numeric']) == 10
        assert ad['non-numeric'][0] == 'hello'

    def test_infer_batch_size(self):
        """Test batch size inference."""
        ad = ArrayDict({
            'foo': jnp.zeros((10, 4, 3)),
            'bar': jnp.ones((10, 2)),
        })
        
        assert ad.batch_size == (10,)

    def test_infer_batch_size_from_sequence(self):
        """Test batch size inference from sequences."""
        ad = ArrayDict({
            'data': jnp.zeros((10, 4)),
            'labels': ['a'] * 10,
        })
        
        assert ad.batch_size == (10,)

    def test_batch_size_mismatch_error(self):
        """Test error on batch size mismatch."""
        with pytest.raises(ValueError):
            ArrayDict({
                'foo': jnp.zeros((10, 4)),
                'bar': jnp.ones((5, 2)),
            })

    def test_complex_nested_structure(self):
        """Test complex nested structure."""
        ad = ArrayDict({
            'foo': jnp.zeros((10, 4, 3)),
            'bar': {
                'baz': jnp.ones((10, 2, 5)),
                'qux': {
                    'nested': jnp.full((10, 3), 7),
                }
            },
            'baz': {
                'a': jnp.arange(10),
                'b': jnp.linspace(0, 1, 10),
            },
            'non-numeric': ['hello' for _ in range(10)],
            ('tuple', 'key'): jax.random.normal(jax.random.PRNGKey(0), (10, 8)),
        }, batch_size=[10])
        
        assert ad['foo'].shape == (10, 4, 3)
        assert ad['bar']['baz'].shape == (10, 2, 5)
        assert ad['bar']['qux']['nested'].shape == (10, 3)
        assert ad['baz']['a'].shape == (10,)


class TestIndexing:
    """Test indexing and slicing."""

    def test_single_index(self):
        """Test getting a single element."""
        ad = ArrayDict({
            'foo': jnp.zeros((10, 4, 3)),
            'bar': jnp.ones((10, 2)),
        }, batch_size=[10])
        
        elem = ad[0]
        assert elem.batch_size == ()
        assert elem['foo'].shape == (4, 3)
        assert elem['bar'].shape == (2,)

    def test_slice_indexing(self):
        """Test slice indexing."""
        ad = ArrayDict({
            'foo': jnp.arange(100).reshape(10, 10),
            'bar': jnp.ones((10, 2)),
        }, batch_size=[10])
        
        sliced = ad[2:5]
        assert sliced.batch_size == (3,)
        assert sliced['foo'].shape == (3, 10)
        assert jnp.array_equal(sliced['foo'], ad['foo'][2:5])

    def test_gather_with_list(self):
        """Test gather with list of indices."""
        ad = ArrayDict({
            'foo': jnp.arange(10).reshape(10, 1),
            'bar': jnp.ones((10, 2)),
        }, batch_size=[10])
        
        gathered = ad.gather([0, 2, 4])
        assert gathered.batch_size == (3,)
        assert gathered['foo'].shape == (3, 1)
        assert jnp.array_equal(gathered['foo'], jnp.array([[0], [2], [4]]))

    def test_gather_with_array(self):
        """Test gather with numpy/jax array."""
        ad = ArrayDict({
            'foo': jnp.arange(10).reshape(10, 1),
        }, batch_size=[10])
        
        gathered = ad.gather(jnp.array([0, 2, 4]))
        assert gathered.batch_size == (3,)
        assert jnp.array_equal(gathered['foo'], jnp.array([[0], [2], [4]]))

    def test_boolean_mask(self):
        """Test boolean masking."""
        ad = ArrayDict({
            'foo': jnp.arange(10),
            'bar': jnp.ones((10, 2)),
        }, batch_size=[10])
        
        mask = jnp.array([i % 2 == 0 for i in range(10)])
        masked = ad[mask]
        assert masked.batch_size == (5,)
        assert jnp.array_equal(masked['foo'], jnp.array([0, 2, 4, 6, 8]))

    def test_nested_indexing(self):
        """Test indexing on nested ArrayDict."""
        ad = ArrayDict({
            'foo': jnp.arange(10),
            'bar': {
                'baz': jnp.ones((10, 5)),
            }
        }, batch_size=[10])
        
        elem = ad[3]
        assert elem['foo'].shape == ()
        assert elem['bar']['baz'].shape == (5,)

    def test_bracket_syntax_for_gather(self):
        """Test using bracket syntax for gather."""
        ad = ArrayDict({
            'foo': jnp.arange(10),
        }, batch_size=[10])
        
        gathered = ad[[0, 2, 4]]
        assert gathered.batch_size == (3,)


class TestReshape:
    """Test reshape operations."""

    def test_reshape(self):
        """Test reshaping batch dimensions."""
        ad = ArrayDict({
            'foo': jnp.arange(20).reshape(10, 2),
            'bar': jnp.ones((10, 3, 4)),
        }, batch_size=[10])
        
        reshaped = ad.reshape((5, 2))
        assert reshaped.batch_size == (5, 2)
        assert reshaped['foo'].shape == (5, 2, 2)
        assert reshaped['bar'].shape == (5, 2, 3, 4)

    def test_reshape_preserves_data(self):
        """Test that reshape preserves data."""
        ad = ArrayDict({
            'foo': jnp.arange(10),
        }, batch_size=[10])
        
        reshaped = ad.reshape((5, 2))
        original_flat = ad['foo'].flatten()
        reshaped_flat = reshaped['foo'].reshape(-1)
        assert jnp.array_equal(original_flat, reshaped_flat)


class TestSplit:
    """Test split operations."""

    def test_split(self):
        """Test splitting into multiple ArrayDicts."""
        ad = ArrayDict({
            'foo': jnp.arange(10),
            'bar': jnp.ones((10, 2)),
        }, batch_size=[10])
        
        splits = ad.split(5)
        assert len(splits) == 5
        assert all(s.batch_size == (2,) for s in splits)
        
        # Verify data integrity
        for i, split in enumerate(splits):
            expected_start = i * 2
            expected_end = (i + 1) * 2
            assert jnp.array_equal(split['foo'], jnp.arange(expected_start, expected_end))

    def test_split_uneven_error(self):
        """Test error on uneven split."""
        ad = ArrayDict({
            'foo': jnp.arange(10),
        }, batch_size=[10])
        
        with pytest.raises(ValueError):
            ad.split(3)


class TestStackAndConcat:
    """Test stack and concat operations."""

    def test_stack_arrays(self):
        """Test stacking ArrayDicts."""
        ad1 = ArrayDict({'foo': jnp.zeros((5, 3))}, batch_size=[5])
        ad2 = ArrayDict({'foo': jnp.ones((5, 3))}, batch_size=[5])
        ad3 = ArrayDict({'foo': jnp.full((5, 3), 2)}, batch_size=[5])
        
        stacked = stack([ad1, ad2, ad3], axis=0)
        assert stacked.batch_size == (3, 5)
        assert stacked['foo'].shape == (3, 5, 3)
        assert jnp.allclose(stacked['foo'][0], 0)
        assert jnp.allclose(stacked['foo'][1], 1)
        assert jnp.allclose(stacked['foo'][2], 2)

    def test_concat_arrays(self):
        """Test concatenating ArrayDicts."""
        ad1 = ArrayDict({'foo': jnp.zeros((5, 3))}, batch_size=[5])
        ad2 = ArrayDict({'foo': jnp.ones((5, 3))}, batch_size=[5])
        ad3 = ArrayDict({'foo': jnp.full((5, 3), 2)}, batch_size=[5])
        
        concatenated = concat([ad1, ad2, ad3], axis=0)
        assert concatenated.batch_size == (15,)
        assert concatenated['foo'].shape == (15, 3)
        
        # Check values
        assert jnp.allclose(concatenated['foo'][:5], 0)
        assert jnp.allclose(concatenated['foo'][5:10], 1)
        assert jnp.allclose(concatenated['foo'][10:15], 2)

    def test_stack_multiple_keys(self):
        """Test stacking with multiple keys."""
        ad1 = ArrayDict({
            'foo': jnp.zeros((3, 2)),
            'bar': jnp.ones((3, 4)),
        }, batch_size=[3])
        ad2 = ArrayDict({
            'foo': jnp.full((3, 2), 5),
            'bar': jnp.full((3, 4), 10),
        }, batch_size=[3])
        
        stacked = stack([ad1, ad2])
        assert stacked.batch_size == (2, 3)
        assert stacked['foo'].shape == (2, 3, 2)
        assert stacked['bar'].shape == (2, 3, 4)

    def test_stack_structure_mismatch_error(self):
        """Test error on structure mismatch."""
        ad1 = ArrayDict({'foo': jnp.zeros((5, 3))}, batch_size=[5])
        ad2 = ArrayDict({'bar': jnp.ones((5, 3))}, batch_size=[5])
        
        with pytest.raises(ValueError):
            stack([ad1, ad2])


class TestKeys:
    """Test key operations."""

    def test_keys_iterator(self):
        """Test keys() method."""
        ad = ArrayDict({
            'foo': jnp.zeros((10, 4)),
            'bar': {
                'baz': jnp.ones((10, 2)),
                'qux': jnp.full((10, 6), 7),
            },
            'non-numeric': ['hello' for _ in range(10)],
            ('tuple', 'key'): jax.random.normal(jax.random.PRNGKey(0), (10, 8)),
        }, batch_size=[10])
        
        keys_list = list(ad.keys())
        assert 'foo' in keys_list
        assert ('bar', 'baz') in keys_list
        assert ('bar', 'qux') in keys_list
        assert 'non-numeric' in keys_list
        assert ('tuple', 'key') in keys_list

    def test_values_iterator(self):
        """Test values() method."""
        ad = ArrayDict({
            'foo': jnp.zeros((10, 4)),
            'bar': jnp.ones((10, 2)),
        }, batch_size=[10])
        
        values_list = list(ad.values())
        assert len(values_list) == 2

    def test_items_iterator(self):
        """Test items() method."""
        ad = ArrayDict({
            'foo': jnp.zeros((10, 4)),
            'bar': jnp.ones((10, 2)),
        }, batch_size=[10])
        
        items_list = list(ad.items())
        assert len(items_list) == 2
        assert items_list[0][0] == 'foo'
        assert items_list[1][0] == 'bar'


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_batch(self):
        """Test with empty batch dimension."""
        ad = ArrayDict({
            'foo': jnp.zeros((0, 4)),
            'bar': jnp.ones((0, 2)),
        })
        
        assert ad.batch_size == (0,)

    def test_scalar_batch(self):
        """Test with scalar batch size (no batch dimension)."""
        ad = ArrayDict.__new__(ArrayDict)
        ad._data = {'foo': 5, 'bar': 'hello'}
        ad._batch_size = ()
        
        assert ad.batch_size == ()

    def test_deep_nesting(self):
        """Test deeply nested structures."""
        ad = ArrayDict({
            'level1': {
                'level2': {
                    'level3': {
                        'data': jnp.arange(10),
                    }
                }
            }
        }, batch_size=[10])
        
        assert ad['level1']['level2']['level3']['data'].shape == (10,)

    def test_list_of_lists(self):
        """Test nested list handling."""
        ad = ArrayDict({
            'data': jnp.arange(10),
            'nested_lists': [['a', 'b'] for _ in range(10)],
        }, batch_size=[10])
        
        elem = ad[0]
        # Should preserve nested structure
        assert isinstance(elem['nested_lists'], list)

    def test_randomized_data(self):
        """Test with randomized data."""
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        
        ad = ArrayDict({
            'random1': jax.random.normal(subkey, (100, 10)),
            'random2': {
                'nested': jax.random.uniform(key, (100, 5, 3)),
            }
        }, batch_size=[100])
        
        assert ad['random1'].shape == (100, 10)
        assert ad['random2']['nested'].shape == (100, 5, 3)
        
        # Test indexing preserves randomness
        elem = ad[50]
        assert elem['random1'].shape == (10,)


class TestNumpyTorchCompatibility:
    """Test compatibility with numpy and torch arrays."""

    def test_numpy_arrays(self):
        """Test with numpy arrays."""
        ad = ArrayDict({
            'numpy_data': np.arange(20).reshape(10, 2),
            'jax_data': jnp.ones((10, 3)),
        }, batch_size=[10])
        
        assert isinstance(ad['numpy_data'], np.ndarray)
        assert ad['numpy_data'].shape == (10, 2)

    @pytest.mark.skipif(
        not _has_torch(), reason="torch not installed"
    )
    def test_torch_tensors(self):
        """Test with torch tensors."""
        import torch
        
        ad = ArrayDict({
            'torch_data': torch.ones(10, 3),
            'jax_data': jnp.ones((10, 3)),
        }, batch_size=[10])
        
        assert isinstance(ad['torch_data'], torch.Tensor)
        assert ad['torch_data'].shape == (10, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
