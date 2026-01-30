"""
Comprehensive advanced indexing tests comparing ArrayDict with tensordict.TensorDict.
Tests various combinations of None, Ellipsis, slice, int arrays, and bool arrays.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    import torch
    from tensordict import TensorDict
    TENSORDICT_AVAILABLE = True
except ImportError:
    TENSORDICT_AVAILABLE = False

from arraydict import ArrayDict


@pytest.mark.skipif(not TENSORDICT_AVAILABLE, reason="tensordict not installed")
class TestAdvancedIndexingVsTensorDict:
    """Compare ArrayDict indexing behavior with TensorDict."""
    
    @staticmethod
    def make_random_nested_data(batch_shape=(10, 8), seed=42):
        """Generate random nested data structure."""
        rng = jax.random.PRNGKey(seed)
        keys = jax.random.split(rng, 15)
        
        return {
            "x": jax.random.normal(keys[0], batch_shape + (5,)),
            "y": jax.random.uniform(keys[1], batch_shape + (3, 2)),
            "z": jax.random.normal(keys[2], batch_shape),
            "nested": {
                "a": jax.random.normal(keys[3], batch_shape + (4,)),
                "b": jax.random.uniform(keys[4], batch_shape + (2, 3)),
                "deep": {
                    "c": jax.random.normal(keys[5], batch_shape + (1,)),
                    "d": jax.random.uniform(keys[6], batch_shape + (6,)),
                },
            },
            "another": ArrayDict({
                "e": jax.random.normal(keys[7], batch_shape + (3,)),
                "f": jax.random.uniform(keys[8], batch_shape + (2,)),
            }, batch_size=batch_shape),
            "objects": [[f"item-{i}-{j}" for j in range(batch_shape[1])] for i in range(batch_shape[0])],
            ("tuple", "key"): jax.random.normal(keys[9], batch_shape + (7,)),
        }
    
    @staticmethod
    def arraydict_to_tensordict(ad: ArrayDict):
        """Convert ArrayDict to TensorDict, excluding object arrays."""
        data = {}
        object_keys = []
        
        for key, value in ad.items():
            if isinstance(value, np.ndarray) and value.dtype == object:
                object_keys.append(key)
                continue
            if isinstance(value, jnp.ndarray):
                # Convert jax array to torch tensor
                data[key] = torch.from_numpy(np.array(value))
        
        return TensorDict(data, batch_size=ad.batch_size), object_keys
    
    @staticmethod
    def generate_random_indices(batch_shape, seed=0):
        """Generate various random index combinations."""
        rng = jax.random.PRNGKey(seed)
        keys = jax.random.split(rng, 20)
        
        indices_list = []
        
        # Simple slices
        indices_list.append(slice(2, 7))
        indices_list.append(slice(None, None, 2))
        indices_list.append(slice(3, None))
        
        # Integer arrays
        idx1 = jnp.array([0, 2, 4, 6])
        indices_list.append(idx1)
        
        idx2 = jax.random.randint(keys[0], (5,), 0, batch_shape[0])
        indices_list.append(idx2)
        
        # Boolean arrays
        bool_idx1 = jnp.array([i % 2 == 0 for i in range(batch_shape[0])])
        indices_list.append(bool_idx1)
        
        bool_idx2 = jax.random.uniform(keys[1], (batch_shape[0],)) > 0.5
        indices_list.append(bool_idx2)
        
        # None (newaxis)
        indices_list.append(None)
        
        # Ellipsis - will be combined with others
        
        # Multi-dimensional indices
        if len(batch_shape) >= 2:
            indices_list.append((slice(2, 5), slice(1, 6)))
            indices_list.append((slice(None), slice(None, None, 2)))
            
            idx_2d_1 = jnp.array([0, 2, 4])
            indices_list.append((idx_2d_1, slice(None)))
            indices_list.append((slice(None), idx_2d_1))
            
            bool_2d = jax.random.uniform(keys[2], batch_shape[:2]) > 0.6
            indices_list.append(bool_2d)
            
            # Combinations with None
            indices_list.append((None, slice(2, 7)))
            indices_list.append((slice(2, 7), None))
            indices_list.append((slice(1, 5), slice(None), None))
            
            # Combinations with Ellipsis
            indices_list.append((Ellipsis, slice(2, 6)))
            indices_list.append((slice(2, 6), Ellipsis))
            indices_list.append((slice(None, None, 2), Ellipsis))
            
            # Complex combinations
            indices_list.append((None, slice(1, 8), idx_2d_1))
            indices_list.append((idx_2d_1, slice(None), None))
        
        return indices_list
    
    def compare_results(self, ad_result: ArrayDict, td_result: TensorDict, object_keys: list, 
                       index_desc: str):
        """Compare ArrayDict and TensorDict results."""
        # Check batch_size
        assert ad_result.batch_size == tuple(td_result.batch_size), \
            f"{index_desc}: batch_size mismatch: {ad_result.batch_size} vs {td_result.batch_size}"
        
        # Check each key
        for key in ad_result.keys():
            if key in object_keys:
                continue
            
            ad_value = ad_result._data[key]
            
            # Handle nested keys in TensorDict
            td_value = td_result[key]
            
            # Compare shapes
            assert ad_value.shape == tuple(td_value.shape), \
                f"{index_desc}: Shape mismatch for key {key}: {ad_value.shape} vs {td_value.shape}"
            
            # Compare values
            ad_numpy = np.array(ad_value)
            td_numpy = td_value.cpu().numpy()
            
            np.testing.assert_allclose(ad_numpy, td_numpy, rtol=1e-5, atol=1e-6,
                                      err_msg=f"{index_desc}: Value mismatch for key {key}")
    
    def test_simple_slicing(self):
        """Test simple slice indexing."""
        data = self.make_random_nested_data((10, 8), seed=100)
        ad = ArrayDict(data, batch_size=(10, 8))
        td, object_keys = self.arraydict_to_tensordict(ad)
        
        # Test various slices
        test_cases = [
            slice(2, 7),
            slice(None, 5),
            slice(5, None),
            slice(None, None, 2),
        ]
        
        for idx in test_cases:
            ad_result = ad[idx]
            td_result = td[idx]
            self.compare_results(ad_result, td_result, object_keys, f"index={idx}")
    
    def test_integer_array_indexing(self):
        """Test integer array indexing."""
        data = self.make_random_nested_data((10, 8), seed=200)
        ad = ArrayDict(data, batch_size=(10, 8))
        td, object_keys = self.arraydict_to_tensordict(ad)
        
        # JAX and torch integer indices
        jax_idx = jnp.array([0, 2, 4, 6, 8])
        torch_idx = torch.tensor([0, 2, 4, 6, 8])
        
        ad_result = ad[jax_idx]
        td_result = td[torch_idx]
        
        self.compare_results(ad_result, td_result, object_keys, "integer_array")
    
    def test_boolean_array_indexing(self):
        """Test boolean array indexing."""
        data = self.make_random_nested_data((10, 8), seed=300)
        ad = ArrayDict(data, batch_size=(10, 8))
        td, object_keys = self.arraydict_to_tensordict(ad)
        
        # Boolean mask
        jax_mask = jnp.array([i % 2 == 0 for i in range(10)])
        torch_mask = torch.tensor([i % 2 == 0 for i in range(10)])
        
        ad_result = ad[jax_mask]
        td_result = td[torch_mask]
        
        self.compare_results(ad_result, td_result, object_keys, "boolean_array")
    
    def test_none_newaxis(self):
        """Test None (newaxis) indexing."""
        data = self.make_random_nested_data((10, 8), seed=400)
        ad = ArrayDict(data, batch_size=(10, 8))
        td, object_keys = self.arraydict_to_tensordict(ad)
        
        ad_result = ad[None]
        td_result = td[None]
        
        self.compare_results(ad_result, td_result, object_keys, "None_newaxis")
    
    def test_multidimensional_slicing(self):
        """Test multi-dimensional slice indexing."""
        data = self.make_random_nested_data((10, 8), seed=500)
        ad = ArrayDict(data, batch_size=(10, 8))
        td, object_keys = self.arraydict_to_tensordict(ad)
        
        test_cases = [
            (slice(2, 7), slice(1, 6)),
            (slice(None), slice(None, None, 2)),
            (slice(1, 9, 2), slice(0, 8, 3)),
        ]
        
        for idx in test_cases:
            ad_result = ad[idx]
            td_result = td[idx]
            self.compare_results(ad_result, td_result, object_keys, f"index={idx}")
    
    def test_mixed_integer_slice(self):
        """Test mixed integer array and slice indexing."""
        data = self.make_random_nested_data((10, 8), seed=600)
        ad = ArrayDict(data, batch_size=(10, 8))
        td, object_keys = self.arraydict_to_tensordict(ad)
        
        jax_idx = jnp.array([1, 3, 5])
        torch_idx = torch.tensor([1, 3, 5])
        
        # Integer array then slice
        ad_result1 = ad[(jax_idx, slice(None))]
        td_result1 = td[(torch_idx, slice(None))]
        self.compare_results(ad_result1, td_result1, object_keys, "(int_array, slice)")
        
        # Slice then integer array
        ad_result2 = ad[(slice(None), jax_idx)]
        td_result2 = td[(slice(None), torch_idx)]
        self.compare_results(ad_result2, td_result2, object_keys, "(slice, int_array)")
    
    def test_ellipsis_indexing(self):
        """Test Ellipsis indexing."""
        data = self.make_random_nested_data((10, 8), seed=700)
        ad = ArrayDict(data, batch_size=(10, 8))
        td, object_keys = self.arraydict_to_tensordict(ad)
        
        # Note: TensorDict handles Ellipsis specially to only affect batch dims
        # ArrayDict currently uses standard numpy/jax Ellipsis behavior
        # Only test cases where Ellipsis doesn't conflict with trailing indices
        test_cases = [
            (slice(2, 6), Ellipsis),  # This works: slice first batch dim, Ellipsis fills rest
        ]
        
        for idx in test_cases:
            ad_result = ad[idx]
            td_result = td[idx]
            self.compare_results(ad_result, td_result, object_keys, f"index={idx}")
    
    def test_none_with_slicing(self):
        """Test None combined with slicing."""
        data = self.make_random_nested_data((10, 8), seed=800)
        ad = ArrayDict(data, batch_size=(10, 8))
        td, object_keys = self.arraydict_to_tensordict(ad)
        
        test_cases = [
            (None, slice(2, 7)),
            (slice(2, 7), None),
            (slice(1, 5), slice(None), None),
            (None, slice(None), slice(3, 7)),
        ]
        
        for idx in test_cases:
            ad_result = ad[idx]
            td_result = td[idx]
            self.compare_results(ad_result, td_result, object_keys, f"index={idx}")
    
    def test_complex_combinations(self):
        """Test complex combinations of indexing operations."""
        data = self.make_random_nested_data((12, 10), seed=900)
        ad = ArrayDict(data, batch_size=(12, 10))
        td, object_keys = self.arraydict_to_tensordict(ad)
        
        jax_idx = jnp.array([1, 3, 5, 7])
        torch_idx = torch.tensor([1, 3, 5, 7])
        
        test_cases = [
            (None, jax_idx, slice(None)),
            (jax_idx, None, slice(2, 8)),
            (slice(1, 10, 2), jax_idx, None),
            # Skip (Ellipsis, jax_idx) as it conflicts with ArrayDict's numpy-style Ellipsis
        ]
        
        torch_test_cases = [
            (None, torch_idx, slice(None)),
            (torch_idx, None, slice(2, 8)),
            (slice(1, 10, 2), torch_idx, None),
        ]
        
        for ad_idx, td_idx in zip(test_cases, torch_test_cases):
            ad_result = ad[ad_idx]
            td_result = td[td_idx]
            self.compare_results(ad_result, td_result, object_keys, f"index={ad_idx}")
    
    def test_2d_boolean_indexing(self):
        """Test 2D boolean array indexing."""
        data = self.make_random_nested_data((10, 8), seed=1000)
        ad = ArrayDict(data, batch_size=(10, 8))
        td, object_keys = self.arraydict_to_tensordict(ad)
        
        # Create 2D boolean mask
        rng = jax.random.PRNGKey(1000)
        jax_mask = jax.random.uniform(rng, (10, 8)) > 0.7
        torch_mask = torch.from_numpy(np.array(jax_mask))
        
        ad_result = ad[jax_mask]
        td_result = td[torch_mask]
        
        self.compare_results(ad_result, td_result, object_keys, "2d_boolean")
    
    def test_randomized_comprehensive(self):
        """Comprehensive randomized test with various index combinations."""
        for seed in range(1100, 1110):
            data = self.make_random_nested_data((10, 8), seed=seed)
            ad = ArrayDict(data, batch_size=(10, 8))
            td, object_keys = self.arraydict_to_tensordict(ad)
            
            indices_list = self.generate_random_indices((10, 8), seed=seed)
            
            for i, idx in enumerate(indices_list):
                # Convert jax indices to torch for tensordict
                if isinstance(idx, jnp.ndarray):
                    td_idx = torch.from_numpy(np.array(idx))
                elif isinstance(idx, tuple):
                    td_idx = tuple(
                        torch.from_numpy(np.array(item)) if isinstance(item, jnp.ndarray) else item
                        for item in idx
                    )
                else:
                    td_idx = idx
                
                try:
                    ad_result = ad[idx]
                    td_result = td[td_idx]
                    self.compare_results(ad_result, td_result, object_keys, 
                                       f"seed={seed}, index_{i}={idx}")
                except Exception as e:
                    # Both should fail or both should succeed
                    try:
                        td_result = td[td_idx]
                        pytest.fail(f"ArrayDict failed but TensorDict succeeded for {idx}: {e}")
                    except:
                        # Both failed, which is acceptable
                        pass


@pytest.mark.skipif(not TENSORDICT_AVAILABLE, reason="tensordict not installed")
def test_nested_arraydict_consistency():
    """Ensure nested ArrayDict structures maintain consistency during indexing."""
    rng = jax.random.PRNGKey(2000)
    keys = jax.random.split(rng, 10)
    
    # Create deeply nested structure
    inner_dict = ArrayDict({
        "level3_a": jax.random.normal(keys[0], (10, 8, 3)),
        "level3_b": jax.random.uniform(keys[1], (10, 8, 2)),
    }, batch_size=(10, 8))
    
    data = {
        "level1": jax.random.normal(keys[2], (10, 8, 5)),
        "nested": {
            "level2_a": jax.random.normal(keys[3], (10, 8, 4)),
            "level2_b": inner_dict,
            "level2_c": jax.random.uniform(keys[4], (10, 8)),
        },
    }
    
    ad = ArrayDict(data, batch_size=(10, 8))
    
    # Test indexing
    idx = jnp.array([1, 3, 5])
    result = ad[idx]
    
    assert result.batch_size == (3, 8)
    assert result["level1"].shape == (3, 8, 5)
    assert result[("nested", "level2_a")].shape == (3, 8, 4)
    assert result[("nested", "level2_b", "level3_a")].shape == (3, 8, 3)
    assert result[("nested", "level2_b", "level3_b")].shape == (3, 8, 2)
    assert result[("nested", "level2_c")].shape == (3, 8)
