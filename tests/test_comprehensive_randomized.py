"""Comprehensive randomized tests with complex nested structures and operations.

Tests ArrayDict against TensorDict with:
- Nested depths: 0-5
- Batch dimensions: 0-5
- Feature dimensions: 0-5
- Numeric (jax.Array) and non-numeric (string, Path) fields
- Multi-step operations and dynamic state-based testing
"""
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from arraydict import ArrayDict, concat, stack


def generate_random_shape(max_dims: int = 5, max_size: int = 8) -> Tuple[int, ...]:
    """Generate random shape with 0-max_dims dimensions."""
    ndims = random.randint(0, max_dims)
    return tuple(random.randint(1, max_size) for _ in range(ndims))


def generate_batch_size(max_dims: int = 5, max_size: int = 6) -> Tuple[int, ...]:
    """Generate random batch_size with 0-max_dims dimensions."""
    ndims = random.randint(0, max_dims)
    return tuple(random.randint(1, max_size) for _ in range(ndims))


def generate_nested_structure(
    depth: int,
    batch_size: Tuple[int, ...],
    numeric_only: bool = False,
    non_numeric_only: bool = False,
    rng_key: Any = None,
) -> Dict[str, Any]:
    """Generate nested structure with specified depth and batch_size.
    
    Args:
        depth: Nesting depth (0=flat, 5=deep nesting)
        batch_size: Batch dimensions
        numeric_only: Only create numeric fields
        non_numeric_only: Only create non-numeric fields
        rng_key: JAX random key
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(random.randint(0, 10000))
    
    result = {}
    num_fields = random.randint(1, 4)
    
    for i in range(num_fields):
        rng_key, subkey = jax.random.split(rng_key)
        
        # Decide field type
        if numeric_only:
            field_type = 'numeric'
        elif non_numeric_only:
            field_type = 'non_numeric'
        else:
            field_type = random.choice(['numeric', 'non_numeric', 'nested'] if depth > 0 else ['numeric', 'non_numeric'])
        
        if field_type == 'numeric':
            # Numeric field: jax.Array
            feature_shape = generate_random_shape(max_dims=5, max_size=4)
            full_shape = batch_size + feature_shape
            result[f'num_{i}'] = jax.random.normal(subkey, full_shape)
        
        elif field_type == 'non_numeric':
            # Non-numeric field: string or Path
            feature_shape = generate_random_shape(max_dims=5, max_size=3)
            full_shape = batch_size + feature_shape
            
            if random.choice([True, False]):
                # String field
                data = np.empty(full_shape, dtype=object)
                for idx in np.ndindex(full_shape):
                    data[idx] = f"str_{'_'.join(map(str, idx))}"
                result[f'str_{i}'] = data
            else:
                # Path field
                data = np.empty(full_shape, dtype=object)
                for idx in np.ndindex(full_shape):
                    data[idx] = Path(f"file_{'_'.join(map(str, idx))}.txt")
                result[f'path_{i}'] = data
        
        elif field_type == 'nested' and depth > 0:
            # Nested structure
            result[f'nested_{i}'] = generate_nested_structure(
                depth - 1, batch_size, numeric_only, non_numeric_only, subkey
            )
    
    return result


def to_tensordict(data: Dict[str, Any], batch_size: Tuple[int, ...]) -> TensorDict:
    """Convert ArrayDict-style data to TensorDict."""
    flat_data = {}
    
    def flatten(d: Dict, prefix: str = ''):
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flatten(value, full_key)
            elif isinstance(value, jnp.ndarray):
                flat_data[full_key] = torch.from_numpy(np.array(value))
            elif isinstance(value, np.ndarray):
                # Handle object arrays (strings, Paths) by converting to tensor if possible
                if value.dtype == object:
                    # Store as nested TensorDict for non-numeric
                    # For testing purposes, skip non-numeric in TensorDict
                    continue
                flat_data[full_key] = torch.from_numpy(value)
    
    flatten(data)
    return TensorDict(flat_data, batch_size=list(batch_size))


class TestComprehensiveRandomized:
    """Comprehensive randomized tests comparing ArrayDict with TensorDict."""
    
    @pytest.mark.parametrize("depth", [0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize("batch_dims", [0, 1, 2, 3, 4, 5])
    def test_nested_depth_and_batch_dims(self, depth: int, batch_dims: int):
        """Test various nesting depths and batch dimensions."""
        random.seed(42 + depth * 10 + batch_dims)
        
        # Generate batch_size
        batch_size = tuple(random.randint(2, 5) for _ in range(batch_dims)) if batch_dims > 0 else ()
        
        # Create structure
        data = generate_nested_structure(depth, batch_size, numeric_only=True)
        ad = ArrayDict(data, batch_size=batch_size)
        
        assert ad.batch_size == batch_size
        
        # Test basic indexing if batch_size not empty
        if batch_dims > 0:
            indexed = ad[0]
            assert indexed.batch_size == batch_size[1:]
    
    @pytest.mark.parametrize("field_type", ["numeric_only", "non_numeric_only", "mixed"])
    def test_field_types(self, field_type: str):
        """Test numeric-only, non-numeric-only, and mixed fields."""
        random.seed(100)
        batch_size = (3, 4)
        
        if field_type == "numeric_only":
            data = generate_nested_structure(2, batch_size, numeric_only=True)
        elif field_type == "non_numeric_only":
            data = generate_nested_structure(2, batch_size, non_numeric_only=True)
        else:
            data = generate_nested_structure(2, batch_size)
        
        ad = ArrayDict(data, batch_size=batch_size)
        assert ad.batch_size == batch_size
        
        # Test column access
        for key in list(ad.keys())[:2]:
            col = ad[key]
            assert col is not None
    
    def test_multi_step_operations(self):
        """Test complex multi-step operations like arraydict[0][None][:][..., 0:1]."""
        random.seed(200)
        batch_size = (4, 3, 2)
        
        data = generate_nested_structure(2, batch_size, numeric_only=True)
        ad = ArrayDict(data, batch_size=batch_size)
        
        # Multi-step: [0][None][:2]
        result = ad[0][None][:2]
        # [0] → (3, 2), [None] → (1, 3, 2), [:2] → (1, 3, 2) (no change if dim0 size is 1)
        assert result.batch_size[1:] == (3, 2)
        
        # Multi-step: [1:3][None, :]
        result = ad[1:3][None, :]
        assert result.batch_size == (1, 2, 3, 2)
        
        # Multi-step with squeeze/unsqueeze
        result = ad[0].unsqueeze(0)[0].unsqueeze(1)
        assert result.batch_size == (3, 1, 2)
    
    def test_dynamic_operations(self):
        """Test dynamically determined operations based on current state."""
        random.seed(300)
        batch_size = (5, 4)
        
        data = generate_nested_structure(1, batch_size, numeric_only=True)
        ad = ArrayDict(data, batch_size=batch_size)
        
        # Dynamically apply operations
        operations = []
        current = ad
        
        for _ in range(5):
            current_bs = current.batch_size
            
            if len(current_bs) == 0:
                # Scalar batch, can only unsqueeze
                current = current.unsqueeze(0)
                operations.append('unsqueeze(0)')
            elif len(current_bs) > 0:
                # Choose random operation
                op = random.choice(['index', 'slice', 'unsqueeze', 'squeeze_if_possible'])
                
                if op == 'index':
                    current = current[0]
                    operations.append('[0]')
                elif op == 'slice':
                    if current_bs[0] > 1:
                        end = random.randint(1, current_bs[0])
                        current = current[:end]
                        operations.append(f'[:{end}]')
                elif op == 'unsqueeze':
                    dim = random.randint(0, len(current_bs))
                    current = current.unsqueeze(dim)
                    operations.append(f'unsqueeze({dim})')
                elif op == 'squeeze_if_possible' and 1 in current_bs:
                    dim = current_bs.index(1)
                    current = current.squeeze(dim)
                    operations.append(f'squeeze({dim})')
        
        # Verify operations don't crash
        assert current.batch_size is not None
    
    def test_stack_and_concat_operations(self):
        """Test stack and concat with various shapes."""
        random.seed(400)
        batch_size = (3, 2)
        
        # Create items with same structure
        base_data = generate_nested_structure(1, batch_size, numeric_only=True)
        rng_key = jax.random.PRNGKey(400)
        
        items = []
        for i in range(3):
            rng_key, subkey = jax.random.split(rng_key)
            # Create same keys with different values
            data = {}
            for key in base_data.keys():
                if isinstance(base_data[key], dict):
                    data[key] = {}
                    for subkey_name in base_data[key].keys():
                        shape = base_data[key][subkey_name].shape
                        rng_key, k = jax.random.split(rng_key)
                        data[key][subkey_name] = jax.random.normal(k, shape)
                else:
                    shape = base_data[key].shape
                    rng_key, k = jax.random.split(rng_key)
                    data[key] = jax.random.normal(k, shape)
            items.append(ArrayDict(data, batch_size=batch_size))
        
        # Stack
        stacked = stack(items, axis=0)
        assert stacked.batch_size == (3,) + batch_size
        
        # Concat
        concatenated = concat(items, axis=0)
        assert concatenated.batch_size == (9, 2)
    
    def test_column_insertion(self):
        """Test inserting new columns dynamically."""
        random.seed(500)
        batch_size = (4, 3)
        
        data = generate_nested_structure(1, batch_size, numeric_only=True)
        ad = ArrayDict(data, batch_size=batch_size)
        
        # Insert numeric column
        rng_key = jax.random.PRNGKey(500)
        new_data = jax.random.normal(rng_key, batch_size + (2,))
        ad['new_num'] = new_data
        
        assert ('new_num',) in ad.keys()
        assert ad['new_num'].shape == batch_size + (2,)
        
        # Insert non-numeric column
        str_data = np.empty(batch_size, dtype=object)
        for idx in np.ndindex(batch_size):
            str_data[idx] = f"inserted_{idx}"
        ad['new_str'] = str_data
        
        assert ('new_str',) in ad.keys()
        
        # Insert nested structure
        nested_ad = ArrayDict({'inner': jax.random.normal(rng_key, batch_size)}, batch_size=batch_size)
        ad['new_nested'] = nested_ad
        
        assert isinstance(ad['new_nested'], ArrayDict)
    
    def test_comparison_with_tensordict(self):
        """Compare ArrayDict behavior with TensorDict on numeric data."""
        random.seed(600)
        batch_size = (3, 4)
        
        data = generate_nested_structure(2, batch_size, numeric_only=True)
        ad = ArrayDict(data, batch_size=batch_size)
        td = to_tensordict(data, batch_size)
        
        assert ad.batch_size == tuple(td.batch_size)
        
        # Test indexing
        ad_indexed = ad[0]
        td_indexed = td[0]
        assert ad_indexed.batch_size == tuple(td_indexed.batch_size)
        
        # Test slicing
        ad_sliced = ad[:2]
        td_sliced = td[:2]
        assert ad_sliced.batch_size == tuple(td_sliced.batch_size)
        
        # Test unsqueeze
        ad_unsqueezed = ad.unsqueeze(0)
        td_unsqueezed = td.unsqueeze(0)
        assert ad_unsqueezed.batch_size == tuple(td_unsqueezed.batch_size)
    
    def test_extreme_nesting_and_dimensions(self):
        """Test with extreme cases: max nesting depth and dimensions."""
        random.seed(700)
        
        # Max depth
        batch_size = (2, 2)
        data = generate_nested_structure(5, batch_size, numeric_only=True)
        ad = ArrayDict(data, batch_size=batch_size)
        assert ad.batch_size == batch_size
        
        # Max batch dims
        batch_size = tuple(2 for _ in range(5))
        data = generate_nested_structure(1, batch_size, numeric_only=True)
        ad = ArrayDict(data, batch_size=batch_size)
        assert ad.batch_size == batch_size
        
        # Scalar batch
        batch_size = ()
        data = generate_nested_structure(0, batch_size, numeric_only=True)
        ad = ArrayDict(data, batch_size=batch_size)
        assert ad.batch_size == ()
    
    def test_mixed_operations_workflow(self):
        """Test realistic workflow with mixed operations."""
        random.seed(800)
        batch_size = (10, 5)
        
        # Create initial ArrayDict
        data = generate_nested_structure(2, batch_size)
        ad = ArrayDict(data, batch_size=batch_size)
        
        # Apply various operations
        # 1. Slice
        ad = ad[:8]
        assert ad.batch_size == (8, 5)
        
        # 2. Add new column
        ad['extra'] = jax.random.normal(jax.random.PRNGKey(800), (8, 5, 3))
        
        # 3. Index
        row = ad[2]
        assert row.batch_size == (5,)
        
        # 4. Expand
        row_expanded = row[None]
        assert row_expanded.batch_size == (1, 5)
        
        # 5. Squeeze if possible
        if 1 in ad.batch_size:
            ad = ad.squeeze(ad.batch_size.index(1))
        
        # 6. Stack multiple
        items = [ad[:2], ad[2:4], ad[4:6]]
        stacked = stack(items, axis=0)
        assert stacked.batch_size[0] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
