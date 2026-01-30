#!/usr/bin/env python3
"""Generate simplified advanced indexing test logs."""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Any, Union, Tuple

from arraydict import ArrayDict


def format_index(idx: Any) -> str:
    """Format an index for display."""
    if isinstance(idx, jnp.ndarray):
        # For arrays, show shape and dtype instead of full content
        if idx.size <= 5:
            return f"jnp.array({idx.tolist()})"
        else:
            return f"jnp.array(shape={idx.shape}, dtype={idx.dtype})"
    elif isinstance(idx, tuple):
        return f"({', '.join(format_index(i) for i in idx)})"
    elif isinstance(idx, slice):
        return repr(idx)
    elif idx is None or idx is Ellipsis:
        return repr(idx)
    else:
        return repr(idx)


def format_arraydict(ad: ArrayDict, indent: int = 0) -> str:
    """Format an ArrayDict for display."""
    prefix = "  " * indent
    
    if not ad._data:
        return f"ArrayDict(batch_size={ad.batch_size})"
    
    # Show batch_size and keys with shapes
    lines = [f"ArrayDict(batch_size={ad.batch_size}, {{"]
    
    for key in sorted(ad._data.keys()):
        value = ad._data[key]
        if isinstance(value, jnp.ndarray):
            feature_dims = value.shape[len(ad.batch_size):]
            lines.append(f"{prefix}  {key}: shape={feature_dims},")
        elif isinstance(value, ArrayDict):
            lines.append(f"{prefix}  {key}: ArrayDict(...),")
        else:
            lines.append(f"{prefix}  {key}: ...,")
    
    lines.append(f"{prefix}}})")
    return "\n".join(lines)


def make_test_data(batch_shape=(10, 8), seed=42) -> ArrayDict:
    """Create test data."""
    rng = jax.random.PRNGKey(seed)
    keys = jax.random.split(rng, 8)
    
    return ArrayDict({
        "x": jax.random.normal(keys[0], batch_shape + (5,)),
        "y": jax.random.uniform(keys[1], batch_shape + (3, 2)),
        "z": jax.random.normal(keys[2], batch_shape),
        "nested": {
            "a": jax.random.normal(keys[3], batch_shape + (4,)),
            "b": jax.random.uniform(keys[4], batch_shape + (2, 3)),
        },
        ("tuple", "key"): jax.random.normal(keys[5], batch_shape + (7,)),
    }, batch_size=batch_shape)


def generate_indexing_log() -> str:
    """Generate comprehensive indexing test log."""
    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("ArrayDict Advanced Indexing Test Log")
    log_lines.append("=" * 80)
    log_lines.append("")
    
    ad = make_test_data()
    log_lines.append("Initial ArrayDict:")
    log_lines.append(repr(ad))
    log_lines.append("")
    
    # Test cases with different index types
    test_cases = [
        ("Single int", 0),
        ("Negative int", -1),
        ("Slice", slice(1, 5)),
        ("Slice with step", slice(None, None, 2)),
        ("None (add dim)", None),
        ("Ellipsis", ...),
        ("Int array", jnp.array([0, 2, 4, 6])),
        ("Bool array", jnp.ones(10, dtype=bool)),
        ("Tuple with int", (2, 3)),
        ("Tuple with slice", (slice(1, 4), 2)),
        ("Complex tuple", (None, slice(2, 6), ...)),
    ]
    
    log_lines.append("Single Index Tests:")
    log_lines.append("-" * 80)
    for name, idx in test_cases:
        try:
            result = ad[idx]
            idx_str = format_index(idx)
            result_repr = repr(result)
            log_lines.append(f"ad[{idx_str}]")
            log_lines.append(f"  → batch_size={result.batch_size}")
            log_lines.append("")
        except Exception as e:
            idx_str = format_index(idx)
            log_lines.append(f"ad[{idx_str}]")
            log_lines.append(f"  → ERROR: {type(e).__name__}: {e}")
            log_lines.append("")
    
    # Column selection tests
    log_lines.append("Column Selection Tests:")
    log_lines.append("-" * 80)
    col_tests = [
        ("String key", "x"),
        ("Tuple key", ("tuple", "key")),
        ("Nested key", ("nested", "a")),
    ]
    
    for name, key in col_tests:
        try:
            result = ad[key]
            log_lines.append(f"ad[{repr(key)}]")
            if isinstance(result, jnp.ndarray):
                log_lines.append(f"  → shape={result.shape}, dtype={result.dtype}")
            elif isinstance(result, ArrayDict):
                log_lines.append(f"  → {repr(result)}")
            else:
                log_lines.append(f"  → {type(result).__name__}")
            log_lines.append("")
        except Exception as e:
            log_lines.append(f"ad[{repr(key)}]")
            log_lines.append(f"  → ERROR: {type(e).__name__}: {e}")
            log_lines.append("")
    
    # Random combination tests
    log_lines.append("Random Combination Tests (with array indices):")
    log_lines.append("-" * 80)
    rng = jax.random.PRNGKey(123)
    
    for i in range(10):
        rng, subkey = jax.random.split(rng)
        
        # Generate random index
        if jax.random.uniform(subkey) < 0.3:
            # Array index
            idx = jax.random.randint(jax.random.fold_in(subkey, i), (3,), 0, 10)
        else:
            # Simple index
            idx_type = jax.random.randint(jax.random.fold_in(subkey, i), (), 0, 4)
            if idx_type == 0:
                idx = int(jax.random.uniform(jax.random.fold_in(subkey, i)) * 10)
            elif idx_type == 1:
                idx = slice(int(jax.random.uniform(jax.random.fold_in(subkey, i)) * 5), 
                           int(jax.random.uniform(jax.random.fold_in(subkey, i + 1)) * 10))
            elif idx_type == 2:
                idx = None
            else:
                idx = ...
        
        try:
            result = ad[idx]
            idx_str = format_index(idx)
            log_lines.append(f"ad[{idx_str}]")
            log_lines.append(f"  → batch_size={result.batch_size}")
            log_lines.append("")
        except Exception as e:
            idx_str = format_index(idx)
            log_lines.append(f"ad[{idx_str}]")
            log_lines.append(f"  → ERROR: {type(e).__name__}")
            log_lines.append("")
    
    log_lines.append("=" * 80)
    log_lines.append("Test Log Generated Successfully")
    log_lines.append("=" * 80)
    
    return "\n".join(log_lines)


if __name__ == "__main__":
    log_content = generate_indexing_log()
    print(log_content)
    
    # Save to file
    output_path = Path(__file__).parent.parent / "tests" / "indexing_test_log.txt"
    output_path.write_text(log_content)
    print(f"\nLog saved to {output_path}")
