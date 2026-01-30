#!/usr/bin/env python3
"""Generate advanced indexing test logs with TensorDict-like repr."""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Any

from arraydict import ArrayDict


def format_index(idx: Any) -> str:
    """Format an index for display."""
    if isinstance(idx, jnp.ndarray):
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
    log_lines.append("=" * 100)
    log_lines.append("ArrayDict Advanced Indexing Test Log")
    log_lines.append("=" * 100)
    log_lines.append("")
    
    ad = make_test_data()
    log_lines.append("Initial ArrayDict:")
    log_lines.append("ad = " + repr(ad))
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
        ("Tuple with ints", (2, 3)),
        ("Tuple with slice", (slice(1, 4), 2)),
        ("Complex tuple", (None, slice(2, 6), ...)),
    ]
    
    log_lines.append("Single Index Tests:")
    log_lines.append("-" * 100)
    for name, idx in test_cases:
        try:
            result = ad[idx]
            idx_str = format_index(idx)
            result_repr = repr(result)
            log_lines.append(f"ad[{idx_str}] ==")
            # Indent the result repr
            for line in result_repr.split("\n"):
                log_lines.append(f"    {line}")
            log_lines.append("")
        except Exception as e:
            idx_str = format_index(idx)
            log_lines.append(f"ad[{idx_str}]")
            log_lines.append(f"    ERROR: {type(e).__name__}: {e}")
            log_lines.append("")
    
    # Column selection tests
    log_lines.append("Column Selection Tests:")
    log_lines.append("-" * 100)
    col_tests = [
        ("String key 'x'", "x"),
        ("String key 'z'", "z"),
        ("Tuple key", ("tuple", "key")),
        ("Nested key", ("nested", "a")),
        ("Nested dict", "nested"),
    ]
    
    for name, key in col_tests:
        try:
            result = ad[key]
            if isinstance(result, jnp.ndarray):
                log_lines.append(f"ad[{repr(key)}]")
                log_lines.append(f"    → shape={result.shape}, dtype={result.dtype}")
            elif isinstance(result, ArrayDict):
                log_lines.append(f"ad[{repr(key)}] ==")
                for line in repr(result).split("\n"):
                    log_lines.append(f"    {line}")
            else:
                log_lines.append(f"ad[{repr(key)}]")
                log_lines.append(f"    → {type(result).__name__}")
            log_lines.append("")
        except Exception as e:
            log_lines.append(f"ad[{repr(key)}]")
            log_lines.append(f"    ERROR: {type(e).__name__}: {e}")
            log_lines.append("")
    
    # Random combination tests
    log_lines.append("Random Combination Tests:")
    log_lines.append("-" * 100)
    rng = jax.random.PRNGKey(123)
    
    for i in range(12):
        rng, subkey = jax.random.split(rng)
        
        # Generate random index
        if jax.random.uniform(subkey) < 0.35:
            # Array index
            idx = jax.random.randint(jax.random.fold_in(subkey, i), (np.random.randint(1, 4),), 0, 10)
        else:
            # Simple index
            idx_type = jax.random.randint(jax.random.fold_in(subkey, i), (), 0, 4)
            if idx_type == 0:
                idx = int(jax.random.uniform(jax.random.fold_in(subkey, i)) * 10)
            elif idx_type == 1:
                start = int(jax.random.uniform(jax.random.fold_in(subkey, i)) * 5)
                stop = int(jax.random.uniform(jax.random.fold_in(subkey, i + 1)) * 10)
                idx = slice(start, stop)
            elif idx_type == 2:
                idx = None
            else:
                idx = ...
        
        try:
            result = ad[idx]
            idx_str = format_index(idx)
            result_repr = repr(result)
            log_lines.append(f"ad[{idx_str}] ==")
            for line in result_repr.split("\n"):
                log_lines.append(f"    {line}")
            log_lines.append("")
        except Exception as e:
            idx_str = format_index(idx)
            log_lines.append(f"ad[{idx_str}]")
            log_lines.append(f"    ERROR: {type(e).__name__}")
            log_lines.append("")
    
    log_lines.append("=" * 100)
    log_lines.append("Test Log Generated Successfully")
    log_lines.append("=" * 100)
    
    return "\n".join(log_lines)


if __name__ == "__main__":
    log_content = generate_indexing_log()
    print(log_content)
    
    # Save to file
    output_path = Path(__file__).parent.parent / "tests" / "indexing_test_log.txt"
    output_path.write_text(log_content)
    print(f"\nLog saved to {output_path}")
