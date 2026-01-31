"""Test indexing behavior on non-numeric (object/string) fields."""
import jax.numpy as jnp
import pytest

from arraydict import ArrayDict


def test_non_numeric_field_indexing_scalar():
    """Non-numeric fields should behave like numeric arrays under indexing."""
    ad = ArrayDict({"foo": [["aaa", "bbb"]]}, batch_size=[1, 2])
    
    # Extract scalar
    indexed = ad[0, 0]
    assert indexed.batch_size == ()
    assert isinstance(indexed["foo"], jnp.ndarray) or hasattr(indexed["foo"], "shape")
    
    # Expand scalar back to array
    expanded = indexed[None]
    assert expanded.batch_size == (1,)
    assert expanded["foo"].shape == (1,)
    assert expanded["foo"].dtype == object


def test_non_numeric_field_with_none_index():
    """Test [None] (newaxis) on non-numeric field."""
    ad = ArrayDict({"messages": [["hello", "world"]]}, batch_size=[1, 2])
    
    expanded = ad[None]
    assert expanded.batch_size == (1, 1, 2)
    assert expanded["messages"].shape == (1, 1, 2)


def test_non_numeric_field_slicing():
    """Test slicing on non-numeric field."""
    ad = ArrayDict(
        {"items": [["a", "b", "c"], ["d", "e", "f"]]},
        batch_size=[2, 3],
    )
    
    sliced = ad[0:1]
    assert sliced.batch_size == (1, 3)
    assert sliced["items"].shape == (1, 3)
    assert (sliced["items"] == [["a", "b", "c"]]).all()


def test_path_field_indexing():
    """Test indexing with Path objects."""
    from pathlib import Path
    
    paths = [
        [Path("a.txt"), Path("b.txt")],
        [Path("c.txt"), Path("d.txt")],
    ]
    ad = ArrayDict({"files": paths}, batch_size=[2, 2])
    
    # Scalar access
    indexed = ad[0, 0]
    assert indexed.batch_size == ()
    assert isinstance(indexed["files"], jnp.ndarray) or isinstance(indexed["files"], object)
    
    # Expand back
    expanded = indexed[None]
    assert expanded.batch_size == (1,)
    assert expanded["files"].shape == (1,)
