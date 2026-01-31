import jax.numpy as jnp
import pytest

from arraydict import ArrayDict


def test_squeeze_removes_dimension_of_size_one():
    arraydict = ArrayDict(
        {"x": jnp.zeros((1, 10, 5)), "y": jnp.ones((1, 10, 3))},
        batch_size=(1, 10),
    )

    squeezed = arraydict.squeeze(0)
    assert squeezed.batch_size == (10,)
    assert squeezed["x"].shape == (10, 5)
    assert squeezed["y"].shape == (10, 3)


def test_unsqueeze_inserts_dimension():
    arraydict = ArrayDict(
        {"x": jnp.zeros((10, 5)), "y": jnp.ones((10, 3))}, batch_size=(10,)
    )

    unsqueezed_at_0 = arraydict.unsqueeze(0)
    assert unsqueezed_at_0.batch_size == (1, 10)
    assert unsqueezed_at_0["x"].shape == (1, 10, 5)
    assert unsqueezed_at_0["y"].shape == (1, 10, 3)

    unsqueezed_at_1 = arraydict.unsqueeze(1)
    assert unsqueezed_at_1.batch_size == (10, 1)
    assert unsqueezed_at_1["x"].shape == (10, 1, 5)
    assert unsqueezed_at_1["y"].shape == (10, 1, 3)


def test_squeeze_unsqueeze_roundtrip():
    arraydict = ArrayDict(
        {"x": jnp.zeros((1, 10, 5)), "y": jnp.ones((1, 10, 3))},
        batch_size=(1, 10),
    )

    squeezed = arraydict.squeeze(0)
    unsqueezed = squeezed.unsqueeze(0)
    assert unsqueezed.batch_size == arraydict.batch_size
    assert (unsqueezed["x"] == arraydict["x"]).all()
    assert (unsqueezed["y"] == arraydict["y"]).all()


def test_squeeze_raises_on_non_one_dimension():
    arraydict = ArrayDict({"x": jnp.zeros((2, 10, 5))}, batch_size=(2, 10))

    with pytest.raises(ValueError, match="must be 1"):
        arraydict.squeeze(0)


def test_squeeze_raises_on_out_of_range():
    arraydict = ArrayDict(
        {"x": jnp.zeros((1, 10, 5))}, batch_size=(1, 10)
    )

    with pytest.raises(ValueError, match="out of range"):
        arraydict.squeeze(5)

    with pytest.raises(ValueError, match="out of range"):
        arraydict.squeeze(-1)


def test_unsqueeze_raises_on_out_of_range():
    arraydict = ArrayDict({"x": jnp.zeros((10, 5))}, batch_size=(10,))

    with pytest.raises(ValueError, match="out of range"):
        arraydict.unsqueeze(5)

    with pytest.raises(ValueError, match="out of range"):
        arraydict.unsqueeze(-1)
