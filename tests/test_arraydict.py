import jax
import jax.numpy as jnp
import numpy as np

import arraydict as ad
from arraydict import ArrayDict


def _make_nested_source(key_seed: int = 0) -> dict:
    rng = jax.random.PRNGKey(key_seed)
    rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
    inner = ArrayDict(
        {
            "t": jax.random.normal(k3, (10, 3)),
            "u": jax.random.uniform(k4, (10, 2, 2)),
        },
        batch_size=10,
    )
    return {
        "foo": jnp.zeros((10, 4, 3)),
        "bar": {
            "baz": jnp.ones((10, 2, 5)),
            "qux": jnp.full((10, 6), 7),
            "deep": {"alpha": jax.random.normal(k1, (10, 1))},
        },
        "baz": {
            "a": jnp.arange(10),
            "b": jnp.linspace(0, 1, 10),
            "inner": inner,
        },
        "non-numeric": [f"hello-{i}" for i in range(10)],
        ("tuple", "key"): jax.random.normal(k2, (10, 8)),
    }


def test_construction_and_column_view():
    arraydict = ArrayDict(_make_nested_source(), batch_size=10)
    assert ("foo",) in arraydict.keys()
    assert ("bar", "baz") in arraydict.keys()
    assert ("bar", "qux") in arraydict.keys()
    assert ("baz", "a") in arraydict.keys()
    assert ("baz", "b") in arraydict.keys()
    assert ("baz", "inner", "t") in arraydict.keys()
    assert ("tuple", "key") in arraydict.keys()

    foo = arraydict["foo"]
    assert isinstance(foo, jnp.ndarray)
    assert foo.shape == (10, 4, 3)

    bar = arraydict[("bar",)]
    assert isinstance(bar, ArrayDict)
    assert ("baz",) in bar.keys()
    assert ("qux",) in bar.keys()

    tuple_key = arraydict[("tuple",)]
    assert isinstance(tuple_key, ArrayDict)
    assert ("key",) in tuple_key.keys()


def test_row_view():
    arraydict = ArrayDict(_make_nested_source(), batch_size=10)
    rows = arraydict.to_rows()
    assert len(rows) == 10
    row0 = rows[0]
    assert "foo" in row0
    assert "bar" in row0
    assert "baz" in row0
    assert "non-numeric" in row0
    assert ("tuple", "key") not in row0
    assert "tuple" in row0


def test_batch_indexing_and_reshape():
    arraydict = ArrayDict(_make_nested_source(), batch_size=10)
    sliced = arraydict[2:5]
    assert sliced.batch_size == (3,)
    assert sliced["foo"].shape == (3, 4, 3)

    expanded = arraydict[None]
    assert expanded.batch_size == (1, 10)
    assert expanded["foo"].shape == (1, 10, 4, 3)

    indices = jnp.array([1, 3, 4])
    advanced = arraydict[indices]
    assert advanced.batch_size == (3,)
    assert advanced["bar"].batch_size == (3,)

    reshaped = arraydict.reshape((5, 2))
    assert reshaped.batch_size == (5, 2)
    assert reshaped["foo"].shape == (5, 2, 4, 3)


def test_split_and_gather():
    arraydict = ArrayDict(_make_nested_source(), batch_size=10)
    parts = arraydict.split(5, axis=0)
    assert len(parts) == 5
    assert all(part.batch_size == (2,) for part in parts)
    assert parts[0]["foo"].shape == (2, 4, 3)

    gathered = arraydict.gather(jnp.array([0, 2, 4]), axis=0)
    assert gathered.batch_size == (3,)
    assert gathered["bar"].batch_size == (3,)


def test_stack_and_concat():
    first = ArrayDict(_make_nested_source(1), batch_size=10)
    second = ArrayDict(_make_nested_source(2), batch_size=10)

    stacked = ad.stack([first, second], axis=0)
    assert stacked.batch_size == (2, 10)
    assert stacked["foo"].shape == (2, 10, 4, 3)

    concatenated = ad.concat([first, second], axis=0)
    assert concatenated.batch_size == (20,)
    assert concatenated["bar"].batch_size == (20,)


def test_batch_size_inference():
    source = {
        "x": jnp.ones((10, 2)),
        "y": np.zeros((10,)),
        "z": ["a" for _ in range(10)],
    }
    arraydict = ArrayDict(source)
    assert arraydict.batch_size == (10,)
    assert arraydict["z"].shape == (10,)


def test_setitem_with_str_and_tuple_keys():
    arraydict = ArrayDict({"base": jnp.zeros((10, 2))}, batch_size=10)

    nested_value = ArrayDict({"a": {"b": jnp.ones((10, 3))}}, batch_size=10)
    arraydict["group"] = nested_value

    group = arraydict["group"]
    assert isinstance(group, ArrayDict)
    assert group["a"]["b"].shape == (10, 3)

    grouped_row = arraydict[1]["group"]["a"]["b"]
    assert grouped_row.shape == (3,)

    flat_value = ArrayDict({"leaf": jnp.arange(10)}, batch_size=10)
    arraydict[("nested", "inner")] = flat_value

    nested_inner = arraydict[("nested", "inner")]
    assert isinstance(nested_inner, ArrayDict)
    assert nested_inner["leaf"].shape == (10,)

    nested_row = arraydict[2]["nested"]["inner"]["leaf"]
    assert nested_row.shape == ()


def test_set_method_returns_new_instance():
    arraydict = ArrayDict({"base": jnp.zeros((10, 2))}, batch_size=10)
    nested_value = ArrayDict({"a": {"b": jnp.ones((10, 3))}}, batch_size=10)

    updated = arraydict.set("group", nested_value)
    assert "group" not in arraydict.to_nested_dict()
    assert isinstance(updated["group"], ArrayDict)
    assert updated["group"]["a"]["b"].shape == (10, 3)

    flat_value = ArrayDict({"leaf": jnp.arange(10)}, batch_size=10)
    updated2 = updated.set(("nested", "inner"), flat_value)
    assert ("nested", "inner") not in updated.keys()
    assert updated2[("nested", "inner")]["leaf"].shape == (10,)
    assert updated2[1]["nested"]["inner"]["leaf"].shape == ()
