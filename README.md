# ArrayDict
ArrayDict is a lightweight, JAX-backed container inspired by tensordict. It stores a mapping of arrays (and nested ArrayDict instances) that share leading batch dimensions, and supports batch-wise operations like indexing, reshaping, splitting, and gathering.

## Installation
```bash
# Basic installation
pip install -e .

# Development installation (includes torch and tensordict for testing)
pip install -e '.[dev]'
```

## Features

Assuming following code snippet has been executed:

```python
import jax
import jax.numpy as jnp
import arraydict as ad
from arraydict import ArrayDict, KeyType, ValueType

arraydict = ArrayDict({
    'foo': jnp.zeros((10, 4, 3)),
    'bar': ArrayDict({
        'baz': jnp.ones((10, 2, 5)),
        'qux': jnp.full((10, 6), 7),
    }),
    'baz': {
        'a': jnp.arange(10),
        'b': jnp.linspace(0, 1, 10),
    },
    'non-numeric': ['hello' for _ in range(10)],
    ('tuple', 'key'): jnp.random.normal(jax.random.PRNGKey(0), (10, 8)),
}, batch_size=[10])
```

- `arraydict` can be seen as a jax.Array with shape `(10,)` where each element is a dictionary with the specified structure.
- `ad.stack([arraydict1, arraydict2, ...], axis=0)` stacks multiple ArrayDict instances along a new batch dimension.
- `ad.concat([arraydict1, arraydict2, ...], axis=0)` concatenates multiple ArrayDict instances along an existing batch dimension.
- `arraydict['foo']` returns an array of shape `(10, 4, 3)`.
- `arraydict['bar']['baz']` returns an array of shape `(10, 2, 5)`.
- `arraydict['baz']['a']` returns an array of shape `(10,)`.
- `arraydict['non-numeric']` returns a list of length 10.
- `arraydict[0]` returns a new ArrayDict with batch size `()`, containing the first elements of each array.
- `arraydict[2:5]` returns a new ArrayDict with batch size `(3,)`, containing elements from index 2 to 4.
- `arraydict.reshape((5, 2))` returns a new ArrayDict with batch size `(5, 2)`.
- `arraydict.split(5)` returns a list of 5 ArrayDicts, each with batch size `(2,)`.
- `arraydict.gather([0, 2, 4])` or `arraydict.gather(jnp.array([0, 2, 4]))` or `arraydict[[0, 2, 4]]` returns a new ArrayDict with batch size `(3,)`, containing elements at indices 0, 2, and 4.
- `arraydict.keys()` returns an iterator over the keys: `['foo', ('bar', 'baz'), ('bar', 'qux'), ('baz', 'a'), ('baz', 'b'), 'non-numeric', ('tuple', 'key')]`. Note that nested keys are represented as tuples, and nested containers are flattened.
- `arraydict['baz']` returns a ArrayDict with keys `['a', 'b']`. This ArrayDict instance is created on-the-fly and is a view of the original data.
- `arraydict.values()` returns an iterator over the values.
- `arraydict.items()` returns an iterator over key-value pairs.
- `KeyType` is a type alias for valid keys (str, tuple of str).
- `ValueType` is a type alias for valid values (JAX arrays, ArrayDict, Mapping[KeyType, ValueType], Sequence[ValueType]).
- The shape or length of non-numeric values (e.g., lists of Any) is checked to match the batch size during initialization. if batch size is not provided, it is inferred from the longest leading dimension or length among numeric arrays and sequences.
- `arraydict[jnp.array([True, False, True, ...])]` supports boolean masking to select elements along the batch dimension.