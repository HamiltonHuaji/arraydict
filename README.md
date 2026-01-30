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

source = {
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
}
arraydict = ArrayDict(source, batch_size=[10]) # or batch_size=10
```

- `source` is first traversed recursively. its keys are first 'unraveled' to tuple. E.g., key `'foo'` becomes `('foo',)`, and `'qux'` inside nested dictionary `'bar'` becomes `('bar', 'qux')`.
- Nested ArrayDict instances are treated as dictionaries during this traversal, so their keys are also unraveled accordingly. E.g., key `'baz'` inside the nested ArrayDict `'bar'` becomes `('bar', 'baz')`.
- Leaf values that are jax.Arrays are kept as is.
- Leaf values that are mappings (like dict) are further traversed recursively.
- Leaf values that are sequences (like list or tuple) are converted to np.arrays with dtype=object.
- Then, unraveled keys and their corresponding values are stored in a flat dict inside `arraydict`. The values must share the same leading batch dimensions specified by `batch_size` (here `(10,)`). If `batch_size` is not provided, it is inferred from the longest leading dimension or length among numeric arrays and sequences.

- `arraydict` can be seen in two ways:
  - Column-wise view: arrays can be retrieved by their keys. Available keys are:
    - `'foo'`: jax.Array with shape `(10, 4, 3)`
    - `('foo',)`: also jax.Array with shape `(10, 4, 3)`
    - `('bar', 'baz')`: jax.Array with shape `(10, 2, 5)`
    - `('bar', 'qux')`: jax.Array with shape `(10, 6)`
    - `('baz', 'a')`: jax.Array with shape `(10,)`
    - `('baz', 'b')`: jax.Array with shape `(10,)`
    - `'non-numeric'`: np.ndarray with shape `(10,)` and dtype=object
    - `('tuple', 'key')`: jax.Array with shape `(10, 8)`
    - `('tuple',)`: new ArrayDict with key `('key',)` and value jax.Array with shape `(10, 8)`
    - `('baz',)`: new ArrayDict with keys `('a',)` and `('b',)` and their corresponding arrays.
  - Row-wise view: a jax.Array with shape `(10,)` where each element is a dictionary with the specified structure.
- `ad.stack([arraydict1, arraydict2, ...], axis=0)` stacks multiple ArrayDict instances along a new batch dimension.
- `ad.concat([arraydict1, arraydict2, ...], axis=0)` concatenates multiple ArrayDict instances along an existing batch dimension.

- other advanced indexing operations are also supported. basically, these operations can be seen as applying the same operation to each array stored in the ArrayDict, while preserving the structure and returning a new ArrayDict.
- `arraydict[None]` adds a new leading batch dimension.
- `arraydict[2:5]` slices the leading batch dimension.
- `arraydict.reshape((5, 2, -1))` reshapes the leading batch dimensions.
- `arraydict.split(5, axis=0)` splits the ArrayDict into multiple ArrayDict instances along the leading batch dimension, and returns a list of ArrayDicts.
- `arraydict.gather(indices, axis=0)` gathers elements along the leading batch dimension based on the provided indices.
- `arraydict.reshape((5, 2, -1))[None, 2:5, :, ..., None]` combines multiple indexing operations.
- `arraydict[indices, :]` is advanced indexing with jax arrays of indices.
- these operations are all non-inplace, returning new ArrayDict instances. the batch_size of the resulting ArrayDict is updated accordingly.
