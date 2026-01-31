"""Demo script to showcase improved __repr__ for ArrayDict."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from arraydict import ArrayDict

print("=" * 80)
print("ArrayDict __repr__ Demo - Various Complex Cases")
print("=" * 80)
print()

# Case 1: Mixed numeric and string fields with nesting
print("Case 1: Mixed numeric, string, and Path fields with nesting")
print("-" * 80)
rng = jax.random.PRNGKey(42)
keys = jax.random.split(rng, 10)

ad1 = ArrayDict({
    'data': jax.random.normal(keys[0], (5, 3, 4)),
    'scores': jax.random.uniform(keys[1], (5, 3)),
    'names': np.array([
        ['Alice', 'Bob', 'Charlie'],
        ['David', 'Eve', 'Frank'],
        ['Grace', 'Henry', 'Iris'],
        ['Jack', 'Kate', 'Liam'],
        ['Mia', 'Noah', 'Olivia']
    ], dtype=object),
    'files': np.array([
        [Path('a.txt'), Path('b.txt'), Path('c.txt')],
        [Path('d.txt'), Path('e.txt'), Path('f.txt')],
        [Path('g.txt'), Path('h.txt'), Path('i.txt')],
        [Path('j.txt'), Path('k.txt'), Path('l.txt')],
        [Path('m.txt'), Path('n.txt'), Path('o.txt')]
    ], dtype=object),
    'metadata': {
        'timestamps': jax.random.normal(keys[2], (5, 3, 2)),
        'labels': np.array([['class_A', 'class_B', 'class_C']] * 5, dtype=object),
    }
}, batch_size=(5, 3))

print(repr(ad1))
print()

# Case 2: Deep nesting with various types
print("Case 2: Deep nesting with various field types")
print("-" * 80)
ad2 = ArrayDict({
    'level1': {
        'numeric': jax.random.normal(keys[3], (4, 2, 3)),
        'strings': np.array([['hello', 'world'], ['foo', 'bar'], ['baz', 'qux'], ['test', 'data']], dtype=object),
        'level2': {
            'deep_data': jax.random.uniform(keys[4], (4, 2)),
            'paths': np.array([[Path('deep/path1.json'), Path('deep/path2.json')]] * 4, dtype=object),
        }
    },
    'root_tensor': jax.random.normal(keys[5], (4, 2, 5, 3)),
}, batch_size=(4, 2))

print(repr(ad2))
print()

# Case 3: Long strings that get truncated
print("Case 3: Long strings with truncation")
print("-" * 80)
long_strings = np.array([
    ['This is a very long string that should be truncated in the repr output',
     'Another extremely long string for demonstration purposes'],
    ['Short', 'Medium length string here']
], dtype=object)

ad3 = ArrayDict({
    'long_text': long_strings,
    'normal': jax.random.normal(keys[6], (2, 2, 3)),
}, batch_size=(2, 2))

print(repr(ad3))
print()

# Case 4: Scalar batch with various types
print("Case 4: Scalar batch_size (empty tuple)")
print("-" * 80)
ad4 = ArrayDict({
    'single_value': jax.random.normal(keys[7], (3,)),
    'single_string': np.array('hello world', dtype=object),
    'single_path': np.array(Path('file.txt'), dtype=object),
}, batch_size=())

print(repr(ad4))
print()

# Case 5: High dimensional tensor
print("Case 5: High dimensional tensors")
print("-" * 80)
ad5 = ArrayDict({
    'high_dim': jax.random.normal(keys[8], (2, 3, 4, 5, 6, 7)),
    'messages': np.array([[['msg1', 'msg2', 'msg3', 'msg4'],
                          ['msg5', 'msg6', 'msg7', 'msg8'],
                          ['msg9', 'msg10', 'msg11', 'msg12']],
                         [['msgA', 'msgB', 'msgC', 'msgD'],
                          ['msgE', 'msgF', 'msgG', 'msgH'],
                          ['msgI', 'msgJ', 'msgK', 'msgL']]], dtype=object),
}, batch_size=(2, 3, 4))

print(repr(ad5))
print()

# Case 6: After indexing operations
print("Case 6: After various indexing operations")
print("-" * 80)
indexed = ad1[0]
print("After ad1[0]:")
print(repr(indexed))
print()

expanded = indexed[None]
print("After ad1[0][None]:")
print(repr(expanded))
print()

# Case 7: Mixed object array with different types
print("Case 7: Mixed types in object array")
print("-" * 80)
mixed_data = np.empty((2, 2), dtype=object)
mixed_data[0, 0] = "string"
mixed_data[0, 1] = 123
mixed_data[1, 0] = Path("file.txt")
mixed_data[1, 1] = [1, 2, 3]

ad7 = ArrayDict({
    'mixed': mixed_data,
    'pure_numeric': jax.random.normal(keys[9], (2, 2, 4)),
}, batch_size=(2, 2))

print(repr(ad7))
print()

print("=" * 80)
print("Demo Complete")
print("=" * 80)
