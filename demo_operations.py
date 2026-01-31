"""
Demonstrate typical multi-step ArrayDict operations with executable examples.
This validates that our randomized tests cover meaningful use cases.
"""

import jax.numpy as jnp
import numpy as np
from pathlib import Path
from arraydict import ArrayDict, stack, concat


def print_case(title: str, operations: list[tuple[str, any]]):
    """Print a case with title, initial state, and operations."""
    print("=" * 80)
    print(f"Case: {title}")
    print("=" * 80)
    for expr, result in operations:
        if expr.startswith("arraydict="):
            print(f"\n{expr}")
            print(result)
        else:
            print(f"\n{expr}")
            print(result)
    print()


# Case 1: Basic indexing and batch manipulation
print_case(
    "Basic indexing and batch dimension manipulation",
    [
        ("arraydict=ArrayDict({'x': jnp.ones((3, 4, 5)), 'y': jnp.zeros((3, 4))})",
         ArrayDict({'x': jnp.ones((3, 4, 5)), 'y': jnp.zeros((3, 4))})),
        
        ("arraydict[0]",
         ArrayDict({'x': jnp.ones((3, 4, 5)), 'y': jnp.zeros((3, 4))})[0]),
        
        ("arraydict[0][None]",
         ArrayDict({'x': jnp.ones((3, 4, 5)), 'y': jnp.zeros((3, 4))})[0][None]),
        
        ("arraydict[:2]",
         ArrayDict({'x': jnp.ones((3, 4, 5)), 'y': jnp.zeros((3, 4))})[:2]),
        
        ("arraydict[0, 1]",
         ArrayDict({'x': jnp.ones((3, 4, 5)), 'y': jnp.zeros((3, 4))})[0, 1]),
    ]
)

# Case 2: Squeeze and unsqueeze operations
ad2 = ArrayDict({'data': jnp.ones((5, 1, 3, 1)), 'label': jnp.zeros((5, 1))})
print_case(
    "Squeeze and unsqueeze operations",
    [
        ("arraydict=ArrayDict({'data': jnp.ones((5, 1, 3, 1)), 'label': jnp.zeros((5, 1))})", ad2),
        
        ("arraydict.squeeze(1)",
         ad2.squeeze(1)),
        
        ("arraydict.squeeze(1).unsqueeze(0)",
         ad2.squeeze(1).unsqueeze(0)),
        
        ("arraydict.squeeze(1).unsqueeze(0)[0]",
         ad2.squeeze(1).unsqueeze(0)[0]),
    ]
)

# Case 3: Nested structures
ad3 = ArrayDict({
    'outer': jnp.ones((2, 3)),
    'nested': {
        'inner1': jnp.zeros((2, 3, 4)),
        'inner2': {
            'deep': jnp.full((2, 3, 5), 7.0)
        }
    }
})
print_case(
    "Nested structure access and indexing",
    [
        ("arraydict=ArrayDict({'outer': jnp.ones((2,3)), 'nested': {'inner1': jnp.zeros((2,3,4)), 'inner2': {'deep': jnp.full((2,3,5), 7.0)}}})", ad3),
        
        ("arraydict['nested']",
         ad3['nested']),
        
        ("arraydict['nested', 'inner2']",
         ad3['nested', 'inner2']),
        
        ("arraydict['nested', 'inner2', 'deep']",
         ad3['nested', 'inner2', 'deep']),
        
        ("arraydict[0]['nested', 'inner1']",
         ad3[0]['nested', 'inner1']),
        
        ("arraydict[:1]['nested', 'inner2', 'deep']",
         ad3[:1]['nested', 'inner2', 'deep']),
    ]
)

# Case 4: Stack and concat operations
ad4a = ArrayDict({'x': jnp.array([1., 2.]), 'y': jnp.array([3., 4.])})
ad4b = ArrayDict({'x': jnp.array([5., 6.]), 'y': jnp.array([7., 8.])})
print_case(
    "Stack and concat operations",
    [
        ("ad_a=ArrayDict({'x': jnp.array([1.,2.]), 'y': jnp.array([3.,4.])})", ad4a),
        ("ad_b=ArrayDict({'x': jnp.array([5.,6.]), 'y': jnp.array([7.,8.])})", ad4b),
        
        ("stack([ad_a, ad_b], axis=0)",
         stack([ad4a, ad4b], axis=0)),
        
        ("concat([ad_a, ad_b], axis=0)",
         concat([ad4a, ad4b], axis=0)),
        
        ("stack([ad_a, ad_b], axis=0)[1]",
         stack([ad4a, ad4b], axis=0)[1]),
        
        ("concat([ad_a, ad_b], axis=0)[:2]",
         concat([ad4a, ad4b], axis=0)[:2]),
    ]
)

# Case 5: Non-numeric fields (strings and paths)
ad5 = ArrayDict({
    'names': np.array(['Alice', 'Bob', 'Charlie'], dtype=object),
    'files': np.array([Path('a.txt'), Path('b.txt'), Path('c.txt')], dtype=object),
    'scores': jnp.array([95.5, 88.0, 92.3])
})
print_case(
    "Non-numeric fields with indexing",
    [
        ("arraydict=ArrayDict({'names': np.array(['Alice','Bob','Charlie'], dtype=object), 'files': np.array([Path('a.txt'),Path('b.txt'),Path('c.txt')], dtype=object), 'scores': jnp.array([95.5,88.0,92.3])})", ad5),
        
        ("arraydict[0]",
         ad5[0]),
        
        ("arraydict[:2]",
         ad5[:2]),
        
        ("arraydict[0][None]",
         ad5[0][None]),
        
        ("arraydict['names']",
         ad5['names']),
    ]
)

# Case 6: Column insertion with set() and __setitem__
ad6 = ArrayDict({'x': jnp.ones((2, 3))})
print_case(
    "Column insertion (immutable set vs mutable __setitem__)",
    [
        ("arraydict=ArrayDict({'x': jnp.ones((2,3))})", ad6),
        
        ("arraydict_new = arraydict.set('y', jnp.zeros((2, 3, 4)))",
         ad6.set('y', jnp.zeros((2, 3, 4)))),
        
        ("arraydict  # original unchanged",
         ad6),
    ]
)

# Demonstrate mutable setitem
ad6_mut = ArrayDict({'x': jnp.ones((2, 3))})
ad6_mut['z'] = jnp.full((2, 3, 5), 7.0)
print_case(
    "Mutable column insertion with __setitem__",
    [
        ("arraydict=ArrayDict({'x': jnp.ones((2,3))})\narraydict['z'] = jnp.full((2,3,5), 7.0)", ad6_mut),
    ]
)

# Case 7: Reshape operations
ad7 = ArrayDict({'data': jnp.arange(24).reshape(2, 3, 4)})
print_case(
    "Reshape operations",
    [
        ("arraydict=ArrayDict({'data': jnp.arange(24).reshape(2,3,4)})", ad7),
        
        ("arraydict.reshape((6, 4))",
         ad7.reshape((6, 4))),
        
        ("arraydict.reshape((6, 4))[::2]",
         ad7.reshape((6, 4))[::2]),
    ]
)

# Case 8: Split operations
ad8 = ArrayDict({'x': jnp.arange(30).reshape(6, 5), 'y': jnp.ones((6, 5, 2))})
print_case(
    "Split operations",
    [
        ("arraydict=ArrayDict({'x': jnp.arange(30).reshape(6,5), 'y': jnp.ones((6,5,2))})", ad8),
        
        ("parts = arraydict.split(3, axis=0)  # split into 3 parts",
         ad8.split(3, axis=0)),
        
        ("parts[0]  # first part",
         ad8.split(3, axis=0)[0]),
        
        ("parts[1]  # second part",
         ad8.split(3, axis=0)[1]),
    ]
)

# Case 9: Gather operations
ad9 = ArrayDict({'values': jnp.arange(15).reshape(3, 5), 'tags': np.array(['a', 'b', 'c'], dtype=object)})
print_case(
    "Gather operations",
    [
        ("arraydict=ArrayDict({'values': jnp.arange(15).reshape(3,5), 'tags': np.array(['a','b','c'], dtype=object)})", ad9),
        
        ("arraydict.gather(jnp.array([0, 2]), axis=0)",
         ad9.gather(jnp.array([0, 2]), axis=0)),
        
        ("arraydict.gather(jnp.array([0, 2]), axis=0)['tags']",
         ad9.gather(jnp.array([0, 2]), axis=0)['tags']),
    ]
)

# Case 10: Complex multi-step workflow
ad10 = ArrayDict({
    'features': jnp.ones((4, 3, 8)),
    'metadata': {
        'labels': np.array(['cat', 'dog', 'bird', 'fish'], dtype=object),
        'ids': jnp.arange(4)
    }
})
print_case(
    "Complex multi-step workflow",
    [
        ("arraydict=ArrayDict({'features': jnp.ones((4,3,8)), 'metadata': {'labels': np.array(['cat','dog','bird','fish'], dtype=object), 'ids': jnp.arange(4)}})", ad10),
        
        ("step1 = arraydict[:2]  # select first 2 samples",
         ad10[:2]),
        
        ("step2 = step1.unsqueeze(0)  # add batch dimension",
         ad10[:2].unsqueeze(0)),
        
        ("step3 = step2[0, 1]  # index into batch",
         ad10[:2].unsqueeze(0)[0, 1]),
        
        ("step4 = step3['metadata', 'labels']  # access nested field",
         ad10[:2].unsqueeze(0)[0, 1]['metadata', 'labels']),
    ]
)

# Case 11: Nested structure with mixed types
ad11 = ArrayDict({
    'numeric': jnp.ones((2, 3)),
    'level1': {
        'strings': np.array([['a', 'b', 'c'], ['d', 'e', 'f']], dtype=object),
        'level2': {
            'mixed': np.array([[1, 'x', 2.5], [2, 'y', 3.5]], dtype=object),
            'paths': np.array([[Path('1.txt'), Path('2.txt'), Path('3.txt')], [Path('4.txt'), Path('5.txt'), Path('6.txt')]], dtype=object)
        }
    }
})
print_case(
    "Deeply nested structure with mixed field types",
    [
        ("arraydict=ArrayDict({'numeric': jnp.ones((2,3)), 'level1': {'strings': np.array([['a','b','c'],['d','e','f']], dtype=object), 'level2': {'mixed': np.array([[1,'x',2.5],[2,'y',3.5]], dtype=object), 'paths': np.array([[Path('1.txt'),Path('2.txt'),Path('3.txt')],[Path('4.txt'),Path('5.txt'),Path('6.txt')]], dtype=object)}}})", ad11),
        
        ("arraydict[0]",
         ad11[0]),
        
        ("arraydict['level1', 'level2']",
         ad11['level1', 'level2']),
        
        ("arraydict[1]['level1', 'strings']",
         ad11[1]['level1', 'strings']),
        
        ("arraydict[:, 1]['level1', 'level2', 'paths']",
         ad11[:, 1]['level1', 'level2', 'paths']),
    ]
)

# Case 12: Scalar batch_size edge cases
ad12 = ArrayDict({'value': jnp.array([42.0]), 'name': np.array(['answer'], dtype=object)}, batch_size=(1,))
print_case(
    "Scalar-like batch operations",
    [
        ("arraydict=ArrayDict({'value': jnp.array([42.0]), 'name': np.array(['answer'], dtype=object)}, batch_size=(1,))", ad12),
        
        ("arraydict[0]  # extract single element",
         ad12[0]),
        
        ("arraydict[None]  # add dimension",
         ad12[None]),
        
        ("arraydict['value']",
         ad12['value']),
    ]
)

print("=" * 80)
print("All cases completed successfully!")
print("=" * 80)
