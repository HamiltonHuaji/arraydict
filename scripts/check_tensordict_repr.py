#!/usr/bin/env python3
"""Check TensorDict repr format."""

try:
    import torch
    from tensordict import TensorDict
    
    # Create sample TensorDict similar to our ArrayDict
    data = {
        "x": torch.randn(10, 8, 5),
        "y": torch.randn(10, 8, 3, 2),
        "z": torch.randn(10, 8),
        "nested": {
            "a": torch.randn(10, 8, 4),
            "b": torch.randn(10, 8, 2, 3),
        },
        "tuple_key": torch.randn(10, 8, 7),
    }
    
    td = TensorDict(data, batch_size=[10, 8])
    
    print("=" * 100)
    print("TensorDict repr:")
    print("=" * 100)
    print(repr(td))
    print()
    
    print("=" * 100)
    print("TensorDict[0] repr:")
    print("=" * 100)
    print(repr(td[0]))
    print()
    
    print("=" * 100)
    print("TensorDict[0:3] repr:")
    print("=" * 100)
    print(repr(td[0:3]))
    print()
    
    print("=" * 100)
    print("TensorDict['x'] (column selection):")
    print("=" * 100)
    print(repr(td["x"]))
    print()
    
    print("=" * 100)
    print("TensorDict['nested'] (nested dict):")
    print("=" * 100)
    print(repr(td["nested"]))
    print()
    
except ImportError as e:
    print(f"TensorDict not available: {e}")
