from pathlib import Path
from arraydict import ArrayDict
import numpy as np

# 创建 batch_size=(5,), files 是 object array (5, 3)
paths_list = [[Path(f'file_{i}_{j}.txt') for j in range(3)] for i in range(5)]
ad = ArrayDict({'files': paths_list}, batch_size=[5, 3])

print('=== Initial state ===')
print('batch_size:', ad.batch_size)
print('files shape:', ad['files'].shape)
print('files dtype:', ad['files'].dtype)
print()

# Step 1: arraydict[0]
print('=== Step 1: arraydict[0] ===')
indexed = ad[0]
print('batch_size:', indexed.batch_size)
print('files value:', indexed['files'])
print('files type:', type(indexed['files']))
print()

# Step 2: arraydict[0][None]
print('=== Step 2: arraydict[0][None] ===')
try:
    expanded = indexed[None]
    print('batch_size:', expanded.batch_size)
    print('files shape:', expanded['files'].shape)
    print('files dtype:', expanded['files'].dtype)
    print('files:', expanded['files'])
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

