# ArrayDict Development & Architecture

> Complete design and development documentation for ArrayDict library developers and maintainers. For user-facing documentation, see [README.md](README.md).

## Documentation Structure

- **README.md** - User guide with features, examples, and basic usage
- **ARCHITECTURE.md** (this file) - Complete developer documentation:
  - Core invariants maintained across all operations
  - Abstract layer design and type dispatch system
  - Refactoring approach and code simplification
  - Testing strategy and extension points
  - Demonstration scripts and performance considerations

---

## Quick Links

- [Core Invariants](#core-invariants) - The four guarantees that make ArrayDict reliable
- [Abstractions](#abstractions) - Type dispatch and validation helpers
- [Refactoring](#refactoring对比) - How the code was simplified
- [Operations](#操作分类) - Classification of all operations
- [Testing](#测试策略) - Test coverage and demo scripts
- [Extension](#扩展点) - How to add new features

---

ArrayDict 是一个轻量级容器，存储共享批次维度的数组映射。本文档说明内部架构、不变量和代码组织。

### 核心设计

```
外部 API (用户视角)
    ├─ 嵌套字典 API: ad['nested', 'key']
    ├─ 批次索引: ad[0], ad[:5], ad[0, 1]
    └─ 维度操作: squeeze, unsqueeze, reshape

        ↓ _normalize_key() / _flatten_mapping()

内部存储 (实现视角)
    ├─ batch_size: Tuple[int, ...]
    ├─ _data: Dict[Tuple[str, ...], ArrayLike]
    │   ├─ keys: 全部转为元组
    │   └─ values: jax.Array 或 np.ndarray(dtype=object)
    └─ 所有字段共享相同的批次维度
```

---

## 不变量 (Invariants)

### 1. 键表示 (Key Representation)

**规则**: `_data` 中所有键都是 `Tuple[str, ...]`

```python
# 外部输入形式                     内部存储形式
ad['field']           →  ('field',)
ad['nested', 'key']   →  ('nested', 'key')
ad[{'x': v}]          →  ('x',)   # 嵌套字典被扁平化
```

**强制器**: `_normalize_key(key) → Tuple[Any, ...]`

**用途**: 统一处理，简化嵌套查询逻辑

---

### 2. 值类型 (Value Types)

**规则**: `_data` 中的值只能是两种类型

| 类型 | 用途 | 创建方式 |
|------|------|---------|
| `jax.Array` | 数值数据（float, int, bool） | 直接存储 JAX 数组 |
| `np.ndarray(dtype=object)` | 非数值数据（字符串、路径、混合类型） | `np.array(value, dtype=object)` |

```python
# 数值字段
jnp.ones((5, 3))           →  jax.Array(shape=(5,3), dtype=float32)

# 非数值字段
np.array(['a', 'b', 'c'], dtype=object)  →  np.ndarray(shape=(3,), dtype=object)

# 标量自动包装
'string'                   →  np.array('string', dtype=object)
Path('file.txt')           →  np.array(Path('file.txt'), dtype=object)
```

**强制器**: 
- 构造函数检查并包装输入
- `_apply_index()` 对标量进行包装
- `_dispatch_array_op()` 确保操作使用正确的库

---

### 3. 批次维度 (Batch Dimensions)

**规则**: 
- `batch_size` 总是 `Tuple[int, ...]`
- 所有字段的前 `len(batch_size)` 个维度必须匹配
- 后续维度（特征维度）可以不同

```python
# 有效配置
batch_size=(5, 3)
'x': shape=(5, 3, 10)      # ✓ 匹配: (5, 3) 前缀
'y': shape=(5, 3)          # ✓ 匹配: (5, 3) 前缀
'z': shape=(5, 3, 4, 8)    # ✓ 匹配: (5, 3) 前缀

# 无效配置
'w': shape=(5, 2)          # ✗ 第二维不匹配 (3 ≠ 2)
'v': shape=(3, 5, 10)      # ✗ 批次维度顺序错误
```

**强制器**: `_resolve_batch_size(batch_size, shapes) → Tuple[int, ...]`

**操作保证**:
- `squeeze(dim)`: 删除一个维度，必须验证大小为 1
- `unsqueeze(dim)`: 插入大小为 1 的维度
- `reshape(new_shape)`: 改变批次维度，保留特征维度
- `gather/split`: 沿批次轴操作，不改变其他维度

---

### 4. 嵌套结构 (Nested Structure)

**规则**: 
- 内部平铺存储：`Dict[Tuple[str, ...], ArrayLike]`
- 外部嵌套 API：`{'key1': {'key2': value}}`
- 双向一致转换

```python
# 外部 API (嵌套)
{
    'metadata': {
        'labels': [...],
        'ids': [...]
    },
    'data': [...]
}

        ↓ _flatten_mapping()

# 内部存储 (平铺)
{
    ('metadata', 'labels'): array([...]),
    ('metadata', 'ids'): array([...]),
    ('data'): array([...])
}

        ↓ _nest_from_flat()  (恢复原样)
```

---

## 抽象层 (Abstractions)

### 类型检测函数

```python
def _is_array(value) → bool
    """JAX 或 NumPy 数组"""
    return isinstance(value, (jnp.ndarray, np.ndarray))

def _is_pure_numpy(value) → bool
    """纯 NumPy 数组（非 JAX 包装）"""
    return isinstance(value, np.ndarray) and not isinstance(value, jnp.ndarray)

def _is_jax_array(value) → bool
    """JAX 数组"""
    return isinstance(value, jnp.ndarray)

def _is_object_array(value) → bool
    """Object dtype 数组（非数值数据）"""
    return isinstance(value, np.ndarray) and value.dtype == object
```

---

### 中心分发器 (Central Dispatcher)

**问题**: 许多操作需要不同的 JAX vs NumPy 实现

```python
# 重复的模式（之前在 squeeze, unsqueeze, reshape 等处出现）
if isinstance(value, np.ndarray) and not isinstance(value, jnp.ndarray):
    result = np.squeeze(value, axis=dim)
elif _is_array(value):
    result = jnp.squeeze(value, axis=dim)
else:
    result = value
```

**解决方案**: 单一分发函数

```python
def _dispatch_array_op(value, jax_op, numpy_op, *args, **kwargs) → Any:
    """
    根据数组类型分发操作。
    
    Invariant: 确保纯 NumPy 数组用 numpy_op，JAX 数组用 jax_op
    """
    if _is_pure_numpy(value):
        return numpy_op(value, *args, **kwargs)
    elif _is_jax_array(value):
        return jax_op(value, *args, **kwargs)
    return value
```

**用处**:
- `squeeze/unsqueeze`: `_dispatch_array_op(v, jnp.squeeze, np.squeeze, axis=dim)`
- `_apply_take`: `_dispatch_array_op(v, jnp.take, np.take, indices, axis=axis)`
- `_apply_split`: 包装 split 逻辑后分发
- `_apply_stack/concat`: 类似

**收益**:
- 消除 5+ 处重复的类型检查
- 单一真相来源
- 易于维护和测试

---

### 维度验证函数

```python
def _validate_dimension(dim, batch_size, operation) → None
    """验证维度索引在范围内"""

def _validate_squeeze_dimension(dim, batch_size) → None
    """验证维度可以被 squeeze（大小必须为 1）"""

def _validate_unsqueeze_dimension(dim, batch_size) → None
    """验证维度可以被 unsqueeze（可在末尾）"""
```

---

## 操作分类

### 批次操作 (Batch Operations)

改变 `batch_size` 的操作：

| 操作 | 输入 | 输出 batch_size | 特征维度 |
|------|------|-----------------|---------|
| `[idx]` | 整数或切片 | 减少/保留/改变 | 不变 |
| `[None]` | None | 增加 1 个维度 | 不变 |
| `squeeze(dim)` | 维度索引 | 移除一个维度 | 不变 |
| `unsqueeze(dim)` | 维度索引 | 插入 1 大小维度 | 不变 |
| `reshape(new_shape)` | 新形状 | 改变 | 不变 |
| `split(num, axis)` | 分割数 | 沿 axis 分割 | 不变 |

### 字段操作 (Field Operations)

操作字段集合：

| 操作 | 作用 |
|------|------|
| `set(key, value)` | 返回新实例（不可变） |
| `__setitem__(key, value)` | 修改当前实例（可变） |
| `gather(indices, axis)` | 沿轴选择元素 |

### 结构操作 (Structure Operations)

全局操作：

| 函数 | 作用 |
|------|------|
| `stack(items, axis)` | 堆叠多个 ArrayDict |
| `concat(items, axis)` | 连接多个 ArrayDict |
| `to_nested_dict()` | 转为嵌套字典 |
| `to_rows()` | 转为行列表 |

---

## 代码组织

### 内部函数命名 (Internal Function Naming)

**前缀 `_`** - 只在模块内使用

```
_is_*              类型检测
_validate_*        维度验证
_normalize_*       输入标准化
_apply_*           应用操作
_flatten_*         嵌套转平铺
_nest_*            平铺转嵌套
_dispatch_*        分发操作
_resolve_*         解析参数
_ensure_*          确保前置条件
```

---

## 重构对比

### squeeze 操作：13 行 → 6 行 (-54%)

**重构前** (类型检查重复):
```python
def squeeze(self, dim):
    if dim < 0 or dim >= len(self.batch_size):
        raise ValueError(...)
    if self.batch_size[dim] != 1:
        raise ValueError(...)
    
    new_batch = ...
    new_data = {}
    for key, value in self._data.items():
        if isinstance(value, np.ndarray) and not isinstance(value, jnp.ndarray):
            new_data[key] = np.squeeze(value, axis=dim)
        elif _is_array(value):
            new_data[key] = jnp.squeeze(value, axis=dim)
        else:
            new_data[key] = value
    return ArrayDict(...)
```

**重构后** (抽象化):
```python
def squeeze(self, dim):
    _validate_squeeze_dimension(dim, self.batch_size)
    
    new_batch = self.batch_size[:dim] + self.batch_size[dim + 1:]
    new_data = {k: _dispatch_array_op(v, jnp.squeeze, np.squeeze, axis=dim) 
                for k, v in self._data.items()}
    return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)
```

**改进**: 
- 代码行数: -54%
- 类型检查集中化
- 使用字典推导式
- 清晰的意图

### 其他操作改进

| 操作 | 代码行数 | 改进 |
|------|----------|------|
| `unsqueeze()` | 13 → 6 | -54% |
| `_apply_take()` | 6 → 2 | -67% |
| `_apply_split()` | 重写 | 统一分发 |
| `_apply_stack()` | 重写 | 统一分发 |
| `_apply_concat()` | 重写 | 统一分发 |

---

## 扩展点 (Extension Points)

### 1. 新增操作类型

添加新的批次操作：
1. 在 `ArrayDict` 中创建方法
2. 更新 `batch_size` 逻辑
3. 使用 `_dispatch_array_op()` 处理值
4. 添加验证函数（如需要）

### 2. 新增字段类型

支持新的字段类型（如 Series）：
1. 更新 `_is_array()` 分支逻辑
2. 添加相应的 `_is_my_type()` 检测函数
3. 在 `_dispatch_array_op()` 中添加分支

### 3. 性能优化

可能的优化点：
- 缓存 `_flatten_mapping()` 结果
- 批量操作优化（多值同时处理）
- 懒求值策略

---

## 测试策略

### 随机化测试 (Randomized Testing)

`tests/test_comprehensive_randomized.py` 包含 46 个参数化测试：

```python
@pytest.mark.parametrize("depth", range(6))           # 嵌套深度: 0-5
@pytest.mark.parametrize("batch_dims", range(6))      # 批次维度: 0-5
@pytest.mark.parametrize("feature_dims", range(6))    # 特征维度: 0-5
@pytest.mark.parametrize("field_type", [              # 字段类型
    "numeric_only",
    "non_numeric_only", 
    "mixed"
])
```

**测试覆盖**:
- ✓ 极端情况：空容器、标量、高维
- ✓ 多步操作：squeeze → unsqueeze → reshape
- ✓ 嵌套结构：3+ 层深度
- ✓ 字段类型：float, int, str, Path, 混合
- ✓ Object arrays 的 squeeze/unsqueeze
- ✓ Stack/concat 操作
- ✓ 与 TensorDict 对比

**运行**: `pytest tests/test_comprehensive_randomized.py -v` (46/46 通过)

### 演示脚本 (Demo Script)

`demo_operations.py` 展示 12 个真实使用场景，包括：

1. 基础索引和批次操作
2. Squeeze/unsqueeze 与 object arrays
3. 嵌套结构访问
4. Stack/concat 操作
5. 非数值字段（字符串、路径）
6. 列插入（可变和不可变）
7. Reshape 操作
8. Split 操作
9. Gather 操作
10. 复杂多步工作流
11. 深度嵌套结构
12. 标量样操作

**输出**: `operations_output.txt` 包含 500+ 行演示结果

---

## 常见模式

### 创建 ArrayDict

```python
# 数值字段
ad = ArrayDict({'x': jnp.ones((5, 3)), 'y': jnp.zeros((5, 3))})

# 非数值字段
ad = ArrayDict({'names': np.array(['a', 'b', 'c'], dtype=object)})

# 混合类型
ad = ArrayDict({
    'data': jnp.ones((5, 3)),
    'labels': np.array(['cat', 'dog'], dtype=object)
})

# 嵌套
ad = ArrayDict({
    'features': jnp.ones((5, 10)),
    'metadata': {
        'labels': np.array(['A', 'B'], dtype=object),
        'ids': jnp.arange(5)
    }
})
```

### 批次索引

```python
ad[0]            # 第一个样本
ad[:5]           # 前 5 个样本
ad[0, 1]         # 多维索引
ad[None]         # 添加维度
ad[..., 0]       # 高级索引
```

### 维度操作

```python
ad.squeeze(1)           # 移除大小为 1 的维度 1
ad.unsqueeze(0)         # 在维度 0 插入大小 1 的维度
ad.reshape((20, 3))     # 改变批次维度
ad.split(4, axis=0)     # 沿轴 0 分成 4 部分
ad.gather([0, 2, 4])    # 选择特定索引
```

### 嵌套访问

```python
ad['field']                    # 简单字段
ad['nested', 'field']          # 嵌套字段
ad[0]['nested', 'field']       # 索引后嵌套访问
ad['nested']['field']          # 字典式访问
```

---

## 性能考虑

### 当前限制

1. **嵌套转平铺**: O(n) 其中 n 是字段数
2. **批次索引**: 对所有字段应用索引，O(n * m) 其中 m 是字段大小
3. **标量包装**: 每次索引时可能创建新数组

### 优化机会

1. 缓存扁平化映射（如果嵌套结构稳定）
2. 向量化批量操作
3. 避免不必要的标量包装

---

## 相关资源

- **用户文档**: 见 README.md
- **操作演示**: demo_operations.py 和 operations_output.txt
- **使用示例**: tests/test_comprehensive_randomized.py
- **参考项目**: TensorDict (https://github.com/pytorch-labs/tensordict)

# 'x' has feature dims (5,)
# 'y' has feature dims ()
```

### 4. Nested Structure
- **Flat storage**: Internally stored as `Dict[Tuple[str, ...], ArrayLike]`
- **Nested API**: Presented to users as nested dict/ArrayDict structure
- **Bidirectional conversion**:
  - Input: `_flatten_mapping()` converts nested dicts to flat tuple keys
  - Output: `_nest_from_flat()` reconstructs nested structure for display

Example:
```python
# Input (user perspective)
{'a': {...}, 'b': {'c': 1}}

# Internal storage
{('a', ...): ..., ('b', 'c'): 1}

# Output (via __repr__ or to_nested_dict)
{'a': {...}, 'b': {'c': ...}}
```

---

## Internal Abstractions

### Type Detection & Dispatch

**Problem**: Many operations need different implementations for JAX vs NumPy arrays.

**Solution**: Centralized dispatch mechanism with type detection helpers.

#### Type Detection Functions
```python
def _is_pure_numpy(value) -> bool
    """Check if pure NumPy array (not JAX-wrapped)."""

def _is_jax_array(value) -> bool
    """Check if JAX array."""

def _is_object_array(value) -> bool
    """Check if has object dtype (non-numeric data)."""
```

#### Dispatch Function
```python
def _dispatch_array_op(value, jax_op, numpy_op, *args, **kwargs) -> Any:
    """
    Dispatch operation based on array type.
    
    Invariants enforced:
    - Pure numpy arrays → numpy_op
    - JAX arrays → jax_op
    - Non-arrays → unchanged
    """
```

**Benefits**:
- Single source of truth for type-based dispatch
- Eliminates repeated if/elif type checking (5+ instances)
- Guarantees consistent handling of object arrays

**Usage Example**:
```python
# Instead of:
if isinstance(value, np.ndarray) and not isinstance(value, jnp.ndarray):
    result = np.squeeze(value, axis=dim)
elif isinstance(value, jnp.ndarray):
    result = jnp.squeeze(value, axis=dim)

# We write:
result = _dispatch_array_op(value, jnp.squeeze, np.squeeze, axis=dim)
```

### Dimension Validation

**Problem**: Batch operations need consistent dimension validation.

**Solution**: Specialized validation functions.

```python
def _validate_dimension(dim, batch_size, operation) -> None
    """Validate dimension is within batch_size range."""

def _validate_squeeze_dimension(dim, batch_size) -> None
    """Validate dimension can be squeezed (size == 1)."""

def _validate_unsqueeze_dimension(dim, batch_size) -> None
    """Validate dimension for unsqueeze (can be at end)."""
```

### Value Operations

All array operations are abstracted to handle both JAX and NumPy:

```python
_apply_index(value, index)           # Indexing with scalar wrapping
_apply_take(value, indices, axis)    # Take along axis
_apply_split(value, num, axis)       # Split into num parts
_apply_stack(values, axis)           # Stack multiple arrays
_apply_concat(values, axis)          # Concatenate arrays
_reshape_with_batch(value, new_batch, old_batch)  # Reshape preserving features
```

---

## Operation Patterns

### Pattern 1: Batch Operations (squeeze, unsqueeze, reshape)

These operations modify batch dimensions uniformly across all fields:

```python
def operation(self, ...):
    # 1. Validate invariants
    _validate_operation_dimension(...)
    
    # 2. Calculate new batch_size
    new_batch = calculate_new_batch_size(self.batch_size, ...)
    
    # 3. Apply to all fields uniformly
    new_data = {k: _dispatch_array_op(v, jax_fn, numpy_fn, ...)
                for k, v in self._data.items()}
    
    # 4. Return new ArrayDict maintaining invariants
    return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)
```

### Pattern 2: Array Indexing

Indexing preserves structure while modifying batch dimensions:

```python
def __getitem__(self, key):
    # 1. Normalize key to tuple representation
    normalized_key = _normalize_key(key)
    
    # 2. Determine if column access or batch indexing
    if is_column_access(normalized_key):
        # Return nested structure for that column(s)
        return get_column(normalized_key)
    else:
        # Apply batch indexing to all fields
        return apply_batch_index(normalized_key)
```

### Pattern 3: Multi-field Operations (stack, concat)

Combining multiple ArrayDicts along an axis:

```python
def operation(arraydict_list, axis):
    # 1. Validate all have same structure
    # 2. Apply operation to each field across all instances
    # 3. Calculate new batch_size
    # 4. Return new ArrayDict
```

---

## Data Flow Examples

### Example 1: Creating ArrayDict

```python
source = {
    'x': jnp.ones((10, 5)),
    'nested': {'y': jnp.zeros((10,))}
}
ad = ArrayDict(source)

# Internal representation:
ad._data = {
    ('x',): jnp.ones((10, 5)),
    ('nested', 'y'): jnp.zeros((10,))
}
ad.batch_size = (10,)
```

### Example 2: Squeeze Operation

```python
ad = ArrayDict({
    'x': jnp.ones((5, 1, 3)),
    'y': np.array(['a', 'b', 'c', 'd', 'e'], dtype=object).reshape(5, 1)
})
ad.batch_size = (5, 1)

result = ad.squeeze(1)

# Step 1: Validate - dimension 1 has size 1 ✓
# Step 2: new_batch = (5,) + () = (5,)
# Step 3: Apply to each field:
#   'x': _dispatch_array_op(jnp_arr, jnp.squeeze, np.squeeze, axis=1)
#        → jnp.squeeze(..., axis=1)  [JAX path]
#   'y': _dispatch_array_op(np_obj_arr, jnp.squeeze, np.squeeze, axis=1)
#        → np.squeeze(..., axis=1)   [NumPy path]
# Step 4: Return new ArrayDict with batch_size=(5,)
```

### Example 3: Non-numeric Field Indexing

```python
ad = ArrayDict({
    'names': np.array(['Alice', 'Bob', 'Charlie'], dtype=object)
})
ad.batch_size = (3,)

single = ad[0]

# Step 1: Apply indexing to 'names'
#   _apply_index(np.array([...], dtype=object), 0)
# Step 2: Result is scalar string 'Alice'
# Step 3: Wrap as object array: np.array('Alice', dtype=object)
# Step 4: Return ArrayDict with batch_size=() and that wrapped scalar
```

---

## Code Quality Metrics

### Before/After Refactoring

| Operation | Before | After | Reduction |
|-----------|--------|-------|-----------|
| `squeeze()` | 13 lines | 6 lines | -54% |
| `unsqueeze()` | 13 lines | 6 lines | -54% |
| `_apply_take()` | 6 lines | 2 lines | -67% |
| `_apply_split()` | 6 lines | ~8 lines | Better structured |
| Type dispatch | 5+ copies | 1 function | 100% DRY |

### Test Coverage

- ✅ 46 comprehensive randomized tests
- ✅ Nesting depth: 0-5 levels
- ✅ Batch dimensions: 0-5
- ✅ Field types: numeric, non-numeric, mixed
- ✅ Operations: indexing, squeeze/unsqueeze, stack/concat, split, gather
- ✅ Edge cases: scalar batches, empty arrays, deep nesting

---

## Future Optimization Opportunities

### 1. Batch Operation Template
Create a higher-level template for common batch operation patterns.

### 2. Value Normalization Layer
Explicit layer for value wrapping and validation.

### 3. Constructor Refactoring
Extract validation and normalization into separate methods.

### 4. Performance Caching
Cache type detection results for repeated operations on same arrays.

---

## Key Design Decisions

### Why Flat Internal Storage?
- Simpler to iterate over all fields uniformly
- Easier to implement batch operations
- Natural representation for nested structures with tuple keys

### Why Object Arrays for Non-numeric?
- JAX doesn't support object dtype directly
- NumPy's object arrays handle any Python object (strings, Path, custom types)
- Consistent behavior: scalar indexing wraps result back as object array

### Why Dispatch Instead of Virtual Methods?
- Functions are simpler than dispatch tables
- Explicit control flow is clearer than hidden methods
- Easy to test dispatch logic independently

### Why Mutable ArrayDict?
- Allows `__setitem__` for column insertion
- Provides both immutable `set()` and mutable `__setitem__` patterns
- More flexible for data manipulation workflows

---

## For Developers

### Adding a New Operation

1. **Decide on invariants**: What does the operation preserve? What changes?
2. **Write validation**: Use/create appropriate `_validate_*()` function
3. **Implement using dispatch**: Use `_dispatch_array_op()` or similar
4. **Update tests**: Add to `test_comprehensive_randomized.py`
5. **Document invariants**: Add to operation docstring

### Debugging

The `__repr__` method shows the complete internal structure:
```python
print(arraydict)  # Shows all fields with types and shapes
```

Check internal state directly:
```python
print(arraydict._data)       # Flat dict with tuple keys
print(arraydict.batch_size)  # Batch dimensions
```

### Type Checking

Use type detection functions to debug array handling:
```python
for key, value in arraydict._data.items():
    print(f"{key}: is_jax={_is_jax_array(value)}, is_numpy={_is_pure_numpy(value)}")
```

---

## References

- See `README.md` for usage examples
- See `demo_operations.py` for 12 real-world usage patterns
- See `tests/test_comprehensive_randomized.py` for comprehensive test suite
