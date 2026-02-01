# ArrayDict 开发指南

这个文档面向对 ArrayDict 进行开发和维护的开发者。

## 项目结构

```
arraydict/
├── src/arraydict/
│   ├── arraydict.py          主类实现 (862 行)
│   ├── einops.py             einops 后端实现 (328 行)
│   └── __init__.py           包初始化
├── tests/
│   ├── test_arraydict.py           核心功能测试
│   ├── test_comprehensive_randomized.py  综合随机化测试
│   └── test_einops_backend.py       einops 后端测试
├── pyproject.toml            项目配置
├── README.md                 用户文档（推荐首先阅读）
└── DEVELOPER.md              本文档
```

## 核心架构

### ArrayDict 的设计思想

ArrayDict 存储一个 **共享 batch 维度** 的数组映射。关键概念：

- **batch_size**：批处理维度，是所有数组的共同前导维度
- **特征维度**：每个数组在 batch 维度之后的维度，可以不同
- **shape 属性**：返回 `batch_size`（仅公开维度），特征维度是私有的

```python
# 示例
ad = ArrayDict({
    'x': np.ones((2, 3, 4)),      # batch=(2, 3), features=(4,)
    'y': np.ones((2, 3, 5, 6))    # batch=(2, 3), features=(5, 6)
}, batch_size=(2, 3))

print(ad.shape)      # (2, 3) - 仅 batch 维度
print(ad['x'].shape) # (2, 3, 4) - 包含特征维度
```

### 主要类和方法

#### ArrayDict 类（src/arraydict/arraydict.py）

**初始化**
- `__init__(source, batch_size)`：创建 ArrayDict，支持自动 batch_size 推断

**索引与切片**
- `__getitem__`、`__setitem__`：获取/设置元素或字段
- 支持高级索引：切片、None（扩展维度）、Ellipsis 等

**批量操作**
- `reshape(shape)`：重塑 batch 维度
- `split(size, axis)`：沿 batch 维度分割
- `gather(indices, axis)`：收集指定索引的元素
- `squeeze(dim)`、`unsqueeze(dim)`：移除/添加大小为 1 的维度
- `repeat(repeats)`：重复 batch 维度（高效实现）

**视图接口**
- `RowView`：行视图（每行是一个字典）
- `ColumnView`：列视图（每列是一个字段）
- 属性：`rows`、`columns`

**嵌套支持**
- `keys()`、`values()`、`items()`：迭代接口
- 支持任意深度的嵌套结构
- 支持字符串和元组键

### einops 后端集成（src/arraydict/einops.py）

ArrayDict 实现了 einops 的后端 API，支持以下操作：

```python
import einops
from arraydict import ArrayDict

ad = ArrayDict({'x': np.ones((2, 3, 4))}, batch_size=(2, 3))

# rearrange - 重新排列 batch 维度
result = einops.rearrange(ad, 'b1 b2 ... -> (b1 b2) ...')

# transpose - 转置 batch 维度
result = einops.rearrange(ad, 'b1 b2 ... -> b2 b1 ...')

# add_axis - 添加维度
result = einops.rearrange(ad, 'b1 b2 ... -> b1 1 b2 ...')

# repeat - 重复维度（推荐直接调用实例方法）
result = einops.repeat(ad, 'b1 b2 ... -> (2 b1) (3 b2) ...')
# 或直接调用实例方法（更简洁，无需 einops）
result = ad.repeat((2, 3))
```

**架构设计**：
- `ArrayDictBackend` 类实现 einops 的 `AbstractBackend` 接口
- 后端自动注册到 einops（导入 `arraydict` 时）
- 每个 einops 操作都是对 ArrayDict 方法的适配（转发层）

**关键方法映射**：
- `shape(x)` → `x.batch_size`
- `reshape(x, shape)` → 内部重塑 batch 维度
- `transpose(x, axes)` → 内部转置 batch 维度
- `add_axis(x, pos)` → 内部扩展 batch 维度
- `repeat(x, repeats)` → `x.repeat(repeats)`
- `stack_on_zeroth_dimension(arrays)` → `ArrayDict.stack(...)`
- `concat(arrays, axis)` → `ArrayDict.concat(...)`

**特征维度处理**：
- einops 操作只影响 batch 维度
- 每个字段的特征维度独立保留
- 支持不同字段有不同的特征维度形状

## 实现细节

### repeat() 的高效实现

repeat 操作是核心优化案例。它使用 `jnp.repeat` 而非 `concat`：

```python
def repeat(self, repeats):
    # 逐轴应用 jnp.repeat，从最后一个轴向前处理
    for axis in range(batch_ndim - 1, -1, -1):
        if repeats[axis] > 1:
            result = jnp.repeat(result, repeats[axis], axis=axis)
    
    # 优点：
    # - 无中间副本（concat 会产生多倍存储）
    # - 内存占用仅为输出大小
    # - JAX 可能对 jnp.repeat 进行进一步优化
```

### 类型系统

- **KeyType**：`Union[str, Tuple[Any, ...]]`
- **ValueType**：`Any`（数组、字典、标量等）
- **BatchIndex**：支持整数、切片、数组索引等

所有公共 API 都有完整的类型注解。

### 错误处理

关键的验证函数（私有）：
- `_validate_batch_size`：检查所有数组的 batch 维度一致
- `_validate_squeeze_dimension`：检查 squeeze 目标维度大小为 1
- `_validate_unsqueeze_dimension`：检查 unsqueeze 位置有效

## 测试覆盖

**98 个测试，全部通过**

运行测试：
```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_arraydict.py -v

# 运行特定测试
pytest tests/test_arraydict.py::test_arraydict_repeat -v

# 快速测试（不显示详细输出）
pytest tests/ -q

# 显示测试覆盖率（需要 pytest-cov）
pytest tests/ --cov=src/arraydict --cov-report=html
```

### 核心功能测试 (test_arraydict.py) - 15 个测试
- 构造与列视图
- 行视图
- 批量索引与重塑
- split、gather、stack、concat
- batch_size 推断
- 字符串和元组键
- set 方法
- 路径列表存储（object 数组）
- **repeat 实例方法（5 个新测试）**：
  - 基本重复
  - 多维度重复
  - 多字段保留
  - 非数值字段保留
  - 单整数参数

### 综合随机化测试 (test_comprehensive_randomized.py) - 63 个测试
- 嵌套深度与维度变化（36 个参数化测试）
- 字段类型混合（numeric、non-numeric、mixed）
- 多步骤操作链
- 动态操作
- 堆叠/连接操作
- TensorDict 比较
- 极端嵌套和维度
- 边界条件：
  - 空 batch_size 推断
  - 空 batch reshape
  - 零长度 batch 维度
  - 零长度特征维度
  - 复杂零场景
  - squeeze/unsqueeze 与零
  - stack/concat/gather/split 与零

### einops 后端测试 (test_einops_backend.py) - 20 个测试
- 后端注册（2 个）
- rearrange 操作（6 个）：
  - 合并 batch 维度
  - 转置 batch 维度
  - 扩展 batch
  - 多字段保留
  - 嵌套键支持
  - 非数值字段保留
- 后端方法（3 个）
- 不同特征维度支持（1 个）
- **repeat 操作（5 个）**：
  - 简单重复
  - 多字段保留
  - 非数值字段保留
  - 多维度重复
  - 选择性维度重复
- 边界情况（3 个）

## 性能考虑

| 操作 | 时间复杂度 | 空间复杂度 | 优化 |
|------|-----------|-----------|------|
| 索引 | O(1) | O(1) | 直接访问 |
| reshape | O(n) | O(1) | 形状变换只需改变 batch_size |
| split | O(n) | O(1) | 共享底层数据 |
| gather | O(k) | O(k) | k = 索引数量 |
| stack | O(n) | O(n) | JAX 优化 |
| concat | O(n) | O(n) | JAX 优化 |
| repeat | O(n) | O(n) | jnp.repeat，无多倍副本 ✓ |

其中 n = 总数组元素数

## 设计决策

### 1. shape = batch_size

**决策**：`shape` 属性只返回 batch_size，不包含特征维度

**理由**：
- 符合 einops 的设计：einops 只操作公开的 batch 维度
- 清晰的公开/私有分离
- 支持不同字段有不同的特征维度

### 2. repeat() 是实例方法

**决策**：repeat 在 ArrayDict 类中实现为实例方法，einops 后端只是转发

**理由**：
- 符合 NumPy 的 API 设计（np.ndarray.repeat()）
- 无需依赖 einops 即可使用
- 代码单一来源，易于维护

### 3. 平面存储

**决策**：内部使用平面字典存储，键是元组

**理由**：
- 统一处理嵌套和非嵌套键
- 简化索引和迭代逻辑
- 易于支持任意深度嵌套

## 添加新功能的指南

### 添加新的批量操作

1. 在 ArrayDict 类中实现方法
2. 在其中调用 `_dispatch_array_op` 或类似的辅助函数
3. 更新 batch_size（如果必要）
4. 添加类型注解
5. 添加 docstring 和测试

示例：
```python
def my_operation(self, param):
    """Operation description."""
    # 验证参数
    # 应用操作到每个字段
    new_data = {k: _apply_operation(v, param) for k, v in self._data.items()}
    # 计算新的 batch_size
    new_batch = ...
    return ArrayDict(_nest_from_flat(new_data), batch_size=new_batch)
```

### 扩展 einops 后端

1. 在 ArrayDictBackend 中添加方法（如果 einops 需要）
2. 通常只需转发到 ArrayDict 的实例方法
3. 添加 docstring 说明映射关系
4. 在 einops 后端测试中添加测试用例

## 常见问题

**Q：为什么 shape 不包含特征维度？**

A：这是设计选择，使 ArrayDict 的公开 API 更清晰。batch 维度是统一的（所有字段共享），特征维度是私有的（每个字段不同）。

**Q：如何添加一个总是作用于特征维度的操作？**

A：这样的操作与 ArrayDict 的设计理念冲突。ArrayDict 的批量操作都作用于 batch 维度。如果需要特征维度操作，建议直接对 `ad['field']` 进行操作。

**Q：能否改变已存在 ArrayDict 的 batch_size？**

A：不能。batch_size 在创建时确定，后续操作都会创建新的 ArrayDict。这符合函数式编程风格。

## 依赖管理

**必需**：
- JAX >= 0.4.0
- NumPy >= 1.20

**可选**：
- einops >= 0.8.0（用于 einops 集成）
- torch（仅测试，用于 TensorDict 比较）
- tensordict（仅测试，用于比较）

在 `pyproject.toml` 的 `dev` 组中配置。

## 构建和发布

```bash
# 安装开发环境
pip install -e '.[dev]'

# 运行测试
pytest tests/ -v

# 构建包
pip install build
python -m build

# 验证
twine check dist/*
```

## 代码风格

- 使用 4 空格缩进
- 类型注解：所有公共 API 都必须有
- Docstring：NumPy 风格，包含 Args、Returns、Example
- 测试：为每个主要功能编写测试

## 其他资源

- **README.md**：用户文档，从这里开始
- **pyproject.toml**：项目配置和依赖
- **src/arraydict/arraydict.py**：主类，包含详细注释

---

**最后更新**：2024
**测试状态**：98/98 通过 ✅
