"""
Additional verification: compare actual values between ArrayDict and TensorDict results.
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tensordict import TensorDict

from arraydict import ArrayDict


def verify_values_match():
    """Verify that values match between ArrayDict and TensorDict for various operations."""
    print("=" * 70)
    print("验证 ArrayDict 和 TensorDict 的实际值匹配")
    print("=" * 70)
    
    # Test 1: Simple case
    print("\n测试 1: 简单 1D 切片")
    print("-" * 70)
    rng = jax.random.PRNGKey(100)
    keys = jax.random.split(rng, 2)
    
    data = {
        "x": jax.random.normal(keys[0], (10, 3)),
        "y": jax.random.uniform(keys[1], (10, 2)),
    }
    
    ad = ArrayDict(data, batch_size=10)
    td = TensorDict({
        ("x",): torch.from_numpy(np.array(data["x"])),
        ("y",): torch.from_numpy(np.array(data["y"])),
    }, batch_size=10)
    
    idx = slice(2, 5)
    ad_result = ad[idx]
    td_result = td[idx]
    
    print(f"原始 batch_size: {ad.batch_size}")
    print(f"索引: {idx}")
    print(f"结果 batch_size: {ad_result.batch_size}")
    
    # Compare x values
    ad_x = ad_result["x"]
    td_x = td_result[("x",)]
    print(f"\nArrayDict['x'] 形状: {ad_x.shape}")
    print(f"TensorDict['x'] 形状: {tuple(td_x.shape)}")
    print(f"ArrayDict['x'][0, :]: {ad_x[0, :]}")
    print(f"TensorDict['x'][0, :]: {td_x[0, :].numpy()}")
    
    # Check if values match
    np.testing.assert_allclose(np.array(ad_x), td_x.numpy(), rtol=1e-5, atol=1e-6)
    print("✓ 值完全匹配!")
    
    # Test 2: Integer array indexing
    print("\n\n测试 2: 整数数组索引")
    print("-" * 70)
    
    idx2 = jnp.array([1, 3, 7])
    idx2_torch = torch.tensor([1, 3, 7])
    
    ad_result2 = ad[idx2]
    td_result2 = td[idx2_torch]
    
    print(f"索引: {idx2}")
    print(f"结果 batch_size: {ad_result2.batch_size}")
    
    ad_y = ad_result2["y"]
    td_y = td_result2[("y",)]
    print(f"\nArrayDict['y'] 形状: {ad_y.shape}")
    print(f"TensorDict['y'] 形状: {tuple(td_y.shape)}")
    print(f"ArrayDict['y']:\n{ad_y}")
    print(f"TensorDict['y']:\n{td_y.numpy()}")
    
    np.testing.assert_allclose(np.array(ad_y), td_y.numpy(), rtol=1e-5, atol=1e-6)
    print("✓ 值完全匹配!")
    
    # Test 3: Boolean indexing
    print("\n\n测试 3: 布尔索引")
    print("-" * 70)
    
    bool_mask = jnp.array([i % 2 == 0 for i in range(10)])
    bool_mask_torch = torch.tensor([i % 2 == 0 for i in range(10)])
    
    ad_result3 = ad[bool_mask]
    td_result3 = td[bool_mask_torch]
    
    print(f"布尔掩码: {bool_mask}")
    print(f"True 的数量: {bool_mask.sum()}")
    print(f"结果 batch_size: {ad_result3.batch_size}")
    
    ad_x3 = ad_result3["x"]
    td_x3 = td_result3[("x",)]
    print(f"\nArrayDict['x'] 形状: {ad_x3.shape}")
    print(f"TensorDict['x'] 形状: {tuple(td_x3.shape)}")
    
    np.testing.assert_allclose(np.array(ad_x3), td_x3.numpy(), rtol=1e-5, atol=1e-6)
    print("✓ 值完全匹配!")
    
    # Test 4: 2D slicing
    print("\n\n测试 4: 2D 切片")
    print("-" * 70)
    
    rng4 = jax.random.PRNGKey(200)
    keys4 = jax.random.split(rng4, 2)
    
    data4 = {
        "a": jax.random.normal(keys4[0], (8, 6, 4)),
        "b": jax.random.uniform(keys4[1], (8, 6, 2)),
    }
    
    ad4 = ArrayDict(data4, batch_size=(8, 6))
    td4 = TensorDict({
        ("a",): torch.from_numpy(np.array(data4["a"])),
        ("b",): torch.from_numpy(np.array(data4["b"])),
    }, batch_size=(8, 6))
    
    idx4 = (slice(1, 5), slice(2, 5))
    ad_result4 = ad4[idx4]
    td_result4 = td4[idx4]
    
    print(f"原始 batch_size: {ad4.batch_size}")
    print(f"索引: {idx4}")
    print(f"结果 batch_size: {ad_result4.batch_size}")
    
    ad_a4 = ad_result4["a"]
    td_a4 = td_result4[("a",)]
    print(f"\nArrayDict['a'] 形状: {ad_a4.shape}")
    print(f"TensorDict['a'] 形状: {tuple(td_a4.shape)}")
    print(f"ArrayDict['a'][0, 0, :]: {ad_a4[0, 0, :]}")
    print(f"TensorDict['a'][0, 0, :]: {td_a4[0, 0, :].numpy()}")
    
    np.testing.assert_allclose(np.array(ad_a4), td_a4.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.array(ad_result4["b"]), td_result4[("b",)].numpy(), rtol=1e-5, atol=1e-6)
    print("✓ 所有值完全匹配!")
    
    # Test 5: None (newaxis)
    print("\n\n测试 5: None (newaxis)")
    print("-" * 70)
    
    idx5 = None
    ad_result5 = ad4[idx5]
    td_result5 = td4[idx5]
    
    print(f"原始 batch_size: {ad4.batch_size}")
    print(f"索引: None")
    print(f"结果 batch_size: {ad_result5.batch_size}")
    
    ad_b5 = ad_result5["b"]
    td_b5 = td_result5[("b",)]
    print(f"\nArrayDict['b'] 形状: {ad_b5.shape}")
    print(f"TensorDict['b'] 形状: {tuple(td_b5.shape)}")
    
    np.testing.assert_allclose(np.array(ad_b5), td_b5.numpy(), rtol=1e-5, atol=1e-6)
    print("✓ 值完全匹配!")
    
    # Test 6: Complex nested
    print("\n\n测试 6: 复杂嵌套结构")
    print("-" * 70)
    
    rng6 = jax.random.PRNGKey(300)
    keys6 = jax.random.split(rng6, 5)
    
    nested_ad = ArrayDict({
        "deep": jax.random.normal(keys6[0], (10, 3)),
    }, batch_size=10)
    
    data6 = {
        "top": jax.random.uniform(keys6[1], (10, 4)),
        "mid": {
            "a": jax.random.normal(keys6[2], (10, 2)),
            "nested": nested_ad,
        },
    }
    
    ad6 = ArrayDict(data6, batch_size=10)
    td6 = TensorDict({
        ("top",): torch.from_numpy(np.array(data6["top"])),
        ("mid", "a"): torch.from_numpy(np.array(data6["mid"]["a"])),
        ("mid", "nested", "deep"): torch.from_numpy(np.array(nested_ad._data[("deep",)])),
    }, batch_size=10)
    
    idx6 = jnp.array([0, 2, 4, 6])
    idx6_torch = torch.tensor([0, 2, 4, 6])
    
    ad_result6 = ad6[idx6]
    td_result6 = td6[idx6_torch]
    
    print(f"索引: {idx6}")
    print(f"结果 batch_size: {ad_result6.batch_size}")
    print(f"Keys: {list(ad_result6.keys())}")
    
    # Verify nested key
    ad_nested = ad_result6[("mid", "nested", "deep")]
    td_nested = td_result6[("mid", "nested", "deep")]
    print(f"\nArrayDict['mid', 'nested', 'deep'] 形状: {ad_nested.shape}")
    print(f"TensorDict['mid', 'nested', 'deep'] 形状: {tuple(td_nested.shape)}")
    print(f"ArrayDict 嵌套值[0]: {ad_nested[0]}")
    print(f"TensorDict 嵌套值[0]: {td_nested[0].numpy()}")
    
    np.testing.assert_allclose(np.array(ad_nested), td_nested.numpy(), rtol=1e-5, atol=1e-6)
    print("✓ 嵌套值完全匹配!")
    
    print("\n" + "=" * 70)
    print("✓✓✓ 所有测试通过！ArrayDict 与 TensorDict 行为完全一致 ✓✓✓")
    print("=" * 70)


if __name__ == "__main__":
    verify_values_match()
