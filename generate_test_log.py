"""
Generate and log test cases for ArrayDict indexing to verify test comprehensiveness.
Records: original ArrayDict, indices, indexed results, and TensorDict comparison.
"""

import json
import jax
import jax.numpy as jnp
import numpy as np
import torch
from tensordict import TensorDict

from arraydict import ArrayDict


def serialize_array(arr):
    """Convert array to serializable format."""
    if isinstance(arr, (jnp.ndarray, np.ndarray)):
        return {
            "type": "jax_array" if isinstance(arr, jnp.ndarray) else "numpy_array",
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "values": arr.tolist() if arr.size < 100 else f"<{arr.size} elements>",
        }
    return str(arr)


def serialize_index(idx):
    """Convert index to serializable format."""
    if isinstance(idx, (jnp.ndarray, np.ndarray)):
        return {
            "type": "array_index",
            "shape": list(idx.shape),
            "dtype": str(idx.dtype),
            "values": idx.tolist(),
        }
    elif isinstance(idx, slice):
        return {
            "type": "slice",
            "start": idx.start,
            "stop": idx.stop,
            "step": idx.step,
        }
    elif isinstance(idx, tuple):
        return {
            "type": "tuple",
            "elements": [serialize_index(item) for item in idx],
        }
    elif idx is None:
        return {"type": "None"}
    elif idx is Ellipsis:
        return {"type": "Ellipsis"}
    else:
        return {"type": "other", "value": str(idx)}


def serialize_arraydict(ad, max_keys=None):
    """Serialize ArrayDict to dict format."""
    result = {
        "batch_size": ad.batch_size,
        "keys": {},
    }
    
    keys_to_show = list(ad.keys())
    if max_keys and len(keys_to_show) > max_keys:
        keys_to_show = keys_to_show[:max_keys]
    
    for key in keys_to_show:
        value = ad._data[key]
        if isinstance(value, np.ndarray) and value.dtype == object:
            result["keys"][str(key)] = {
                "type": "object_array",
                "shape": list(value.shape),
                "sample": str(value.flat[0]) if value.size > 0 else "empty",
            }
        else:
            result["keys"][str(key)] = serialize_array(value)
    
    if max_keys and len(ad.keys()) > max_keys:
        result["total_keys"] = len(ad.keys())
        result["showing_first"] = max_keys
    
    return result


def generate_test_cases():
    """Generate comprehensive test cases and log them."""
    test_log = []
    
    # Test case 1: Simple 1D slicing
    print("Generating test case 1: Simple 1D slicing...")
    rng = jax.random.PRNGKey(1000)
    keys = jax.random.split(rng, 5)
    
    data1 = {
        "x": jax.random.normal(keys[0], (10, 5)),
        "y": jax.random.uniform(keys[1], (10, 3)),
        "nested": {
            "a": jax.random.normal(keys[2], (10, 2)),
        },
    }
    ad1 = ArrayDict(data1, batch_size=10)
    td1 = TensorDict({
        ("x",): torch.from_numpy(np.array(data1["x"])),
        ("y",): torch.from_numpy(np.array(data1["y"])),
        ("nested", "a"): torch.from_numpy(np.array(data1["nested"]["a"])),
    }, batch_size=10)
    
    idx1 = slice(2, 7)
    ad1_result = ad1[idx1]
    td1_result = td1[idx1]
    
    test_log.append({
        "case_id": 1,
        "description": "Simple 1D slicing with slice(2, 7)",
        "original_arraydict": serialize_arraydict(ad1),
        "original_tensordict_batch_size": list(td1.batch_size),
        "index": serialize_index(idx1),
        "result_arraydict": serialize_arraydict(ad1_result),
        "result_tensordict_batch_size": list(td1_result.batch_size),
        "shapes_match": ad1_result.batch_size == tuple(td1_result.batch_size),
    })
    
    # Test case 2: 2D slicing
    print("Generating test case 2: 2D slicing...")
    data2 = {
        "x": jax.random.normal(keys[3], (10, 8, 5)),
        "y": jax.random.uniform(keys[4], (10, 8, 3, 2)),
    }
    ad2 = ArrayDict(data2, batch_size=(10, 8))
    td2 = TensorDict({
        ("x",): torch.from_numpy(np.array(data2["x"])),
        ("y",): torch.from_numpy(np.array(data2["y"])),
    }, batch_size=(10, 8))
    
    idx2 = (slice(2, 7), slice(1, 6))
    ad2_result = ad2[idx2]
    td2_result = td2[idx2]
    
    test_log.append({
        "case_id": 2,
        "description": "2D slicing with (slice(2, 7), slice(1, 6))",
        "original_arraydict": serialize_arraydict(ad2),
        "original_tensordict_batch_size": list(td2.batch_size),
        "index": serialize_index(idx2),
        "result_arraydict": serialize_arraydict(ad2_result),
        "result_tensordict_batch_size": list(td2_result.batch_size),
        "shapes_match": ad2_result.batch_size == tuple(td2_result.batch_size),
    })
    
    # Test case 3: Integer array indexing
    print("Generating test case 3: Integer array indexing...")
    rng3 = jax.random.PRNGKey(2000)
    keys3 = jax.random.split(rng3, 4)
    
    data3 = {
        "a": jax.random.normal(keys3[0], (12, 6)),
        "b": jax.random.uniform(keys3[1], (12, 4, 3)),
    }
    ad3 = ArrayDict(data3, batch_size=12)
    td3 = TensorDict({
        ("a",): torch.from_numpy(np.array(data3["a"])),
        ("b",): torch.from_numpy(np.array(data3["b"])),
    }, batch_size=12)
    
    idx3 = jnp.array([1, 3, 5, 7, 9])
    idx3_torch = torch.tensor([1, 3, 5, 7, 9])
    ad3_result = ad3[idx3]
    td3_result = td3[idx3_torch]
    
    test_log.append({
        "case_id": 3,
        "description": "Integer array indexing",
        "original_arraydict": serialize_arraydict(ad3),
        "original_tensordict_batch_size": list(td3.batch_size),
        "index": serialize_index(idx3),
        "result_arraydict": serialize_arraydict(ad3_result),
        "result_tensordict_batch_size": list(td3_result.batch_size),
        "shapes_match": ad3_result.batch_size == tuple(td3_result.batch_size),
    })
    
    # Test case 4: Boolean array indexing
    print("Generating test case 4: Boolean array indexing...")
    data4 = {
        "x": jax.random.normal(keys3[2], (10, 5)),
        "y": jax.random.uniform(keys3[3], (10, 3)),
    }
    ad4 = ArrayDict(data4, batch_size=10)
    td4 = TensorDict({
        ("x",): torch.from_numpy(np.array(data4["x"])),
        ("y",): torch.from_numpy(np.array(data4["y"])),
    }, batch_size=10)
    
    idx4 = jnp.array([i % 2 == 0 for i in range(10)])
    idx4_torch = torch.tensor([i % 2 == 0 for i in range(10)])
    ad4_result = ad4[idx4]
    td4_result = td4[idx4_torch]
    
    test_log.append({
        "case_id": 4,
        "description": "Boolean array indexing",
        "original_arraydict": serialize_arraydict(ad4),
        "original_tensordict_batch_size": list(td4.batch_size),
        "index": serialize_index(idx4),
        "result_arraydict": serialize_arraydict(ad4_result),
        "result_tensordict_batch_size": list(td4_result.batch_size),
        "shapes_match": ad4_result.batch_size == tuple(td4_result.batch_size),
    })
    
    # Test case 5: None (newaxis)
    print("Generating test case 5: None (newaxis)...")
    rng5 = jax.random.PRNGKey(3000)
    keys5 = jax.random.split(rng5, 3)
    
    data5 = {
        "x": jax.random.normal(keys5[0], (8, 6, 4)),
        "y": jax.random.uniform(keys5[1], (8, 6, 2)),
    }
    ad5 = ArrayDict(data5, batch_size=(8, 6))
    td5 = TensorDict({
        ("x",): torch.from_numpy(np.array(data5["x"])),
        ("y",): torch.from_numpy(np.array(data5["y"])),
    }, batch_size=(8, 6))
    
    idx5 = None
    ad5_result = ad5[idx5]
    td5_result = td5[idx5]
    
    test_log.append({
        "case_id": 5,
        "description": "None (newaxis) indexing",
        "original_arraydict": serialize_arraydict(ad5),
        "original_tensordict_batch_size": list(td5.batch_size),
        "index": serialize_index(idx5),
        "result_arraydict": serialize_arraydict(ad5_result),
        "result_tensordict_batch_size": list(td5_result.batch_size),
        "shapes_match": ad5_result.batch_size == tuple(td5_result.batch_size),
    })
    
    # Test case 6: Complex nested structure
    print("Generating test case 6: Complex nested structure...")
    rng6 = jax.random.PRNGKey(4000)
    keys6 = jax.random.split(rng6, 10)
    
    inner_ad = ArrayDict({
        "deep1": jax.random.normal(keys6[0], (10, 8, 3)),
        "deep2": jax.random.uniform(keys6[1], (10, 8, 2)),
    }, batch_size=(10, 8))
    
    data6 = {
        "top": jax.random.normal(keys6[2], (10, 8, 5)),
        "mid": {
            "a": jax.random.normal(keys6[3], (10, 8, 4)),
            "b": jax.random.uniform(keys6[4], (10, 8, 6)),
            "nested": inner_ad,
        },
        "objects": [[f"s{i}{j}" for j in range(8)] for i in range(10)],
    }
    ad6 = ArrayDict(data6, batch_size=(10, 8))
    
    # For tensordict, flatten the structure
    td6 = TensorDict({
        ("top",): torch.from_numpy(np.array(data6["top"])),
        ("mid", "a"): torch.from_numpy(np.array(data6["mid"]["a"])),
        ("mid", "b"): torch.from_numpy(np.array(data6["mid"]["b"])),
        ("mid", "nested", "deep1"): torch.from_numpy(np.array(inner_ad._data[("deep1",)])),
        ("mid", "nested", "deep2"): torch.from_numpy(np.array(inner_ad._data[("deep2",)])),
    }, batch_size=(10, 8))
    
    idx6 = (slice(2, 8), jnp.array([1, 3, 5]))
    idx6_torch = (slice(2, 8), torch.tensor([1, 3, 5]))
    ad6_result = ad6[idx6]
    td6_result = td6[idx6_torch]
    
    test_log.append({
        "case_id": 6,
        "description": "Complex nested structure with mixed indexing",
        "original_arraydict": serialize_arraydict(ad6, max_keys=8),
        "original_tensordict_batch_size": list(td6.batch_size),
        "index": serialize_index(idx6),
        "result_arraydict": serialize_arraydict(ad6_result, max_keys=8),
        "result_tensordict_batch_size": list(td6_result.batch_size),
        "shapes_match": ad6_result.batch_size == tuple(td6_result.batch_size),
    })
    
    # Test case 7: 2D boolean indexing
    print("Generating test case 7: 2D boolean indexing...")
    rng7 = jax.random.PRNGKey(5000)
    keys7 = jax.random.split(rng7, 4)
    
    data7 = {
        "x": jax.random.normal(keys7[0], (6, 8, 4)),
        "y": jax.random.uniform(keys7[1], (6, 8, 3)),
    }
    ad7 = ArrayDict(data7, batch_size=(6, 8))
    td7 = TensorDict({
        ("x",): torch.from_numpy(np.array(data7["x"])),
        ("y",): torch.from_numpy(np.array(data7["y"])),
    }, batch_size=(6, 8))
    
    idx7 = jax.random.uniform(keys7[2], (6, 8)) > 0.6
    idx7_torch = torch.from_numpy(np.array(idx7))
    ad7_result = ad7[idx7]
    td7_result = td7[idx7_torch]
    
    test_log.append({
        "case_id": 7,
        "description": "2D boolean indexing",
        "original_arraydict": serialize_arraydict(ad7),
        "original_tensordict_batch_size": list(td7.batch_size),
        "index": {
            "type": "2d_boolean_array",
            "shape": list(idx7.shape),
            "true_count": int(idx7.sum()),
        },
        "result_arraydict": serialize_arraydict(ad7_result),
        "result_tensordict_batch_size": list(td7_result.batch_size),
        "shapes_match": ad7_result.batch_size == tuple(td7_result.batch_size),
    })
    
    # Test case 8: Multiple None dimensions
    print("Generating test case 8: Multiple None dimensions...")
    data8 = {
        "x": jax.random.normal(keys7[3], (10, 5)),
    }
    ad8 = ArrayDict(data8, batch_size=10)
    td8 = TensorDict({
        ("x",): torch.from_numpy(np.array(data8["x"])),
    }, batch_size=10)
    
    idx8 = (None, slice(2, 7), None)
    ad8_result = ad8[idx8]
    td8_result = td8[idx8]
    
    test_log.append({
        "case_id": 8,
        "description": "Multiple None dimensions",
        "original_arraydict": serialize_arraydict(ad8),
        "original_tensordict_batch_size": list(td8.batch_size),
        "index": serialize_index(idx8),
        "result_arraydict": serialize_arraydict(ad8_result),
        "result_tensordict_batch_size": list(td8_result.batch_size),
        "shapes_match": ad8_result.batch_size == tuple(td8_result.batch_size),
    })
    
    return test_log


def main():
    """Generate test cases and save to file."""
    print("Generating comprehensive test case log...\n")
    test_log = generate_test_cases()
    
    # Save to JSON file
    output_file = "test_cases_log.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_log, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Generated {len(test_log)} test cases")
    print(f"✓ Saved to {output_file}")
    
    # Print summary
    print("\nSummary:")
    for case in test_log:
        match_status = "✓" if case["shapes_match"] else "✗"
        print(f"  {match_status} Case {case['case_id']}: {case['description']}")
    
    # Generate human-readable report
    report_file = "test_cases_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# ArrayDict 高级索引测试用例报告\n\n")
        f.write(f"生成时间: {jax.random.PRNGKey(0)}\n")
        f.write(f"测试用例数量: {len(test_log)}\n\n")
        
        for case in test_log:
            f.write(f"## 测试用例 {case['case_id']}: {case['description']}\n\n")
            
            f.write("### 原始 ArrayDict\n")
            f.write(f"- Batch size: `{case['original_arraydict']['batch_size']}`\n")
            f.write(f"- Keys 数量: {len(case['original_arraydict']['keys'])}\n")
            if 'showing_first' in case['original_arraydict']:
                f.write(f"- (仅显示前 {case['original_arraydict']['showing_first']} 个，共 {case['original_arraydict']['total_keys']} 个)\n")
            f.write("\n**Keys 详情:**\n")
            for key, info in case['original_arraydict']['keys'].items():
                if info.get('type') == 'object_array':
                    f.write(f"- `{key}`: object array, shape={info['shape']}, sample={info['sample']}\n")
                else:
                    f.write(f"- `{key}`: {info['type']}, shape={info['shape']}, dtype={info.get('dtype', 'N/A')}\n")
            
            f.write("\n### 索引操作\n")
            f.write(f"```python\n")
            idx_desc = case['index']
            if idx_desc['type'] == 'slice':
                f.write(f"index = slice({idx_desc['start']}, {idx_desc['stop']}, {idx_desc['step']})\n")
            elif idx_desc['type'] == 'tuple':
                f.write(f"index = (...complex tuple...)\n")
            elif idx_desc['type'] == 'array_index':
                f.write(f"index = jnp.array(...) # shape={idx_desc['shape']}\n")
            elif idx_desc['type'] == '2d_boolean_array':
                f.write(f"index = boolean_array # shape={idx_desc['shape']}, true_count={idx_desc['true_count']}\n")
            else:
                f.write(f"index = {idx_desc['type']}\n")
            f.write(f"```\n\n")
            
            f.write("### 结果\n")
            f.write(f"- ArrayDict batch_size: `{case['result_arraydict']['batch_size']}`\n")
            f.write(f"- TensorDict batch_size: `{case['result_tensordict_batch_size']}`\n")
            f.write(f"- **匹配状态: {'✓ 一致' if case['shapes_match'] else '✗ 不一致'}**\n\n")
            
            f.write("**结果 Keys:**\n")
            for key, info in case['result_arraydict']['keys'].items():
                if info.get('type') == 'object_array':
                    f.write(f"- `{key}`: object array, shape={info['shape']}\n")
                else:
                    f.write(f"- `{key}`: {info['type']}, shape={info['shape']}\n")
            
            f.write("\n---\n\n")
    
    print(f"✓ Generated human-readable report: {report_file}")


if __name__ == "__main__":
    main()
