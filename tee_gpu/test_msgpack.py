#!/usr/bin/env python3
"""
测试 msgpack 序列化
"""
import msgpack
import numpy as np

print("Testing msgpack serialization...")

# 测试 1: numpy array (会失败)
try:
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    data = {"array": arr}
    packed = msgpack.packb(data, use_bin_type=True)
    print("✗ Test 1 FAILED: numpy array should not be serializable")
except Exception as e:
    print(f"✓ Test 1 PASSED: numpy array cannot be serialized (expected)")
    print(f"  Error: {type(e).__name__}")

# 测试 2: bytes (应该成功)
try:
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    data = {
        "buffer": arr.tobytes(),
        "shape": list(arr.shape),
        "dtype": "float32"
    }
    packed = msgpack.packb(data, use_bin_type=True)
    unpacked = msgpack.unpackb(packed, raw=False)
    
    # 重建数组
    arr_restored = np.frombuffer(unpacked["buffer"], dtype=np.float32).reshape(unpacked["shape"])
    
    if np.allclose(arr, arr_restored):
        print("✓ Test 2 PASSED: bytes serialization works")
        print(f"  Original: {arr}")
        print(f"  Restored: {arr_restored}")
    else:
        print("✗ Test 2 FAILED: arrays don't match")
except Exception as e:
    print(f"✗ Test 2 FAILED: {e}")

# 测试 3: 嵌套字典
try:
    data = {
        "config": {
            "num_layers": 22,
            "hidden_size": 2048,
        },
        "weights": {
            "layer_0": {
                "weight": np.random.randn(2048).astype(np.float32).tobytes(),
                "shape": [2048],
                "eps": 1e-6,
            }
        }
    }
    packed = msgpack.packb(data, use_bin_type=True)
    unpacked = msgpack.unpackb(packed, raw=False)
    
    print("✓ Test 3 PASSED: nested dict with bytes works")
    print(f"  Packed size: {len(packed)} bytes")
except Exception as e:
    print(f"✗ Test 3 FAILED: {e}")

print("\nAll tests completed!")
