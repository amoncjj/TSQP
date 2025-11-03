#!/usr/bin/env python3
"""
测试共享内存通信
验证小数据(<10MB)使用共享内存，大数据使用ZeroMQ
"""
import numpy as np
import torch

# 测试数据大小
test_cases = [
    ("Small - 1KB", (1, 256)),           # ~1KB
    ("Medium - 100KB", (1, 25600)),      # ~100KB
    ("Large - 1MB", (1, 256000)),        # ~1MB
    ("Very Large - 5MB", (1, 1280000)),  # ~5MB
    ("Huge - 15MB", (1, 3840000)),       # ~15MB (超过10MB阈值)
]

print("="*70)
print("Shared Memory Communication Test")
print("="*70)
print(f"Threshold: 10MB")
print(f"Expected behavior:")
print(f"  - Data < 10MB: Use shared memory ring buffer")
print(f"  - Data >= 10MB: Use ZeroMQ")
print("="*70)

for name, shape in test_cases:
    tensor = torch.randn(*shape, dtype=torch.float32)
    size_bytes = tensor.numel() * 4  # float32 = 4 bytes
    size_mb = size_bytes / 1024 / 1024
    
    expected_method = "Shared Memory" if size_bytes < 10 * 1024 * 1024 else "ZeroMQ"
    
    print(f"\n{name}:")
    print(f"  Shape: {shape}")
    print(f"  Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    print(f"  Expected: {expected_method}")

print("\n" + "="*70)
print("To run the actual test:")
print("1. Start server: python tee_gpu/server_optimized.py")
print("2. Run client:   python tee_gpu/tee_runner_optimized.py")
print("="*70)
