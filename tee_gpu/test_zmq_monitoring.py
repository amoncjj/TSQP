#!/usr/bin/env python3
"""
测试ZeroMQ性能监控功能
创建模拟的RPC调用来验证日志记录
"""
import time
import zmq
import msgpack
import numpy as np
from tee_runner_optimized import GPUClient


def test_monitoring():
    """测试监控功能"""
    print("="*80)
    print("Testing ZeroMQ Performance Monitoring")
    print("="*80)
    
    # 创建测试客户端
    print("\n1. Creating test client...")
    
    # 注意: 这需要server_optimized.py正在运行
    try:
        client = GPUClient(
            ipc_path="ipc:///tmp/tsqp_gpu_server.ipc",
            log_file="test_zmq_performance.log"
        )
        print("✓ Client created successfully")
    except Exception as e:
        print(f"✗ Failed to create client: {e}")
        print("\nPlease start the server first:")
        print("  python server_optimized.py")
        return
    
    # 测试Init调用
    print("\n2. Testing Init call...")
    try:
        init_data = client.init()
        print(f"✓ Init successful, got {len(init_data)} config items")
    except Exception as e:
        print(f"✗ Init failed: {e}")
        return
    
    # 测试Embedding调用
    print("\n3. Testing Embedding call...")
    try:
        import torch
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        embeddings = client.embedding(input_ids)
        print(f"✓ Embedding successful, shape: {embeddings.shape}")
    except Exception as e:
        print(f"✗ Embedding failed: {e}")
    
    # 测试BatchLinear调用
    print("\n4. Testing BatchLinear call...")
    try:
        hidden_states = torch.randn(1, 10, 2048)
        outputs = client.batch_linear(0, ["q_proj", "k_proj"], hidden_states)
        print(f"✓ BatchLinear successful, got {len(outputs)} outputs")
    except Exception as e:
        print(f"✗ BatchLinear failed: {e}")
    
    # 打印统计
    print("\n5. Printing statistics...")
    client.print_stats()
    
    # 关闭客户端
    client.close()
    
    print("\n6. Analyzing log file...")
    import subprocess
    try:
        result = subprocess.run(
            ["python", "analyze_zmq_performance.py", "test_zmq_performance.log"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        print("\nYou can manually run:")
        print("  python analyze_zmq_performance.py test_zmq_performance.log")
    
    print("\n" + "="*80)
    print("Test completed!")
    print("Log file: test_zmq_performance.log")
    print("="*80)


if __name__ == "__main__":
    test_monitoring()
