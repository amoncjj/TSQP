#!/usr/bin/env python3
"""
简单测试脚本 - 用于诊断问题
"""
import os
import sys
import zmq
import msgpack
import numpy as np
import torch

# 配置
IPC_PATH = "ipc:///tmp/tsqp_gpu_server.ipc"

def test_connection():
    """测试连接"""
    print("=" * 80)
    print("测试 1: ZeroMQ 连接")
    print("=" * 80)
    
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 秒超时
        socket.connect(IPC_PATH)
        print(f"✓ 连接到 {IPC_PATH}")
        
        # 测试 Init
        print("\n测试 Init 请求...")
        request = {"method": "Init", "request": {}}
        socket.send(msgpack.packb(request, use_bin_type=True))
        
        response_bytes = socket.recv()
        response = msgpack.unpackb(response_bytes, raw=False)
        
        if response["status"] == "success":
            print("✓ Init 成功")
            config = response["response"]["config"]
            print(f"  - num_layers: {config['num_layers']}")
            print(f"  - hidden_size: {config['hidden_size']}")
            print(f"  - num_heads: {config['num_heads']}")
        else:
            print(f"✗ Init 失败: {response.get('error', 'Unknown error')}")
            if "traceback" in response:
                print(response["traceback"])
            return False
        
        socket.close()
        context.term()
        return True
        
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding():
    """测试 Embedding"""
    print("\n" + "=" * 80)
    print("测试 2: Embedding")
    print("=" * 80)
    
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 5000)
        socket.connect(IPC_PATH)
        
        # 创建测试输入
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
        print(f"输入: {input_ids.shape}")
        
        request = {
            "method": "Embedding",
            "request": {
                "buffer": input_ids.numpy().tobytes(),
                "shape": list(input_ids.shape),
            }
        }
        
        socket.send(msgpack.packb(request, use_bin_type=True))
        response_bytes = socket.recv()
        response = msgpack.unpackb(response_bytes, raw=False)
        
        if response["status"] == "success":
            output_shape = response["response"]["shape"]
            print(f"✓ Embedding 成功")
            print(f"  输出形状: {output_shape}")
        else:
            print(f"✗ Embedding 失败: {response.get('error', 'Unknown error')}")
            if "traceback" in response:
                print(response["traceback"])
            return False
        
        socket.close()
        context.term()
        return True
        
    except Exception as e:
        print(f"✗ Embedding 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_linear():
    """测试 BatchLinear"""
    print("\n" + "=" * 80)
    print("测试 3: BatchLinear")
    print("=" * 80)
    
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 5000)
        socket.connect(IPC_PATH)
        
        # 创建测试输入 (batch_size=1, seq_len=5, hidden_size=2048)
        hidden_states = torch.randn(1, 5, 2048, dtype=torch.float32)
        print(f"输入: {hidden_states.shape}")
        
        request = {
            "method": "BatchLinear",
            "request": {
                "layer_idx": 0,
                "module_names": ["q_proj", "k_proj", "v_proj"],
                "hidden_states": {
                    "buffer": hidden_states.numpy().tobytes(),
                    "shape": list(hidden_states.shape),
                }
            }
        }
        
        print("发送请求...")
        socket.send(msgpack.packb(request, use_bin_type=True))
        
        print("等待响应...")
        response_bytes = socket.recv()
        response = msgpack.unpackb(response_bytes, raw=False)
        
        if response["status"] == "success":
            outputs = response["response"]["outputs"]
            print(f"✓ BatchLinear 成功")
            print(f"  输出数量: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"  输出 {i}: {output['shape']}")
        else:
            print(f"✗ BatchLinear 失败: {response.get('error', 'Unknown error')}")
            if "traceback" in response:
                print("\n完整错误堆栈:")
                print(response["traceback"])
            return False
        
        socket.close()
        context.term()
        return True
        
    except Exception as e:
        print(f"✗ BatchLinear 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("TSQP 诊断测试")
    print("=" * 80)
    
    # 检查服务器是否运行
    if not os.path.exists("/tmp/tsqp_gpu_server.ipc"):
        print("⚠️  警告: IPC 文件不存在，服务器可能未运行")
        print("   请先运行: python server_optimized.py")
        return
    
    # 运行测试
    tests = [
        ("连接测试", test_connection),
        ("Embedding 测试", test_embedding),
        ("BatchLinear 测试", test_batch_linear),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except KeyboardInterrupt:
            print("\n\n用户中断")
            break
        except Exception as e:
            print(f"\n✗ {name} 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")


if __name__ == "__main__":
    main()
