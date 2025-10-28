#!/usr/bin/env python3
"""
诊断传输方式和性能
"""
import os
import time
import zmq
import msgpack
import numpy as np

def test_transport(address: str, data_size_mb: float = 10.0):
    """测试传输性能"""
    print(f"\n{'='*80}")
    print(f"测试地址: {address}")
    print(f"数据大小: {data_size_mb} MB")
    print(f"{'='*80}")
    
    # 创建测试数据
    num_elements = int(data_size_mb * 1024 * 1024 / 4)  # float32 = 4 bytes
    test_data = np.random.randn(num_elements).astype(np.float32)
    
    # 服务器
    context_server = zmq.Context()
    socket_server = context_server.socket(zmq.REP)
    socket_server.bind(address)
    print(f"✓ 服务器绑定到: {address}")
    
    # 客户端
    context_client = zmq.Context()
    socket_client = context_client.socket(zmq.REQ)
    socket_client.connect(address)
    print(f"✓ 客户端连接到: {address}")
    
    # 预热
    for _ in range(3):
        request = {"data": test_data.tobytes(), "shape": list(test_data.shape)}
        socket_client.send(msgpack.packb(request, use_bin_type=True))
        response_bytes = socket_server.recv()
        socket_server.send(response_bytes)
        socket_client.recv()
    
    print("✓ 预热完成")
    
    # 性能测试
    num_iterations = 10
    times = []
    
    for i in range(num_iterations):
        # 序列化
        t0 = time.perf_counter()
        request = {"data": test_data.tobytes(), "shape": list(test_data.shape)}
        request_bytes = msgpack.packb(request, use_bin_type=True)
        serialize_time = time.perf_counter() - t0
        
        # 发送
        t0 = time.perf_counter()
        socket_client.send(request_bytes)
        send_time = time.perf_counter() - t0
        
        # 服务器接收
        t0 = time.perf_counter()
        response_bytes = socket_server.recv()
        recv_time = time.perf_counter() - t0
        
        # 服务器发送
        socket_server.send(response_bytes)
        
        # 客户端接收
        t0 = time.perf_counter()
        response_bytes = socket_client.recv()
        client_recv_time = time.perf_counter() - t0
        
        # 反序列化
        t0 = time.perf_counter()
        response = msgpack.unpackb(response_bytes, raw=False)
        deserialize_time = time.perf_counter() - t0
        
        total_time = serialize_time + send_time + recv_time + client_recv_time + deserialize_time
        times.append({
            "total": total_time,
            "serialize": serialize_time,
            "send": send_time,
            "recv": recv_time,
            "client_recv": client_recv_time,
            "deserialize": deserialize_time,
        })
    
    # 统计
    avg_times = {
        key: np.mean([t[key] for t in times]) * 1000  # 转换为毫秒
        for key in times[0].keys()
    }
    
    print(f"\n性能统计 (平均 {num_iterations} 次):")
    print(f"  序列化:     {avg_times['serialize']:8.3f} ms")
    print(f"  发送:       {avg_times['send']:8.3f} ms")
    print(f"  服务器接收: {avg_times['recv']:8.3f} ms")
    print(f"  客户端接收: {avg_times['client_recv']:8.3f} ms")
    print(f"  反序列化:   {avg_times['deserialize']:8.3f} ms")
    print(f"  总计:       {avg_times['total']:8.3f} ms")
    print(f"\n  吞吐量:     {data_size_mb / (avg_times['total'] / 1000):.2f} MB/s")
    print(f"  往返延迟:   {avg_times['total']:.3f} ms")
    
    # 清理
    socket_client.close()
    socket_server.close()
    context_client.term()
    context_server.term()
    
    return avg_times['total']


def main():
    """主函数"""
    print("ZeroMQ 传输性能诊断")
    print("="*80)
    
    # 测试不同传输方式
    tests = [
        ("IPC", "ipc:///tmp/test_zmq_perf.ipc", 10.0),
        ("TCP (localhost)", "tcp://127.0.0.1:15555", 10.0),
    ]
    
    results = []
    
    for name, address, data_size in tests:
        try:
            # 清理旧的 IPC 文件
            if "ipc://" in address:
                ipc_file = address.replace("ipc://", "")
                if os.path.exists(ipc_file):
                    os.remove(ipc_file)
            
            latency = test_transport(address, data_size)
            results.append((name, latency))
            
            # 清理 IPC 文件
            if "ipc://" in address:
                ipc_file = address.replace("ipc://", "")
                if os.path.exists(ipc_file):
                    os.remove(ipc_file)
            
            time.sleep(0.5)  # 等待端口释放
            
        except Exception as e:
            print(f"\n✗ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print(f"\n{'='*80}")
    print("性能对比")
    print(f"{'='*80}")
    
    if len(results) >= 2:
        ipc_latency = results[0][1]
        tcp_latency = results[1][1]
        speedup = tcp_latency / ipc_latency
        
        print(f"IPC 延迟:  {ipc_latency:.3f} ms")
        print(f"TCP 延迟:  {tcp_latency:.3f} ms")
        print(f"IPC 比 TCP 快: {speedup:.1f} 倍")
        
        print(f"\n预期性能:")
        print(f"  如果当前 RPC 延迟是 332ms (TCP)")
        print(f"  切换到 IPC 后应该是: {332 / speedup:.3f} ms")
    
    print(f"\n{'='*80}")
    print("诊断建议:")
    print(f"{'='*80}")
    
    if len(results) >= 1:
        ipc_latency = results[0][1]
        if ipc_latency > 10:
            print("⚠️  IPC 延迟过高 (>10ms)，可能的原因:")
            print("  1. 系统负载过高")
            print("  2. 数据量过大，序列化成为瓶颈")
            print("  3. 需要使用共享内存零拷贝")
        elif ipc_latency > 1:
            print("⚠️  IPC 延迟偏高 (>1ms)，建议:")
            print("  1. 减少数据传输量（使用 float16/bfloat16）")
            print("  2. 使用共享内存零拷贝")
        else:
            print("✓ IPC 性能正常")
    
    print(f"\n如果实际 RPC 延迟是 332ms，说明:")
    print("  1. 可能使用了 TCP 而不是 IPC")
    print("  2. 检查环境变量 LLAMA_IPC_PATH")
    print("  3. 检查 IPC 文件是否创建成功")
    print("  4. 查看服务器和客户端的连接日志")


if __name__ == "__main__":
    main()
