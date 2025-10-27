#!/usr/bin/env python3
"""
简单的连接测试脚本
用于验证 ZeroMQ 服务器是否正常运行
"""
import zmq
import msgpack
import sys

def test_connection(endpoint: str = "localhost:50051") -> bool:
    """测试与服务器的连接"""
    print(f"Testing connection to {endpoint}...")
    
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时
        socket.connect(f"tcp://{endpoint}")
        
        # 发送一个简单的测试请求
        test_message = {
            "method": "RegisterClient",
            "request": {"module_names": []}
        }
        
        message_bytes = msgpack.packb(test_message, use_bin_type=True)
        socket.send(message_bytes)
        
        # 接收响应
        response_bytes = socket.recv()
        response = msgpack.unpackb(response_bytes, raw=False)
        
        socket.close()
        context.term()
        
        if response.get("status") == "success":
            print("✓ Connection successful!")
            print(f"  Server response: {response}")
            return True
        else:
            print("✗ Connection failed!")
            print(f"  Error: {response.get('error', 'Unknown error')}")
            return False
            
    except zmq.error.Again:
        print("✗ Connection timeout!")
        print("  Make sure the server is running.")
        return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False

if __name__ == "__main__":
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "localhost:50051"
    success = test_connection(endpoint)
    sys.exit(0 if success else 1)
