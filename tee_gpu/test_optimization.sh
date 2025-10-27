#!/bin/bash

echo "=========================================="
echo "TSQP Performance Optimization Test"
echo "=========================================="
echo ""

# 检查 CUDA
echo "1. Checking CUDA..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# 清理旧的 IPC 文件
echo "2. Cleaning up old IPC files..."
rm -f /tmp/tsqp_gpu_server.ipc
echo "✓ Cleaned"
echo ""

# 启动优化的服务器（后台）
echo "3. Starting optimized GPU server..."
python3 server_optimized.py &
SERVER_PID=$!
echo "✓ Server started (PID: $SERVER_PID)"
sleep 3
echo ""

# 检查 IPC 文件
echo "4. Checking IPC connection..."
if [ -e /tmp/tsqp_gpu_server.ipc ]; then
    echo "✓ IPC file exists: /tmp/tsqp_gpu_server.ipc"
    ls -lh /tmp/tsqp_gpu_server.ipc
else
    echo "✗ IPC file not found!"
    kill $SERVER_PID
    exit 1
fi
echo ""

# 运行优化的客户端
echo "5. Running optimized client..."
echo "=========================================="
python3 tee_runner_optimized.py
echo "=========================================="
echo ""

# 清理
echo "6. Cleaning up..."
kill $SERVER_PID
rm -f /tmp/tsqp_gpu_server.ipc
echo "✓ Server stopped"
echo ""

echo "=========================================="
echo "Test completed!"
echo "=========================================="
