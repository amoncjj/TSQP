#!/bin/bash
# 快速测试脚本 - 共享内存优化版本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}共享内存优化版本测试${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查环境变量
if [ -z "$LLAMA_MODEL_PATH" ]; then
    echo -e "${YELLOW}警告: LLAMA_MODEL_PATH 未设置，使用默认路径${NC}"
    export LLAMA_MODEL_PATH="/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b"
fi

if [ -z "$LLAMA_IPC_PATH" ]; then
    export LLAMA_IPC_PATH="ipc:///tmp/tsqp_gpu_server.ipc"
fi

echo -e "${GREEN}配置:${NC}"
echo -e "  模型路径: ${LLAMA_MODEL_PATH}"
echo -e "  IPC路径:  ${LLAMA_IPC_PATH}"
echo -e "  GPU设备:  ${LLAMA_GPU_DEVICE:-cuda:0}"
echo ""

# 清理旧的IPC文件
if [ -e "/tmp/tsqp_gpu_server.ipc" ]; then
    echo -e "${YELLOW}清理旧的IPC文件...${NC}"
    rm -f /tmp/tsqp_gpu_server.ipc
fi

# 启动服务端（后台）
echo -e "${GREEN}启动GPU服务端...${NC}"
python tee_gpu/server_optimized.py &
SERVER_PID=$!

# 等待服务端启动
sleep 3

# 检查服务端是否运行
if ! ps -p $SERVER_PID > /dev/null; then
    echo -e "${RED}错误: 服务端启动失败${NC}"
    exit 1
fi

echo -e "${GREEN}服务端已启动 (PID: $SERVER_PID)${NC}"
echo ""

# 运行客户端
echo -e "${GREEN}运行TEE客户端...${NC}"
echo ""
python tee_gpu/tee_runner_optimized.py

# 清理
echo ""
echo -e "${YELLOW}清理...${NC}"
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo -e "${GREEN}测试完成！${NC}"
echo ""
echo -e "${GREEN}查看详细日志:${NC}"
echo -e "  cat zmq_performance.log"
