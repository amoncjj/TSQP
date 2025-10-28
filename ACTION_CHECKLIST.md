# 立即行动清单

## 🎯 目标

将推理时间从 44 秒优化到 1-2 秒（22-44 倍提升）

## ✅ 今天必须完成的任务

### 任务 1: 诊断传输方式 (10 分钟)

```bash
cd /home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu
python diagnose_transport.py
```

**预期输出**:
```
IPC 延迟:  0.5-2 ms
TCP 延迟:  50-100 ms
IPC 比 TCP 快: 50-100 倍
```

**如果 IPC 延迟 > 10ms**: 有问题，需要检查系统配置

---

### 任务 2: 确认使用 IPC (5 分钟)

```bash
# 停止所有进程
pkill -f server_optimized
pkill -f tee_runner_optimized

# 清理 IPC 文件
rm -f /tmp/tsqp_gpu_server.ipc

# 确保没有环境变量覆盖
unset LLAMA_IPC_PATH

# 启动服务器
python server_optimized.py
```

**检查服务器输出**:
```
✓ ZeroMQ server started on ipc:///tmp/tsqp_gpu_server.ipc
✓ Using IPC for zero-copy local communication
```

**在另一个终端启动客户端**:
```bash
python tee_runner_optimized.py
```

**检查客户端输出**:
```
✓ Connected to GPU server at ipc:///tmp/tsqp_gpu_server.ipc
```

**检查性能统计**:
```
RPC Time: 0.02-0.1s (0.2-1ms/call)  ← 应该在这个范围
```

**如果 RPC 延迟仍然是 332ms**: 说明没有使用 IPC，继续排查

---

### 任务 3: 切换到 bfloat16 (2 分钟)

```bash
# 停止所有进程
pkill -f server_optimized
pkill -f tee_runner_optimized

# 设置环境变量
export LLAMA_DTYPE="bfloat16"

# 启动服务器
python server_optimized.py &

# 启动客户端
python tee_runner_optimized.py
```

**检查服务器输出**:
```
Device: cuda:0, Dtype: torch.bfloat16  ← 应该是 bfloat16
```

**检查性能统计**:
```
Data Sent:    1664 MB  ← 应该是原来的一半
Data Received: 1828 MB
```

---

### 任务 4: 记录性能提升 (1 分钟)

记录优化前后的性能数据：

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 总时间 | 44.25s | ___s | ___x |
| RPC 延迟 | 332ms | ___ms | ___x |
| 数据发送 | 3328MB | ___MB | ___x |
| 数据接收 | 3657MB | ___MB | ___x |

---

## 🔍 故障排查

### 问题 1: IPC 文件不存在

```bash
ls -la /tmp/tsqp_gpu_server.ipc
```

**如果不存在**:
- 检查服务器是否正常启动
- 检查是否有权限创建文件
- 尝试手动创建: `touch /tmp/tsqp_gpu_server.ipc`

---

### 问题 2: RPC 延迟仍然很高

**检查实际连接**:
```bash
lsof -p $(pgrep -f server_optimized) | grep socket
lsof -p $(pgrep -f tee_runner_optimized) | grep socket
```

**检查环境变量**:
```bash
echo $LLAMA_IPC_PATH
# 应该是空的或 ipc:///tmp/tsqp_gpu_server.ipc
```

**添加调试日志**:
在 `tee_runner_optimized.py` 的 `GPUClient.__init__()` 中添加:
```python
print(f"  Actual IPC path: {ipc_path}")
print(f"  Transport: {'IPC' if 'ipc://' in ipc_path else 'TCP'}")
```

---

### 问题 3: 数据量没有减半

**检查 dtype**:
```bash
# 在服务器输出中查找
grep "Dtype" server_output.log
```

**确认环境变量**:
```bash
echo $LLAMA_DTYPE
# 应该是 bfloat16
```

**重启服务器**:
```bash
pkill -f server_optimized
export LLAMA_DTYPE="bfloat16"
python server_optimized.py
```

---

## 📊 预期结果

### 如果一切正常

**阶段 1 完成后**:
```
总时间: 15-20 秒 (从 44 秒)
RPC 延迟: 0.2-1 ms (从 332 ms)
数据传输: 1.6-1.8 GB (从 3.3 GB)

提升: 2-3 倍
```

### 如果仍然很慢

**可能原因**:
1. 系统负载过高
2. 磁盘 I/O 瓶颈
3. 网络配置问题
4. ZeroMQ 版本过低

**进一步诊断**:
```bash
# 检查系统负载
top

# 检查 ZeroMQ 版本
python -c "import zmq; print(zmq.zmq_version())"

# 检查磁盘 I/O
iostat -x 1 5
```

---

## 📝 完成检查清单

- [ ] 运行 `diagnose_transport.py`
- [ ] 确认 IPC 延迟 < 10ms
- [ ] 确认服务器绑定到 IPC 地址
- [ ] 确认客户端连接到 IPC 地址
- [ ] 确认 RPC 延迟 < 1ms
- [ ] 设置 `LLAMA_DTYPE="bfloat16"`
- [ ] 确认服务器使用 bfloat16
- [ ] 确认数据量减半
- [ ] 记录性能提升数据
- [ ] 总时间 < 25 秒

---

## 🚀 下一步

完成今天的任务后：

1. **本周**: 实现共享内存零拷贝
2. **下周**: GPU Kernel 融合
3. **最终**: 1-2 秒完成推理

---

## 📞 需要帮助？

如果遇到问题，提供以下信息：

1. **诊断输出**:
```bash
python diagnose_transport.py > diagnostic.log 2>&1
```

2. **服务器日志**:
```bash
python server_optimized.py > server.log 2>&1
```

3. **客户端日志**:
```bash
python tee_runner_optimized.py > client.log 2>&1
```

4. **环境信息**:
```bash
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import zmq; print(f'ZeroMQ: {zmq.zmq_version()}')"
nvidia-smi
```

---

**现在就开始！第一步：运行诊断脚本！** 🎯
