# TSQP - TEE-Separated Query Processing

**最新版本**: v2.3  
**更新日期**: 2025-10-27

## 🎯 项目目标

将 LLM 推理计算分离到 TEE 和 GPU，实现安全的隐私计算：
- **TEE 端**: 执行 Softmax, RMSNorm, RotaryEmbedding, SiLU
- **GPU 端**: 执行所有 Linear, Embedding, Matmul 操作
- **通信**: 使用 ZeroMQ IPC 实现高性能通信

## 🚀 快速开始

### 环境要求

```bash
Python 3.8+
CUDA GPU
pip install -r requirements.txt
```

### 运行优化版本（推荐）

```bash
cd tee_gpu

# 终端 1: 启动 GPU 服务器
export LLAMA_DTYPE="bfloat16"  # 可选，提升性能
python server_optimized.py

# 终端 2: 运行 TEE 客户端
export LLAMA_DTYPE="bfloat16"  # 与服务器保持一致
python tee_runner_optimized.py
```

## 📊 当前状态

### ✅ 已修复的问题

- **v2.2**: msgpack 序列化错误
- **v2.3**: 梯度追踪错误

### 🔴 性能问题

**当前性能**:
- 总时间: 44.25 秒 (1024 tokens)
- RPC 延迟: 332ms/call
- 数据传输: 3.3GB

**问题**: RPC 延迟应该是 0.2ms（IPC），但实际是 332ms，慢了 **1660 倍**！

### 🎯 优化目标

**目标**: 从 44 秒优化到 **1-2 秒**，提升 **22-44 倍**！

## 🔧 立即优化

### 步骤 1: 诊断传输方式

```bash
cd tee_gpu
python diagnose_transport.py
```

这会告诉你实际使用的是 IPC 还是 TCP。

### 步骤 2: 确保使用 IPC

```bash
# 停止所有进程
pkill -f server_optimized
pkill -f tee_runner_optimized

# 清理 IPC 文件
rm -f /tmp/tsqp_gpu_server.ipc

# 启动服务器
python server_optimized.py

# 在另一个终端启动客户端
python tee_runner_optimized.py
```

### 步骤 3: 使用 bfloat16

```bash
export LLAMA_DTYPE="bfloat16"
python server_optimized.py &
python tee_runner_optimized.py
```

**预期效果**: 44s → 15-20s (2-3 倍提升)

## 📚 文档

### 技术文档
- **`PERFORMANCE_ANALYSIS.md`** - 深度性能分析和优化方案
- **`OPTIMIZATION_ROADMAP.md`** - 完整优化路线图
- **`BUGFIX_MSGPACK.md`** - msgpack 序列化错误修复
- **`BUGFIX_NO_GRAD.md`** - 梯度追踪错误修复

### 操作指南
- **`QUICK_REFERENCE.md`** - 快速参考卡片
- **`TROUBLESHOOTING.md`** - 故障排查指南
- **`SUMMARY.md`** - 项目完成总结

### 项目管理
- **`STATUS.md`** - 项目状态报告
- **`CHANGELOG.md`** - 变更日志

## 🛠️ 工具

### 诊断工具
- **`diagnose_transport.py`** - 传输性能诊断
- **`test_simple.py`** - RPC 功能测试

## 📈 优化路线图

| 阶段 | 优化内容 | 预期时间 | 提升倍数 |
|------|---------|---------|---------|
| 当前 | 无 | 44.25s | 1x |
| 阶段1 | IPC + bfloat16 | 15-20s | 2-3x |
| 阶段2 | 共享内存 + 合并 | 3-5s | 9-15x |
| 阶段3 | GPU优化 + Pipeline | 1-2s | 22-44x |

## 🔍 性能分析

### 当前瓶颈

1. **RPC 通信** (73.67%)
   - 延迟: 332ms/call (应该是 0.2ms)
   - 序列化: 46ms/call
   - 反序列化: 25ms/call

2. **GPU 计算** (98.25%)
   - Matmul: 29.54s (66.75%)
   - Linear: 13.92s (31.45%)

3. **数据传输**
   - 发送: 3328 MB
   - 接收: 3657 MB

### 优化方向

1. **确保使用 IPC** → RPC 延迟 332ms → 0.2ms (1660x)
2. **使用 bfloat16** → 数据量减半 (2x)
3. **共享内存零拷贝** → 序列化开销几乎为 0 (100x+)
4. **整层合并** → RPC 次数减少 (1.5-2x)
5. **GPU Kernel 融合** → GPU 计算提升 (2-4x)

## 🎓 最佳实践

### 1. 推理代码必须使用 `@torch.no_grad()`

```python
@torch.no_grad()
def forward(self, input_ids):
    # 所有推理代码
    pass
```

### 2. 使用 IPC 而不是 TCP

```python
# 正确
ipc_path = "ipc:///tmp/tsqp_gpu_server.ipc"

# 错误（慢 50-500 倍）
tcp_path = "tcp://127.0.0.1:5555"
```

### 3. 使用 bfloat16 进行推理

```bash
export LLAMA_DTYPE="bfloat16"
```

## 🐛 常见问题

### Q1: RPC 延迟很高（>100ms）

**A**: 可能使用了 TCP 而不是 IPC
- 运行 `diagnose_transport.py` 诊断
- 检查环境变量 `LLAMA_IPC_PATH`
- 确认 IPC 文件存在: `ls -la /tmp/tsqp_gpu_server.ipc`

### Q2: 出现 "Can't call numpy() on Tensor that requires grad"

**A**: 已在 v2.3 修复
- 确保使用最新代码
- `forward()` 方法有 `@torch.no_grad()` 装饰器

### Q3: msgpack 序列化错误

**A**: 已在 v2.2 修复
- 确保使用最新代码
- 所有 numpy 类型已转换为 Python 基本类型

## 📞 获取帮助

如果遇到问题，请提供：
1. `diagnose_transport.py` 的输出
2. 服务器和客户端的日志
3. 性能统计输出
4. 环境信息（Python 版本、PyTorch 版本、CUDA 版本）

## 📝 版本历史

- **v2.3** (2025-10-27): 修复梯度追踪错误，添加性能分析
- **v2.2** (2025-10-27): 修复 msgpack 序列化错误
- **v2.1** (2025-10-27): 代码精简与优化
- **v2.0** (2025-10-27): 从 gRPC 迁移到 ZeroMQ

## 🎯 下一步

1. **立即**: 运行诊断脚本，确认使用 IPC
2. **今天**: 切换到 bfloat16，测试性能
3. **本周**: 实现共享内存零拷贝
4. **下周**: GPU Kernel 融合，异步 Pipeline

**目标**: 1-2 秒完成 1024 tokens 推理！🚀
