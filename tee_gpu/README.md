# TSQP - TEE + GPU 协同推理

基于 ZeroMQ 的 TEE 与 GPU 分离式 LLaMA 模型推理系统，用于 Prefill 阶段性能测试。

## 架构概述

```
┌─────────────────┐         ZeroMQ          ┌─────────────────┐
│   TEE Client    │ ◄──────────────────────► │   GPU Server    │
│  (tee_runner)   │    msgpack 序列化        │    (server)     │
├─────────────────┤                          ├─────────────────┤
│ • 非线性层      │                          │ • Linear 层     │
│ • RMSNorm       │                          │ • Embedding 层  │
│ • Attention     │                          │ • Matmul 加速   │
│ • 控制流程      │                          │ • GPU 计算      │
└─────────────────┘                          └─────────────────┘
```

## 核心文件

- `server.py` - GPU 服务端，托管 Linear/Embedding 层
- `tee_runner.py` - TEE 客户端，执行 prefill 测试
- `Makefile` - Gramine 构建脚本
- `*.manifest.template` - Gramine SGX 配置模板

## 快速开始

### 1. 安装依赖

```bash
pip install -r ../requirements.txt
```

### 2. 启动 GPU 服务器

```bash
# 设置环境变量（可选）
export LLAMA_MODEL_PATH="/path/to/llama/model"
export LLAMA_GPU_DEVICE="cuda:0"
export LLAMA_GPU_PORT="50051"

# 启动服务
python server.py
```

### 3. 运行 TEE 客户端测试

```bash
# 设置环境变量（可选）
export LLAMA_MODEL_PATH="/path/to/llama/model"
export LLAMA_GPU_ENDPOINT="localhost:50051"

# 运行 prefill 测试
python tee_runner.py
```

## 环境变量

### 服务端 (server.py)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLAMA_MODEL_PATH` | `/home/.../llama3.2-1b` | 模型路径 |
| `LLAMA_GPU_DEVICE` | `cuda:0` | GPU 设备 |
| `LLAMA_DTYPE` | `float32` | 数据类型 |
| `LLAMA_GPU_PORT` | `50051` | 服务端口 |

### 客户端 (tee_runner.py)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLAMA_MODEL_PATH` | `meta-llama/Llama-3.2-1B-Instruct` | 模型路径 |
| `LLAMA_GPU_ENDPOINT` | `localhost:50051` | 服务器地址 |

## Prefill 测试配置

在 `tee_runner.py` 中修改：

```python
PREFILL_TOKEN_LENGTH = 128  # 修改 token 长度
```

## 通信协议

使用 ZeroMQ REQ-REP 模式 + msgpack 序列化：

### 支持的 RPC 方法

1. **RegisterClient** - 注册模块
2. **Forward** - 前向传播（Linear/Embedding）
3. **Matmul** - 矩阵乘法（Attention）
4. **FetchNonLinearTensors** - 获取非线性层参数

## Gramine SGX 构建

```bash
# 生成 manifest
make

# 生成 SGX 签名
make SGX=1

# 清理
make clean
```

## 性能测试输出

```
============================================================
Running Prefill Benchmark
============================================================
Token length: 128
============================================================
Prefill time: 2.3456 seconds
Throughput: 54.56 tokens/sec
============================================================
```

## 项目特点

- ✅ 使用 ZeroMQ 替代 gRPC，更轻量级
- ✅ 使用 msgpack 序列化，无需 protobuf
- ✅ 专注于 prefill 阶段测试
- ✅ 清晰的模块分离架构
- ✅ 支持 Gramine SGX 环境

## 注意事项

1. 确保 GPU 服务器先启动
2. 模型路径需要包含配置文件和权重
3. 客户端和服务端的模型路径应一致
4. 默认端口 50051，确保未被占用
