# TSQP - TEE + GPU 协同推理框架

基于 Gramine 的 LLaMA 模型 TEE 推理系统，使用 ZeroMQ 实现 TEE 与 GPU 的高效通信。

## 项目概述

本项目演示如何将 LLaMA 模型部署在可信执行环境（TEE）中，同时将计算密集型的 Linear 和 Embedding 层卸载到 GPU，实现安全性与性能的平衡。

### 核心特性

- ✅ **TEE + GPU 分离架构** - Linear/Embedding 层在 GPU，非线性层在 TEE
- ✅ **ZeroMQ 通信** - 轻量级、高效的进程间通信
- ✅ **Prefill 性能测试** - 专注于推理的 prefill 阶段
- ✅ **Gramine SGX 支持** - 可在 Intel SGX 环境中运行

## 目录结构

```
TSQP/
├── tee_gpu/                    # TEE + GPU 协同推理
│   ├── server.py               # GPU 服务端（托管 Linear/Embedding）
│   ├── tee_runner.py           # TEE 客户端（prefill 测试）
│   ├── Makefile                # Gramine 构建脚本
│   ├── *.manifest.template     # SGX 配置模板
│   └── README.md               # 详细使用说明
├── tee_only_llama/             # 纯 TEE 推理（对比基线）
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `pyzmq` - ZeroMQ Python 绑定
- `msgpack` - 高效序列化
- `torch` - PyTorch 深度学习框架
- `transformers` - HuggingFace 模型库

### 2. 准备模型

下载 LLaMA 模型权重并设置路径：

```bash
export LLAMA_MODEL_PATH="/path/to/llama-3.2-1b"
```

### 3. 启动 GPU 服务器

```bash
cd tee_gpu
python server.py
```

输出示例：
```
Loading model from: /path/to/llama-3.2-1b
Device: cuda:0, Dtype: torch.float32
✓ Registered 154 remote modules
✓ ZeroMQ server started on port 50051
```

### 4. 运行 TEE 客户端测试

在另一个终端：

```bash
cd tee_gpu
python tee_runner.py
```

输出示例：
```
✓ Connected to server at localhost:50051
✓ Found 154 linear/embedding modules
✓ Model setup complete

============================================================
Running Prefill Benchmark
============================================================
Token length: 128
============================================================
Prefill time: 2.3456 seconds
Throughput: 54.56 tokens/sec
============================================================
```

## 架构说明

### 计算分离策略

```
┌─────────────────────────────────────────────────────────┐
│                    LLaMA Model                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐              ┌──────────────┐        │
│  │  TEE (CPU)   │   ZeroMQ     │  GPU Server  │        │
│  ├──────────────┤ ◄──────────► ├──────────────┤        │
│  │ • RMSNorm    │   msgpack    │ • Linear     │        │
│  │ • SiLU       │              │ • Embedding  │        │
│  │ • Attention  │              │ • Matmul     │        │
│  │ • 控制流程   │              │              │        │
│  └──────────────┘              └──────────────┘        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 通信流程

1. **注册阶段** - TEE 客户端向 GPU 服务器注册需要的模块
2. **状态同步** - 获取非线性层的参数和 buffer
3. **推理阶段** - Linear/Embedding 通过 ZeroMQ 远程调用

## 配置选项

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLAMA_MODEL_PATH` | - | 模型路径 |
| `LLAMA_GPU_DEVICE` | `cuda:0` | GPU 设备 |
| `LLAMA_GPU_PORT` | `50051` | 服务端口 |
| `LLAMA_GPU_ENDPOINT` | `localhost:50051` | 服务器地址 |
| `LLAMA_DTYPE` | `float32` | 数据类型 |

### Prefill Token 长度

在 `tee_gpu/tee_runner.py` 中修改：

```python
PREFILL_TOKEN_LENGTH = 128  # 修改为所需长度
```

## Gramine SGX 部署

### 构建 Manifest

```bash
cd tee_gpu
make
```

### 生成 SGX 签名

```bash
make SGX=1
```

### 运行

```bash
# GPU 服务器（可选在 SGX 中运行）
gramine-sgx server

# TEE 客户端（在 SGX 中运行）
gramine-sgx tee_runner
```

## 性能优化建议

1. **批量处理** - 增加 batch size
2. **数据类型** - 使用 `float16` 或 `bfloat16`
3. **网络优化** - 使用 IPC 或 RDMA（如果支持）
4. **模型量化** - 减少传输数据量

## 技术细节

### ZeroMQ vs gRPC

| 特性 | ZeroMQ | gRPC |
|------|--------|------|
| 依赖 | 轻量级 | 需要 protobuf |
| 性能 | 更快 | 较慢 |
| 灵活性 | 高 | 需要预定义 schema |
| 适用场景 | 点对点通信 | 微服务架构 |

### 消息序列化

使用 msgpack 进行高效序列化：
- 比 JSON 更快、更紧凑
- 支持二进制数据（张量传输）
- 无需预定义 schema

## 常见问题

### 1. 连接失败

确保 GPU 服务器已启动，端口未被占用：

```bash
netstat -an | grep 50051
```

### 2. 模型加载失败

检查模型路径和文件完整性：

```bash
ls -la $LLAMA_MODEL_PATH
```

### 3. GPU 内存不足

减小模型或使用更小的数据类型：

```bash
export LLAMA_DTYPE="float16"
```

## 项目演进

- ✅ v1.0 - gRPC 通信实现
- ✅ v2.0 - 迁移到 ZeroMQ
- ✅ v2.1 - 精简代码，专注 prefill 测试
- 🔄 v3.0 - 支持多客户端并发（计划中）

## 参考资料

- [Gramine 文档](https://gramine.readthedocs.io/)
- [ZeroMQ 指南](https://zeromq.org/get-started/)
- [LLaMA 模型](https://ai.meta.com/llama/)

## 许可证

本项目基于 Apache 2.0 许可证开源。

## 贡献

欢迎提交 Issue 和 Pull Request！
