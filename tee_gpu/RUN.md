# TEE-GPU 运行指南

## 正确的运行方式

### 1. 构建 Manifest

```bash
cd /home/junjie_chen@idm.teecertlabs.com/TSQP/tee_gpu
make clean
make
```

这会生成：
- `tee_runner.manifest` - TEE 客户端配置
- `server.manifest` - GPU 服务器配置
- `msg_pb2.py` 和 `msg_pb2_grpc.py` - gRPC 通信代码

### 2. 启动 GPU 服务器

**方式 1: 直接运行（推荐用于测试）**
```bash
python server.py
```

**方式 2: 在 Gramine 中运行**
```bash
gramine-direct server server.py
```

### 3. 运行 TEE 客户端

**在另一个终端中：**

```bash
cd /home/junjie_chen@idm.teecertlabs.com/TSQP/tee_gpu

# Direct 模式（不使用 SGX）
gramine-direct tee_runner tee_runner.py

# 或 SGX 模式
make SGX=1
gramine-sgx tee_runner tee_runner.py
```

## 常见错误

### 错误 1: `can't open file '//tee_runner.py'`

**原因**: 命令格式错误

❌ 错误：`gramine-direct ./tee_runner tee_runner.py`  
✅ 正确：`gramine-direct tee_runner tee_runner.py`

### 错误 2: `RESOURCE_EXHAUSTED: Received message larger than max`

**原因**: gRPC 消息大小限制

**解决**: 已在代码中修复，设置了 512MB 限制

### 错误 3: 找不到模型文件

**解决**: 设置环境变量
```bash
export LLAMA_MODEL_PATH=/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b
```

## 当前配置

代码中已硬编码：
- **Token 长度**: 128（在 `tee_runner.py` 第 15 行修改 `PREFILL_TOKEN_LENGTH`）
- **输出**: 只显示 prefill 时间，不输出其他结果

## 修改 Token 长度

编辑 `tee_runner.py`：
```python
PREFILL_TOKEN_LENGTH = 256  # 改成你想要的数量
```

然后重新运行（不需要重新 make）。

## 完整测试流程

```bash
# 1. 进入目录
cd /home/junjie_chen@idm.teecertlabs.com/TSQP/tee_gpu

# 2. 构建
make clean && make

# 3. 终端 1: 启动 GPU 服务器
python server.py

# 4. 终端 2: 运行 TEE 客户端
gramine-direct tee_runner tee_runner.py
```

## 预期输出

**GPU 服务器（终端 1）:**
```
Loading model from: /home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b
Is local path: True
GPU remote module service started on port 50051
```

**TEE 客户端（终端 2）:**
```
Loading model from: /home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b
Loading model config from: /home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b
Is local path: True
Connecting to GPU server at localhost:50051
Running prefill with 128 tokens...
Prefill token length: 128

Prefill time: 0.1234 seconds
```
