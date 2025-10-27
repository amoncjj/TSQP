# TSQP - TEE + GPU 协同推理

基于 ZeroMQ 的 TEE 与 GPU 细粒度分离式 LLaMA 模型推理系统，用于 Prefill 阶段性能测试。

## 架构概述

### 计算分离策略

```
┌─────────────────────────────────────────────────────────────┐
│                      LLaMA Forward Pass                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ZeroMQ        ┌─────────────┐│
│  │   TEE (CPU)      │ ◄──────────────────► │ GPU Server  ││
│  ├──────────────────┤      msgpack         ├─────────────┤│
│  │ • Softmax        │                      │ • Linear    ││
│  │ • RMSNorm        │                      │ • Embedding ││
│  │ • RotaryEmbed    │                      │ • Matmul    ││
│  │ • SiLU (激活)    │                      │ • LM Head   ││
│  │ • 控制流程       │                      │             ││
│  └──────────────────┘                      └─────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 每层的计算流程

```
Input IDs
    ↓
[GPU] Embedding
    ↓
For each layer (0 to N-1):
    ├─ [TEE] RMSNorm (input_layernorm)
    ├─ Attention:
    │   ├─ [GPU] Q/K/V Projections (Linear)
    │   ├─ [TEE] Reshape
    │   ├─ [TEE] Apply RotaryEmbedding
    │   ├─ [TEE] Repeat KV (for GQA)
    │   ├─ [GPU] Q @ K^T (Matmul)
    │   ├─ [TEE] Softmax
    │   ├─ [GPU] Attention @ V (Matmul)
    │   └─ [GPU] Output Projection (Linear)
    ├─ [TEE] Residual Add
    ├─ [TEE] RMSNorm (post_attention_layernorm)
    ├─ MLP:
    │   ├─ [GPU] Gate & Up Projections (Linear)
    │   ├─ [TEE] SiLU Activation
    │   ├─ [TEE] Element-wise Multiply
    │   └─ [GPU] Down Projection (Linear)
    └─ [TEE] Residual Add
    ↓
[TEE] Final RMSNorm
    ↓
[GPU] LM Head (Linear)
    ↓
Logits
```

## 核心文件

- `server.py` - GPU 服务端，执行所有 GPU 密集型计算
- `tee_runner.py` - TEE 客户端，执行 Softmax/RMSNorm/RoPE/SiLU
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

输出示例：
```
Loading model from: /path/to/llama-3.2-1b
Device: cuda:0, Dtype: torch.float32
✓ Model loaded: 22 layers, hidden_size=2048
✓ ZeroMQ server started on port 50051
```

### 3. 运行 TEE 客户端测试

```bash
# 设置环境变量（可选）
export LLAMA_MODEL_PATH="/path/to/llama/model"
export LLAMA_GPU_ENDPOINT="localhost:50051"

# 运行 prefill 测试
python tee_runner.py
```

输出示例：
```
✓ Connected to GPU server at localhost:50051
Initializing model from GPU server...
✓ TEE model initialized: 22 layers

============================================================
Running Prefill Benchmark
============================================================
Token length: 128
TEE operations: Softmax, RMSNorm, RotaryEmbedding, SiLU
GPU operations: Linear, Embedding, Matmul
============================================================
Prefill time: 2.3456 seconds
Throughput: 54.56 tokens/sec
Logits shape: torch.Size([1, 1, 32000])
============================================================
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
| `LLAMA_MODEL_PATH` | `meta-llama/Llama-3.2-1B-Instruct` | 模型路径（仅用于 tokenizer）|
| `LLAMA_GPU_ENDPOINT` | `localhost:50051` | 服务器地址 |

## Prefill 测试配置

在 `tee_runner.py` 中修改：

```python
PREFILL_TOKEN_LENGTH = 128  # 修改 token 长度
```

## 通信协议

使用 ZeroMQ REQ-REP 模式 + msgpack 序列化：

### 支持的 RPC 方法

1. **Init** - 初始化，获取模型配置和参数
   - 返回：模型配置、RotaryEmbedding 参数、所有 RMSNorm 权重

2. **Embedding** - Embedding 层前向传播
   - 输入：input_ids
   - 输出：embeddings

3. **Linear** - Linear 层前向传播
   - 输入：layer_idx, module_name, hidden_states
   - 输出：transformed hidden_states
   - 支持的模块：q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

4. **Matmul** - 矩阵乘法
   - 输入：两个张量 a, b
   - 输出：a @ b

5. **LMHead** - LM Head 前向传播
   - 输入：hidden_states
   - 输出：logits

## TEE 端实现细节

### RMSNorm

```python
class TEERMSNorm(nn.Module):
    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return self.weight * hidden_states
```

### RotaryEmbedding

```python
class TEERotaryEmbedding(nn.Module):
    def forward(self, x, position_ids):
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
        return cos, sin
```

### SiLU 激活函数

```python
# 使用 PyTorch 内置
gate = F.silu(gate)
```

### Softmax

```python
# 在 Attention 中使用
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
```

## 性能分析

### 通信开销

对于 LLaMA 3.2-1B (22 层) 的 prefill 阶段：

**每个 token 的 RPC 调用次数：**
- Embedding: 1 次
- 每层 Attention: 4 次 Linear + 2 次 Matmul = 6 次
- 每层 MLP: 3 次 Linear = 3 次
- LM Head: 1 次
- **总计**: 1 + 22×(6+3) + 1 = **200 次 RPC 调用**

**128 tokens prefill：**
- 约 200 次 RPC 调用（与 token 数量无关，因为是批量处理）
- 每次调用传输 `[batch_size, seq_len, hidden_size]` 的数据

### 优化建议

1. **使用更高效的数据类型** - float16/bfloat16
2. **批量处理** - 增加 batch size
3. **网络优化** - 使用 IPC 或共享内存（如果在同一机器）
4. **张量压缩** - 量化或稀疏化

## Gramine SGX 构建

```bash
# 生成 manifest
make

# 生成 SGX 签名
make SGX=1

# 清理
make clean
```

## 项目特点

- ✅ **细粒度计算分离** - TEE 只执行必要的非线性操作
- ✅ **最小化 TEE 计算** - 将密集计算卸载到 GPU
- ✅ **清晰的接口设计** - 基于操作类型的 RPC 方法
- ✅ **高效的通信** - ZeroMQ + msgpack
- ✅ **专注 Prefill** - 简化测试流程

## 注意事项

1. 确保 GPU 服务器先启动
2. 模型路径需要包含配置文件和权重
3. TEE 端只需要 tokenizer，不需要完整模型
4. 默认端口 50051，确保未被占用
5. 所有张量传输使用 float32 格式

## 故障排除

### 连接失败
```bash
# 检查服务器是否运行
netstat -an | grep 50051

# 检查防火墙设置
```

### 形状不匹配错误
- 检查模型配置是否一致
- 确认 batch_size 和 seq_len 正确

### 性能问题
- 使用 float16 减少传输量
- 检查网络延迟
- 考虑使用本地 IPC
