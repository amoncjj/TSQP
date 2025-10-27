# TSQP 架构设计文档

## 概述

本项目实现了 TEE 与 GPU 的细粒度计算分离，将 LLaMA 模型的推理过程分解为两部分：
- **TEE 端**：执行轻量级的非线性操作（Softmax、RMSNorm、RotaryEmbedding、SiLU）
- **GPU 端**：执行计算密集型操作（Linear、Embedding、Matmul）

## 设计原则

### 1. 最小化 TEE 计算负担
- TEE 只执行必须在安全环境中进行的操作
- 将所有密集计算卸载到 GPU

### 2. 细粒度操作分离
- 不是按模块分离（如整个 Attention 层），而是按操作类型分离
- 每个操作都可以独立优化

### 3. 清晰的接口设计
- 基于操作类型的 RPC 方法
- 统一的张量传输格式

## 计算分离详解

### Embedding 层
```
[GPU] input_ids → Embedding → embeddings
```
- **原因**：Embedding 表很大，适合 GPU 存储和计算

### Attention 层

```
输入: hidden_states [batch, seq_len, hidden_size]

1. [TEE] RMSNorm (input_layernorm)
   └─ 轻量级归一化操作

2. [GPU] Q/K/V Projections
   ├─ q_proj: Linear(hidden_size, num_heads * head_dim)
   ├─ k_proj: Linear(hidden_size, num_kv_heads * head_dim)
   └─ v_proj: Linear(hidden_size, num_kv_heads * head_dim)
   └─ 大矩阵乘法，适合 GPU

3. [TEE] Reshape
   └─ 简单的张量重组

4. [TEE] Apply RotaryEmbedding
   ├─ 计算 cos/sin
   └─ 应用旋转变换
   └─ 位置编码，需要在 TEE 中保护

5. [TEE] Repeat KV (GQA)
   └─ 简单的张量复制

6. [GPU] Attention Scores (Q @ K^T)
   └─ 大规模矩阵乘法

7. [TEE] Softmax
   └─ 非线性操作，相对轻量

8. [GPU] Attention Output (Attn @ V)
   └─ 大规模矩阵乘法

9. [GPU] Output Projection
   └─ o_proj: Linear(num_heads * head_dim, hidden_size)

10. [TEE] Residual Add
    └─ 简单的加法
```

### MLP 层

```
输入: hidden_states [batch, seq_len, hidden_size]

1. [TEE] RMSNorm (post_attention_layernorm)
   └─ 轻量级归一化操作

2. [GPU] Gate & Up Projections
   ├─ gate_proj: Linear(hidden_size, intermediate_size)
   └─ up_proj: Linear(hidden_size, intermediate_size)
   └─ 大矩阵乘法

3. [TEE] SiLU Activation
   └─ gate = gate * sigmoid(gate)
   └─ 非线性激活函数

4. [TEE] Element-wise Multiply
   └─ intermediate = gate * up
   └─ 简单的逐元素乘法

5. [GPU] Down Projection
   └─ down_proj: Linear(intermediate_size, hidden_size)
   └─ 大矩阵乘法

6. [TEE] Residual Add
   └─ 简单的加法
```

### LM Head
```
[TEE] Final RMSNorm
  ↓
[GPU] LM Head: Linear(hidden_size, vocab_size)
  ↓
Logits
```

## 通信协议设计

### 消息格式

所有消息使用 msgpack 序列化，格式如下：

```python
# 请求
{
    "method": str,      # RPC 方法名
    "request": dict     # 请求参数
}

# 响应
{
    "status": str,      # "success" 或 "error"
    "response": dict,   # 响应数据（成功时）
    "error": str,       # 错误信息（失败时）
    "traceback": str    # 错误堆栈（失败时）
}
```

### 张量传输格式

```python
{
    "output": bytes,        # 张量数据（numpy tobytes）
    "shape": List[int],     # 张量形状
    "dtype": str            # 数据类型（"torch.float32" 等）
}
```

### RPC 方法详解

#### 1. Init
**用途**：初始化，获取模型配置和参数

**请求**：
```python
{}  # 空请求
```

**响应**：
```python
{
    "config": {
        "num_layers": int,
        "hidden_size": int,
        "num_heads": int,
        "num_kv_heads": int,
        "head_dim": int
    },
    "rotary_emb_params": {
        "inv_freq": bytes,
        "inv_freq_shape": List[int],
        "attention_scaling": float
    },
    "norm_weights": {
        "layer_0_input_layernorm": {
            "weight": bytes,
            "shape": List[int],
            "eps": float
        },
        ...
    }
}
```

#### 2. Embedding
**用途**：Embedding 层前向传播

**请求**：
```python
{
    "input_ids": bytes,
    "input_shape": List[int],
    "dtype": str
}
```

**响应**：
```python
{
    "output": bytes,
    "shape": List[int],
    "dtype": str
}
```

#### 3. Linear
**用途**：Linear 层前向传播

**请求**：
```python
{
    "layer_idx": int,           # 层索引 (0-21)
    "module_name": str,         # 模块名称
    "hidden_states": bytes,
    "shape": List[int],
    "dtype": str
}
```

**module_name 可选值**：
- `"q_proj"` - Query projection
- `"k_proj"` - Key projection
- `"v_proj"` - Value projection
- `"o_proj"` - Output projection
- `"gate_proj"` - Gate projection (MLP)
- `"up_proj"` - Up projection (MLP)
- `"down_proj"` - Down projection (MLP)

**响应**：
```python
{
    "output": bytes,
    "shape": List[int],
    "dtype": str
}
```

#### 4. Matmul
**用途**：矩阵乘法

**请求**：
```python
{
    "a_buffer": bytes,
    "a_shape": List[int],
    "b_buffer": bytes,
    "b_shape": List[int],
    "dtype": str
}
```

**响应**：
```python
{
    "output": bytes,
    "shape": List[int],
    "dtype": str
}
```

#### 5. LMHead
**用途**：LM Head 前向传播

**请求**：
```python
{
    "hidden_states": bytes,
    "shape": List[int],
    "dtype": str
}
```

**响应**：
```python
{
    "output": bytes,
    "shape": List[int],
    "dtype": str
}
```

## 性能分析

### 通信开销

对于 LLaMA 3.2-1B (22 层, hidden_size=2048)：

**每层的 RPC 调用**：
- Attention: 4 次 Linear + 2 次 Matmul = 6 次
- MLP: 3 次 Linear = 3 次
- **每层总计**: 9 次

**整个 Prefill 的 RPC 调用**：
- Embedding: 1 次
- 22 层 × 9 次/层 = 198 次
- LM Head: 1 次
- **总计**: 200 次

**数据传输量（128 tokens）**：
- 每次 Linear: ~1 MB (1 × 128 × 2048 × 4 bytes)
- 每次 Matmul: ~0.5-2 MB（取决于维度）
- **总传输量**: ~200-400 MB

### 计算时间分布（估算）

假设：
- GPU Linear: 0.5 ms
- GPU Matmul: 1 ms
- TEE RMSNorm: 0.1 ms
- TEE Softmax: 0.2 ms
- TEE SiLU: 0.1 ms
- 网络延迟: 0.1 ms/次

**每层时间**：
- GPU 计算: 4×0.5 + 2×1 + 3×0.5 = 5.5 ms
- TEE 计算: 2×0.1 + 0.2 + 0.1 = 0.5 ms
- 网络通信: 9×0.1 = 0.9 ms
- **每层总计**: ~7 ms

**总时间（22 层）**：
- ~154 ms (不含 Embedding 和 LM Head)

### 优化方向

1. **减少通信次数**
   - 批量合并多个 Linear 调用
   - 使用异步通信

2. **减少数据传输量**
   - 使用 float16/bfloat16
   - 张量压缩

3. **优化网络**
   - 使用 IPC（同机器）
   - 使用 RDMA（不同机器）

4. **流水线并行**
   - TEE 和 GPU 并行执行不同层

## 安全性考虑

### TEE 端保护的内容

1. **位置信息**
   - RotaryEmbedding 的计算
   - Position IDs

2. **Attention 模式**
   - Softmax 后的 attention weights
   - 揭示了 token 之间的关系

3. **激活值**
   - 中间层的激活值
   - 可能泄露输入信息

### GPU 端暴露的内容

1. **模型权重**
   - 所有 Linear 层的权重
   - Embedding 表

2. **部分中间结果**
   - Linear 层的输出
   - Matmul 的结果

### 安全性权衡

- **优势**：关键的非线性操作在 TEE 中执行
- **劣势**：大量中间结果需要传输到 GPU
- **适用场景**：模型权重不敏感，但推理过程需要保护

## 扩展性

### 支持其他模型

只需修改：
1. `GPUComputeService.linear()` - 适配不同的 Linear 模块名称
2. `TEELlamaModel` - 适配不同的层结构

### 支持 Decode 阶段

需要添加：
1. KV Cache 管理
2. 增量计算支持
3. 采样逻辑

### 支持批处理

当前已支持 batch_size > 1，只需：
1. 修改 input_ids 的形状
2. 确保 GPU 内存足够

## 总结

本架构实现了 TEE 与 GPU 的细粒度计算分离，在保护关键操作的同时，充分利用 GPU 的计算能力。通过清晰的接口设计和高效的通信协议，实现了灵活且高性能的推理系统。
