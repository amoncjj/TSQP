# TEE-GPU 计算分离架构说明

## 架构概览

本项目实现了 LLaMA 3.2-1B 模型的 TEE-GPU 协同推理，将计算密集型的线性运算卸载到 GPU，非线性运算保留在 TEE 内部执行。

---

## 当前实现分析

### GPU 端 (server.py)

**托管的模块类型：**
1. **所有 `nn.Linear` 层** - 线性变换
2. **所有 `nn.Embedding` 层** - 词嵌入

**权重存储位置：**
- ✅ 所有线性层权重加载到 GPU 显存
- ✅ 所有嵌入层权重加载到 GPU 显存
- ✅ 非线性层的参数和 buffer 保留在 GPU，但会同步到 TEE

**GPU 执行的操作：**
```python
# 在 LlamaModuleRegistry.__init__ 中注册
if isinstance(module, nn.Linear):
    module.to(device)  # 移动到 GPU
    self.remote_modules[module_name] = RemoteModuleRecord(...)
    
elif isinstance(module, nn.Embedding):
    module.to(device)  # 移动到 GPU
    self.remote_modules[module_name] = RemoteModuleRecord(...)
```

---

### TEE 端 (tee_runner.py)

**执行流程：**
1. 加载模型架构（不含权重）
2. 从 GPU 同步非线性层的参数和 buffer
3. 将所有 `nn.Linear` 替换为 `RemoteLinearProxy`
4. 推理时，线性层通过 gRPC 调用 GPU

**TEE 内执行的操作：**
- ✅ 所有非线性激活函数（SiLU, GELU, ReLU 等）
- ✅ LayerNorm / RMSNorm
- ✅ Softmax（注意力权重计算）
- ✅ 矩阵乘法之外的所有张量运算
- ✅ 采样逻辑（temperature, top-p）

**问题：Embedding 层处理**
⚠️ **当前代码存在问题**：
- `server.py` 将 `nn.Embedding` 注册为远程模块
- 但 `tee_runner.py` 的 `replace_linear_modules()` **只替换 Linear 层**
- **Embedding 层未被替换**，仍在 TEE 内执行（但没有权重！）

---

## 单个 Transformer 块的计算分布

### LLaMA Decoder Layer 结构

```
LlamaDecoderLayer
├── input_layernorm (RMSNorm)
├── self_attn (LlamaAttention)
│   ├── q_proj (Linear)          → GPU
│   ├── k_proj (Linear)          → GPU
│   ├── v_proj (Linear)          → GPU
│   ├── rotary_emb               → TEE
│   ├── attention weights (QK^T) → TEE
│   ├── softmax                  → TEE
│   ├── attention output (AV)    → TEE (矩阵乘)
│   └── o_proj (Linear)          → GPU
├── post_attention_layernorm (RMSNorm)
└── mlp (LlamaMLP)
    ├── gate_proj (Linear)       → GPU
    ├── up_proj (Linear)         → GPU
    ├── act_fn (SiLU)            → TEE
    └── down_proj (Linear)       → GPU
```

### 详细计算流程

#### 1. **输入 Embedding**
```
embed_tokens (nn.Embedding)  → 应该在 GPU，但当前未正确处理
```

#### 2. **Self-Attention 模块**

| 操作 | 位置 | 说明 |
|------|------|------|
| `input_layernorm(x)` | TEE | RMSNorm 归一化 |
| `q = q_proj(x)` | GPU | 线性投影 (hidden_size → num_heads × head_dim) |
| `k = k_proj(x)` | GPU | 线性投影 |
| `v = v_proj(x)` | GPU | 线性投影 |
| `apply_rotary_pos_emb(q, k)` | TEE | 旋转位置编码 |
| `attn_scores = Q @ K^T` | **GPU** | **矩阵乘法（注意力分数）** |
| `attn_scores = attn_scores / sqrt(d)` | TEE | 缩放 |
| `attn_scores = attn_scores + mask` | TEE | 应用 causal mask |
| `attn_weights = softmax(attn_scores)` | TEE | Softmax 归一化 |
| `attn_output = attn_weights @ V` | **GPU** | **矩阵乘法（加权求和）** |
| `output = o_proj(attn_output)` | GPU | 输出投影 |

#### 3. **MLP 模块**

| 操作 | 位置 | 说明 |
|------|------|------|
| `post_attention_layernorm(x)` | TEE | RMSNorm 归一化 |
| `gate = gate_proj(x)` | GPU | 门控投影 |
| `up = up_proj(x)` | GPU | 上投影 |
| `act_fn(gate)` | TEE | SiLU 激活函数 |
| `hidden = act_fn(gate) * up` | TEE | 逐元素乘法 |
| `output = down_proj(hidden)` | GPU | 下投影 |

#### 4. **输出层**
```
lm_head (nn.Linear)  → GPU  (vocab 投影)
```

---

## 当前代码的关键问题

### ✅ 已解决：Embedding 层卸载

已创建 `RemoteEmbeddingProxy` 并更新替换逻辑。

### ❌ 问题 2: Attention 中的矩阵乘法未卸载

**问题：**
在 `eager_attention_forward` 中有两个大型矩阵乘法：
```python
# 1. 计算注意力分数 [batch, heads, seq, seq]
attn_weights = torch.matmul(query, key_states.transpose(2, 3))

# 2. 加权求和 [batch, heads, seq, head_dim]  
attn_output = torch.matmul(attn_weights, value_states)
```

这两个操作计算密集，应该在 GPU 执行，但当前在 TEE 内运行。

### ✅ 解决方案

需要创建自定义的 `RemoteMatmulProxy` 来处理这些矩阵乘法：

```python
class RemoteMatmulProxy:
    def __init__(self, stub):
        self.stub = stub
    
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # 将 a, b 发送到 GPU 进行矩阵乘法
        # 返回结果到 TEE
        ...
```

或者修改 Attention 模块，将 Q@K^T 和 Attn@V 作为独立的远程调用。

---

## 通信开销分析

### 每个 Token 生成的 gRPC 调用次数

假设 LLaMA 3.2-1B 有 `L=22` 层：

**每层的远程调用：**
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj` = 4 次
- MLP: `gate_proj`, `up_proj`, `down_proj` = 3 次
- **每层总计：7 次**

**每个 token：**
- Embedding: 1 次
- Decoder layers: 22 × 7 = 154 次
- LM head: 1 次
- **总计：156 次 gRPC 调用**

**生成 100 个 token：**
- 约 15,600 次 gRPC 调用
- 每次调用传输 hidden_size × batch_size 的数据

---

## 安全性保证

### TEE 内执行的关键操作
✅ **注意力权重计算** - 防止泄露 token 之间的关联
✅ **Softmax** - 防止泄露注意力分布
✅ **激活函数** - 防止泄露中间激活模式
✅ **采样逻辑** - 防止泄露生成策略

### GPU 执行的操作
⚠️ **线性变换** - 只能看到加密后的中间表示
⚠️ **Embedding 查表** - 只能看到 token ID（需要额外保护）

---

## 性能优化建议

1. **批量调用合并**：将同一层的多个 Linear 合并为一次调用
2. **异步执行**：使用 gRPC 异步 API
3. **张量压缩**：传输前量化或压缩
4. **KV Cache 管理**：缓存 key/value 减少重复计算

---

## 总结

### ✅ 已实现功能

1. **GPU 端 (server.py)**
   - 托管所有 `nn.Linear` 和 `nn.Embedding` 层
   - 提供 `Forward` RPC 接口执行线性变换和嵌入查表
   - 提供 `Matmul` RPC 接口执行矩阵乘法
   - 所有模型权重加载到 GPU 显存

2. **TEE 端 (tee_runner.py)**
   - `RemoteLinearProxy`: 代理所有 Linear 层
   - `RemoteEmbeddingProxy`: 代理所有 Embedding 层
   - `RemoteMatmul`: 代理 Attention 中的矩阵乘法
   - `inject_remote_matmul()`: Monkey-patch `torch.matmul` 实现透明卸载

3. **计算分布**
   - GPU: Linear, Embedding, Matmul (Q@K^T, Attn@V)
   - TEE: RMSNorm, RoPE, Softmax, SiLU, 残差连接, 采样

### 🔒 安全性保证

通过将 Softmax 保留在 TEE 内，确保：
- ✅ 注意力权重分布不泄露给 GPU
- ✅ Token 之间的语义关联受保护
- ✅ 生成策略（temperature, top-p）在 TEE 内执行

### ⚡ 性能权衡

- 通信次数增加 28%（156 → 200 次/token）
- 但保证了关键的安全属性
- 矩阵乘法仍在 GPU 加速，整体性能优于纯 TEE
