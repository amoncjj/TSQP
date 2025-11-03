# 传输量分析报告

## 问题概述

在使用 **LLaMA 3.2-1B** 模型、**1024 tokens** 的情况下，部分操作的传输量异常大。

## 数据分析

### 总体统计
- **总 RPC 调用**: 98 次
- **总发送数据**: 3328.02 MB
- **总接收数据**: 3656.50 MB
- **总传输量**: **6984.52 MB (~7GB)**
- **平均吞吐量**: 162.43 MB/s

### 主要操作传输量

| 操作类型 | 发送量 (KB) | 接收量 (KB) | 总量 (MB) | 占比 |
|---------|------------|------------|----------|------|
| **Matmul** | 16384 或 139264 | 131072 或 8192 | **128-144 MB/次** | **最大** |
| BatchLinear | 8192 或 32768 | 8192-65536 | 8-72 MB/次 | 中等 |
| Embedding | 8 | 8192 | 8 MB/次 | 小 |
| LMHead | 8 | 501 | 0.5 MB/次 | 最小 |

## 🔴 核心问题：Matmul 传输量过大

### 问题详情

从日志中可以看到，**Matmul** 操作有两种模式：

#### 模式 1：Q @ K^T（Attention Scores）
```
ID 4: Matmul
  Sent:     16384.08 KB  (16 MB)   ← Q: [1, 32, 1024, 64]
  Received: 131072.05 KB (128 MB)  ← Scores: [1, 32, 1024, 1024]
  Total:    144 MB
```

#### 模式 2：Scores @ V（Attention Output）
```
ID 5: Matmul
  Sent:     139264.08 KB (136 MB)  ← Scores: [1, 32, 1024, 1024] + V: [1, 32, 1024, 64]
  Received: 8192.05 KB   (8 MB)    ← Output: [1, 32, 1024, 64]
  Total:    144 MB
```

### 🎯 根本原因

**Attention Scores 矩阵过大！**

```python
# Attention 计算流程
Q: [batch=1, heads=32, seq_len=1024, head_dim=64]  # 8 MB
K: [batch=1, heads=32, seq_len=1024, head_dim=64]  # 8 MB

# 问题在这里 ↓
Scores = Q @ K^T  # [1, 32, 1024, 1024]  ← 128 MB！！！

# 然后
Output = Scores @ V  # [1, 32, 1024, 64]  # 8 MB
```

### 数学计算

**Attention Scores 大小**：
```
Shape: [batch, num_heads, seq_len, seq_len]
     = [1, 32, 1024, 1024]
     
Size = 1 × 32 × 1024 × 1024 × 4 bytes (float32)
     = 134,217,728 bytes
     = 128 MB
```

**每层 Attention 的传输量**：
- Q @ K^T: 发送 16MB，接收 128MB = **144 MB**
- Scores @ V: 发送 136MB，接收 8MB = **144 MB**
- **每层总计**: **288 MB**

**16 层 Decoder 的总传输量**：
```
16 layers × 288 MB/layer = 4608 MB ≈ 4.5 GB
```

这与日志中的总传输量 **~7GB** 基本吻合（还包括 BatchLinear、Embedding 等）。

## 🔍 详细分解

### LLaMA 3.2-1B 模型配置
```python
num_layers = 16
hidden_size = 2048
num_heads = 32
head_dim = 64
seq_len = 1024  # Prefill length
```

### 每层的传输量计算

#### 1. Attention 部分
```
Q/K/V Projections (BatchLinear):
  Input:  [1, 1024, 2048] = 8 MB
  Output: [1, 1024, 2048] × 3 = 24 MB
  Total:  32 MB

Q @ K^T (Matmul):
  Q:      [1, 32, 1024, 64] = 8 MB
  K^T:    [1, 32, 64, 1024] = 8 MB (已包含在 Q 中)
  Scores: [1, 32, 1024, 1024] = 128 MB  ← 问题！
  Total:  144 MB

Scores @ V (Matmul):
  Scores: [1, 32, 1024, 1024] = 128 MB
  V:      [1, 32, 1024, 64] = 8 MB
  Output: [1, 32, 1024, 64] = 8 MB
  Total:  144 MB

O Projection (BatchLinear):
  Input:  [1, 1024, 2048] = 8 MB
  Output: [1, 1024, 2048] = 8 MB
  Total:  16 MB

Attention 小计: 32 + 144 + 144 + 16 = 336 MB
```

#### 2. MLP 部分
```
Gate/Up Projections (BatchLinear):
  Input:  [1, 1024, 2048] = 8 MB
  Output: [1, 1024, 8192] × 2 = 64 MB
  Total:  72 MB

Down Projection (BatchLinear):
  Input:  [1, 1024, 8192] = 32 MB
  Output: [1, 1024, 2048] = 8 MB
  Total:  40 MB

MLP 小计: 72 + 40 = 112 MB
```

#### 3. 每层总计
```
Attention: 336 MB
MLP:       112 MB
Total:     448 MB/layer
```

#### 4. 全模型总计
```
Embedding:  8 MB
16 Layers:  448 × 16 = 7168 MB ≈ 7 GB
LM Head:    0.5 MB
Total:      ~7.2 GB
```

**与日志吻合！** ✅

## 🚨 为什么传输量这么大？

### 原因 1：Attention Scores 是二次复杂度
```
Scores = Q @ K^T
Shape: [batch, heads, seq_len, seq_len]
                              ↑       ↑
                              这两个维度相乘！

当 seq_len = 1024 时：
Size ∝ seq_len²
    = 1024² = 1,048,576 个元素/head
    × 32 heads
    × 4 bytes
    = 128 MB
```

### 原因 2：当前架构需要传输中间结果
```
TEE 端                          GPU 端
  ↓                               ↓
Q, K, V ──────────────────────> Linear
  ↓                               ↓
Reshape, RoPE                     ↓
  ↓                               ↓
Q, K ─────────────────────────> Matmul (Q @ K^T)
  ↓                               ↓
Scores (128MB!) <───────────────  ↓
  ↓                               ↓
Softmax                           ↓
  ↓                               ↓
Scores (128MB!) ─────────────> Matmul (Scores @ V)
  ↓                               ↓
Output <─────────────────────────  ↓
```

**每次都要传输 128MB 的 Scores！**

### 原因 3：Float32 精度
```
当前使用 float32 (4 bytes)
如果使用 bfloat16 (2 bytes)，可以减半：
  128 MB → 64 MB
```

## 💡 优化方案

### 方案 1：使用 bfloat16（立即可行）✅
```python
# 修改 wire_dtype
wire_dtype = "bfloat16"  # 从 float32 改为 bfloat16

预期效果：
  传输量减半：7 GB → 3.5 GB
  性能提升：~2x
```

### 方案 2：Fused Attention（推荐）⭐
```python
# 将整个 Attention 放在 GPU 端执行
def fused_attention_gpu(Q, K, V):
    # 在 GPU 端完成所有操作，不传输中间结果
    scores = Q @ K.T
    scores = softmax(scores)
    output = scores @ V
    return output

优势：
  - 不传输 Scores (128MB × 2)
  - 每层节省 256 MB
  - 16 层节省 4 GB
  - 总传输量：7 GB → 3 GB
```

### 方案 3：Flash Attention（最优）🚀
```python
# 使用 Flash Attention 算法
# 不需要显式计算完整的 Scores 矩阵

优势：
  - 内存占用 O(N) 而不是 O(N²)
  - 不传输 Scores
  - 速度更快
  - 总传输量：7 GB → 2.5 GB
```

### 方案 4：分块传输（辅助优化）
```python
# 对于大矩阵，分块传输
chunk_size = 256
for i in range(0, seq_len, chunk_size):
    chunk = scores[:, :, i:i+chunk_size, :]
    # 处理 chunk

优势：
  - 降低峰值内存
  - 可以与共享内存结合
```

## 📊 优化效果预测

| 方案 | 传输量 | 延迟 | 实现难度 |
|-----|-------|------|---------|
| 当前 (float32) | 7.0 GB | 基准 | - |
| bfloat16 | 3.5 GB | -50% | ⭐ 简单 |
| Fused Attention | 3.0 GB | -60% | ⭐⭐ 中等 |
| Flash Attention | 2.5 GB | -70% | ⭐⭐⭐ 困难 |
| 组合优化 | 1.5 GB | -80% | ⭐⭐⭐⭐ 很难 |

## 🎯 立即行动建议

### 短期（1-2天）
1. **启用 bfloat16**
   ```python
   # 在 init() 中设置
   wire_dtype = "bfloat16"
   ```
   - 预期效果：传输量减半
   - 风险：低（已有代码支持）

2. **验证共享内存优化**
   - 当前 10MB 阈值对 Scores (128MB) 无效
   - 考虑提高阈值或使用压缩

### 中期（1-2周）
3. **实现 Fused Attention**
   ```python
   def handle_fused_attention(self, request):
       Q, K, V = request["Q"], request["K"], request["V"]
       # 在 GPU 端完成所有 Attention 计算
       output = fused_attention(Q, K, V)
       return output
   ```

4. **优化 BatchLinear**
   - 合并多个 Linear 操作
   - 减少往返次数

### 长期（1个月+）
5. **集成 Flash Attention**
   - 使用 `flash-attn` 库
   - 需要修改模型结构

6. **动态阈值调整**
   - 根据数据大小自动选择传输方式
   - 自适应压缩

## 📈 性能对比

### 当前性能
```
Prefill (1024 tokens):
  Total Time: ~43 seconds (98 calls × 438ms/call)
  Throughput: 162 MB/s
  Bottleneck: Matmul (Scores 传输)
```

### 优化后预期
```
使用 bfloat16 + Fused Attention:
  Total Time: ~15 seconds (估计)
  Throughput: 400+ MB/s
  改进: 3x 加速
```

## 🔧 代码修改示例

### 1. 启用 bfloat16
```python
# tee_runner_optimized.py
def init(self) -> Dict:
    meta = {
        "wire_dtype": "bfloat16",  # ← 改这里
        "max_chunks": 10,
    }
    # ...
```

### 2. 实现 Fused Attention（服务端）
```python
# server_optimized.py
@torch.no_grad()
def fused_attention(self, layer_idx: int, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Fused Attention - 在 GPU 端完成所有计算"""
    # Q, K, V: [batch, heads, seq_len, head_dim]
    
    # 1. Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 2. Scale
    scores = scores * (self.head_dim ** -0.5)
    
    # 3. Softmax
    scores = F.softmax(scores, dim=-1)
    
    # 4. Scores @ V
    output = torch.matmul(scores, V)
    
    return output  # 只返回最终结果，不传输中间 Scores
```

### 3. 客户端调用
```python
# tee_runner_optimized.py
def attention(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    # 1. QKV projections (GPU)
    qkv = self.gpu.batch_linear(layer_idx, ["q_proj", "k_proj", "v_proj"], hidden_states)
    Q, K, V = qkv
    
    # 2. Reshape (TEE)
    Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    K = K.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    
    # 3. RoPE (TEE)
    cos, sin = self.rotary_emb(V, position_ids)
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
    K = repeat_kv(K, self.num_key_value_groups)
    V = repeat_kv(V, self.num_key_value_groups)
    
    # 4. Fused Attention (GPU) ← 新增
    attn_output = self.gpu.fused_attention(layer_idx, Q, K, V)
    
    # 5. Reshape (TEE)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
    
    # 6. O projection (GPU)
    attn_output = self.gpu.batch_linear(layer_idx, ["o_proj"], attn_output)[0]
    
    return attn_output
```

## 总结

### 问题根源
**Attention Scores 矩阵的二次复杂度**导致传输量巨大：
- Scores: [1, 32, 1024, 1024] = **128 MB**
- 每层传输 2 次 = **256 MB/layer**
- 16 层 = **4 GB** (占总传输量的 57%)

### 解决方案优先级
1. ✅ **立即**: 启用 bfloat16（减半传输量）
2. ⭐ **推荐**: 实现 Fused Attention（减少 60% 传输量）
3. 🚀 **最优**: 集成 Flash Attention（减少 70% 传输量）

### 预期效果
通过组合优化，可以将总传输量从 **7 GB 降至 1.5 GB**，性能提升 **3-5 倍**。
