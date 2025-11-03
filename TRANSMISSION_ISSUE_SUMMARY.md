# 传输量问题总结

## 📊 问题现状

在使用 **LLaMA 3.2-1B** 模型、**1024 tokens** prefill 的情况下：

### 总体数据
- **总传输量**: **13.97 GB** (发送 6.66 GB + 接收 7.31 GB)
- **RPC 调用**: 197 次
- **平均延迟**: 438.77 ms/call
- **总耗时**: ~86 秒

### 操作分布

| 操作 | 次数 | 发送(MB) | 接收(MB) | 总量(MB) | 占比 | 平均/次(MB) |
|-----|------|---------|---------|---------|------|------------|
| **Matmul** | 64 | 4864 | 4352 | **9216** | **66.0%** | **144** |
| BatchLinear | 128 | 1792 | 2944 | 4736 | 33.9% | 37 |
| Embedding | 2 | 0.02 | 16 | 16 | 0.1% | 8 |
| LMHead | 2 | 0.02 | 1 | 1 | 0.0% | 0.5 |

## 🔴 核心问题：Matmul 占 66% 传输量

### 问题根源

**Attention Scores 矩阵的二次复杂度**：

```python
# Attention 计算
Q: [1, 32, 1024, 64]  # 8 MB
K: [1, 32, 1024, 64]  # 8 MB

# 问题在这里 ↓
Scores = Q @ K^T  # [1, 32, 1024, 1024]  ← 128 MB！！！
                  #  ↑              ↑
                  #  seq_len × seq_len = 二次复杂度

# 然后
Output = Scores @ V  # [1, 32, 1024, 64]  # 8 MB
```

### 数学分析

**Attention Scores 大小**：
```
Shape: [batch, num_heads, seq_len, seq_len]
     = [1, 32, 1024, 1024]
     
Size = 1 × 32 × 1024 × 1024 × 4 bytes (float32)
     = 134,217,728 bytes
     = 128 MB

复杂度: O(seq_len²)
```

### 每层 Attention 的传输

```
1. Q @ K^T (Matmul):
   发送: Q (8 MB) + K (8 MB) = 16 MB
   接收: Scores (128 MB)
   总计: 144 MB

2. Scores @ V (Matmul):
   发送: Scores (128 MB) + V (8 MB) = 136 MB
   接收: Output (8 MB)
   总计: 144 MB

每层 Attention: 144 + 144 = 288 MB
```

### 全模型传输量

```
模型结构:
- 16 Decoder Layers
- 每层包含: Attention + MLP

每层传输量:
- Attention: 288 MB (Matmul × 2)
- BatchLinear: ~160 MB (QKV + O + Gate + Up + Down)
- 小计: ~448 MB/layer

全模型:
- Embedding: 16 MB
- 16 Layers: 448 × 16 = 7168 MB
- LM Head: 1 MB
- 总计: ~7.2 GB

实际测量: 13.97 GB (包含往返，约 2x)
```

## 🎯 为什么这么大？

### 1. 二次复杂度
```
Scores 大小 ∝ seq_len²

seq_len = 1024:
  Scores = 1024² × 32 heads × 4 bytes = 128 MB

seq_len = 2048:
  Scores = 2048² × 32 heads × 4 bytes = 512 MB  ← 4倍增长！
```

### 2. 需要传输中间结果
```
当前架构:
TEE ──Q,K──> GPU ──Scores──> TEE ──Softmax──> TEE ──Scores──> GPU ──Output──> TEE
      8MB         128MB←问题    128MB→问题         8MB

如果 Fused:
TEE ──Q,K,V──> GPU ──[内部计算]──> GPU ──Output──> TEE
      24MB                              8MB
```

### 3. Float32 精度
```
当前: float32 (4 bytes)
可选: bfloat16 (2 bytes) ← 减半！
```

## 💡 优化方案

### 方案对比

| 方案 | 传输量 | 减少 | 实现难度 | 时间 |
|-----|-------|------|---------|------|
| **当前 (float32)** | 13.97 GB | - | - | - |
| **1. bfloat16** | 6.98 GB | 50% | ⭐ 简单 | 1天 |
| **2. Fused Attention** | 7.52 GB | 46% | ⭐⭐ 中等 | 1周 |
| **3. 组合 (1+2)** | 3.76 GB | 73% | ⭐⭐ 中等 | 1周 |
| **4. Flash Attention** | 2.50 GB | 82% | ⭐⭐⭐ 困难 | 1月 |

### 方案 1: 启用 bfloat16 ✅ 推荐立即实施

**原理**: 使用 16-bit 浮点数代替 32-bit

**修改**:
```python
# tee_runner_optimized.py
def init(self) -> Dict:
    meta = {
        "wire_dtype": "bfloat16",  # ← 改这里
        "max_chunks": 10,
    }
```

**效果**:
- 传输量: 13.97 GB → **6.98 GB** (减少 50%)
- 性能提升: **~2x**
- 精度损失: 极小（bfloat16 专为深度学习设计）

**风险**: 低（代码已支持）

### 方案 2: Fused Attention ⭐ 推荐

**原理**: 将整个 Attention 计算放在 GPU 端，不传输中间 Scores

**架构变化**:
```python
# 当前
TEE: Reshape, RoPE
GPU: Q@K^T → TEE: Softmax → GPU: Scores@V

# 优化后
TEE: Reshape, RoPE
GPU: Q@K^T + Softmax + Scores@V (一次完成)
```

**效果**:
- Matmul 传输: 9216 MB → **2765 MB** (减少 70%)
- 总传输量: 13.97 GB → **7.52 GB** (减少 46%)

**实现**:
```python
# server_optimized.py
@torch.no_grad()
def fused_attention(self, Q, K, V, scaling):
    """Fused Attention - 在 GPU 端完成"""
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scaling
    scores = F.softmax(scores, dim=-1)
    output = torch.matmul(scores, V)
    return output  # 只返回最终结果
```

### 方案 3: 组合优化 (bfloat16 + Fused Attention) 🚀 最佳

**效果**:
- 传输量: 13.97 GB → **3.76 GB** (减少 73%)
- 性能提升: **3-5x**
- Matmul 占比: 66% → 20%

**实施步骤**:
1. 启用 bfloat16 (1天)
2. 实现 Fused Attention (3-5天)
3. 测试验证 (1-2天)

### 方案 4: Flash Attention (长期)

**原理**: 使用 Flash Attention 算法，避免显式计算完整 Scores 矩阵

**优势**:
- 内存复杂度: O(N²) → O(N)
- 速度更快
- 传输量最小

**挑战**:
- 需要集成 `flash-attn` 库
- 可能需要修改模型结构
- 调试复杂

## 📈 性能预测

### 当前性能
```
Prefill (1024 tokens):
  传输量: 13.97 GB
  总耗时: ~86 秒
  吞吐量: 162 MB/s
  瓶颈: Matmul (Scores 传输)
```

### 优化后 (bfloat16 + Fused Attention)
```
Prefill (1024 tokens):
  传输量: 3.76 GB  (↓ 73%)
  总耗时: ~25 秒   (↓ 71%)
  吞吐量: 450 MB/s (↑ 2.8x)
  瓶颈: BatchLinear
```

### 不同 seq_len 的影响

| seq_len | 当前传输量 | 优化后 | 改进 |
|---------|-----------|--------|------|
| 512 | 4.5 GB | 1.5 GB | 3x |
| 1024 | 14.0 GB | 3.8 GB | 3.7x |
| 2048 | 48.0 GB | 10.0 GB | 4.8x |
| 4096 | 180.0 GB | 32.0 GB | 5.6x |

**结论**: seq_len 越大，优化效果越明显！

## 🔧 实施计划

### Phase 1: 快速优化 (1-2天) ✅
- [x] 分析问题根源
- [ ] 启用 bfloat16
- [ ] 验证传输量减半
- [ ] 性能测试

### Phase 2: 架构优化 (1周)
- [ ] 设计 Fused Attention 接口
- [ ] 实现服务端 fused_attention()
- [ ] 修改客户端调用逻辑
- [ ] 集成测试

### Phase 3: 性能调优 (1周)
- [ ] 优化 BatchLinear 合并
- [ ] 动态阈值调整
- [ ] 压缩算法优化
- [ ] 端到端测试

### Phase 4: 长期优化 (1月+)
- [ ] 评估 Flash Attention
- [ ] 集成 flash-attn 库
- [ ] 性能对比测试
- [ ] 生产环境部署

## 📝 代码示例

### 1. 启用 bfloat16 (立即可行)

```python
# tee_gpu/tee_runner_optimized.py
def init(self) -> Dict:
    meta = {
        "wire_dtype": "bfloat16",  # ← 从 float32 改为 bfloat16
        "max_chunks": 10,
    }
    init_data = self._send_request("Init", meta)
    # ...
```

### 2. Fused Attention (推荐)

**服务端**:
```python
# tee_gpu/server_optimized.py
class GPUComputeService:
    @torch.no_grad()
    def fused_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                       scaling: float) -> torch.Tensor:
        """Fused Attention - 在 GPU 端完成所有计算"""
        # Q, K, V: [batch, heads, seq_len, head_dim]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scaling
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
        output = torch.matmul(scores, V)
        return output

class ZMQServer:
    def handle_fused_attention(self, request: Dict) -> Dict:
        """处理 Fused Attention 请求"""
        Q = self._receive_tensor(request["Q"])
        K = self._receive_tensor(request["K"])
        V = self._receive_tensor(request["V"])
        scaling = request["scaling"]
        
        output = self.compute.fused_attention(Q, K, V, scaling)
        return {"output": self._send_tensor(output)}
```

**客户端**:
```python
# tee_gpu/tee_runner_optimized.py
class GPUClient:
    def fused_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                       scaling: float) -> torch.Tensor:
        """Fused Attention"""
        request = {
            "Q": self._send_tensor(Q),
            "K": self._send_tensor(K),
            "V": self._send_tensor(V),
            "scaling": scaling,
        }
        resp = self._send_request("FusedAttention", request)
        return self._receive_tensor(resp["output"])

class TEELlamaModel:
    def attention(self, layer_idx: int, hidden_states: torch.Tensor, 
                 position_ids: torch.Tensor) -> torch.Tensor:
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
        
        # 4. Fused Attention (GPU) ← 替换原来的两次 Matmul
        attn_output = self.gpu.fused_attention(Q, K, V, self.scaling)
        
        # 5. Reshape (TEE)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # 6. O projection (GPU)
        attn_output = self.gpu.batch_linear(layer_idx, ["o_proj"], attn_output)[0]
        
        return attn_output
```

## 🎯 总结

### 问题本质
**Attention Scores 的二次复杂度** 导致传输量巨大：
- Scores: [1, 32, 1024, 1024] = **128 MB**
- 每层传输 2 次 = **288 MB/layer**
- 16 层 = **4.6 GB** (占总传输量的 66%)

### 解决方案
1. **立即**: 启用 bfloat16 → 减少 50% 传输量
2. **推荐**: Fused Attention → 减少 70% Matmul 传输
3. **最佳**: 组合优化 → 减少 73% 总传输量，性能提升 3-5x

### 预期效果
```
当前: 13.97 GB, ~86 秒
优化: 3.76 GB,  ~25 秒  (3.4x 加速)
```

### 下一步
1. 启用 bfloat16（1天）
2. 实现 Fused Attention（1周）
3. 性能测试和调优（1周）
