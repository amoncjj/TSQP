# 性能优化路线图 V2 (基于实际测量)

## 📊 当前状态

**实测性能**: 332ms/token
**目标性能**: 10ms/token
**提升倍数**: 33x

### 性能分解

```
总延迟: 332ms (100%)
├─ 序列化/反序列化: 136ms (41%)  ← 最大瓶颈
├─ GPU计算: 100ms (30%)
├─ RPC开销: 50ms (15%)
└─ TEE计算: 46ms (14%)

RPC统计:
- 总调用次数: 98次
- 平均延迟: 3.4ms/次
- 总数据量: 7.5GB
```

### 关键发现

1. **诊断测试误导**: 单次10MB传输21ms,但实际98次调用332ms
2. **真正瓶颈**: 序列化 > GPU计算 > RPC次数 > 传输协议
3. **IPC优势有限**: 对大数据(10MB+),IPC vs TCP差异<2%

详见: [性能差距分析](PERFORMANCE_GAP_ANALYSIS.md)

---

## 🎯 优化阶段

### 阶段 1: 共享内存 (1-2天)

**目标**: 332ms → 206ms (1.6x提升)

#### 核心思路

消除msgpack序列化开销,使用共享内存零拷贝传输

#### 实现方案

```python
# 1. 创建共享内存
import posix_ipc
import mmap

shm = posix_ipc.SharedMemory("/tsqp_shm", size=100*1024*1024)
mem = mmap.mmap(shm.fd, shm.size)

# 2. 写入数据(零拷贝)
offset = 0
data = array.tobytes()
mem[offset:offset+len(data)] = data

# 3. 只传输元数据
metadata = {
    "shm_offset": offset,
    "shape": [1, 1024, 2048],
    "dtype": "float32"
}
message = msgpack.packb({"method": "BatchLinear", "metadata": metadata})
```

#### 预期效果

- 序列化: 136ms → 10ms (13.6x)
- 内存拷贝: 大幅减少
- **总延迟**: 332ms → 206ms

#### 实现文件

- `tee_gpu/shared_memory.py` - 共享内存管理器
- 修改 `GPUClient._send_request()` - 使用共享内存
- 修改 `GPUServer.handle_*()` - 从共享内存读取

---

### 阶段 2: 算子融合 (2-3天)

**目标**: 206ms → 50ms (4.1x提升)

#### 核心思路

减少RPC调用次数,将多个GPU操作合并为一次RPC

#### 当前问题

```python
# 每层6次RPC调用
qkv = gpu.batch_linear(...)      # RPC 1
attn1 = gpu.matmul(Q, K.T)       # RPC 2
attn2 = gpu.matmul(attn, V)      # RPC 3
o = gpu.batch_linear(...)        # RPC 4
gate_up = gpu.batch_linear(...)  # RPC 5
down = gpu.batch_linear(...)     # RPC 6

# 16层 × 6次 = 96次
# + embedding(1次) + lm_head(1次) = 98次
```

#### 优化方案

```python
# 方案A: 每层1次RPC (激进)
output = gpu.fused_layer(input, layer_idx)  # 包含所有GPU操作

# 方案B: 每层2次RPC (保守)
qkv_attn = gpu.fused_attention(input, layer_idx)  # QKV + Matmul
output = gpu.fused_mlp(attn_output, layer_idx)    # Gate/Up + Down
```

#### 实现步骤

1. **服务端添加融合算子**:

```python
# server_optimized.py
@torch.no_grad()
def fused_attention(self, layer_idx: int, hidden_states: torch.Tensor) -> Dict:
    \"\"\"融合的Attention操作\"\"\"
    layer = self.layers[layer_idx].self_attn
    
    # QKV projections
    q = layer.q_proj(hidden_states)
    k = layer.k_proj(hidden_states)
    v = layer.v_proj(hidden_states)
    
    # 返回给TEE做RoPE + Softmax
    return {"q": q, "k": k, "v": v}

@torch.no_grad()
def fused_mlp(self, layer_idx: int, hidden_states: torch.Tensor, 
              gate_up: torch.Tensor) -> torch.Tensor:
    \"\"\"融合的MLP操作\"\"\"
    layer = self.layers[layer_idx].mlp
    
    # Down projection
    return layer.down_proj(gate_up)
```

2. **客户端调用融合算子**:

```python
# tee_runner_optimized.py
def decoder_layer(self, layer_idx, hidden_states, position_ids):
    # TEE: Input norm
    hidden_states = self.input_layernorms[layer_idx](hidden_states)
    
    # GPU: Fused attention (1次RPC)
    qkv = self.gpu.fused_attention(layer_idx, hidden_states)
    
    # TEE: RoPE + Softmax + Matmul
    attn_output = self._tee_attention(qkv, position_ids)
    
    # GPU: O projection (1次RPC)
    attn_output = self.gpu.batch_linear(layer_idx, ["o_proj"], attn_output)[0]
    
    # ... MLP类似
```

#### 预期效果

- RPC次数: 98 → 18 (5.4x)
- RPC开销: 50ms → 10ms
- **总延迟**: 206ms → 50ms

---

### 阶段 3: GPU优化 (1周)

**目标**: 50ms → 10ms (5x提升)

#### 优化方向

1. **bfloat16精度**
   - 数据量减半: 7.5GB → 3.75GB
   - GPU计算加速: 1.5-2x
   
2. **Flash Attention**
   - 减少Matmul开销
   - 内存效率提升
   
3. **Kernel融合**
   - 减少GPU kernel启动开销
   - 提升内存带宽利用率

#### 实现方案

```python
# 1. bfloat16
model = model.to(torch.bfloat16)

# 2. Flash Attention
from flash_attn import flash_attn_func

attn_output = flash_attn_func(q, k, v, causal=True)

# 3. Kernel融合
# 使用torch.compile或手写CUDA kernel
```

#### 预期效果

- GPU计算: 100ms → 20ms (5x)
- 数据传输: 减半
- **总延迟**: 50ms → 10ms

---

## 📈 性能预测总结

| 阶段 | 优化 | 延迟 | 提升 | 难度 | 时间 |
|------|------|------|------|------|------|
| 当前 | - | 332ms | 1x | - | - |
| 阶段1 | 共享内存 | 206ms | 1.6x | 中 | 1-2天 |
| 阶段2 | 算子融合 | 50ms | 6.6x | 中 | 2-3天 |
| 阶段3 | GPU优化 | 10ms | 33x | 高 | 1周 |

---

## 🚀 立即行动

### 优先级1: 共享内存POC (今天)

1. **安装依赖**:
```bash
pip install posix_ipc
```

2. **创建测试脚本** `test_shared_memory.py`:
```python
import posix_ipc
import mmap
import numpy as np
import time

# 创建共享内存
shm = posix_ipc.SharedMemory("/test_shm", posix_ipc.O_CREAT, size=100*1024*1024)
mem = mmap.mmap(shm.fd, shm.size)

# 测试写入
data = np.random.rand(1024, 2048).astype(np.float32)
t0 = time.perf_counter()
mem[0:data.nbytes] = data.tobytes()
write_time = time.perf_counter() - t0

# 测试读取
t0 = time.perf_counter()
data2 = np.frombuffer(mem[0:data.nbytes], dtype=np.float32).reshape(1024, 2048)
read_time = time.perf_counter() - t0

print(f"写入: {write_time*1000:.2f}ms")
print(f"读取: {read_time*1000:.2f}ms")
print(f"数据量: {data.nbytes/1024/1024:.2f}MB")

# 清理
shm.unlink()
```

3. **运行测试**:
```bash
python test_shared_memory.py
```

**预期结果**: 写入/读取时间 < 1ms (远快于msgpack的7.5ms)

### 优先级2: 实现共享内存传输 (明天)

修改 `GPUClient` 和 `GPUServer` 使用共享内存

### 优先级3: 性能验证 (后天)

运行完整推理,验证性能提升

---

## 📚 参考资料

- [POSIX共享内存](https://docs.python.org/3/library/multiprocessing.shared_memory.html)
- [ZeroMQ性能优化](https://zeromq.org/socket-api/)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [PyTorch性能优化](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
