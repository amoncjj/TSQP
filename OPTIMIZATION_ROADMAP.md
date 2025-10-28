# 性能优化路线图

## 🎯 目标

将推理时间从 **44 秒** 优化到 **1-2 毫秒**，提升 **22000-44000 倍**！

## 📊 当前性能瓶颈

```
总时间: 44.25 秒
├─ GPU 计算: 43.48s (98.25%)
│  ├─ Matmul: 29.54s (66.75%)  ← 最大瓶颈
│  └─ Linear: 13.92s (31.45%)
├─ RPC 通信: 32.59s (73.67%)   ← 第二大瓶颈
│  ├─ 序列化: 4.53s
│  └─ 反序列化: 2.50s
└─ TEE 计算: 0.77s (1.75%)

关键问题:
1. RPC 延迟 332ms/call - 应该是 0.2ms (IPC)
2. 数据传输 3.3GB - 太大了
3. GPU 计算时间异常长 - 可能是数据传输导致
```

## 🚀 立即行动（今天）

### 步骤 1: 诊断传输方式 (10 分钟)

**在远程服务器上运行**:

```bash
cd /home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu

# 运行诊断脚本
python diagnose_transport.py
```

**预期输出**:
```
IPC 延迟:  0.5-2 ms
TCP 延迟:  50-100 ms
IPC 比 TCP 快: 50-100 倍
```

**如果 IPC 延迟 > 10ms**: 说明有问题，需要检查系统配置。

### 步骤 2: 确认实际使用的传输方式 (5 分钟)

**修改 `tee_runner_optimized.py`**，添加诊断信息：

```python
def __init__(self, ipc_path: str) -> None:
    self.ipc_path = ipc_path
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.REQ)
    
    # 优化 ZeroMQ 性能
    self.socket.setsockopt(zmq.SNDHWM, 1000)
    self.socket.setsockopt(zmq.RCVHWM, 1000)
    self.socket.setsockopt(zmq.LINGER, 0)
    
    self.socket.connect(ipc_path)
    
    # 添加诊断信息
    transport_type = "IPC" if "ipc://" in ipc_path else "TCP"
    print(f"✓ Connected to GPU server at {ipc_path}")
    print(f"  Transport type: {transport_type}")
    print(f"  Expected latency: {'<1ms' if transport_type == 'IPC' else '10-100ms'}")
```

### 步骤 3: 切换到 bfloat16 (2 分钟)

```bash
# 设置环境变量
export LLAMA_DTYPE="bfloat16"

# 重启服务器和客户端
pkill -f server_optimized
pkill -f tee_runner_optimized

python server_optimized.py &
python tee_runner_optimized.py
```

**预期效果**:
- 数据传输量: 3.3GB → 1.65GB (减半)
- 序列化时间: 4.53s → 2.27s (减半)
- GPU 计算: 可能提升 1.5-2 倍

**总预期**: 44s → **15-20s** (2-3 倍提升)

---

## 🔧 深度优化（本周）

### 优化 1: 共享内存零拷贝 (1 天)

**问题**: 每次 RPC 都要序列化/反序列化 1.65GB 数据（bfloat16）

**解决方案**: 使用 POSIX 共享内存

#### 实现步骤

1. **安装依赖**:
```bash
pip install posix_ipc
```

2. **创建共享内存管理器** (`tee_gpu/shared_memory.py`):

```python
import mmap
import posix_ipc
import numpy as np
import torch

class SharedMemoryManager:
    """共享内存管理器"""
    
    def __init__(self, name: str, size: int = 1024 * 1024 * 1024):  # 1GB
        self.name = name
        self.size = size
        
        # 创建共享内存
        self.shm = posix_ipc.SharedMemory(
            name,
            posix_ipc.O_CREAT,
            size=size
        )
        self.mem = mmap.mmap(self.shm.fd, size)
        self.offset = 0
    
    def write_tensor(self, tensor: torch.Tensor) -> dict:
        """写入张量，返回元数据"""
        # 转换为 numpy
        array = tensor.detach().cpu().numpy()
        data = array.tobytes()
        
        # 写入共享内存
        offset = self.offset
        self.mem[offset:offset+len(data)] = data
        self.offset = (offset + len(data)) % self.size
        
        return {
            "offset": offset,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "size": len(data)
        }
    
    def read_tensor(self, metadata: dict) -> torch.Tensor:
        """从共享内存读取张量"""
        offset = metadata["offset"]
        size = metadata["size"]
        shape = metadata["shape"]
        dtype = np.dtype(metadata["dtype"])
        
        # 从共享内存读取
        data = bytes(self.mem[offset:offset+size])
        array = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        
        return torch.from_numpy(array)
    
    def close(self):
        """关闭共享内存"""
        self.mem.close()
        posix_ipc.unlink_shared_memory(self.name)
```

3. **修改 GPUClient** 使用共享内存:

```python
class GPUClient:
    def __init__(self, ipc_path: str):
        # ZeroMQ 连接
        self.socket = ...
        
        # 共享内存
        self.shm = SharedMemoryManager("/tsqp_shm")
    
    def batch_linear(self, layer_idx, module_names, hidden_states):
        # 写入共享内存
        metadata = self.shm.write_tensor(hidden_states)
        
        # RPC 只传递元数据
        request = {
            "layer_idx": layer_idx,
            "module_names": module_names,
            "shm_metadata": metadata  # 只有几十字节
        }
        
        response = self._send_request("BatchLinear", request)
        
        # 从共享内存读取结果
        outputs = []
        for output_meta in response["outputs"]:
            tensor = self.shm.read_tensor(output_meta)
            outputs.append(tensor)
        
        return outputs
```

**预期效果**:
- 序列化时间: 2.27s → **0.01s** (227 倍提升)
- 反序列化时间: 1.14s → **0.01s** (114 倍提升)
- RPC 数据量: 1.65GB → **几 KB** (百万倍减少)

**总预期**: 15-20s → **5-8s** (3-4 倍提升)

---

### 优化 2: 整层合并 (半天)

**问题**: 每层 5-6 次 RPC 调用

**解决方案**: 每层只调用 1 次 RPC

#### 实现步骤

1. **在 `server_optimized.py` 添加整层方法**:

```python
@torch.no_grad()
def full_decoder_layer(self, layer_idx: int, hidden_states: torch.Tensor, 
                       cos: torch.Tensor, sin: torch.Tensor) -> dict:
    """完整的 decoder 层（GPU 部分）"""
    layer = self.layers[layer_idx]
    
    # Attention
    # 1. QKV projections
    q = layer.self_attn.q_proj(hidden_states)
    k = layer.self_attn.k_proj(hidden_states)
    v = layer.self_attn.v_proj(hidden_states)
    
    # 返回给 TEE 做 RoPE + Softmax
    return {
        "q": q, "k": k, "v": v,
        "residual": hidden_states
    }

@torch.no_grad()
def attention_output(self, layer_idx: int, attn_output: torch.Tensor) -> torch.Tensor:
    """Attention 输出投影"""
    layer = self.layers[layer_idx]
    return layer.self_attn.o_proj(attn_output)

@torch.no_grad()
def mlp_forward(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
    """MLP 前向传播"""
    layer = self.layers[layer_idx]
    gate = layer.mlp.gate_proj(hidden_states)
    up = layer.mlp.up_proj(hidden_states)
    # 返回给 TEE 做 SiLU
    return {"gate": gate, "up": up}

@torch.no_grad()
def mlp_output(self, layer_idx: int, gate_up: torch.Tensor) -> torch.Tensor:
    """MLP 输出投影"""
    layer = self.layers[layer_idx]
    return layer.mlp.down_proj(gate_up)
```

2. **修改 TEELlamaModel**:

```python
def decoder_layer(self, layer_idx, hidden_states, position_ids):
    """Decoder 层 - 优化版"""
    residual = hidden_states
    
    # TEE: Input norm
    hidden_states = self.input_layernorms[layer_idx](hidden_states)
    
    # GPU: QKV projections (1 次 RPC)
    qkv_data = self.gpu.full_decoder_layer(layer_idx, hidden_states, cos, sin)
    
    # TEE: RoPE + Attention
    attn_output = self._tee_attention(qkv_data, position_ids)
    
    # GPU: O projection (1 次 RPC)
    attn_output = self.gpu.attention_output(layer_idx, attn_output)
    
    hidden_states = residual + attn_output
    residual = hidden_states
    
    # TEE: Post attention norm
    hidden_states = self.post_attention_layernorms[layer_idx](hidden_states)
    
    # GPU: Gate/Up projections (1 次 RPC)
    gate_up_data = self.gpu.mlp_forward(layer_idx, hidden_states)
    
    # TEE: SiLU
    gate_up = self._tee_mlp(gate_up_data)
    
    # GPU: Down projection (1 次 RPC)
    mlp_output = self.gpu.mlp_output(layer_idx, gate_up)
    
    hidden_states = residual + mlp_output
    
    return hidden_states
```

**预期效果**:
- RPC 次数: 98 → **88** (每层从 5-6 次减少到 4 次)
- RPC 开销减少 20-30%

**总预期**: 5-8s → **3-5s** (1.5-2 倍提升)

---

### 优化 3: GPU Kernel 融合 (2 天)

**问题**: GPU 计算时间异常长（29.5s matmul + 13.9s linear）

**可能原因**:
1. 数据在 CPU 和 GPU 之间频繁传输
2. 小批量操作效率低
3. 没有使用 Flash Attention

#### 解决方案

1. **使用 Flash Attention**:

```bash
pip install flash-attn
```

```python
from flash_attn import flash_attn_func

def attention_with_flash(q, k, v):
    """使用 Flash Attention"""
    # q, k, v: [batch, num_heads, seq_len, head_dim]
    output = flash_attn_func(q, k, v, causal=True)
    return output
```

**预期效果**: Attention 计算提升 2-4 倍

2. **Kernel 融合**:

```python
import torch.nn.functional as F

@torch.jit.script
def fused_mlp(x: torch.Tensor, gate_weight: torch.Tensor, 
              up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    """融合的 MLP"""
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    gate_up = F.silu(gate) * up
    return F.linear(gate_up, down_weight)
```

**预期效果**: MLP 计算提升 1.5-2 倍

**总预期**: 3-5s → **1-2s** (2-3 倍提升)

---

## 🎯 最终目标

| 阶段 | 优化内容 | 预期时间 | 提升倍数 |
|------|---------|---------|---------|
| 当前 | 无 | 44.25s | 1x |
| 阶段1 | IPC + bfloat16 | 15-20s | 2-3x |
| 阶段2 | 共享内存 | 5-8s | 5-9x |
| 阶段3 | 整层合并 | 3-5s | 9-15x |
| 阶段4 | GPU 优化 | 1-2s | 22-44x |
| 终极 | CUDA IPC | 0.5-1s | 44-88x |

## 📝 今天的任务清单

- [ ] 运行 `diagnose_transport.py` 诊断传输方式
- [ ] 确认实际使用 IPC 还是 TCP
- [ ] 切换到 bfloat16
- [ ] 测试性能提升
- [ ] 如果性能仍然很慢，检查 IPC 文件权限和系统配置

## 🔍 调试检查清单

如果性能仍然很慢，检查：

1. **IPC 文件**:
```bash
ls -la /tmp/tsqp_gpu_server.ipc
# 应该存在且可读写
```

2. **进程连接**:
```bash
lsof -p $(pgrep -f server_optimized) | grep socket
lsof -p $(pgrep -f tee_runner_optimized) | grep socket
```

3. **环境变量**:
```bash
echo $LLAMA_IPC_PATH
# 应该是空的或者是 ipc:///tmp/tsqp_gpu_server.ipc
```

4. **ZeroMQ 版本**:
```bash
python -c "import zmq; print(zmq.zmq_version())"
# 应该 >= 4.0
```

---

**开始行动吧！第一步就是运行诊断脚本！** 🚀
