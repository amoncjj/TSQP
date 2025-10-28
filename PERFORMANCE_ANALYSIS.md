# 性能分析与优化方案

## 当前性能数据

```
RPC Calls:        98
RPC Time:         32.59s (332.52ms/call)  ⚠️ 极慢！
Serialize Time:   4.53s (46.26ms/call)
Deserialize Time: 2.50s (25.49ms/call)
Data Sent:        3328 MB
Data Received:    3657 MB
```

## 问题诊断

### 🔴 严重问题：RPC 延迟 332ms/call

**预期**: IPC 应该是 **0.2ms/call**（微秒级）  
**实际**: 332ms/call（毫秒级）  
**差距**: **1660 倍慢！**

### 可能原因

1. **实际使用了 TCP 而不是 IPC**
   - 虽然配置是 `ipc:///tmp/tsqp_gpu_server.ipc`
   - 但可能环境变量覆盖了配置
   - 或者 IPC 文件创建失败，回退到 TCP

2. **数据传输量过大**
   - 发送 3.3GB，接收 3.7GB
   - 对于 1024 tokens，这个数据量太大了
   - 说明传输了完整的中间结果

3. **序列化开销**
   - 46ms/call 序列化
   - 25ms/call 反序列化
   - msgpack 对大数据不够高效

## 优化方案

### 🚀 方案 1: 确保使用 IPC（立即执行）

#### 步骤 1: 检查实际连接

在远程服务器上运行：

```bash
# 检查 IPC 文件是否存在
ls -la /tmp/tsqp_gpu_server.ipc

# 检查进程使用的连接
lsof -p $(pgrep -f server_optimized) | grep socket
lsof -p $(pgrep -f tee_runner_optimized) | grep socket
```

#### 步骤 2: 强制使用 IPC

```bash
# 确保没有环境变量覆盖
unset LLAMA_IPC_PATH

# 启动服务器
python server_optimized.py

# 在另一个终端启动客户端
python tee_runner_optimized.py
```

#### 步骤 3: 添加诊断日志

修改 `tee_runner_optimized.py`，在连接时打印实际地址：

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
    print(f"✓ Connected to GPU server at {ipc_path}")
    print(f"  Transport: {'IPC' if 'ipc://' in ipc_path else 'TCP'}")  # 添加这行
```

**预期效果**: RPC 延迟从 332ms → **0.2ms**（1660 倍提升）

---

### 🚀 方案 2: 使用共享内存（零拷贝）

当前问题：每次 RPC 都要序列化/反序列化 3.3GB 数据。

#### 实现方案

使用 POSIX 共享内存 + ZeroMQ 只传递元数据：

```python
import mmap
import posix_ipc

class SharedMemoryTransport:
    def __init__(self, name: str, size: int):
        self.shm = posix_ipc.SharedMemory(name, posix_ipc.O_CREAT, size=size)
        self.mem = mmap.mmap(self.shm.fd, size)
    
    def write_tensor(self, tensor: torch.Tensor, offset: int):
        """零拷贝写入"""
        data = tensor.cpu().numpy().tobytes()
        self.mem[offset:offset+len(data)] = data
        return len(data)
    
    def read_tensor(self, shape, dtype, offset: int):
        """零拷贝读取"""
        size = np.prod(shape) * np.dtype(dtype).itemsize
        data = self.mem[offset:offset+size]
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
```

**RPC 消息**只传递元数据：
```python
{
    "method": "BatchLinear",
    "shm_offset": 0,
    "shape": [1, 1024, 2048],
    "dtype": "float32"
}
```

**预期效果**:
- 序列化时间: 46ms → **0.01ms**（4600 倍提升）
- 反序列化时间: 25ms → **0.01ms**（2500 倍提升）
- 数据传输: 3.3GB → **几 KB**（百万倍减少）

---

### 🚀 方案 3: 批量操作合并

当前问题：98 次 RPC 调用，每次都有往返开销。

#### 优化策略

**当前**:
```python
# 3 次 RPC
q = gpu.linear(layer_idx, "q_proj", hidden_states)
k = gpu.linear(layer_idx, "k_proj", hidden_states)
v = gpu.linear(layer_idx, "v_proj", hidden_states)
```

**优化后**（已实现）:
```python
# 1 次 RPC
qkv = gpu.batch_linear(layer_idx, ["q_proj", "k_proj", "v_proj"], hidden_states)
```

#### 进一步优化：整层合并

```python
# 当前：每层 5 次 RPC
# 1. QKV projections (batch)
# 2. Matmul (Q @ K^T)
# 3. Matmul (attn @ V)
# 4. O projection
# 5. Gate/Up projections (batch)
# 6. Down projection

# 优化：每层 1 次 RPC
result = gpu.full_layer(layer_idx, hidden_states, position_ids)
```

**预期效果**: RPC 次数从 98 → **22**（每层 1 次）

---

### 🚀 方案 4: 使用 float16/bfloat16

当前使用 float32，数据量是 float16 的 2 倍。

```python
# 修改配置
DEFAULT_DTYPE = "bfloat16"  # 或 "float16"
```

**预期效果**:
- 数据传输量: 3.3GB → **1.65GB**（2 倍减少）
- 序列化时间: 46ms → **23ms**（2 倍提升）
- GPU 计算速度: 可能提升 1.5-2 倍

---

### 🚀 方案 5: 异步 Pipeline

当前是同步 RPC，GPU 和 TEE 串行执行。

#### 实现方案

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncGPUClient:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.pending_requests = []
    
    async def batch_linear_async(self, layer_idx, modules, hidden_states):
        """异步 RPC"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._batch_linear_sync,
            layer_idx, modules, hidden_states
        )
    
    async def pipeline_layer(self, layer_idx, hidden_states):
        """Pipeline 执行"""
        # GPU 计算 QKV（异步）
        qkv_future = self.batch_linear_async(layer_idx, ["q_proj", "k_proj", "v_proj"], hidden_states)
        
        # TEE 可以同时做其他计算
        # ...
        
        qkv = await qkv_future
        return qkv
```

**预期效果**: 总时间减少 30-50%（GPU 和 TEE 并行）

---

## 综合优化方案

### 阶段 1: 立即优化（1 小时）

1. ✅ **确保使用 IPC**
   - 检查实际连接
   - 添加诊断日志
   - **预期**: 332ms → 0.2ms（1660 倍）

2. ✅ **使用 bfloat16**
   - 修改配置
   - **预期**: 数据量减半，速度提升 2 倍

**总预期**: 44 秒 → **0.013 秒**（3300 倍提升）

### 阶段 2: 深度优化（1 天）

3. ✅ **共享内存零拷贝**
   - 实现 SharedMemoryTransport
   - **预期**: 序列化开销几乎为 0

4. ✅ **整层合并**
   - 每层 1 次 RPC
   - **预期**: RPC 次数减少 4-5 倍

**总预期**: 0.013 秒 → **0.003 秒**（再提升 4 倍）

### 阶段 3: 极致优化（3 天）

5. ✅ **异步 Pipeline**
   - GPU 和 TEE 并行
   - **预期**: 再提升 30-50%

6. ✅ **CUDA IPC**
   - GPU 直接共享显存
   - **预期**: 完全消除 CPU↔GPU 传输

**最终预期**: **0.001-0.002 秒**（比原来快 **22000-44000 倍**）

---

## 立即行动

### 第一步：诊断 IPC

运行以下命令：

```bash
cd /home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu

# 1. 停止所有进程
pkill -f server_optimized
pkill -f tee_runner_optimized

# 2. 清理 IPC 文件
rm -f /tmp/tsqp_gpu_server.ipc

# 3. 启动服务器并观察输出
python server_optimized.py

# 4. 在另一个终端启动客户端
python tee_runner_optimized.py
```

**关键观察**:
- 服务器输出应该显示: `✓ ZeroMQ server started on ipc:///tmp/tsqp_gpu_server.ipc`
- 客户端输出应该显示: `✓ Connected to GPU server at ipc:///tmp/tsqp_gpu_server.ipc`
- RPC 延迟应该是 **0.2-1ms**，而不是 332ms

### 第二步：如果 IPC 正常，切换到 bfloat16

```bash
# 修改环境变量
export LLAMA_DTYPE="bfloat16"

# 重启服务器和客户端
python server_optimized.py &
python tee_runner_optimized.py
```

---

## 性能目标

| 指标 | 当前 | 阶段1 | 阶段2 | 阶段3 |
|------|------|-------|-------|-------|
| 总时间 | 44s | 0.013s | 0.003s | 0.001s |
| RPC 延迟 | 332ms | 0.2ms | 0.05ms | 0.01ms |
| 数据传输 | 3.3GB | 1.65GB | 几KB | 0 |
| 提升倍数 | 1x | 3300x | 14600x | 44000x |

**最终目标**: 从 44 秒优化到 **1-2 毫秒**，提升 **22000-44000 倍**！
