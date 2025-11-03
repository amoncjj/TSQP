# 共享内存通信优化

## 概述

参考 `shm_broadcast.py` 的设计，优化了 server 和 tee 之间的通信逻辑，实现了智能数据传输：
- **小数据（<10MB）**：使用共享内存环形缓冲区（零拷贝）
- **大数据（≥10MB）**：使用 ZeroMQ（避免共享内存溢出）

## 主要改进

### 1. 共享内存环形缓冲区（ShmRingBuffer）

基于 `shm_broadcast.py` 的设计，实现了一个简化的环形缓冲区：

```python
class ShmRingBuffer:
    """
    Buffer memory layout:
    +-------------------------------+----------------------------------------+
    | chunk0 | chunk1 | ... | chunk | metadata0 | metadata1 | ... | metadata |
    +-------------------------------+----------------------------------------+
    | max_chunks x max_chunk_bytes  | max_chunks x 2 bytes (writer+reader)   |
    
    metadata: [written_flag, read_flag]
    """
```

**特点**：
- 环形缓冲区，避免频繁分配/释放内存
- 使用元数据标志位同步读写
- 支持超时机制，避免死锁
- 最大块大小：10MB
- 默认块数量：10

### 2. 智能传输选择

系统会根据数据大小自动选择传输方式：

```python
def _send_tensor(self, tensor: torch.Tensor) -> Dict:
    data, shape, dtype = self._serialize_tensor_to_bytes(tensor)
    data_size = len(data)
    
    if data_size < MAX_SHM_CHUNK_BYTES:  # 10MB
        # 使用共享内存
        with self.shm_ring_tx.acquire_write(timeout=5.0) as buf:
            buf[:4] = data_size.to_bytes(4, byteorder='little')
            buf[4:4+data_size] = data
        return {"use_shm": True, "shape": shape, "dtype": dtype}
    else:
        # 使用 ZeroMQ
        return {"use_shm": False, "data": data, "shape": shape, "dtype": dtype}
```

### 3. 统计信息

新增了传输方式统计：

```
Transfer Method Breakdown:
  Shared Memory:      245 transfers (95.3%)
                    12.45 MB (78.2%)
  ZeroMQ:              12 transfers ( 4.7%)
                     3.47 MB (21.8%)
```

## 文件修改

### server_optimized.py

1. **新增 ShmRingBuffer 类**
   - 环形缓冲区管理
   - 读写同步机制

2. **修改 ZMQServer 类**
   - 初始化时创建环形缓冲区
   - `_send_tensor()`: 智能选择传输方式
   - `_receive_tensor()`: 从共享内存或ZeroMQ接收
   - 新增传输统计

3. **简化处理函数**
   - `handle_embedding()`: 使用统一的发送/接收接口
   - `handle_batch_linear()`: 同上
   - `handle_matmul()`: 同上
   - `handle_lm_head()`: 同上

### tee_runner_optimized.py

1. **新增 ShmRingBuffer 类**
   - 客户端版本的环形缓冲区

2. **修改 GPUClient 类**
   - 初始化时连接环形缓冲区
   - `_send_tensor()`: 智能选择传输方式
   - `_receive_tensor()`: 从共享内存或ZeroMQ接收
   - 新增传输统计

3. **简化调用接口**
   - `embedding()`: 使用统一的发送/接收接口
   - `batch_linear()`: 同上
   - `matmul()`: 同上
   - `lm_head()`: 同上

## 性能优势

### 共享内存 vs ZeroMQ

| 传输方式 | 延迟 | 吞吐量 | 适用场景 |
|---------|------|--------|---------|
| 共享内存 | 极低（~μs级） | 极高 | 小数据（<10MB） |
| ZeroMQ | 低（~ms级） | 高 | 大数据（≥10MB） |

### 预期收益

对于典型的 LLaMA 推理场景：
- **Embedding**: ~1MB → 使用共享内存
- **Linear层**: ~2-5MB → 使用共享内存
- **Attention**: ~1-3MB → 使用共享内存
- **大矩阵乘法**: 可能>10MB → 使用ZeroMQ

预计 **80-90%** 的数据传输会使用共享内存，显著降低延迟。

## 使用方法

### 启动服务端

```bash
export LLAMA_MODEL_PATH="/path/to/llama3.2-1b"
export LLAMA_GPU_DEVICE="cuda:0"
export LLAMA_IPC_PATH="ipc:///tmp/tsqp_gpu_server.ipc"

python tee_gpu/server_optimized.py
```

### 启动客户端

```bash
export LLAMA_MODEL_PATH="/path/to/llama3.2-1b"
export LLAMA_IPC_PATH="ipc:///tmp/tsqp_gpu_server.ipc"

python tee_gpu/tee_runner_optimized.py
```

### 查看统计信息

服务端和客户端都会在退出时打印传输统计：

**服务端**：
```
Server Transfer Statistics
Shared Memory Transfers:      245 (95.3%)
  Data transferred:          12.45 MB
ZeroMQ Transfers:              12 ( 4.7%)
  Data transferred:           3.47 MB
Total Transfers:              257
Total Data:                  15.92 MB
```

**客户端**：
```
Transfer Method Breakdown:
  Shared Memory:      245 transfers (95.3%)
                    12.45 MB (78.2%)
  ZeroMQ:              12 transfers ( 4.7%)
                     3.47 MB (21.8%)
```

## 配置参数

可以通过环境变量调整：

```bash
# 共享内存块大小阈值（默认10MB）
export TSQP_MAX_SHM_CHUNK_BYTES=$((10 * 1024 * 1024))

# 环形缓冲区块数量（默认10）
export TSQP_SHM_MAX_CHUNKS=10
```

## 测试

运行测试脚本查看不同数据大小的传输方式：

```bash
python test_shm_communication.py
```

## 技术细节

### 同步机制

使用两个标志位实现读写同步：
- `written_flag`: 0=未写入，1=已写入
- `read_flag`: 0=未读取，1=已读取

**状态转换**：
1. 初始状态：`[0, 0]` - 可写入
2. 写入后：`[1, 0]` - 可读取
3. 读取后：`[1, 1]` - 可写入
4. 重置：`[0, 0]` - 循环

### 数据格式

共享内存中的数据格式：
```
+--------+------------------+
| 4 bytes| Variable length  |
+--------+------------------+
| Size   | Tensor data      |
+--------+------------------+
```

### 错误处理

- 超时机制：读写操作默认5秒超时
- 环回机制：缓冲区满时自动循环使用
- 异常恢复：连接断开时自动清理共享内存

## 参考

- `shm_broadcast.py`: vLLM 的共享内存广播实现
- `ShmRingBuffer`: 环形缓冲区设计
- ZeroMQ IPC: 本地进程间通信

## 注意事项

1. **内存限制**：确保系统有足够的共享内存（至少 200MB）
2. **清理机制**：程序异常退出时可能需要手动清理共享内存
3. **并发限制**：当前实现为单读单写，不支持多客户端
4. **数据对齐**：共享内存数据已对齐到64字节边界

## 未来优化

1. 支持多客户端（多读者）
2. 动态调整块大小
3. 压缩传输（bfloat16）
4. 异步传输
5. GPU Direct RDMA（跨节点）
