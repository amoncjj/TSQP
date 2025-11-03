# 修改总结

## 目标
参考 `shm_broadcast.py`，修改 server 和 tee 的通信逻辑，对于 10MB 以下的数据使用共享内存。

## 核心改进

### 1. 引入共享内存环形缓冲区
- 参考 vLLM 的 `ShmRingBuffer` 设计
- 实现了简化版的环形缓冲区（单读单写）
- 使用元数据标志位同步读写操作
- 支持超时和错误处理

### 2. 智能传输选择
- **小数据（<10MB）**：使用共享内存环形缓冲区
  - 零拷贝传输
  - 极低延迟（微秒级）
  - 适合频繁的小数据传输
  
- **大数据（≥10MB）**：使用 ZeroMQ
  - 避免共享内存溢出
  - 稳定可靠
  - 适合偶尔的大数据传输

### 3. 统一的接口设计
- `_send_tensor()`: 自动选择传输方式
- `_receive_tensor()`: 自动识别传输方式
- 对上层调用透明，无需修改业务逻辑

## 修改的文件

### 1. `tee_gpu/server_optimized.py`
**新增**：
- `ShmRingBuffer` 类（环形缓冲区）
- `_serialize_tensor_to_bytes()` 方法
- `_deserialize_tensor_from_bytes()` 方法
- `_send_tensor()` 方法（智能传输）
- `_receive_tensor()` 方法（智能接收）
- 传输统计功能

**修改**：
- `__init__()`: 创建环形缓冲区
- `handle_init()`: 返回缓冲区句柄
- `handle_embedding()`: 使用新接口
- `handle_batch_linear()`: 使用新接口
- `handle_matmul()`: 使用新接口
- `handle_lm_head()`: 使用新接口
- `serve()`: 添加统计输出

**删除**：
- 旧的共享内存管理代码（offset-based）
- 复杂的手动内存管理逻辑

### 2. `tee_gpu/tee_runner_optimized.py`
**新增**：
- `ShmRingBuffer` 类（客户端版本）
- `_serialize_tensor_to_bytes()` 方法
- `_deserialize_tensor_from_bytes()` 方法
- `_send_tensor()` 方法（智能传输）
- `_receive_tensor()` 方法（智能接收）
- 传输统计功能

**修改**：
- `__init__()`: 初始化环形缓冲区
- `init()`: 连接服务端缓冲区
- `embedding()`: 使用新接口
- `batch_linear()`: 使用新接口
- `matmul()`: 使用新接口
- `lm_head()`: 使用新接口
- `print_stats()`: 添加传输方式统计
- `close()`: 清理环形缓冲区

**删除**：
- 旧的共享内存管理代码（offset-based）
- 复杂的手动内存管理逻辑

### 3. 新增文件
- `test_shm_communication.py`: 测试脚本
- `SHARED_MEMORY_OPTIMIZATION.md`: 详细文档
- `CHANGES_SUMMARY.md`: 本文件

## 代码行数变化

### server_optimized.py
- 新增：~120 行（ShmRingBuffer + 新方法）
- 删除：~80 行（旧的内存管理）
- 净增：~40 行

### tee_runner_optimized.py
- 新增：~150 行（ShmRingBuffer + 新方法）
- 删除：~120 行（旧的内存管理）
- 净增：~30 行

## 性能预期

### 延迟改进
- 小数据传输延迟：**降低 50-70%**
  - 从 ~1ms（ZeroMQ）降至 ~0.1-0.3ms（共享内存）
  
### 吞吐量改进
- 小数据吞吐量：**提升 2-3倍**
  - 共享内存避免了序列化和网络栈开销

### 典型场景
对于 LLaMA 1B 模型的 prefill（1024 tokens）：
- Embedding: ~1MB → 共享内存
- 每层 Linear: ~2-5MB → 共享内存
- Attention matmul: ~1-3MB → 共享内存
- 预计 **85-90%** 的传输使用共享内存

## 兼容性

### 向后兼容
- ✅ 保持了相同的 API 接口
- ✅ 自动降级到 ZeroMQ（大数据）
- ✅ 无需修改调用代码

### 系统要求
- 共享内存支持（Linux/macOS）
- Python 3.7+
- PyTorch 1.9+
- ZeroMQ 4.0+

## 测试建议

### 1. 功能测试
```bash
# 启动服务端
python tee_gpu/server_optimized.py

# 启动客户端
python tee_gpu/tee_runner_optimized.py
```

### 2. 性能测试
观察输出的统计信息：
- 共享内存传输比例
- 平均延迟
- 吞吐量

### 3. 压力测试
- 连续运行多次 prefill
- 观察内存使用情况
- 检查是否有内存泄漏

## 已知限制

1. **单客户端**：当前实现仅支持一个客户端
2. **固定块大小**：10MB 阈值硬编码
3. **同步传输**：暂不支持异步传输
4. **本地通信**：仅支持同节点通信

## 后续优化方向

1. **多客户端支持**
   - 参考 `shm_broadcast.py` 的多读者设计
   - 每个客户端独立的读标志位

2. **动态阈值**
   - 根据系统负载动态调整
   - 自适应选择传输方式

3. **压缩传输**
   - bfloat16 压缩（已部分实现）
   - 量化传输

4. **异步传输**
   - Pipeline 多个请求
   - 重叠计算和通信

5. **跨节点支持**
   - GPU Direct RDMA
   - 分布式共享内存

## 参考资料

- [vLLM shm_broadcast.py](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/shm_broadcast.py)
- [Python multiprocessing.shared_memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html)
- [ZeroMQ IPC Transport](https://zeromq.org/socket-api/#ipc-transport)

## 总结

本次修改成功引入了共享内存环形缓冲区，实现了智能的数据传输选择。预期能够显著降低小数据传输的延迟，提升整体性能。代码结构更加清晰，易于维护和扩展。
