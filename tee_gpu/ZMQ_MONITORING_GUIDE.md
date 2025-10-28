# ZeroMQ性能监控指南

## 概述

已为`tee_runner_optimized.py`添加详细的ZeroMQ性能监控功能,可以记录每次RPC调用的:
- 序列化时间
- 发送时间
- 接收时间
- 反序列化时间
- 数据传输量
- 吞吐量

## 使用方法

### 1. 运行推理并生成日志

```bash
cd tee_gpu

# 启动GPU服务器
python server_optimized.py &

# 运行TEE客户端(会自动生成zmq_performance.log)
python tee_runner_optimized.py
```

### 2. 查看实时日志

```bash
# 实时查看日志
tail -f zmq_performance.log

# 查看最后20行
tail -20 zmq_performance.log
```

### 3. 分析性能数据

```bash
# 运行分析脚本
python analyze_zmq_performance.py zmq_performance.log
```

## 日志格式

### 详细日志 (zmq_performance.log)

```
ZeroMQ Performance Log - Transport: IPC
====================================================================================================================
ID    Method               Serialize(ms)   Send(ms)     Recv(ms)     Deserialize(ms)   Total(ms)    Sent(KB)     Recv(KB)     Throughput(MB/s)  
====================================================================================================================
1     Init                 0.123           0.045        0.234        0.089             0.491        0.15         125.34       255.23            
2     Embedding            1.234           0.123        0.456        0.234             2.047        32.50        8192.00      4012.45           
3     BatchLinear          2.345           0.234        0.567        0.345             3.491        8192.00      24576.00     9378.12           
...
```

### 字段说明

- **ID**: RPC调用序号
- **Method**: 调用的方法名(Init, Embedding, BatchLinear, Matmul, LMHead)
- **Serialize(ms)**: 序列化时间(毫秒)
- **Send(ms)**: 发送数据时间
- **Recv(ms)**: 接收数据时间
- **Deserialize(ms)**: 反序列化时间
- **Total(ms)**: 总时间
- **Sent(KB)**: 发送的数据量(KB)
- **Recv(KB)**: 接收的数据量(KB)
- **Throughput(MB/s)**: 吞吐量(MB/秒)

## 分析报告示例

运行`analyze_zmq_performance.py`后会生成详细报告:

```
====================================================================================================
                              ZeroMQ Performance Analysis Report                                    
====================================================================================================

OVERALL STATISTICS
----------------------------------------------------------------------------------------------------
Total RPC Calls:      98
Total Time:           332.45 ms
Average Time/Call:    3.393 ms

Time Breakdown (Average per call):
  Serialize:            1.388 ms  ( 40.9%)
  Send:                 0.197 ms  (  5.8%)
  Receive:              0.510 ms  ( 15.0%)
  Deserialize:          0.408 ms  ( 12.0%)

Data Transfer:
  Total Sent:         3750.23 MB  (38.27 KB/call)
  Total Received:     3750.45 MB  (38.27 KB/call)
  Total:              7500.68 MB
  Throughput:         22567.89 MB/s

STATISTICS BY METHOD
----------------------------------------------------------------------------------------------------
Method               Calls    Avg Time(ms)   Total Time(ms)    Avg Sent(KB)    Avg Recv(KB)   % of Total
----------------------------------------------------------------------------------------------------
BatchLinear             96          3.456           331.78           40.50          121.50        99.8%
Embedding                1          2.047             2.047          32.50         8192.00         0.6%
LMHead                   1          1.234             1.234          32.50         4096.00         0.4%

BOTTLENECK ANALYSIS
----------------------------------------------------------------------------------------------------

Top 5 Slowest Calls:
ID       Method               Total(ms)    Serialize        Send      Recv    Deserialize
----------------------------------------------------------------------------------------------------
45       BatchLinear             5.234        2.345      0.234     0.567          0.345
23       BatchLinear             4.891        2.123      0.198     0.534          0.312
67       BatchLinear             4.756        2.089      0.201     0.523          0.298

Top 5 Largest Data Transfers:
ID       Method               Sent(KB)     Recv(KB)     Total(KB)    Throughput(MB/s)
----------------------------------------------------------------------------------------------------
2        Embedding            32.50        8192.00      8224.50           4012.45
3        BatchLinear          8192.00      24576.00     32768.00          9378.12

OPTIMIZATION RECOMMENDATIONS
----------------------------------------------------------------------------------------------------
⚠️  Serialization占40.9%的时间 - 建议:
   1. 使用共享内存替代msgpack序列化
   2. 使用更高效的序列化格式(如protobuf)
   3. 减少传输的数据量(如使用bfloat16)

⚠️  RPC调用次数过多(98次) - 建议:
   1. 算子融合,减少RPC调用次数
   2. 批量处理多个操作
   3. 缓存重复计算的结果
====================================================================================================
```

## 性能优化指南

### 基于监控数据的优化步骤

#### 1. 识别瓶颈

查看分析报告中的"Time Breakdown":
- **序列化 > 30%**: 考虑共享内存
- **发送/接收 > 30%**: 检查传输协议(IPC vs TCP)
- **RPC调用次数 > 50**: 考虑算子融合

#### 2. 针对性优化

**如果序列化是瓶颈**:
```python
# 当前: msgpack序列化
data = msgpack.packb({"buffer": array.tobytes(), "shape": shape})

# 优化: 共享内存
shm.write(array.tobytes())
metadata = msgpack.packb({"shm_offset": 0, "shape": shape})
```

**如果RPC次数过多**:
```python
# 当前: 每层6次RPC
qkv = gpu.batch_linear(...)      # RPC 1
attn1 = gpu.matmul(...)          # RPC 2
attn2 = gpu.matmul(...)          # RPC 3
o = gpu.batch_linear(...)        # RPC 4
gate_up = gpu.batch_linear(...)  # RPC 5
down = gpu.batch_linear(...)     # RPC 6

# 优化: 每层1-2次RPC
output = gpu.fused_layer(input)  # RPC 1
```

**如果数据量过大**:
```python
# 使用bfloat16减少数据量
array = array.astype(np.float16)  # 数据量减半
```

#### 3. 验证优化效果

```bash
# 优化前
python tee_runner_optimized.py
python analyze_zmq_performance.py zmq_performance.log > before.txt

# 应用优化
# ... 修改代码 ...

# 优化后
python tee_runner_optimized.py
python analyze_zmq_performance.py zmq_performance.log > after.txt

# 对比
diff before.txt after.txt
```

## 高级用法

### 自定义日志文件名

```python
# 在tee_runner_optimized.py中
gpu_client = GPUClient(
    ipc_path=DEFAULT_IPC_PATH,
    log_file="my_custom_log.log"  # 自定义日志文件名
)
```

### 只记录特定方法

修改`_send_request`方法:

```python
def _send_request(self, method: str, request: Dict) -> Dict:
    # 只记录BatchLinear和Matmul
    if method in ["BatchLinear", "Matmul"]:
        # ... 记录日志 ...
    else:
        # ... 不记录 ...
```

### 导出为CSV格式

```bash
# 提取数据行(跳过头部和汇总)
grep -E "^[0-9]+" zmq_performance.log > performance.csv

# 添加CSV头
echo "ID,Method,Serialize_ms,Send_ms,Recv_ms,Deserialize_ms,Total_ms,Sent_KB,Recv_KB,Throughput_MBps" | cat - performance.csv > temp && mv temp performance.csv
```

然后可以用Excel或Python pandas分析:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('performance.csv')

# 绘制时间分布
df[['Serialize_ms', 'Send_ms', 'Recv_ms', 'Deserialize_ms']].plot(kind='box')
plt.title('Time Distribution by Stage')
plt.ylabel('Time (ms)')
plt.show()

# 绘制吞吐量趋势
df['Throughput_MBps'].plot()
plt.title('Throughput Over Time')
plt.ylabel('Throughput (MB/s)')
plt.xlabel('Call ID')
plt.show()
```

## 故障排查

### 日志文件为空

**原因**: 程序崩溃或未正常结束

**解决**: 确保程序正常运行完成,或在异常处理中调用`gpu_client.print_stats()`

### 吞吐量异常低

**原因**: 可能使用了TCP而非IPC

**检查**: 查看日志头部的"Transport"字段
```
ZeroMQ Performance Log - Transport: TCP  # 应该是IPC
```

**解决**: 确认`ipc_path`参数正确:
```python
ipc_path = "ipc:///tmp/tsqp_gpu_server.ipc"  # 正确
# 不是 "tcp://127.0.0.1:5555"
```

### 分析脚本报错

**原因**: 日志格式不匹配

**解决**: 确保使用最新版本的`tee_runner_optimized.py`和`analyze_zmq_performance.py`

## 参考资料

- [ZeroMQ性能优化](https://zeromq.org/socket-api/)
- [msgpack性能对比](https://msgpack.org/)
- [共享内存IPC](https://docs.python.org/3/library/multiprocessing.shared_memory.html)
