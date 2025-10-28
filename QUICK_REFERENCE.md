# 快速参考

## 🚨 当前问题

**RPC 延迟 332ms/call - 应该是 0.2ms！**

可能原因：使用了 TCP 而不是 IPC

## ⚡ 立即执行

### 1. 诊断传输方式

```bash
cd /home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu
python diagnose_transport.py
```

### 2. 确认使用 IPC

```bash
# 停止所有进程
pkill -f server_optimized
pkill -f tee_runner_optimized

# 清理 IPC 文件
rm -f /tmp/tsqp_gpu_server.ipc

# 启动服务器
python server_optimized.py

# 在另一个终端启动客户端
python tee_runner_optimized.py
```

### 3. 切换到 bfloat16

```bash
export LLAMA_DTYPE="bfloat16"
python server_optimized.py &
python tee_runner_optimized.py
```

## 📊 性能目标

| 指标 | 当前 | 目标 | 提升 |
|------|------|------|------|
| 总时间 | 44s | 1-2s | 22-44x |
| RPC 延迟 | 332ms | 0.2ms | 1660x |
| 数据传输 | 3.3GB | <1MB | 3300x |

## 📁 关键文件

- `PERFORMANCE_ANALYSIS.md` - 详细性能分析
- `OPTIMIZATION_ROADMAP.md` - 优化路线图
- `diagnose_transport.py` - 传输诊断脚本
- `BUGFIX_NO_GRAD.md` - 梯度错误修复文档

## 🔧 已修复的问题

- ✅ msgpack 序列化错误 (v2.2)
- ✅ 梯度追踪错误 (v2.3)

## 🎯 下一步

1. 运行诊断脚本
2. 确认使用 IPC
3. 切换到 bfloat16
4. 实现共享内存零拷贝
5. 整层合并优化

## 📞 需要帮助？

提供以下信息：
1. `diagnose_transport.py` 的输出
2. 服务器和客户端的连接日志
3. 性能统计输出
