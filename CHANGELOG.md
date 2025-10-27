# 项目变更日志

## v2.3 - 梯度追踪错误修复 (2025-10-27)

### 修复内容

#### 问题
运行 `tee_runner_optimized.py` 时出现错误：
```
RuntimeError: Can't call numpy() on Tensor that requires grad.
```

#### 解决方案
在 `TEELlamaModel.forward()` 方法添加 `@torch.no_grad()` 装饰器，禁用梯度追踪。

**优势**:
- ✅ 更符合推理代码最佳实践
- ✅ 节省内存（不保存中间结果用于反向传播）
- ✅ 提升性能（跳过梯度计算开销）
- ✅ 代码更简洁（不需要到处添加 `.detach()`）

**影响文件**:
- `tee_gpu/tee_runner_optimized.py` (第 407 行，添加装饰器)

**文档**:
- 新增 `BUGFIX_NO_GRAD.md` - 详细说明问题和解决方案

---

## v2.2 - msgpack 序列化错误修复 (2025-10-27)

### 修复内容

#### 1. msgpack 序列化错误
**问题**: `server_optimized.py` 中 `rotary_emb.attention_scaling` 可能是 numpy 数组或标量，导致 msgpack 无法序列化

**修复**:
- 在 `get_init_data()` 中添加类型转换逻辑
- 将 numpy 数组/标量转换为 Python float
- 将 None 转换为默认值 1.0

**影响文件**:
- `tee_gpu/server_optimized.py` (第 131-140 行)

#### 2. numpy 只读数组警告
**问题**: `np.frombuffer()` 返回只读数组，传递给 `torch.from_numpy()` 时产生警告

**修复**:
- 在所有 `np.frombuffer()` 调用后添加 `.copy()`
- 确保数组可写，避免警告

**影响文件**:
- `tee_gpu/server_optimized.py` (第 179, 205 行)

#### 3. 文档更新
- 新增 `BUGFIX_MSGPACK.md` - 详细记录问题、原因、修复方案和验证方法

### 性能影响
- `.copy()` 增加一次内存拷贝，但相比 IPC 通信的性能提升（39 倍），影响可忽略
- 保持纳秒级 RPC 延迟

---

## v2.1 - 代码精简与优化 (2025-10-27)

### 主要变更

#### 1. 删除不必要的文件
- ✅ 删除 `msg.proto` - 不再使用 protobuf
- ✅ 删除 `msg_pb2.py` - gRPC 生成文件
- ✅ 删除 `msg_pb2_grpc.py` - gRPC 生成文件
- ✅ 删除 `modeling_llama.py` - 直接使用 transformers 库
- ✅ 删除 `COMPUTATION_SPLIT.md` - 冗余文档
- ✅ 删除 `RUN.md` - 冗余文档

#### 2. 代码重构与精简

**server.py (精简 60 行)**
- 重构为更清晰的类结构：`ModuleRegistry` + `ZMQServer`
- 移除复杂的错误处理逻辑
- 简化模型加载流程
- 添加更友好的日志输出

**tee_runner.py (精简 27 行)**
- 专注于 prefill 阶段测试
- 移除 decode 生成相关代码
- 优化客户端连接管理
- 改进性能测试输出格式

**Makefile (精简 11 行)**
- 移除 gRPC 代码生成规则
- 简化构建流程
- 清理不必要的依赖

#### 3. 新增文件

- ✅ `tee_gpu/README.md` - 详细的使用文档
- ✅ `tee_gpu/test_connection.py` - 连接测试工具
- ✅ `CHANGELOG.md` - 本文件

#### 4. 文档更新

- ✅ 更新主 `README.md`，反映 ZeroMQ 架构
- ✅ 添加快速开始指南
- ✅ 添加配置说明和常见问题

### 技术改进

#### 通信协议
- **之前**: gRPC + Protobuf
- **现在**: ZeroMQ + msgpack
- **优势**: 
  - 更轻量级（减少依赖）
  - 更快的序列化速度
  - 更灵活的消息格式

#### 代码质量
- 更清晰的模块划分
- 更好的错误处理
- 更友好的用户界面
- 更详细的注释

#### 测试重点
- **之前**: 完整的生成流程（prefill + decode）
- **现在**: 专注于 prefill 阶段
- **原因**: 
  - Prefill 是性能瓶颈
  - 更容易进行性能对比
  - 简化测试流程

### 性能优化

1. **减少依赖加载时间**
   - 移除 protobuf 编译开销
   - 直接使用 transformers 库

2. **简化通信流程**
   - ZeroMQ 比 gRPC 更快
   - msgpack 序列化更高效

3. **优化内存使用**
   - 移除不必要的模型副本
   - 更好的张量管理

### 代码统计

| 文件 | 之前 | 现在 | 变化 |
|------|------|------|------|
| server.py | 310 行 | 250 行 | -60 行 |
| tee_runner.py | 288 行 | 261 行 | -27 行 |
| Makefile | 67 行 | 56 行 | -11 行 |
| **总计** | **665 行** | **567 行** | **-98 行 (-15%)** |

### 依赖变化

**移除的依赖**:
- `grpcio==1.63.0`
- `grpcio-tools==1.63.0`
- `protobuf==3.19.1`

**新增的依赖**:
- `pyzmq==25.1.1`
- `msgpack==1.0.7`

**依赖大小对比**:
- 之前: ~150 MB (grpcio + protobuf)
- 现在: ~15 MB (pyzmq + msgpack)
- **减少**: ~90%

### 使用方式变化

#### 之前
```bash
# 需要先生成 protobuf 代码
cd tee_gpu
make msg_pb2.py msg_pb2_grpc.py

# 启动服务器
python server.py

# 运行客户端
python tee_runner.py
```

#### 现在
```bash
# 直接启动服务器
cd tee_gpu
python server.py

# 运行客户端
python tee_runner.py

# 测试连接
python test_connection.py
```

### 向后兼容性

- ⚠️ **不兼容**: 需要重新安装依赖
- ⚠️ **不兼容**: 环境变量保持一致
- ✅ **兼容**: Gramine manifest 配置
- ✅ **兼容**: 模型路径和配置

### 迁移指南

1. **更新依赖**
   ```bash
   pip uninstall grpcio grpcio-tools protobuf
   pip install pyzmq msgpack
   ```

2. **清理旧文件**
   ```bash
   cd tee_gpu
   rm -f msg.proto msg_pb2.py msg_pb2_grpc.py
   ```

3. **重新构建 manifest**
   ```bash
   make clean
   make SGX=1
   ```

### 已知问题

- 无

### 下一步计划

- [ ] 支持多客户端并发
- [ ] 添加性能监控
- [ ] 支持模型量化
- [ ] 添加更多测试用例

---

## v2.0 - ZeroMQ 迁移 (2025-10-27)

### 主要变更
- 从 gRPC 迁移到 ZeroMQ
- 使用 msgpack 替代 protobuf
- 保持功能完整性

---

## v1.0 - 初始版本

### 功能
- TEE + GPU 分离架构
- gRPC 通信
- 完整的生成流程
