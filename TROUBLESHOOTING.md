# 故障排查指南

## 当前问题

您遇到的错误信息被截断了，无法看到完整的错误类型。错误发生在 `tee_runner_optimized.py` 的 `batch_linear` 调用处。

## 诊断步骤

### 步骤 1: 运行诊断脚本

在**远程服务器**上运行诊断脚本来获取完整错误信息：

```bash
cd /home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu

# 确保服务器正在运行
python server_optimized.py &

# 运行诊断脚本
python test_simple.py
```

这个脚本会：
1. 测试 ZeroMQ 连接
2. 测试 Init 请求
3. 测试 Embedding 请求
4. 测试 BatchLinear 请求（这是出错的地方）

如果 BatchLinear 失败，脚本会打印**完整的错误堆栈**。

### 步骤 2: 检查服务器日志

查看服务器端是否有错误输出：

```bash
# 如果服务器在后台运行，查看输出
fg

# 或者重新启动服务器并观察输出
python server_optimized.py
```

### 步骤 3: 检查常见问题

#### 问题 1: IPC 文件权限

```bash
# 检查 IPC 文件
ls -la /tmp/tsqp_gpu_server.ipc

# 如果文件存在但无法访问，删除并重启服务器
rm -f /tmp/tsqp_gpu_server.ipc
python server_optimized.py
```

#### 问题 2: 模型路径

确认模型路径正确：

```bash
# 检查模型是否存在
ls -la /home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b

# 如果路径不对，设置环境变量
export LLAMA_MODEL_PATH="/正确的/模型/路径"
python server_optimized.py
```

#### 问题 3: GPU 可用性

```bash
# 检查 GPU
nvidia-smi

# 检查 PyTorch 是否能访问 GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 问题 4: 依赖包版本

```bash
# 检查关键包版本
pip list | grep -E "torch|zmq|msgpack|transformers"

# 如果版本不对，重新安装
pip install -r requirements.txt
```

## 可能的错误原因

基于错误发生的位置（`batch_linear` 调用），可能的原因包括：

### 1. 序列化错误
- **症状**: msgpack 无法序列化某些数据类型
- **解决**: 已在 `server_optimized.py` 中修复（v2.2）
- **验证**: 运行 `test_simple.py` 的 BatchLinear 测试

### 2. 张量形状不匹配
- **症状**: 输入张量形状与模型期望不符
- **解决**: 检查 `hidden_states` 的形状是否为 `(batch_size, seq_len, hidden_size)`
- **验证**: 在 `test_simple.py` 中打印张量形状

### 3. 数据类型错误
- **症状**: numpy/torch 数据类型不匹配
- **解决**: 确保所有张量都是 `float32`
- **验证**: 检查 `tensor_cpu.dtype`

### 4. 内存不足
- **症状**: GPU 内存不足
- **解决**: 减小 batch size 或序列长度
- **验证**: 运行 `nvidia-smi` 查看内存使用

## 获取完整错误信息

如果上述步骤都无法解决问题，请提供以下信息：

### 1. 运行诊断脚本的完整输出

```bash
python test_simple.py 2>&1 | tee diagnostic_output.txt
```

### 2. 服务器端的完整输出

```bash
python server_optimized.py 2>&1 | tee server_output.txt
```

### 3. 客户端的完整输出

```bash
python tee_runner_optimized.py 2>&1 | tee client_output.txt
```

### 4. 环境信息

```bash
# Python 版本
python --version

# PyTorch 版本
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# CUDA 版本
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"

# GPU 信息
nvidia-smi

# 依赖包版本
pip list | grep -E "torch|zmq|msgpack|transformers|numpy"
```

## 快速修复建议

如果您急需运行代码，可以尝试：

### 方案 1: 使用标准版本（TCP 通信）

```bash
# 使用 server.py 和 tee_runner.py（已验证可用）
python server.py &
python tee_runner.py
```

### 方案 2: 降低复杂度

修改 `tee_runner_optimized.py`，减小测试规模：

```python
# 第 479 行附近
PREFILL_TOKEN_LENGTH = 32  # 从 1024 改为 32
```

### 方案 3: 添加详细日志

在 `tee_runner_optimized.py` 的 `batch_linear` 方法中添加调试信息：

```python
def batch_linear(self, layer_idx: int, module_names: List[str], hidden_states: torch.Tensor) -> List[torch.Tensor]:
    """批量 Linear"""
    print(f"[DEBUG] batch_linear: layer_idx={layer_idx}, modules={module_names}")
    print(f"[DEBUG] hidden_states: shape={hidden_states.shape}, dtype={hidden_states.dtype}")
    
    tensor_cpu = hidden_states.cpu().contiguous() if hidden_states.is_cuda else hidden_states.contiguous()
    print(f"[DEBUG] tensor_cpu: shape={tensor_cpu.shape}, dtype={tensor_cpu.dtype}")
    
    # ... 其余代码
```

## 联系支持

如果问题仍未解决，请提供：
1. 诊断脚本的完整输出
2. 服务器和客户端的完整日志
3. 环境信息
4. 您所做的任何修改

这将帮助快速定位问题。
