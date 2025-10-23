# Prefill-Only 模式简明指南

## 概述

代码已简化为**只支持 Prefill 模式**，移除了所有 decode 相关代码。

## 核心功能

- ✅ **只进行 Prefill**: 前向传播计算 logits，不生成新 token
- ✅ **可配置长度**: 通过环境变量设置 prefill token 长度
- ✅ **支持批处理**: 可设置 batch size
- ✅ **性能统计**: 自动计算吞吐量（tokens/second）

## 快速使用

### 1. 纯 TEE 模式

```bash
cd /home/junjie_chen@idm.teecertlabs.com/TSQP/tee_only_llama

# 基本用法（默认 prefill_length=128）
python tee_runner.py

# 自定义 prefill 长度
export LLAMA_PREFILL_LENGTH=256
python tee_runner.py

# 批处理
export LLAMA_PREFILL_LENGTH=512
export LLAMA_TEE_BATCH_SIZE=4
python tee_runner.py
```

### 2. TEE+GPU 协同模式

```bash
cd /home/junjie_chen@idm.teecertlabs.com/TSQP/tee_gpu

# 启动 GPU 服务器
python server.py &

# 运行 TEE 客户端
export LLAMA_PREFILL_LENGTH=256
python tee_runner.py
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLAMA_MODEL_PATH` | 模型路径 | `/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b` |
| `LLAMA_PREFILL_LENGTH` | Prefill token 长度 | `128` |
| `LLAMA_TEE_BATCH_SIZE` | 批大小 | `1` |
| `LLAMA_PROMPT_PATH` | Prompt 文件路径 | - |
| `LLAMA_TEE_RESULT_PATH` | 结果输出路径 | `tee_only_results.json` |
| `LLAMA_GPU_ENDPOINT` | GPU 服务端点 | `localhost:50051` |

## 输出示例

```json
{
  "mode": "prefill-only",
  "prompt_count": 1,
  "prefill_length": 128,
  "time_seconds": 2.345,
  "total_tokens": 128,
  "throughput_tokens_per_sec": 54.6,
  "prompts": ["Hello, world!"],
  "input_ids_shape": [1, 128],
  "prefill_length": 128,
  "actual_lengths": [5],
  "last_token_logits_shape": [1, 32000],
  "output_logits_shape": [1, 128, 32000]
}
```

## 代码结构

### tee_only_llama/tee_runner.py (180 行)

```python
# 核心函数
load_model_and_tokenizer()  # 加载模型
run_prefill()               # 执行 prefill
benchmark()                 # 运行 benchmark
main()                      # 主函数
```

### tee_gpu/tee_runner.py (380 行)

```python
# 核心类
RemoteLinearProxy          # 远程 Linear 层代理
RemoteEmbeddingProxy       # 远程 Embedding 层代理
RemoteMatmul               # 远程矩阵乘法
RemoteModuleClient         # 远程模块客户端

# 核心函数
load_split_model()         # 加载分离模型
replace_linear_modules()   # 替换为远程代理
inject_remote_matmul()     # 注入远程矩阵乘法
run_prefill()              # 执行 prefill
benchmark_split_inference() # 运行 benchmark
```

## 性能测试

### 测试不同长度

```bash
for length in 64 128 256 512 1024; do
    export LLAMA_PREFILL_LENGTH=$length
    python tee_only_llama/tee_runner.py
done
```

### 测试不同批大小

```bash
for batch in 1 2 4 8; do
    export LLAMA_TEE_BATCH_SIZE=$batch
    export LLAMA_PREFILL_LENGTH=256
    python tee_only_llama/tee_runner.py
done
```

## 在 Gramine SGX 中运行

```bash
cd /home/junjie_chen@idm.teecertlabs.com/TSQP/tee_only_llama

# 构建
make clean
make SGX=1

# 运行
export LLAMA_PREFILL_LENGTH=256
gramine-sgx ./tee_only.manifest.sgx
```

## 代码改进

相比之前的版本：

1. ✅ **移除了所有 decode 代码** - 不再有 `model.generate()`
2. ✅ **移除了温度、top_p 等采样参数** - 不需要生成配置
3. ✅ **简化了环境变量** - 只保留必要的配置
4. ✅ **统一了输出格式** - 只有一种模式
5. ✅ **添加了吞吐量计算** - 自动计算性能指标
6. ✅ **代码更简洁** - 减少了约 30% 的代码量

## 常见问题

**Q: 为什么只有 prefill？**  
A: Prefill 是性能瓶颈，专注于优化这一阶段。

**Q: 如何选择 prefill_length？**  
A: 根据测试需求：64-128（快速）、256-512（常见）、1024+（压力测试）

**Q: 输出的 logits 有什么用？**  
A: 可用于分析模型预测、计算困惑度、或作为下游任务的输入。

**Q: 支持多 GPU 吗？**  
A: TEE+GPU 模式支持，在 `server.py` 中指定 GPU 设备。
