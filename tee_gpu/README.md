# TEE+GPU 混合推理

基于 Intel TDX 的 TEE+GPU 混合推理实现。

## 架构

- **CPU (TEE)**: RMSNorm, RotaryEmbedding, Softmax, SiLU 等轻量计算
- **GPU**: Linear, Matmul, Embedding, LM Head 等密集计算
- **数据传输**: 通过 `.to(device)` 实现 CPU-GPU 之间的数据传输（GPU Passthrough）

## 配置

编辑 `config.py` 来修改参数：

```python
MODEL_PATH = "/path/to/model"        # 模型路径
PREFILL_TOKEN_LENGTH = 8             # Prefill token 数量
OUTPUT_FILE = "tee_gpu_results.json" # 输出文件
GPU_DEVICE = "cuda:0"                # GPU 设备
CPU_DEVICE = "cpu"                   # CPU 设备 (TEE)
```

## 运行

### 直接运行

```bash
python tee_runner_optimized.py
```

### 使用脚本运行

```bash
bash benchmarks/run_split_benchmark.sh
```

## 输出

性能统计会保存到 `tee_gpu_results.json`，包含：

- **timing**: 各部分耗时（传输、GPU计算、CPU计算、总时间）
- **timing_percentage**: 各部分占比
- **data_transfer**: 数据传输量（CPU<->GPU）
- **operation_counts**: 各类操作的次数
- **benchmark_info**: 基准测试信息

## 性能分析

程序会输出详细的性能统计：

1. **传输开销**: CPU->GPU 和 GPU->CPU 的数据传输时间
2. **GPU 计算**: 在 GPU 上执行的计算时间
3. **CPU 计算 (TEE)**: 在 CPU/TEE 中执行的计算时间
4. **总时间**: 端到端的推理时间

## Intel TDX 模式

在 Intel TDX 环境下：
- CPU 上的计算自动在 TEE 中执行
- GPU 通过 Passthrough 直接访问
- 不需要 Gramine 或其他 TEE 框架
- **无 warmup 步骤**，直接测量实际性能
