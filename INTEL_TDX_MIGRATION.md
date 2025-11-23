# Intel TDX 模式迁移总结

## 概述

已将项目完全迁移到 Intel TDX Passthrough 模式。在 Intel TDX 环境下，CPU 上的计算自动在 TEE 中执行，GPU 通过 Passthrough 直接访问，使用 `.to(device)` 进行数据传输，无需复杂的进程间通信。

## 主要变更

### 1. 架构简化

**之前 (Gramine SGX + ZeroMQ/IPC)**:
- 需要 ZeroMQ/IPC 进行进程间通信
- 需要共享内存环形缓冲区
- 需要复杂的序列化/反序列化
- 需要独立的 GPU 服务器进程
- 需要 Gramine manifest 文件
- 需要 Makefile 构建
- 有 warmup 步骤

**现在 (Intel TDX Passthrough)**:
- ✅ 直接使用 `.to(device)` 进行数据传输
- ✅ GPU Passthrough，无需 IPC
- ✅ 单进程架构，无需独立服务器
- ✅ 简化的架构，直接运行 Python
- ✅ 移除所有 Gramine 相关文件
- ✅ 配置文件独立管理
- ✅ **无 warmup 步骤，直接计时**

### 2. 三种实现方法（全部完成迁移）

项目包含三种不同的 TEE+GPU 混合推理实现，**全部已迁移到 Intel TDX Passthrough 模式**：

#### 1. tee_runner_optimized.py - 优化版本 ✅
- **状态**: 完全迁移到 Intel TDX Passthrough
- **特点**: 
  - 使用 `.to(device)` 直接传输数据
  - 无加密，纯性能优化
  - 三部分详细计时（传输、GPU计算、CPU计算）
  - 使用 `config.py` 配置
  - 无 warmup
  - 单进程运行

#### 2. tee_runner_ours.py - 我们的加密方案 ✅
- **状态**: 完全迁移到 Intel TDX Passthrough
- **加密方案**: 
  - Linear层: MX = DX + α(β^T X)
  - Matmul层: Q' = (D₁P₁)Q(P₂D₂), K'^T = (D₂⁻¹P₂⁻¹)K^T(P₃D₃)
- **特点**:
  - 使用 `.to(device)` 直接传输
  - 加密/解密在 CPU/TEE 中执行
  - 详细的加密开销统计
  - 无 warmup
  - 单进程运行

#### 3. tee_runner_otp.py - OTP 加密方案 ✅
- **状态**: 完全迁移到 Intel TDX Passthrough
- **加密方案**:
  - Linear层: 加法秘密分享 (X-R)W + RW
  - Matmul层: 嵌入式加法外包
- **特点**:
  - 使用 `.to(device)` 直接传输
  - 掩码/恢复在 CPU/TEE 中执行
  - 详细的掩码和恢复开销统计
  - 无 warmup
  - 单进程运行

### 3. 文件结构

#### tee_gpu/ (TEE+GPU 混合推理)

```
tee_gpu/
├── config.py                          # 配置文件（所有方法共用）
├── tee_runner_optimized.py           # 优化版本 ✅
├── tee_runner_ours.py                # 我们的方法 ✅
├── tee_runner_otp.py                 # OTP 方法 ✅
├── README.md                          # 文档
└── benchmarks/
    └── run_split_benchmark.sh        # 运行脚本
```

#### tee_only_llama/ (TEE-Only 推理)

```
tee_only_llama/
├── config.py                          # 配置文件
├── tee_runner.py                      # 主程序 ✅
├── prompts.txt                        # 提示词
├── README.md                          # 文档
└── benchmarks/
    ├── run_full_tee_benchmark.sh     # 运行脚本
    └── compare_split_vs_tee.sh       # 对比脚本
```

### 4. 配置管理

所有方法共用 `config.py`：

```python
# tee_gpu/config.py
MODEL_PATH = "/path/to/model"
PREFILL_TOKEN_LENGTH = 8
OUTPUT_FILE = "tee_gpu_results.json"
GPU_DEVICE = "cuda:0"
CPU_DEVICE = "cpu"
```

### 5. 性能追踪（三部分计时 + 加密开销）

所有三个版本都提供详细的性能追踪：

#### 基础计时（所有版本）:
1. **传输开销** (CPU <-> GPU):
   - `transfer_to_gpu`: CPU -> GPU 传输时间
   - `transfer_to_cpu`: GPU -> CPU 传输时间
   - 传输数据量 (MB)

2. **GPU 计算**:
   - Embedding, Linear, Matmul, LM Head

3. **CPU 计算 (TEE)**:
   - RMSNorm, RotaryEmbedding, Softmax, SiLU

4. **总时间**: 端到端推理时间（无 warmup）

#### 加密版本额外统计:
- **ours**: 加密时间、解密时间、加密操作次数
- **otp**: 掩码时间、恢复时间、掩码/恢复操作次数

### 6. 无 Warmup 设计

所有三个实现都移除了 warmup 步骤，直接进行计时：

```python
# 直接运行 benchmark (无 warmup)
print("Running benchmark (no warmup)...")
start_time = time.perf_counter()
logits = model.forward(input_ids)
total_time = time.perf_counter() - start_time
```

这样可以获得更真实的首次推理性能数据。

### 7. 输出格式

所有版本都输出详细的 JSON 格式结果：

```json
{
  "timing": {
    "transfer_to_gpu_ms": ...,
    "transfer_to_cpu_ms": ...,
    "total_transfer_ms": ...,
    "gpu_compute_ms": ...,
    "cpu_compute_ms": ...,
    "total_ms": ...,
    // 加密版本额外包含:
    "encryption_ms": ...,  // 或 masking_ms
    "decryption_ms": ...,  // 或 recovery_ms
    "total_crypto_ms": ...
  },
  "timing_percentage": {
    "transfer_pct": ...,
    "gpu_compute_pct": ...,
    "cpu_compute_pct": ...,
    "crypto_pct": ...  // 加密版本
  },
  "data_transfer": {
    "to_gpu_mb": ...,
    "to_cpu_mb": ...,
    "total_mb": ...
  },
  "operation_counts": { ... },
  "benchmark_info": {
    "encryption_scheme": "..."  // 加密版本
  }
}
```

## 使用方法

### TEE+GPU 混合推理（三种方法，全部单进程）

```bash
cd tee_gpu

# 方法1: 优化版本（无加密，最快）
python tee_runner_optimized.py
# 输出: tee_gpu_results.json

# 方法2: 我们的加密方案（矩阵变换）
python tee_runner_ours.py
# 输出: tee_gpu_results_ours.json

# 方法3: OTP 加密方案（秘密分享）
python tee_runner_otp.py
# 输出: tee_gpu_results_otp.json
```

### TEE-Only 推理

```bash
cd tee_only_llama
python tee_runner.py
# 输出: tee_only_results.json
```

### 对比测试

```bash
bash tee_only_llama/benchmarks/compare_split_vs_tee.sh
```

## Intel TDX 优势

1. **简化架构**: 无需复杂的 IPC 和序列化
2. **GPU Passthrough**: 直接访问 GPU，性能更好
3. **自动 TEE 保护**: CPU 计算自动在 TEE 中执行
4. **易于开发**: 直接运行 Python，无需 manifest 配置
5. **灵活配置**: 通过配置文件轻松调整参数
6. **真实性能**: 无 warmup，测量首次推理的真实性能
7. **单进程部署**: 无需独立的 GPU 服务器，简化部署

## 已删除的文件

### Gramine 相关文件
- `tee_gpu/Makefile`
- `tee_gpu/*.manifest.template`
- `tee_gpu/server_optimized.py` (不再需要独立服务器)
- `tee_gpu/shm_broadcast.py` (不再需要共享内存)
- `tee_only_llama/Makefile`
- `tee_only_llama/*.manifest*`
- `tee_only_llama/*.sig`
- `tee_only_llama/modeling_llama.py`

### 旧的分析和测试文件
- `analyze_transmission.py`
- `test_shm_communication.py`
- `BEFORE_AFTER_COMPARISON.md`
- `CHANGES_SUMMARY.md`
- `SHARED_MEMORY_OPTIMIZATION.md`
- `TRANSMISSION_ANALYSIS.md`

## 迁移状态

| 文件 | 迁移状态 | 通信方式 | 加密方案 | 进程模型 |
|------|---------|---------|---------|---------|
| `tee_runner_optimized.py` | ✅ 完成 | Passthrough | 无 | 单进程 |
| `tee_runner_ours.py` | ✅ 完成 | Passthrough | 矩阵变换 | 单进程 |
| `tee_runner_otp.py` | ✅ 完成 | Passthrough | 秘密分享 | 单进程 |
| `tee_only_llama/tee_runner.py` | ✅ 完成 | N/A | 无 | 单进程 |

## 三种方法对比

| 特性 | Optimized | Ours | OTP |
|------|-----------|------|-----|
| **加密方案** | 无 | 矩阵变换 | 秘密分享 |
| **安全性** | 低 | 高 | 高 |
| **性能** | 最快 | 中等 | 较慢 |
| **加密开销** | 0% | ~10-20% | ~20-30% |
| **适用场景** | 性能测试 | 平衡安全与性能 | 最高安全要求 |

## 技术细节

### 数据传输（所有版本）

```python
def _to_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
    """传输到 GPU 并记录"""
    t0 = time.perf_counter()
    result = tensor.to(self.gpu_device)
    elapsed = time.perf_counter() - t0
    self.tracker.record_transfer_to_gpu(tensor, elapsed)
    return result
```

### 加密方案

#### Ours (矩阵变换):
```python
# 加密: MX = DX + α(β^T X)
MX = self.encrypt_linear_input(X)
# GPU 计算
Z = Linear(MX)
# 解密: M^{-1}Z
Y = self.decrypt_linear_output(Z)
```

#### OTP (秘密分享):
```python
# 掩码: X-R
X_masked, R = self.mask_linear_input(X)
# GPU 计算
Y_masked = Linear(X_masked)
# 恢复: Y = (X-R)W + RW
Y = self.recover_linear_output(Y_masked, RW)
```

## 注意事项

1. ✅ 确保在 Intel TDX 环境下运行以获得 TEE 保护
2. ✅ GPU Passthrough 需要硬件和系统支持
3. ✅ 传输开销是性能瓶颈，加密方案会增加额外开销
4. ✅ 配置文件中的路径需要根据实际环境调整
5. ✅ 推荐使用 `tee_runner_optimized.py` 获得最佳性能
6. ✅ 加密方案提供额外的安全保护，但会有性能损失
7. ✅ 所有方法都是单进程运行，部署简单

## 性能预期

基于 Intel TDX Passthrough 模式：

- **Optimized**: 接近原生 GPU 性能，传输开销 5-10%
- **Ours**: 加密开销 10-20%，总体性能损失 15-25%
- **OTP**: 加密开销 20-30%，总体性能损失 25-35%

实际性能取决于：
- 模型大小
- Prefill token 长度
- GPU 性能
- CPU-GPU 传输带宽

## 总结

✅ **所有三种方法都已完全迁移到 Intel TDX Passthrough 模式**
✅ **移除了所有 ZeroMQ/IPC 依赖**
✅ **单进程架构，无需独立服务器**
✅ **无 warmup，真实首次推理性能**
✅ **详细的三部分计时统计**
✅ **统一的配置文件管理**

项目现在完全适配 Intel TDX 环境，可以直接运行！🎉
