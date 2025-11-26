# GPU 计算开销分析

## 问题：为什么 TEE+GPU 混合模式的 GPU 计算开销比纯 GPU 还长？

## 1. 纯 GPU 计算模式（假设 tee_runner.py 用 GPU）

### 执行流程（单个 Attention 层）
```
[GPU] Embedding
  ↓
[GPU] RMSNorm → QKV Projection → Rotary → Attention → O Projection
  ↓
[GPU] RMSNorm → Gate/Up Projection → SiLU → Down Projection
  ↓
[GPU] LM Head
```

### 特点
- ✅ **数据始终在 GPU 上**，无需 CPU ↔ GPU 传输
- ✅ **GPU 流水线高效**，多个操作可以并行/流水线执行
- ✅ **无额外开销**，只有纯计算
- ✅ **Kernel Fusion**，框架可以自动合并相邻操作

**GPU 利用率：~80-90%**

---

## 2. TEE+GPU 混合模式（tee_runner_ours.py）

### 执行流程（单个 Attention 层）

#### Attention 部分
```
[CPU/TEE] RMSNorm
  ↓
[CPU/TEE] 加密输入 (encrypt_linear_input)         ← 额外开销 1
  ↓
[传输] CPU → GPU                                   ← 传输开销 1
  ↓
[GPU] QKV Projection (3个 Linear)                  ← 实际 GPU 计算
  ↓
[传输] GPU → CPU (3次)                             ← 传输开销 2
  ↓
[CPU/TEE] 解密 QKV (3次 decrypt_linear_output)    ← 额外开销 2
  ↓
[CPU/TEE] Rotary Embedding
  ↓
[CPU/TEE] 加密 Q, K^T (encrypt_query/encrypt_key) ← 额外开销 3
  ↓
[传输] CPU → GPU (2次)                             ← 传输开销 3
  ↓
[GPU] Q @ K^T                                      ← 实际 GPU 计算
  ↓
[传输] GPU → CPU                                   ← 传输开销 4
  ↓
[CPU/TEE] 解密 attention scores                   ← 额外开销 4
  ↓
[CPU/TEE] Softmax
  ↓
[传输] CPU → GPU (2次: scores + V)                ← 传输开销 5
  ↓
[GPU] Scores @ V                                   ← 实际 GPU 计算
  ↓
[传输] GPU → CPU                                   ← 传输开销 6
  ↓
[CPU/TEE] Reshape
  ↓
[CPU/TEE] 加密输出                                 ← 额外开销 5
  ↓
[传输] CPU → GPU                                   ← 传输开销 7
  ↓
[GPU] O Projection                                 ← 实际 GPU 计算
  ↓
[传输] GPU → CPU                                   ← 传输开销 8
  ↓
[CPU/TEE] 解密输出                                 ← 额外开销 6
```

#### MLP 部分（类似流程）
- 4 次 GPU 计算（Gate, Up, Down projections）
- 8 次 CPU ↔ GPU 传输
- 6 次加密/解密操作

### 问题分析

#### A. 数据传输开销
每层需要 **15-20 次** CPU ↔ GPU 数据传输：
```python
# 每次传输的开销（示例，512 tokens, hidden_size=4096）
数据大小 = 1 × 512 × 4096 × 2 bytes (float16) = 4 MB
PCIe 带宽 = ~16 GB/s (PCIe 3.0 x16)
单次传输延迟 = 4 MB / 16 GB/s + 同步开销 ≈ 0.25-0.5 ms

每层传输总开销 = 15 × 0.3 ms = 4.5 ms
32 层总开销 = 32 × 4.5 ms = 144 ms
```

#### B. GPU 流水线破碎
- 每次传输都是一个 **同步点**，GPU 必须等待数据到达
- 无法进行 **Kernel Fusion**（例如，QKV 三个 projection 无法合并）
- GPU 在等待数据期间处于 **空闲状态**

#### C. 小批量操作导致 GPU 利用率低
- 每次 GPU 计算都是独立的小操作
- GPU Kernel 启动开销相对较大
- 无法充分利用 GPU 的并行计算能力

#### D. 额外的加密/解密计算
虽然在 CPU 上执行，但也会延长整体时间：
```python
# 每层加密/解密次数
Linear 加密/解密: 6 次（QKV in/out, MLP in/out）
Matmul 加密/解密: 4 次（Q, K, attention out）
总计: ~10 次/层

# 假设每次 einsum 操作 ~0.5 ms
每层加密开销 = 10 × 0.5 ms = 5 ms
32 层总开销 = 32 × 5 ms = 160 ms
```

**GPU 利用率：~20-30%**（大部分时间在等待或传输）

---

## 3. OTP 方案（tee_runner_otp.py）

### 差异
- 使用 **加法秘密分享** 而非矩阵变换
- 加密/解密操作更简单（生成随机数 + 加减法）
- 但仍然有 **相同的数据传输模式**

### 开销构成
```
传输开销: ~140 ms（与 ours 类似）
掩码开销: ~80 ms（比矩阵加密更快）
GPU 计算: 与 ours 相同
```

---

## 4. 量化对比

### 假设配置
- 模型：LLaMA-2-7B (32 层)
- Sequence Length: 512
- GPU: 支持 float16

### 理论时间分解（单次 prefill）

| 项目 | 纯 GPU | TEE+GPU (Ours) | TEE+GPU (OTP) |
|------|--------|----------------|---------------|
| GPU 纯计算 | 200 ms | 200 ms | 200 ms |
| CPU ↔ GPU 传输 | 0 ms | **150 ms** | **150 ms** |
| 加密/解密 | 0 ms | **160 ms** | **80 ms** |
| CPU 计算 (Rotary/Softmax) | 0 ms | 50 ms | 50 ms |
| GPU 同步等待 | 0 ms | **100 ms** | **100 ms** |
| **总时间** | **200 ms** | **660 ms** | **580 ms** |
| **GPU 有效利用率** | 90% | 30% | 34% |

---

## 5. 关键问题总结

### 为什么 GPU 计算反而更慢？

1. **测量的"GPU 计算时间"不包含传输和同步开销**
   ```python
   # tee_runner_ours.py 第 509-514 行
   t0 = time.perf_counter()
   q_proj = attn_layer.q_proj(hs_gpu)  # 只测量这一行
   k_proj = attn_layer.k_proj(hs_gpu)
   v_proj = attn_layer.v_proj(hs_gpu)
   elapsed = time.perf_counter() - t0
   self.tracker.record_gpu_compute(elapsed, "gpu_linear")
   
   # 但实际上：
   # - 第 507 行的 _to_gpu() 传输不算在内
   # - 第 517-519 行的 _to_cpu() 传输不算在内
   # - GPU 等待数据到达的时间不算在内
   ```

2. **频繁的同步点破坏了 GPU 流水线**
   - 纯 GPU：一次提交所有操作，GPU 可以自己安排执行顺序
   - TEE+GPU：每次传输都要等待，无法流水线化

3. **GPU 大部分时间在空闲等待**
   - 等待 CPU 加密
   - 等待数据传输
   - 等待 CPU 解密

4. **无法使用 GPU 优化技术**
   - 无 Kernel Fusion
   - 无 Operation Reordering
   - 无 Memory Coalescing

---

## 6. 解决方案建议

### 短期优化
1. **批量传输**：将多个小传输合并成一次大传输
2. **异步传输**：使用 CUDA Streams 实现计算和传输的并行
3. **减少同步点**：尽量将连续的 GPU 操作合并

### 长期优化
1. **GPU 上的同态加密**：将加密计算也放到 GPU
2. **使用 GPU-TEE**：如 NVIDIA Confidential Computing
3. **优化加密方案**：使用计算开销更小的方案

### 理想架构
```
[CPU/TEE] 预处理 + 批量加密
  ↓ (一次大传输)
[GPU] 所有加密计算（多层融合）
  ↓ (一次大传输)
[CPU/TEE] 批量解密 + 后处理
```

这样可以将传输次数从 **~500 次** 减少到 **~2 次**。

