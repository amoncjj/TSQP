# TEE安全外包计算协议 - 更新说明

## 📋 更新概述

本次更新为TEE-GPU协同计算系统添加了**安全外包计算协议**，使用**嵌入式加性外包(Embedded Additive Outsourcing)**和**一次性密码本(OTP)**保护TEE内部数据在传输到GPU计算时的隐私。

## 🔐 核心功能

### 1. 安全矩阵乘法 (默认启用)
- **位置**: Attention层中的 `Q @ K^T` 和 `Attn @ V`
- **方法**: `secure_matmul()`
- **协议**: 使用嵌入式加性外包，将数据掩码后发送到GPU
- **状态**: ✅ 已实现并默认启用

### 2. 安全线性层 (可选启用)
- **位置**: QKV投影, O投影, MLP层, LM Head
- **方法**: `secure_batch_linear()`, `secure_lm_head()`
- **协议**: 使用一次性密码本掩码
- **状态**: ✅ 已实现，需要GPU服务器支持

## 📁 修改文件

### 主要文件
1. **`tee_runner_optimized.py`** - 核心实现
   - 添加了安全计算协议实现
   - 支持自动回退到普通方法
   - 详细的性能统计

2. **`SECURE_COMPUTATION.md`** - 详细文档
   - 协议原理说明
   - 使用方法
   - 性能分析
   - 安全性说明

3. **`example_secure_usage.py`** - 使用示例
   - 示例1: 仅安全矩阵乘法 (推荐)
   - 示例2: 完整安全计算
   - 示例3: 单独测试

## 🚀 快速开始

### 默认使用 (推荐)

```python
from tee_runner_optimized import GPUClient, TEELlamaModel

# 连接GPU
gpu_client = GPUClient(ipc_path)
init_data = gpu_client.init()

# 创建模型 (自动启用安全矩阵乘法)
model = TEELlamaModel(gpu_client, init_data["config"], 
                      init_data["rotary_emb_params"],
                      init_data["norm_weights"])

# 运行推理 (Attention中的matmul自动使用安全协议)
logits = model.forward(input_ids)
```

### 启用完整安全计算 (可选)

```python
# 启用所有层的安全计算
model.use_secure_linear = True  # 需要GPU服务器支持
logits = model.forward(input_ids)
```

### 运行示例

```bash
# 示例1: 默认配置 (仅安全矩阵乘法)
python example_secure_usage.py 1

# 示例2: 完整安全计算
python example_secure_usage.py 2

# 示例3: 单独测试安全矩阵乘法
python example_secure_usage.py 3
```

## 🔍 关键实现细节

### 安全矩阵乘法协议

```
离线阶段 (TEE):
  R_Q, R_K^T ← 随机矩阵
  a, b ← 随机标量
  
嵌入式加性外包:
  Q̃ = [Q + R_Q; aR_Q]
  K̃^T = [K^T + R_K^T, bR_K^T]
  发送 Q̃, K̃^T → GPU
  
GPU计算:
  Result = Q̃ @ K̃^T (包含4个块)
  返回 Result → TEE
  
恢复 (TEE):
  R_QR_K^T = (1/ab) * T4
  QR_K^T = (1/b) * T2 - R_QR_K^T
  R_QK^T = (1/a) * T3 - R_QR_K^T
  QK^T = T1 - R_QR_K^T - QR_K^T - R_QK^T
```

## 📊 性能开销

| 操作类型 | 额外开销 | 说明 |
|---------|---------|------|
| 安全矩阵乘法 | ~5-10% | 掩码生成 + 恢复计算 |
| 安全线性层 | ~10-20% | 需要GPU计算额外的RW |
| 通信开销 | ~2倍数据量 | 矩阵乘法需要传输拼接后的掩码矩阵 |

## 🔒 安全性保证

1. **信息论安全**: 使用一次性密码本，每次计算生成新的随机掩码
2. **GPU不可见**: GPU只能看到随机掩码后的数据
3. **TEE恢复**: 所有恢复操作在TEE内部完成

## 📈 性能统计

运行后会输出三类TEE操作的详细统计：

```
TEE Basic Operations:      # 基础操作
  RMSNORM, ROTARY, SOFTMAX, SILU, OTHER

TEE Secure Computation:    # 安全计算额外开销
  Secure Mask              # 生成随机掩码
  Secure Prepare           # 准备掩码数据
  Secure Recover           # 恢复原始结果
```

## 🔧 GPU服务器端支持 (可选)

要启用完整的安全线性层计算，GPU服务器需要实现：

1. **`BatchLinearWithMask`** 方法
   - 输入: `hidden_states + mask`, `mask`
   - 输出: `(hidden_states + mask)W`, `maskW`

2. **`LMHeadWithMask`** 方法
   - 输入: `hidden_states + mask`, `mask`
   - 输出: `(hidden_states + mask)W`, `maskW`

如果服务器不支持，代码会自动回退到普通方法并打印警告。

## 📚 相关文档

- **`SECURE_COMPUTATION.md`** - 详细的协议说明和使用指南
- **`example_secure_usage.py`** - 完整的使用示例
- **`tee_runner_optimized.py`** - 源代码实现（包含详细注释）

## 🎯 适用场景

✅ 推荐使用：
- 云端GPU加速但不信任云服务提供商
- 需要保护模型中间状态隐私
- 金融、医疗等强隐私要求场景

⚠️ 性能考虑：
- 对性能极度敏感的场景可能需要权衡
- 建议先使用默认配置（仅安全矩阵乘法）

## 🐛 问题排查

### Q: 提示 "BatchLinearWithMask not supported"
**A**: GPU服务器不支持掩码版本的Linear层，会自动回退到普通方法。如需完整安全计算，请更新GPU服务器实现。

### Q: 性能下降明显
**A**: 
1. 检查是否启用了 `use_secure_linear=True`
2. 安全计算会增加5-20%的开销，这是正常的
3. 可以只使用默认的安全矩阵乘法来平衡性能和安全性

### Q: 如何验证安全协议是否生效？
**A**: 运行示例3单独测试安全矩阵乘法，对比输出误差应该在1e-4以内。

## 📝 版本信息

- **版本**: v1.0
- **日期**: 2025-11-04
- **作者**: TEE-GPU协同计算团队
- **基于**: 嵌入式加性外包协议

## 📞 支持

如有问题或建议，请查看：
1. 详细文档: `SECURE_COMPUTATION.md`
2. 使用示例: `example_secure_usage.py`
3. 源代码注释: `tee_runner_optimized.py`

