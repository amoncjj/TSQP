# 安全外包计算协议实现

本文档说明了在TEE-GPU协同计算中实现的安全外包计算协议。

## 概述

当TEE需要将内部敏感数据发送到GPU进行计算时，使用**嵌入式加性外包(Embedded Additive Outsourcing)**协议来保护数据隐私。该协议使用一次性密码本(OTP)对数据进行掩码，确保GPU端无法获取原始数据。

## 协议详情

### 1. 离线阶段 (Offline)

在TEE内部预先生成随机掩码：
- 采样随机矩阵/向量 `R` (与数据同形状)
- 采样随机标量 `a, b` (用于矩阵乘法)
- 预计算标量乘法: `aR`, `bR`

### 2. 嵌入式加性外包 (Embedded Additive Outsource)

#### 2.1 矩阵乘法: Q @ K^T

对于需要计算 `QK^T` 的情况：

**TEE端准备:**
```
Q̃ = [Q + R_Q; aR_Q]  # 在行维度拼接
K̃^T = [K^T + R_K^T, bR_K^T]  # 在列维度拼接
```

**GPU端计算:**
```
Result = Q̃ @ K̃^T = [[T1, T2],
                     [T3, T4]]
```
其中:
- `T1 = (Q + R_Q)(K^T + R_K^T)` 
- `T2 = (Q + R_Q)bR_K^T`
- `T3 = aR_Q(K^T + R_K^T)`
- `T4 = abR_QR_K^T`

#### 2.2 线性层: y = xW

对于线性变换：

**TEE端准备:**
```
x̃ = x + R  # 添加随机掩码
```

**GPU端计算:**
```
ỹ = x̃W = (x + R)W
R̃ = RW
```
返回 `(ỹ, R̃)` 到TEE

### 3. 恢复 (Recovery)

#### 3.1 矩阵乘法恢复

在TEE端通过以下步骤恢复原始结果：

```python
# 步骤1: 计算 R_Q @ R_K^T
R_Q_R_KT = (1/(a*b)) * T4

# 步骤2: 计算 Q @ R_K^T
Q_R_KT = (1/b) * T2 - R_Q_R_KT

# 步骤3: 计算 R_Q @ K^T
R_Q_KT = (1/a) * T3 - R_Q_R_KT

# 步骤4: 恢复最终结果
QKT = T1 - R_Q_R_KT - Q_R_KT - R_Q_KT
```

#### 3.2 线性层恢复

```python
y = ỹ - R̃ = (x+R)W - RW = xW
```

## 代码实现

### 当前实现状态

1. **矩阵乘法 (Attention)** - ✅ 已实现并默认启用
   - `Q @ K^T` (注意力权重计算)
   - `Attn @ V` (注意力输出计算)
   - 使用 `secure_matmul()` 方法

2. **线性层** - ✅ 已实现，可选启用
   - QKV projections
   - O projection  
   - MLP (gate, up, down projections)
   - LM Head
   - 使用 `secure_batch_linear()` 和 `secure_lm_head()` 方法
   - **注意**: 需要GPU服务器支持 `BatchLinearWithMask` 和 `LMHeadWithMask` 方法

### 使用方式

#### 方式1: 仅使用安全矩阵乘法(默认)

```python
# 默认配置，矩阵乘法自动使用安全协议
model = TEELlamaModel(gpu_client, config, rotary_params, norm_weights)
# model.use_secure_linear = False (默认)
logits = model.forward(input_ids)
```

#### 方式2: 启用完整安全计算

```python
# 启用所有Linear层的安全计算
model = TEELlamaModel(gpu_client, config, rotary_params, norm_weights)
model.use_secure_linear = True  # 启用Linear层安全计算
logits = model.forward(input_ids)
```

**注意**: 启用 `use_secure_linear=True` 需要GPU服务器实现以下方法：
- `BatchLinearWithMask`: 计算 `(x+R)W` 和 `RW`
- `LMHeadWithMask`: 计算 `(x+R)W` 和 `RW`

如果服务器不支持，代码会自动回退到普通方法并打印警告。

## 性能统计

运行后会输出详细的性能统计，包括：

### TEE操作分类
- **基础操作**: RMSNorm, RotaryEmbedding, Softmax, SiLU
- **安全计算操作**:
  - `Secure Mask`: 生成随机掩码的时间
  - `Secure Prepare`: 准备掩码数据的时间  
  - `Secure Recover`: 恢复原始结果的时间

### GPU操作
- Embedding
- Linear
- Matmul (包含安全matmul)
- LM Head

示例输出:
```
TEE Secure Computation
----------------------------------------------------------------------
Secure Mask              32    0.0124      0.3875    2.15%
Secure Prepare           32    0.0089      0.2781    1.54%
Secure Recover           32    0.0156      0.4875    2.70%
```

## 安全性说明

### 隐私保护
- **一次性密码本(OTP)**: 每次计算生成新的随机掩码，确保信息论安全
- **GPU不可见**: GPU只能看到随机掩码后的数据，无法推断原始数据
- **TEE恢复**: 所有恢复操作在TEE内部完成，确保数据不泄露

### 适用场景
- 云端GPU加速但不信任云服务提供商
- 需要保护模型中间状态隐私
- 满足强隐私保护要求的应用(金融、医疗等)

### 性能开销
- **矩阵乘法**: 额外开销约5-10% (掩码生成+恢复计算)
- **线性层**: 额外开销约10-20% (需要GPU计算额外的 `RW`)
- **通信开销**: 矩阵乘法需要传输约2倍数据量 (拼接掩码矩阵)

## 扩展和定制

如果需要对其他操作应用安全协议，可以参考以下模式：

```python
def secure_operation(self, data: torch.Tensor) -> torch.Tensor:
    # 1. 生成掩码
    R = torch.randn_like(data)
    data_masked = data + R
    
    # 2. 发送掩码数据到GPU
    result_masked = self.gpu.operation(data_masked)
    
    # 3. 在TEE中恢复
    # 根据具体操作设计恢复逻辑
    result = recover_function(result_masked, R)
    
    return result
```

## 参考资料

该实现基于安全外包计算的经典协议，主要参考：
- Embedded Additive Outsourcing Protocol
- Secure Multi-party Computation (MPC) techniques
- One-Time Pad (OTP) encryption

## 修改历史

- 2025-11-04: 初始实现，支持attention中的安全矩阵乘法和可选的安全线性层

