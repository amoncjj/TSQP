# 加解密复杂度分析 - tee_runner_ours.py

## 🔴 核心问题：实际复杂度远超理论值

---

## 1. OurEncryptionScheme - Linear 层加密

### 理论设计

加密公式：`MX = DX + α(β^T X)`

**理论复杂度：O(seq_len × hidden_size)**

理由：
- D 是**对角矩阵**，DX 只需逐元素相乘：O(seq_len × hidden_size)
- β^T X 是行向量与矩阵相乘：O(seq_len × hidden_size)
- α(β^T X) 是列向量与行向量相乘：O(seq_len × hidden_size)
- 总计：O(seq_len × hidden_size)

---

### 实际实现（第269-289行）

```python
def encrypt_linear_input(self, X: torch.Tensor) -> torch.Tensor:
    """加密 Linear 层输入: MX = DX + α(β^T X)"""
    # X: (batch, seq_len, in_features)
    batch_size, seq_len, in_features = X.shape
    
    # 确保加密参数与输入的数据类型匹配
    D = self.D.to(X.dtype)  # D: (seq_len, seq_len) - 完整矩阵！
    alpha = self.alpha.to(X.dtype)
    beta = self.beta.to(X.dtype)
    
    # 🔴 问题 1：使用完整矩阵乘法而非对角乘法
    DX = torch.einsum('ij,bjk->bik', D, X)  # ← O(seq_len^2 × in_features)
    
    # β^T X: (1, seq_len) @ (batch, seq_len, in_features) = (batch, 1, in_features)
    beta_T_X = torch.einsum('ij,bjk->bik', beta.T, X)  # O(seq_len × in_features)
    
    # α(β^T X): (seq_len, 1) @ (batch, 1, in_features) = (batch, seq_len, in_features)
    alpha_beta_T_X = torch.einsum('ij,bjk->bik', alpha, beta_T_X)  # O(seq_len × in_features)
    
    MX = DX + alpha_beta_T_X
    return MX
```

**实际复杂度：O(seq_len² × hidden_size)** ❌

**问题根源：**
- 虽然 D 是对角矩阵（第259行：`torch.diag(...)`）
- 但 `torch.einsum('ij,bjk->bik', D, X)` 将 D 视为**密集矩阵**
- PyTorch 执行完整的矩阵乘法，没有利用对角性质

---

### 性能影响（以 seq_len=512, hidden_size=4096 为例）

```
理论操作数：512 × 4096 = 2,097,152 次乘法
实际操作数：512 × 512 × 4096 = 1,073,741,824 次乘法

慢了 512 倍！ 😱
```

**每层加密/解密调用：**
- Linear 加密：6 次（QKV in, Gate/Up in, Down in, O in）
- Linear 解密：6 次
- 总计：12 次

**32 层总计：**
- 384 次 × 1,073,741,824 = **412,316,860,416 次操作**
- 如果用对角优化：384 次 × 2,097,152 = **805,306,368 次操作**

**额外浪费了 511 倍的计算量！**

---

## 2. MatmulEncryptionScheme - Matmul 层加密

### 理论设计

加密公式：`Q' = (D₁P₁)Q(P₂D₂)`

**理论复杂度（利用对角性质）：O(seq_len × head_dim)**

---

### 实际实现（第340-356行）

```python
def encrypt_query(self, Q: torch.Tensor) -> torch.Tensor:
    """加密 Query: Q' = (D₁P₁)Q(P₂D₂)"""
    # Q: (batch, num_heads, seq_len, head_dim)
    batch, num_heads, seq_len, head_dim = Q.shape
    
    # 确保加密参数与输入的数据类型匹配
    D1 = self.D1.to(Q.dtype)  # (seq_len, seq_len) - 完整矩阵
    P1 = self.P1.to(Q.dtype)  # (seq_len, seq_len) - 单位矩阵但存储为完整矩阵
    P2 = self.P2.to(Q.dtype)  # (head_dim, head_dim)
    D2 = self.D2.to(Q.dtype)  # (head_dim, head_dim)
    
    # 🔴 问题 1：双重循环，无并行化
    # 🔴 问题 2：连续4次矩阵乘法，没有合并
    # 🔴 问题 3：完整矩阵乘法而非对角乘法
    Q_encrypted = torch.zeros_like(Q)
    for b in range(batch):
        for h in range(num_heads):
            Q_encrypted[b, h] = D1 @ P1 @ Q[b, h] @ P2 @ D2
            # ↑ 每次执行4个矩阵乘法：
            # 1. D1 @ P1: (seq_len, seq_len) @ (seq_len, seq_len) = O(seq_len^3)
            # 2. (D1P1) @ Q[b,h]: (seq_len, seq_len) @ (seq_len, head_dim) = O(seq_len^2 × head_dim)
            # 3. (...) @ P2: (seq_len, head_dim) @ (head_dim, head_dim) = O(seq_len × head_dim^2)
            # 4. (...) @ D2: (seq_len, head_dim) @ (head_dim, head_dim) = O(seq_len × head_dim^2)
    
    return Q_encrypted
```

**实际复杂度：O(batch × num_heads × seq_len³)** ❌

**多重问题：**

1. **预计算浪费**：每次循环都重新计算 `D1 @ P1`
2. **串行执行**：双重循环强制串行，无法利用 GPU 并行
3. **完整矩阵乘法**：没有利用 D1, D2 是对角矩阵的性质
4. **单位矩阵乘法**：P1, P2, P3 是单位矩阵，完全不需要乘法！

---

### 性能影响（以 seq_len=512, head_dim=128, num_heads=32 为例）

```python
# 每次 Q 加密的操作数
batch = 1
num_heads = 32

# D1 @ P1 (每个 head 都重复计算！)
D1_P1_ops = 512^3 = 134,217,728 次/head
total_D1_P1 = 32 × 134,217,728 = 4,294,967,296 次

# (D1P1) @ Q
DQ_ops = 512^2 × 128 = 33,554,432 次/head
total_DQ = 32 × 33,554,432 = 1,073,741,824 次

# ... @ P2
QP_ops = 512 × 128^2 = 8,388,608 次/head
total_QP = 32 × 8,388,608 = 268,435,456 次

# ... @ D2
QD_ops = 512 × 128^2 = 8,388,608 次/head
total_QD = 32 × 8,388,608 = 268,435,456 次

# 总计单次 Q 加密
total_ops = 4,294,967,296 + 1,073,741,824 + 268,435,456 + 268,435,456
         ≈ 5,905,580,032 次操作

# 每层需要 2 次（Q 和 K）+ 1 次解密
每层 ≈ 3 × 5,905,580,032 = 17,716,740,096 次操作

# 32 层
32 × 17,716,740,096 = 566,935,683,072 次操作
```

**如果优化（利用对角性质 + 去除单位矩阵）：**
```python
# 只需要对角乘法
每层 ≈ 3 × (512 × 128 × 32) = 6,291,456 次操作
32 层 = 201,326,592 次操作

快了 2,817 倍！
```

---

## 3. 总体性能分析

### 每层的加密/解密次数

| 操作 | 调用次数/层 | 类型 |
|------|------------|------|
| Linear 加密 | 5 次 | OurEncryptionScheme |
| Linear 解密 | 5 次 | OurEncryptionScheme |
| Matmul 加密（Q, K^T） | 2 次 | MatmulEncryptionScheme |
| Matmul 解密 | 1 次 | MatmulEncryptionScheme |

### 复杂度对比（seq_len=512）

| 组件 | 理论复杂度 | 实际复杂度 | 浪费倍数 |
|------|-----------|-----------|---------|
| Linear 加密/解密 | O(n × d) | O(n² × d) | **512×** |
| Matmul 加密/解密 | O(n × d) | O(n³ + n² × d) | **>1000×** |

### 时间估算（单层，seq_len=512）

```
Linear 加解密：10 次 × (512^2 × 4096) × 浮点运算时间
             = 10 × 1,073,741,824 × 1ns
             = ~10,737 ms = 10.7 秒

Matmul 加解密：3 次 × 5,905,580,032 × 1ns
             = ~17,717 ms = 17.7 秒

每层总计：~28 秒
32 层总计：~896 秒 = 15 分钟！
```

**但实际测量只有 ~600 ms？**
说明：
1. 实际 CPU 性能比估算好（多核并行、SIMD）
2. 但仍然比理论值慢 **数百倍**

---

## 4. 为什么实际开销这么高？

### 根本原因总结

| 问题 | 位置 | 影响 |
|------|------|------|
| **1. 未利用对角矩阵性质** | 第280行 `torch.einsum` | 慢 512× |
| **2. 双重循环串行执行** | 第352-354行 | 无法 GPU 并行 |
| **3. 重复计算 D1@P1** | 第354行 | 每个 head 重复 |
| **4. 单位矩阵无用乘法** | 第354行 P1, P2 | 纯粹浪费 |
| **5. 连续矩阵乘法未合并** | 第354行 | 多次内存读写 |

---

## 5. 优化方案

### 优化 1：利用对角矩阵性质

**当前实现：**
```python
DX = torch.einsum('ij,bjk->bik', D, X)  # O(seq_len^2 × in_features)
```

**优化实现：**
```python
# 提取对角元素
D_diag = torch.diag(D)  # (seq_len,)
# 对角乘法：每行独立缩放
DX = D_diag.unsqueeze(0).unsqueeze(-1) * X  # O(seq_len × in_features)
```

**代码示例：**
```python
def encrypt_linear_input_optimized(self, X: torch.Tensor) -> torch.Tensor:
    """优化的加密 Linear 层输入"""
    # X: (batch, seq_len, in_features)
    
    # 提取对角元素（只需要一次）
    D_diag = torch.diagonal(self.D).to(X.dtype)  # (seq_len,)
    
    # 对角矩阵乘法：逐行缩放
    # D_diag: (seq_len,) -> (1, seq_len, 1)
    # X: (batch, seq_len, in_features)
    DX = D_diag.view(1, -1, 1) * X  # 广播，O(seq_len × in_features)
    
    # β^T X: (1, seq_len) @ X -> (batch, 1, in_features)
    beta_T_X = torch.einsum('i,bij->bj', self.beta.squeeze(), X)
    beta_T_X = beta_T_X.unsqueeze(1)  # (batch, 1, in_features)
    
    # α(β^T X): (seq_len, 1) × (batch, 1, in_features) -> (batch, seq_len, in_features)
    alpha_beta_T_X = self.alpha.to(X.dtype) * beta_T_X
    
    return DX + alpha_beta_T_X
```

**性能提升：512×**

---

### 优化 2：移除单位矩阵乘法 + 向量化

**当前实现：**
```python
for b in range(batch):
    for h in range(num_heads):
        Q_encrypted[b, h] = D1 @ P1 @ Q[b, h] @ P2 @ D2
```

**优化实现（P1=P2=P3=I）：**
```python
def encrypt_query_optimized(self, Q: torch.Tensor) -> torch.Tensor:
    """优化的 Query 加密"""
    # Q: (batch, num_heads, seq_len, head_dim)
    
    # 提取对角元素
    D1_diag = torch.diagonal(self.D1).to(Q.dtype)  # (seq_len,)
    D2_diag = torch.diagonal(self.D2).to(Q.dtype)  # (head_dim,)
    
    # 向量化对角乘法
    # D1 作用于 seq_len 维度
    Q_encrypted = Q * D1_diag.view(1, 1, -1, 1)  # 广播
    # D2 作用于 head_dim 维度
    Q_encrypted = Q_encrypted * D2_diag.view(1, 1, 1, -1)  # 广播
    
    return Q_encrypted
```

**性能提升：>2000×**

---

### 优化 3：预计算加密矩阵

**如果不能去除 P1, P2**（例如真的是置换矩阵）：
```python
def __init__(self, seq_len: int, head_dim: int, device: torch.device):
    # ... 生成 D1, P1, D2, P2 ...
    
    # 🔑 预计算左侧变换矩阵
    self.left_transform = self.D1 @ self.P1  # 只计算一次！
    self.right_transform = self.P2 @ self.D2  # 只计算一次！
    
    # 预计算逆矩阵
    self.left_transform_inv = self.P1.T @ self.D1_inv
    # ...

def encrypt_query_precomputed(self, Q: torch.Tensor) -> torch.Tensor:
    """使用预计算的加密"""
    # Q: (batch, num_heads, seq_len, head_dim)
    
    # 向量化批量乘法（无循环）
    Q_reshaped = Q.reshape(-1, Q.size(-2), Q.size(-1))  # (batch*num_heads, seq_len, head_dim)
    Q_encrypted = torch.bmm(
        self.left_transform.unsqueeze(0).expand(Q_reshaped.size(0), -1, -1),
        torch.bmm(Q_reshaped, self.right_transform.unsqueeze(0).expand(Q_reshaped.size(0), -1, -1))
    )
    Q_encrypted = Q_encrypted.reshape(Q.shape)
    
    return Q_encrypted
```

**性能提升：~32× (消除 num_heads 次重复计算)**

---

## 6. 建议的完整优化

### 立即可做（不改变算法）：

1. ✅ **利用对角性质**：用逐元素乘法替代矩阵乘法
2. ✅ **去除单位矩阵**：P1=P2=P3=I 时直接跳过
3. ✅ **向量化计算**：消除双重循环
4. ✅ **预计算常量**：一次性计算 D1@P1 等

**预期性能提升：100-1000×**

### 长期优化：

1. 重新设计加密方案，直接使用对角矩阵而非通过 `torch.diag()` 构造
2. 使用稀疏矩阵表示
3. 考虑使用更简单的加密方案（如 OTP）

---

## 7. 总结

### 当前状态
- **理论复杂度**：O(seq_len × hidden_size)
- **实际复杂度**：O(seq_len² × hidden_size) 至 O(seq_len³)
- **性能损失**：**500-3000 倍**

### 核心问题
1. 🔴 对角矩阵被当作密集矩阵处理（最严重）
2. 🔴 双重循环串行执行，无GPU并行
3. 🔴 单位矩阵执行无用乘法
4. 🔴 重复计算未被缓存

### 解决方案
实施上述优化后，加密开销可以从 **~160 ms** 降低到 **<1 ms**！

这就是为什么实际开销这么高的根本原因。

