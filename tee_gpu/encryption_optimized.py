"""
优化的加密方案实现
将复杂度从 O(seq_len^2 × hidden_size) 降低到 O(seq_len × hidden_size)
"""
import torch


class OurEncryptionSchemeOptimized:
    """优化的加密方案：利用对角矩阵性质"""
    
    def __init__(self, seq_len: int, device: torch.device):
        self.seq_len = seq_len
        self.device = device
        
        # 🔑 关键优化：只存储对角元素，不存储完整矩阵
        self.D_diag = torch.randn(seq_len, device=device) + 2.0  # (seq_len,)
        self.alpha = torch.randn(seq_len, 1, device=device)  # (seq_len, 1)
        self.beta = torch.randn(seq_len, 1, device=device)   # (seq_len, 1)
        
        # 预计算逆矩阵的对角元素
        self.D_inv_diag = 1.0 / self.D_diag  # (seq_len,)
        self.D_inv_alpha = self.D_inv_diag.unsqueeze(-1) * self.alpha  # (seq_len, 1)
        self.beta_T_D_inv_alpha = (self.beta.T @ self.D_inv_alpha).item()
        self.scale_factor = 1.0 / (1.0 + self.beta_T_D_inv_alpha)
    
    def encrypt_linear_input(self, X: torch.Tensor) -> torch.Tensor:
        """
        加密 Linear 层输入: MX = DX + α(β^T X)
        优化：利用 D 是对角矩阵的性质
        
        复杂度：O(batch × seq_len × in_features)
        原始：O(batch × seq_len^2 × in_features)
        """
        # X: (batch, seq_len, in_features)
        batch_size, seq_len, in_features = X.shape
        
        # 确保数据类型匹配
        D_diag = self.D_diag.to(X.dtype)
        alpha = self.alpha.to(X.dtype)
        beta = self.beta.to(X.dtype)
        
        # 🚀 优化 1：对角矩阵乘法 -> 逐元素乘法
        # DX = D @ X，但 D 是对角矩阵
        # 等价于每行乘以对应的对角元素
        DX = D_diag.view(1, -1, 1) * X  # 广播：(1, seq_len, 1) × (batch, seq_len, in_features)
        
        # β^T X: (1, seq_len) @ (batch, seq_len, in_features) -> (batch, 1, in_features)
        # 🚀 优化 2：使用更高效的 einsum
        beta_T_X = torch.einsum('si,bsi->bi', beta.T, X).unsqueeze(1)
        
        # α(β^T X): (seq_len, 1) × (batch, 1, in_features) -> (batch, seq_len, in_features)
        alpha_beta_T_X = alpha * beta_T_X  # 广播
        
        MX = DX + alpha_beta_T_X
        return MX
    
    def decrypt_linear_output(self, Z: torch.Tensor) -> torch.Tensor:
        """
        解密 Linear 层输出: M^{-1}Z = D^{-1}Z - scale * D^{-1}α(β^T D^{-1}Z)
        优化：利用 D^{-1} 是对角矩阵的性质
        
        复杂度：O(batch × seq_len × out_features)
        原始：O(batch × seq_len^2 × out_features)
        """
        # Z: (batch, seq_len, out_features)
        
        # 确保数据类型匹配
        D_inv_diag = self.D_inv_diag.to(Z.dtype)
        beta = self.beta.to(Z.dtype)
        D_inv_alpha = self.D_inv_alpha.to(Z.dtype)
        
        # 🚀 优化：对角矩阵乘法
        # D^{-1}Z
        D_inv_Z = D_inv_diag.view(1, -1, 1) * Z
        
        # β^T D^{-1}Z
        beta_T_D_inv_Z = torch.einsum('si,bsi->bi', beta.T, D_inv_Z).unsqueeze(1)
        
        # D^{-1}α(β^T D^{-1}Z)
        D_inv_alpha_term = D_inv_alpha * beta_T_D_inv_Z
        
        # M^{-1}Z
        M_inv_Z = D_inv_Z - self.scale_factor * D_inv_alpha_term
        return M_inv_Z


class MatmulEncryptionSchemeOptimized:
    """
    优化的 Matmul 加密方案
    1. 利用对角矩阵性质
    2. 去除单位矩阵乘法
    3. 向量化计算（无循环）
    """
    
    def __init__(self, seq_len: int, head_dim: int, device: torch.device):
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.device = device
        
        # 🔑 优化：只存储对角元素
        self.D1_diag = torch.randn(seq_len, device=device) + 2.0
        self.D2_diag = torch.randn(head_dim, device=device) + 2.0
        self.D3_diag = torch.randn(seq_len, device=device) + 2.0
        
        # 预计算逆矩阵的对角元素
        self.D1_inv_diag = 1.0 / self.D1_diag
        self.D2_inv_diag = 1.0 / self.D2_diag
        self.D3_inv_diag = 1.0 / self.D3_diag
        
        # 🔑 优化：P1, P2, P3 是单位矩阵，直接跳过
        # 原始：self.P1 = torch.eye(seq_len)
        # 优化：不存储，加密时不使用
    
    def encrypt_query(self, Q: torch.Tensor) -> torch.Tensor:
        """
        加密 Query: Q' = (D₁P₁)Q(P₂D₂)
        由于 P1=P2=I，简化为: Q' = D₁ Q D₂
        
        复杂度：O(batch × num_heads × seq_len × head_dim)
        原始：O(batch × num_heads × seq_len^3) + O(... × seq_len^2 × head_dim)
        
        性能提升：~1000-3000×
        """
        # Q: (batch, num_heads, seq_len, head_dim)
        
        # 确保数据类型匹配
        D1_diag = self.D1_diag.to(Q.dtype)
        D2_diag = self.D2_diag.to(Q.dtype)
        
        # 🚀 向量化对角乘法（无循环！）
        # D1 作用于 seq_len 维度（axis=-2）
        Q_encrypted = Q * D1_diag.view(1, 1, -1, 1)  # 广播
        
        # D2 作用于 head_dim 维度（axis=-1）
        Q_encrypted = Q_encrypted * D2_diag.view(1, 1, 1, -1)  # 广播
        
        return Q_encrypted
    
    def encrypt_key_transpose(self, K_T: torch.Tensor) -> torch.Tensor:
        """
        加密 Key^T: K'^T = (D₂⁻¹P₂⁻¹)K^T(P₃D₃)
        由于 P2=P3=I，简化为: K'^T = D₂⁻¹ K^T D₃
        
        复杂度：O(batch × num_heads × head_dim × seq_len)
        原始：O(batch × num_heads × (seq_len^3 + head_dim^3))
        """
        # K_T: (batch, num_heads, head_dim, seq_len)
        
        # 确保数据类型匹配
        D2_inv_diag = self.D2_inv_diag.to(K_T.dtype)
        D3_diag = self.D3_diag.to(K_T.dtype)
        
        # 🚀 向量化对角乘法
        # D2_inv 作用于 head_dim 维度（axis=-2）
        K_T_encrypted = K_T * D2_inv_diag.view(1, 1, -1, 1)
        
        # D3 作用于 seq_len 维度（axis=-1）
        K_T_encrypted = K_T_encrypted * D3_diag.view(1, 1, 1, -1)
        
        return K_T_encrypted
    
    def decrypt_matmul_output(self, QK_T_encrypted: torch.Tensor) -> torch.Tensor:
        """
        解密 Matmul 输出: QK^T = P₁⁻¹D₁⁻¹Q'K'^TD₃⁻¹P₃⁻¹
        由于 P1=P3=I，简化为: QK^T = D₁⁻¹ Q'K'^T D₃⁻¹
        
        复杂度：O(batch × num_heads × seq_len × seq_len)
        原始：O(batch × num_heads × seq_len^3)
        """
        # QK_T_encrypted: (batch, num_heads, seq_len, seq_len)
        
        # 确保数据类型匹配
        D1_inv_diag = self.D1_inv_diag.to(QK_T_encrypted.dtype)
        D3_inv_diag = self.D3_inv_diag.to(QK_T_encrypted.dtype)
        
        # 🚀 向量化对角乘法
        # D1_inv 作用于第一个 seq_len 维度（axis=-2）
        QK_T_decrypted = QK_T_encrypted * D1_inv_diag.view(1, 1, -1, 1)
        
        # D3_inv 作用于第二个 seq_len 维度（axis=-1）
        QK_T_decrypted = QK_T_decrypted * D3_inv_diag.view(1, 1, 1, -1)
        
        return QK_T_decrypted


# ============================================================================
# 性能对比测试
# ============================================================================

def benchmark_comparison():
    """对比原始实现和优化实现的性能"""
    import time
    
    # 测试参数（模拟 LLaMA-2-7B）
    batch = 1
    seq_len = 512
    hidden_size = 4096
    num_heads = 32
    head_dim = 128
    device = torch.device("cpu")
    
    print("="*80)
    print("加密方案性能对比测试")
    print("="*80)
    print(f"配置: seq_len={seq_len}, hidden_size={hidden_size}")
    print(f"      num_heads={num_heads}, head_dim={head_dim}")
    print("="*80)
    
    # 准备测试数据
    X = torch.randn(batch, seq_len, hidden_size, device=device)
    Q = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
    
    # ========== Linear 加密测试 ==========
    print("\n【Linear 层加密测试】")
    
    # 原始实现（使用完整矩阵）
    from tee_gpu.tee_runner_ours import OurEncryptionScheme
    enc_original = OurEncryptionScheme(seq_len, device)
    
    t0 = time.perf_counter()
    for _ in range(10):
        _ = enc_original.encrypt_linear_input(X)
    time_original = (time.perf_counter() - t0) / 10
    
    # 优化实现
    enc_optimized = OurEncryptionSchemeOptimized(seq_len, device)
    
    t0 = time.perf_counter()
    for _ in range(10):
        _ = enc_optimized.encrypt_linear_input(X)
    time_optimized = (time.perf_counter() - t0) / 10
    
    print(f"原始实现: {time_original*1000:.2f} ms")
    print(f"优化实现: {time_optimized*1000:.2f} ms")
    print(f"加速比: {time_original/time_optimized:.1f}×")
    
    # ========== Matmul 加密测试 ==========
    print("\n【Matmul 层加密测试】")
    
    # 原始实现
    from tee_gpu.tee_runner_ours import MatmulEncryptionScheme
    matmul_original = MatmulEncryptionScheme(seq_len, head_dim, device)
    
    t0 = time.perf_counter()
    for _ in range(10):
        _ = matmul_original.encrypt_query(Q)
    time_original_matmul = (time.perf_counter() - t0) / 10
    
    # 优化实现
    matmul_optimized = MatmulEncryptionSchemeOptimized(seq_len, head_dim, device)
    
    t0 = time.perf_counter()
    for _ in range(10):
        _ = matmul_optimized.encrypt_query(Q)
    time_optimized_matmul = (time.perf_counter() - t0) / 10
    
    print(f"原始实现: {time_original_matmul*1000:.2f} ms")
    print(f"优化实现: {time_optimized_matmul*1000:.2f} ms")
    print(f"加速比: {time_original_matmul/time_optimized_matmul:.1f}×")
    
    # ========== 整体估算 ==========
    print("\n" + "="*80)
    print("【32 层总体性能估算】")
    print("="*80)
    
    # 每层的加密次数
    linear_per_layer = 10  # 5次加密 + 5次解密
    matmul_per_layer = 3   # 2次加密 + 1次解密
    
    total_original = (linear_per_layer * time_original + 
                     matmul_per_layer * time_original_matmul) * 32
    total_optimized = (linear_per_layer * time_optimized + 
                      matmul_per_layer * time_optimized_matmul) * 32
    
    print(f"原始实现总加密时间: {total_original*1000:.2f} ms")
    print(f"优化实现总加密时间: {total_optimized*1000:.2f} ms")
    print(f"节省时间: {(total_original-total_optimized)*1000:.2f} ms")
    print(f"整体加速比: {total_original/total_optimized:.1f}×")
    print("="*80)


if __name__ == "__main__":
    # 运行性能对比
    benchmark_comparison()

