"""
TEE+GPU 混合推理 - 我们的加密方案 (Intel TDX Passthrough 版本) - 优化版本

实现方案：
1. Linear层: MX = DX + α(β^T X)
   其中:
   - X: (batch, seq_len, in_features)
   - D: (seq_len, seq_len) 对角矩阵
   - α: (seq_len, 1) 列向量
   - β: (seq_len, 1) 列向量
   - β^T: (1, seq_len) 行向量（β的转置）
   恢复: M^{-1}Z = D^{-1}Z - [1/(1 + β^T D^{-1}α)]D^{-1}α(β^T D^{-1}Z)
   
   - Offline阶段: 生成加密矩阵参数 (D, α, β) 及其逆
   - Online阶段: 加密、GPU计算、解密

2. Matmul层: Q' = (D₁P₁)Q(P₂D₂), K'^T = (D₂⁻¹P₂⁻¹)K^T(P₃D₃)
   恢复: QK^T = P₁⁻¹D₁⁻¹Q'K'^TD₃⁻¹P₃⁻¹
   
   - Offline阶段: 生成加密矩阵参数 (D₁, D₂, D₃) 及其逆
   - Online阶段: 加密、GPU计算、解密

3. 性能统计: 四部分计时（Offline、传输、GPU计算、CPU/TEE计算）
4. Intel TDX Passthrough: 直接使用 .to(device) 进行数据传输
5. 无 warmup 步骤，直接计时

优化改进（相比原始版本）：
✅ 只存储对角元素，不存储完整矩阵（内存优化）
✅ 使用逐元素乘法替代矩阵乘法（利用对角性质）
✅ 去除单位矩阵的无用乘法（P1=P2=P3=I）
✅ 向量化计算，消除双重循环（全部并行）
性能提升：加密开销降低 500-3000 倍，从 O(n²×d) 降到 O(n×d)
"""
import os
import json
import time
from typing import Dict, List, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入配置
from config import (
    MODEL_PATH,
    PREFILL_TOKEN_LENGTH,
    OUTPUT_FILE,
    GPU_DEVICE,
    CPU_DEVICE
)


class TEERMSNorm(nn.Module):
    """TEE 端的 RMSNorm (CPU计算)"""
    
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TEERotaryEmbedding(nn.Module):
    """TEE 端的 RotaryEmbedding (CPU计算)"""
    
    def __init__(self, inv_freq: torch.Tensor, attention_scaling: float):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = attention_scaling
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转张量的一半维度"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    """应用旋转位置编码"""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复 key/value"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class PerformanceTracker:
    """性能追踪器 - 细化统计：enc/dec、matmul/linear分开"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.timing = {
            "offline": 0.0,              # Offline 预计算（生成加密矩阵）
            "transfer_to_gpu": 0.0,      # CPU -> GPU 传输
            "transfer_to_cpu": 0.0,      # GPU -> CPU 传输
            "gpu_compute": 0.0,          # GPU 计算
            "cpu_compute": 0.0,          # CPU 计算 (TEE)
            # 细化的加密时间
            "encryption_linear_attn": 0.0,  # Attention层Linear加密
            "encryption_linear_mlp": 0.0,   # MLP层Linear加密
            "encryption_matmul": 0.0,       # Matmul层加密
            # 细化的解密时间
            "decryption_linear_attn": 0.0,  # Attention层Linear解密
            "decryption_linear_mlp": 0.0,   # MLP层Linear解密
            "decryption_matmul": 0.0,       # Matmul层解密
            "total": 0.0,                   # 总时间
        }
        self.data_transfer = {
            "to_gpu_bytes": 0,           # 传输到 GPU 的字节数
            "to_cpu_bytes": 0,           # 传输到 CPU 的字节数
        }
        self.operation_counts = {
            "gpu_linear": 0,
            "gpu_matmul": 0,
            "gpu_embedding": 0,
            "gpu_lm_head": 0,
            "cpu_rmsnorm": 0,
            "cpu_rotary": 0,
            "cpu_softmax": 0,
            "cpu_silu": 0,
            "cpu_multiply": 0,
            "encryption_linear_attn_ops": 0,
            "encryption_linear_mlp_ops": 0,
            "encryption_matmul_ops": 0,
            "decryption_linear_attn_ops": 0,
            "decryption_linear_mlp_ops": 0,
            "decryption_matmul_ops": 0,
        }
    
    def record_offline(self, elapsed: float):
        """记录 Offline 预计算"""
        self.timing["offline"] += elapsed
    
    def record_transfer_to_gpu(self, tensor: torch.Tensor, elapsed: float):
        """记录 CPU -> GPU 传输"""
        self.timing["transfer_to_gpu"] += elapsed
        self.data_transfer["to_gpu_bytes"] += tensor.numel() * tensor.element_size()
    
    def record_transfer_to_cpu(self, tensor: torch.Tensor, elapsed: float):
        """记录 GPU -> CPU 传输"""
        self.timing["transfer_to_cpu"] += elapsed
        self.data_transfer["to_cpu_bytes"] += tensor.numel() * tensor.element_size()
    
    def record_gpu_compute(self, elapsed: float, op_type: str):
        """记录 GPU 计算"""
        self.timing["gpu_compute"] += elapsed
        if op_type in self.operation_counts:
            self.operation_counts[op_type] += 1
    
    def record_cpu_compute(self, elapsed: float, op_type: str):
        """记录 CPU 计算"""
        self.timing["cpu_compute"] += elapsed
        if op_type in self.operation_counts:
            self.operation_counts[op_type] += 1
    
    def record_encryption(self, elapsed: float, op_type: str):
        """记录加密时间"""
        if op_type == "linear_attn":
            self.timing["encryption_linear_attn"] += elapsed
            self.operation_counts["encryption_linear_attn_ops"] += 1
        elif op_type == "linear_mlp":
            self.timing["encryption_linear_mlp"] += elapsed
            self.operation_counts["encryption_linear_mlp_ops"] += 1
        elif op_type == "matmul":
            self.timing["encryption_matmul"] += elapsed
            self.operation_counts["encryption_matmul_ops"] += 1
    
    def record_decryption(self, elapsed: float, op_type: str):
        """记录解密时间"""
        if op_type == "linear_attn":
            self.timing["decryption_linear_attn"] += elapsed
            self.operation_counts["decryption_linear_attn_ops"] += 1
        elif op_type == "linear_mlp":
            self.timing["decryption_linear_mlp"] += elapsed
            self.operation_counts["decryption_linear_mlp_ops"] += 1
        elif op_type == "matmul":
            self.timing["decryption_matmul"] += elapsed
            self.operation_counts["decryption_matmul_ops"] += 1
    
    def get_summary(self) -> Dict:
        """获取统计摘要"""
        total_transfer = self.timing["transfer_to_gpu"] + self.timing["transfer_to_cpu"]
        total_compute = self.timing["gpu_compute"] + self.timing["cpu_compute"]
        total_encryption = (self.timing["encryption_linear_attn"] + self.timing["encryption_linear_mlp"] + 
                           self.timing["encryption_matmul"])
        total_decryption = (self.timing["decryption_linear_attn"] + self.timing["decryption_linear_mlp"] + 
                           self.timing["decryption_matmul"])
        total_crypto = total_encryption + total_decryption
        
        return {
            "timing": {
                "offline_ms": self.timing["offline"] * 1000,
                "transfer_to_gpu_ms": self.timing["transfer_to_gpu"] * 1000,
                "transfer_to_cpu_ms": self.timing["transfer_to_cpu"] * 1000,
                "total_transfer_ms": total_transfer * 1000,
                "gpu_compute_ms": self.timing["gpu_compute"] * 1000,
                "cpu_compute_ms": self.timing["cpu_compute"] * 1000,
                "total_compute_ms": total_compute * 1000,
                "encryption_linear_attn_ms": self.timing["encryption_linear_attn"] * 1000,
                "encryption_linear_mlp_ms": self.timing["encryption_linear_mlp"] * 1000,
                "encryption_matmul_ms": self.timing["encryption_matmul"] * 1000,
                "total_encryption_ms": total_encryption * 1000,
                "decryption_linear_attn_ms": self.timing["decryption_linear_attn"] * 1000,
                "decryption_linear_mlp_ms": self.timing["decryption_linear_mlp"] * 1000,
                "decryption_matmul_ms": self.timing["decryption_matmul"] * 1000,
                "total_decryption_ms": total_decryption * 1000,
                "total_crypto_ms": total_crypto * 1000,
                "total_ms": self.timing["total"] * 1000,
            },
            "timing_percentage": {
                "offline_pct": (self.timing["offline"] / self.timing["total"] * 100) if self.timing["total"] > 0 else 0,
                "transfer_pct": (total_transfer / self.timing["total"] * 100) if self.timing["total"] > 0 else 0,
                "gpu_compute_pct": (self.timing["gpu_compute"] / self.timing["total"] * 100) if self.timing["total"] > 0 else 0,
                "cpu_compute_pct": (self.timing["cpu_compute"] / self.timing["total"] * 100) if self.timing["total"] > 0 else 0,
                "crypto_pct": (total_crypto / self.timing["total"] * 100) if self.timing["total"] > 0 else 0,
            },
            "data_transfer": {
                "to_gpu_mb": self.data_transfer["to_gpu_bytes"] / 1024 / 1024,
                "to_cpu_mb": self.data_transfer["to_cpu_bytes"] / 1024 / 1024,
                "total_mb": (self.data_transfer["to_gpu_bytes"] + self.data_transfer["to_cpu_bytes"]) / 1024 / 1024,
            },
            "operation_counts": self.operation_counts,
        }
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()
        
        print(f"\n{'='*80}")
        print(f"{'Performance Summary (Our Encryption Scheme)':^80}")
        print(f"{'='*80}")
        
        print(f"\n{'Timing Breakdown':^80}")
        print(f"{'-'*80}")
        timing = summary["timing"]
        pct = summary["timing_percentage"]
        print(f"  Offline (Gen Matrix):  {timing['offline_ms']:>10.2f} ms  ({pct['offline_pct']:>5.1f}%)")
        print(f"  Transfer (CPU<->GPU):  {timing['total_transfer_ms']:>10.2f} ms  ({pct['transfer_pct']:>5.1f}%)")
        print(f"    - To GPU:            {timing['transfer_to_gpu_ms']:>10.2f} ms")
        print(f"    - To CPU:            {timing['transfer_to_cpu_ms']:>10.2f} ms")
        print(f"  GPU Compute:           {timing['gpu_compute_ms']:>10.2f} ms  ({pct['gpu_compute_pct']:>5.1f}%)")
        print(f"  CPU Compute (TEE):     {timing['cpu_compute_ms']:>10.2f} ms  ({pct['cpu_compute_pct']:>5.1f}%)")
        print(f"  Crypto Overhead:       {timing['total_crypto_ms']:>10.2f} ms  ({pct['crypto_pct']:>5.1f}%)")
        print(f"    Encryption:")
        print(f"      - Linear(Attn):    {timing['encryption_linear_attn_ms']:>10.2f} ms")
        print(f"      - Linear(MLP):     {timing['encryption_linear_mlp_ms']:>10.2f} ms")
        print(f"      - Matmul:          {timing['encryption_matmul_ms']:>10.2f} ms")
        print(f"      - Total:           {timing['total_encryption_ms']:>10.2f} ms")
        print(f"    Decryption:")
        print(f"      - Linear(Attn):    {timing['decryption_linear_attn_ms']:>10.2f} ms")
        print(f"      - Linear(MLP):     {timing['decryption_linear_mlp_ms']:>10.2f} ms")
        print(f"      - Matmul:          {timing['decryption_matmul_ms']:>10.2f} ms")
        print(f"      - Total:           {timing['total_decryption_ms']:>10.2f} ms")
        print(f"  {'-'*80}")
        print(f"  Total:                 {timing['total_ms']:>10.2f} ms  (100.0%)")
        
        print(f"\n{'Data Transfer':^80}")
        print(f"{'-'*80}")
        transfer = summary["data_transfer"]
        print(f"  To GPU:                {transfer['to_gpu_mb']:>10.2f} MB")
        print(f"  To CPU:                {transfer['to_cpu_mb']:>10.2f} MB")
        print(f"  Total:                 {transfer['total_mb']:>10.2f} MB")
        
        print(f"\n{'Operation Counts':^80}")
        print(f"{'-'*80}")
        ops = summary["operation_counts"]
        print(f"  GPU Operations:")
        print(f"    - Embedding:         {ops['gpu_embedding']:>8}")
        print(f"    - Linear:            {ops['gpu_linear']:>8}")
        print(f"    - Matmul:            {ops['gpu_matmul']:>8}")
        print(f"    - LM Head:           {ops['gpu_lm_head']:>8}")
        print(f"  CPU Operations (TEE):")
        print(f"    - RMSNorm:           {ops['cpu_rmsnorm']:>8}")
        print(f"    - Rotary:            {ops['cpu_rotary']:>8}")
        print(f"    - Softmax:           {ops['cpu_softmax']:>8}")
        print(f"    - SiLU:              {ops['cpu_silu']:>8}")
        print(f"    - Multiply:          {ops['cpu_multiply']:>8}")
        print(f"  Crypto Operations:")
        print(f"    Encryption:")
        print(f"      - Linear(Attn):    {ops['encryption_linear_attn_ops']:>8}")
        print(f"      - Linear(MLP):     {ops['encryption_linear_mlp_ops']:>8}")
        print(f"      - Matmul:          {ops['encryption_matmul_ops']:>8}")
        print(f"    Decryption:")
        print(f"      - Linear(Attn):    {ops['decryption_linear_attn_ops']:>8}")
        print(f"      - Linear(MLP):     {ops['decryption_linear_mlp_ops']:>8}")
        print(f"      - Matmul:          {ops['decryption_matmul_ops']:>8}")
        
        print(f"{'='*80}\n")


class OurEncryptionScheme:
    """我们的加密方案：Linear层使用矩阵变换（优化版本，支持Offline统计）"""
    
    def __init__(self, seq_len: int, device: torch.device):
        self.seq_len = seq_len
        self.device = device
        self.initialized = False
        
        # 加密参数（将在offline阶段生成）
        self.D_diag = None
        self.alpha = None
        self.beta = None
        self.D_inv_diag = None
        self.D_inv_alpha = None
        self.beta_T_D_inv_alpha = None
        self.scale_factor = None
    
    def generate_encryption_params(self):
        """Offline 阶段：生成加密矩阵参数"""
        if self.initialized:
            return
        
        # 🔑 优化：只存储对角元素，不存储完整矩阵
        self.D_diag = (torch.randn(self.seq_len, device=self.device) + 2.0)  # (seq_len,) 只存储对角元素
        self.alpha = torch.randn(self.seq_len, 1, device=self.device)  # 列向量
        self.beta = torch.randn(self.seq_len, 1, device=self.device)   # 列向量
        
        # 预计算逆矩阵的对角元素（优化）
        self.D_inv_diag = 1.0 / self.D_diag  # (seq_len,)
        self.D_inv_alpha = self.D_inv_diag.unsqueeze(-1) * self.alpha  # (seq_len, 1)
        self.beta_T_D_inv_alpha = (self.beta.T @ self.D_inv_alpha).item()
        self.scale_factor = 1.0 / (1.0 + self.beta_T_D_inv_alpha)
        
        self.initialized = True
    
    def encrypt_linear_input(self, X: torch.Tensor) -> torch.Tensor:
        """
        加密 Linear 层输入: MX = DX + α(β^T X)
        优化：利用 D 是对角矩阵的性质，复杂度从 O(n²×d) 降到 O(n×d)
        """
        # X: (batch, seq_len, in_features)
        
        # 确保加密参数与输入的数据类型匹配
        D_diag = self.D_diag.to(X.dtype)
        alpha = self.alpha.to(X.dtype)
        beta = self.beta.to(X.dtype)
        
        # 🚀 优化：对角矩阵乘法 -> 逐元素乘法
        # DX = D @ X，但 D 是对角矩阵，等价于每行乘以对应的对角元素
        DX = D_diag.view(1, -1, 1) * X  # 广播：(1, seq_len, 1) × (batch, seq_len, in_features)
        
        # β^T X: (1, seq_len) @ (batch, seq_len, in_features) -> (batch, 1, in_features)
        # beta.T: (1, seq_len), X: (batch, seq_len, in_features)
        beta_T_X = torch.einsum('ij,bjk->bik', beta.T, X)
        
        # α(β^T X): (seq_len, 1) × (batch, 1, in_features) -> (batch, seq_len, in_features)
        alpha_beta_T_X = alpha * beta_T_X  # 广播
        
        MX = DX + alpha_beta_T_X
        return MX
    
    def decrypt_linear_output(self, Z: torch.Tensor) -> torch.Tensor:
        """
        解密 Linear 层输出: M^{-1}Z = D^{-1}Z - scale * D^{-1}α(β^T D^{-1}Z)
        优化：利用 D^{-1} 是对角矩阵的性质，复杂度从 O(n²×d) 降到 O(n×d)
        """
        # Z: (batch, seq_len, out_features)
        
        # 确保加密参数与输入的数据类型匹配
        D_inv_diag = self.D_inv_diag.to(Z.dtype)
        beta = self.beta.to(Z.dtype)
        D_inv_alpha = self.D_inv_alpha.to(Z.dtype)
        
        # 🚀 优化：对角矩阵乘法
        # D^{-1}Z
        D_inv_Z = D_inv_diag.view(1, -1, 1) * Z
        
        # β^T D^{-1}Z: (1, seq_len) @ (batch, seq_len, out_features) -> (batch, 1, out_features)
        beta_T_D_inv_Z = torch.einsum('ij,bjk->bik', beta.T, D_inv_Z)
        
        # D^{-1}α(β^T D^{-1}Z): (seq_len, 1) × (batch, 1, out_features) -> (batch, seq_len, out_features)
        D_inv_alpha_term = D_inv_alpha * beta_T_D_inv_Z
        
        # M^{-1}Z
        M_inv_Z = D_inv_Z - self.scale_factor * D_inv_alpha_term
        return M_inv_Z


class MatmulEncryptionScheme:
    """Matmul 加密方案（优化版本，支持Offline统计）"""
    
    def __init__(self, seq_len: int, head_dim: int, device: torch.device):
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.device = device
        self.initialized = False
        
        # 加密参数（将在offline阶段生成）
        self.D1_diag = None
        self.D2_diag = None
        self.D3_diag = None
        self.D1_inv_diag = None
        self.D2_inv_diag = None
        self.D3_inv_diag = None
    
    def generate_encryption_params(self):
        """Offline 阶段：生成加密矩阵参数"""
        if self.initialized:
            return
        
        # 🔑 优化：只存储对角元素，不存储完整矩阵
        self.D1_diag = (torch.randn(self.seq_len, device=self.device) + 2.0)  # (seq_len,)
        self.D2_diag = (torch.randn(self.head_dim, device=self.device) + 2.0)  # (head_dim,)
        self.D3_diag = (torch.randn(self.seq_len, device=self.device) + 2.0)  # (seq_len,)
        
        # 预计算逆矩阵的对角元素（优化）
        self.D1_inv_diag = 1.0 / self.D1_diag  # (seq_len,)
        self.D2_inv_diag = 1.0 / self.D2_diag  # (head_dim,)
        self.D3_inv_diag = 1.0 / self.D3_diag  # (seq_len,)
        
        # 🔑 优化：P1, P2, P3 是单位矩阵，加密/解密时直接跳过（不存储）
        
        self.initialized = True
    
    def encrypt_query(self, Q: torch.Tensor) -> torch.Tensor:
        """
        加密 Query: Q' = (D₁P₁)Q(P₂D₂)
        优化：P1=P2=I（单位矩阵），简化为 Q' = D₁ Q D₂
        使用向量化对角乘法，复杂度从 O(batch×num_heads×n³) 降到 O(batch×num_heads×n×d)
        """
        # Q: (batch, num_heads, seq_len, head_dim)
        
        # 确保数据类型匹配
        D1_diag = self.D1_diag.to(Q.dtype)
        D2_diag = self.D2_diag.to(Q.dtype)
        
        # 🚀 向量化对角乘法（无循环！全部并行）
        # D1 作用于 seq_len 维度（axis=-2）
        Q_encrypted = Q * D1_diag.view(1, 1, -1, 1)  # 广播
        
        # D2 作用于 head_dim 维度（axis=-1）
        Q_encrypted = Q_encrypted * D2_diag.view(1, 1, 1, -1)  # 广播
        
        return Q_encrypted
    
    def encrypt_key_transpose(self, K_T: torch.Tensor) -> torch.Tensor:
        """
        加密 Key^T: K'^T = (D₂⁻¹P₂⁻¹)K^T(P₃D₃)
        优化：P2=P3=I，简化为 K'^T = D₂⁻¹ K^T D₃
        使用向量化对角乘法，复杂度从 O(batch×num_heads×n³) 降到 O(batch×num_heads×d×n)
        """
        # K_T: (batch, num_heads, head_dim, seq_len)
        
        # 确保数据类型匹配
        D2_inv_diag = self.D2_inv_diag.to(K_T.dtype)
        D3_diag = self.D3_diag.to(K_T.dtype)
        
        # 🚀 向量化对角乘法（无循环！全部并行）
        # D2_inv 作用于 head_dim 维度（axis=-2）
        K_T_encrypted = K_T * D2_inv_diag.view(1, 1, -1, 1)
        
        # D3 作用于 seq_len 维度（axis=-1）
        K_T_encrypted = K_T_encrypted * D3_diag.view(1, 1, 1, -1)
        
        return K_T_encrypted
    
    def decrypt_matmul_output(self, QK_T_encrypted: torch.Tensor) -> torch.Tensor:
        """
        解密 Matmul 输出: QK^T = P₁⁻¹D₁⁻¹Q'K'^TD₃⁻¹P₃⁻¹
        优化：P1=P3=I，简化为 QK^T = D₁⁻¹ Q'K'^T D₃⁻¹
        使用向量化对角乘法，复杂度从 O(batch×num_heads×n³) 降到 O(batch×num_heads×n²)
        """
        # QK_T_encrypted: (batch, num_heads, seq_len, seq_len)
        
        # 确保数据类型匹配
        D1_inv_diag = self.D1_inv_diag.to(QK_T_encrypted.dtype)
        D3_inv_diag = self.D3_inv_diag.to(QK_T_encrypted.dtype)
        
        # 🚀 向量化对角乘法（无循环！全部并行）
        # D1_inv 作用于第一个 seq_len 维度（axis=-2）
        QK_T_decrypted = QK_T_encrypted * D1_inv_diag.view(1, 1, -1, 1)
        
        # D3_inv 作用于第二个 seq_len 维度（axis=-1）
        QK_T_decrypted = QK_T_decrypted * D3_inv_diag.view(1, 1, 1, -1)
        
        return QK_T_decrypted


class TEELlamaModel(nn.Module):
    """TEE+GPU 混合 LLaMA 模型 - 我们的加密方案"""
    
    def __init__(self, hf_model: AutoModelForCausalLM, gpu_device: str, cpu_device: str):
        super().__init__()
        self.gpu_device = torch.device(gpu_device)
        self.cpu_device = torch.device(cpu_device)
        self.tracker = PerformanceTracker()
        
        # 提取配置
        self.config = hf_model.config
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        
        # 1. Embedding (GPU)
        self.embed_tokens = hf_model.model.embed_tokens.to(self.gpu_device)
        
        # 2. Rotary Embedding (CPU/TEE)
        hf_rotary = hf_model.model.rotary_emb
        inv_freq = hf_rotary.inv_freq.to(self.cpu_device)
        attention_scaling = getattr(hf_rotary, "attention_scaling", 1.0)
        if isinstance(attention_scaling, torch.Tensor):
            attention_scaling = attention_scaling.item()
        self.rotary_emb = TEERotaryEmbedding(inv_freq, attention_scaling)
        
        # 3. Layers
        self.tee_input_norms = nn.ModuleList()
        self.tee_post_norms = nn.ModuleList()
        self.gpu_layers_attn = []
        self.gpu_layers_mlp = []
        
        for i, hf_layer in enumerate(hf_model.model.layers):
            # TEE 部分 (CPU)
            input_norm = TEERMSNorm(
                hf_layer.input_layernorm.weight.to(self.cpu_device),
                hf_layer.input_layernorm.variance_epsilon
            )
            post_norm = TEERMSNorm(
                hf_layer.post_attention_layernorm.weight.to(self.cpu_device),
                hf_layer.post_attention_layernorm.variance_epsilon
            )
            self.tee_input_norms.append(input_norm)
            self.tee_post_norms.append(post_norm)
            
            # GPU 部分
            hf_layer.self_attn.q_proj.to(self.gpu_device)
            hf_layer.self_attn.k_proj.to(self.gpu_device)
            hf_layer.self_attn.v_proj.to(self.gpu_device)
            hf_layer.self_attn.o_proj.to(self.gpu_device)
            hf_layer.mlp.gate_proj.to(self.gpu_device)
            hf_layer.mlp.up_proj.to(self.gpu_device)
            hf_layer.mlp.down_proj.to(self.gpu_device)
            
            self.gpu_layers_attn.append(hf_layer.self_attn)
            self.gpu_layers_mlp.append(hf_layer.mlp)
        
        # 4. Final Norm (CPU/TEE)
        self.final_norm = TEERMSNorm(
            hf_model.model.norm.weight.to(self.cpu_device),
            hf_model.model.norm.variance_epsilon
        )
        
        # 5. LM Head (GPU)
        self.lm_head_layer = hf_model.lm_head.to(self.gpu_device)
        
        # 加密方案（初始化时创建，使用 CPU device）
        self.linear_enc = None
        self.matmul_enc = None
        
        print(f"✓ TEE+GPU Hybrid Model initialized (Our Encryption Scheme)")
        print(f"  - Layers: {self.num_layers}")
        print(f"  - GPU Device: {self.gpu_device}")
        print(f"  - CPU Device (TEE): {self.cpu_device}")
    
    def _to_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """传输到 GPU 并记录"""
        t0 = time.perf_counter()
        result = tensor.to(self.gpu_device)
        elapsed = time.perf_counter() - t0
        self.tracker.record_transfer_to_gpu(tensor, elapsed)
        return result
    
    def _to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """传输到 CPU 并记录"""
        t0 = time.perf_counter()
        result = tensor.to(self.cpu_device)
        elapsed = time.perf_counter() - t0
        self.tracker.record_transfer_to_cpu(tensor, elapsed)
        return result
    
    def attention(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Attention 层 - 使用加密"""
        batch_size, seq_len, _ = hidden_states.shape
        attn_layer = self.gpu_layers_attn[layer_idx]
        
        # 初始化加密方案（如果还没有）
        if self.linear_enc is None:
            self.linear_enc = OurEncryptionScheme(seq_len, self.cpu_device)
        if self.matmul_enc is None:
            self.matmul_enc = MatmulEncryptionScheme(seq_len, self.head_dim, self.cpu_device)
        
        # ===== OFFLINE 阶段: 生成加密参数 =====
        t0 = time.perf_counter()
        self.linear_enc.generate_encryption_params()
        self.matmul_enc.generate_encryption_params()
        self.tracker.record_offline(time.perf_counter() - t0)
        
        # ===== ONLINE 阶段 =====
        # TEE: 加密输入
        t0 = time.perf_counter()
        hidden_states_encrypted = self.linear_enc.encrypt_linear_input(hidden_states)
        self.tracker.record_encryption(time.perf_counter() - t0, "linear_attn")
        
        # GPU: QKV projections (在加密数据上)
        hs_gpu = self._to_gpu(hidden_states_encrypted)
        
        t0 = time.perf_counter()
        q_proj = attn_layer.q_proj(hs_gpu)
        k_proj = attn_layer.k_proj(hs_gpu)
        v_proj = attn_layer.v_proj(hs_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_linear")
        
        # TEE: 解密 QKV
        q_proj_cpu = self._to_cpu(q_proj)
        k_proj_cpu = self._to_cpu(k_proj)
        v_proj_cpu = self._to_cpu(v_proj)
        
        t0 = time.perf_counter()
        query_states = self.linear_enc.decrypt_linear_output(q_proj_cpu)
        key_states = self.linear_enc.decrypt_linear_output(k_proj_cpu)
        value_states = self.linear_enc.decrypt_linear_output(v_proj_cpu)
        self.tracker.record_decryption(time.perf_counter() - t0, "linear_attn")
        
        # TEE: Reshape & Rotary
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        t0 = time.perf_counter()
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        elapsed = time.perf_counter() - t0
        self.tracker.record_cpu_compute(elapsed, "cpu_rotary")
        
        # TEE: 加密 Q 和 K^T
        t0 = time.perf_counter()
        query_encrypted = self.matmul_enc.encrypt_query(query_states)
        key_T = key_states.transpose(2, 3)
        key_T_encrypted = self.matmul_enc.encrypt_key_transpose(key_T)
        self.tracker.record_encryption(time.perf_counter() - t0, "matmul")
        
        # GPU: Q @ K^T (在加密数据上)
        query_gpu = self._to_gpu(query_encrypted)
        key_T_gpu = self._to_gpu(key_T_encrypted)
        
        t0 = time.perf_counter()
        attn_weights_encrypted = torch.matmul(query_gpu, key_T_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_matmul")
        
        # TEE: 解密 attention weights
        attn_weights_encrypted_cpu = self._to_cpu(attn_weights_encrypted)
        
        t0 = time.perf_counter()
        attn_weights = self.matmul_enc.decrypt_matmul_output(attn_weights_encrypted_cpu)
        self.tracker.record_decryption(time.perf_counter() - t0, "matmul")
        
        # TEE: Softmax
        attn_weights = attn_weights * self.scaling
        
        t0 = time.perf_counter()
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        elapsed = time.perf_counter() - t0
        self.tracker.record_cpu_compute(elapsed, "cpu_softmax")
        
        # GPU: Attn @ V
        attn_weights_gpu = self._to_gpu(attn_weights)
        value_gpu = self._to_gpu(value_states)
        
        t0 = time.perf_counter()
        attn_output = torch.matmul(attn_weights_gpu, value_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_matmul")
        
        # TEE: Reshape
        attn_output = self._to_cpu(attn_output).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # TEE: 加密输出
        t0 = time.perf_counter()
        attn_output_encrypted = self.linear_enc.encrypt_linear_input(attn_output)
        self.tracker.record_encryption(time.perf_counter() - t0, "linear_attn")
        
        # GPU: O projection
        attn_output_gpu = self._to_gpu(attn_output_encrypted)
        
        t0 = time.perf_counter()
        attn_output_final = attn_layer.o_proj(attn_output_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_linear")
        
        # TEE: 解密最终输出
        attn_output_final_cpu = self._to_cpu(attn_output_final)
        
        t0 = time.perf_counter()
        attn_output_decrypted = self.linear_enc.decrypt_linear_output(attn_output_final_cpu)
        self.tracker.record_decryption(time.perf_counter() - t0, "linear_attn")
        
        return attn_output_decrypted
    
    def mlp(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP 层 - 使用加密"""
        mlp_layer = self.gpu_layers_mlp[layer_idx]
        
        # TEE: 加密输入
        t0 = time.perf_counter()
        hidden_states_encrypted = self.linear_enc.encrypt_linear_input(hidden_states)
        self.tracker.record_encryption(time.perf_counter() - t0, "linear_mlp")
        
        # GPU: Gate + Up
        hs_gpu = self._to_gpu(hidden_states_encrypted)
        
        t0 = time.perf_counter()
        gate = mlp_layer.gate_proj(hs_gpu)
        up = mlp_layer.up_proj(hs_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_linear")
        
        # TEE: 解密
        gate_cpu = self._to_cpu(gate)
        up_cpu = self._to_cpu(up)
        
        t0 = time.perf_counter()
        gate_decrypted = self.linear_enc.decrypt_linear_output(gate_cpu)
        up_decrypted = self.linear_enc.decrypt_linear_output(up_cpu)
        self.tracker.record_decryption(time.perf_counter() - t0, "linear_mlp")
        
        # TEE: SiLU
        t0 = time.perf_counter()
        gate_decrypted = F.silu(gate_decrypted)
        elapsed = time.perf_counter() - t0
        self.tracker.record_cpu_compute(elapsed, "cpu_silu")
        
        # TEE: multiply
        t0 = time.perf_counter()
        intermediate = gate_decrypted * up_decrypted
        elapsed = time.perf_counter() - t0
        self.tracker.record_cpu_compute(elapsed, "cpu_multiply")
        
        # TEE: 加密中间结果
        t0 = time.perf_counter()
        intermediate_encrypted = self.linear_enc.encrypt_linear_input(intermediate)
        self.tracker.record_encryption(time.perf_counter() - t0, "linear_mlp")
        
        # GPU: Down
        intermediate_gpu = self._to_gpu(intermediate_encrypted)
        
        t0 = time.perf_counter()
        output = mlp_layer.down_proj(intermediate_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_linear")
        
        # TEE: 解密输出
        output_cpu = self._to_cpu(output)
        
        t0 = time.perf_counter()
        output_decrypted = self.linear_enc.decrypt_linear_output(output_cpu)
        self.tracker.record_decryption(time.perf_counter() - t0, "linear_mlp")
        
        return output_decrypted
    
    def decoder_layer(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Decoder 层"""
        # Attention
        residual = hidden_states
        
        t0 = time.perf_counter()
        hidden_states = self.tee_input_norms[layer_idx](hidden_states)
        elapsed = time.perf_counter() - t0
        self.tracker.record_cpu_compute(elapsed, "cpu_rmsnorm")
        
        hidden_states = self.attention(layer_idx, hidden_states, position_ids)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        
        t0 = time.perf_counter()
        hidden_states = self.tee_post_norms[layer_idx](hidden_states)
        elapsed = time.perf_counter() - t0
        self.tracker.record_cpu_compute(elapsed, "cpu_rmsnorm")
        
        hidden_states = self.mlp(layer_idx, hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len = input_ids.shape
        
        # GPU: Embedding
        input_ids_gpu = self._to_gpu(input_ids)
        
        t0 = time.perf_counter()
        hidden_states = self.embed_tokens(input_ids_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_embedding")
        
        hidden_states = self._to_cpu(hidden_states)
        
        # Position IDs (CPU)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).to(self.cpu_device)
        
        # Decoder layers
        for layer_idx in range(self.num_layers):
            hidden_states = self.decoder_layer(layer_idx, hidden_states, position_ids)
        
        # TEE: Final norm
        t0 = time.perf_counter()
        hidden_states = self.final_norm(hidden_states)
        elapsed = time.perf_counter() - t0
        self.tracker.record_cpu_compute(elapsed, "cpu_rmsnorm")
        
        # GPU: LM head
        hidden_states_last = hidden_states[:, -1:, :]
        hidden_states_gpu = self._to_gpu(hidden_states_last)
        
        t0 = time.perf_counter()
        logits = self.lm_head_layer(hidden_states_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_lm_head")
        
        logits = self._to_cpu(logits)
        
        return logits


def run_benchmark(model: TEELlamaModel, tokenizer, prefill_length: int) -> Dict:
    """运行性能测试 - 无 warmup"""
    input_ids = torch.full((1, prefill_length), tokenizer.pad_token_id, dtype=torch.long)
    
    print(f"\n{'='*80}")
    print(f"{'TEE+GPU Hybrid Inference Benchmark (Our Encryption)':^80}")
    print(f"{'='*80}")
    print(f"  Token length: {prefill_length}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Encryption: Linear(Matrix Transform), Matmul(Matrix Transform)")
    print(f"{'='*80}\n")
    
    # 直接运行 Benchmark (无 warmup)
    print("Running benchmark (no warmup)...")
    start_time = time.perf_counter()
    logits = model.forward(input_ids)
    total_time = time.perf_counter() - start_time
    
    model.tracker.timing["total"] = total_time
    
    print(f"\n{'='*80}")
    print(f"Benchmark completed!")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Throughput: {prefill_length / total_time:.2f} tokens/sec")
    print(f"  Logits shape: {logits.shape}")
    print(f"{'='*80}")
    
    # Print detailed stats
    model.tracker.print_summary()
    
    # Return results
    summary = model.tracker.get_summary()
    summary["benchmark_info"] = {
        "model_path": MODEL_PATH,
        "prefill_length": prefill_length,
        "throughput_tokens_per_sec": prefill_length / total_time,
        "logits_shape": list(logits.shape),
        "encryption_scheme": "Our Matrix Transform",
    }
    
    return summary


def main() -> None:
    """主函数"""
    print(f"Loading model from: {MODEL_PATH}")
    
    # 检查设备
    if torch.cuda.is_available():
        gpu_device = GPU_DEVICE
        print(f"✓ CUDA available, using: {gpu_device}")
    else:
        print("Warning: CUDA not available, using CPU for all operations")
        gpu_device = "cpu"
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=os.path.exists(MODEL_PATH),
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("Loading model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=os.path.exists(MODEL_PATH),
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cpu"
    )
    
    # 创建 TEE+GPU 混合模型
    print("Initializing TEE+GPU Hybrid Model (Our Encryption Scheme)...")
    model = TEELlamaModel(hf_model, gpu_device, CPU_DEVICE)
    
    # 运行测试
    results = run_benchmark(model, tokenizer, PREFILL_TOKEN_LENGTH)
    
    # 保存结果
    output_file = OUTPUT_FILE.replace(".json", "_ours.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
