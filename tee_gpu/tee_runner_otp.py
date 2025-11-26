"""
TEE+GPU 混合推理 - OTP加密方案 (Intel TDX Passthrough 版本)
实现方案：
1. Linear层: 使用加法秘密分享 (X-R)W + RW
   - 在线计算: 每次生成随机掩码R
   - TEE内部用随机权重代替真实权重，计算RW（模拟计算开销）
   - 发送(X-R)到GPU计算(X-R)W（使用真实权重）
   - 恢复: TEE中计算 Y = (X-R)W + RW
2. Matmul层: 使用嵌入式加法外包 (Embedded Additive Outsource)
3. 性能统计: 三部分计时（传输、GPU计算、CPU/TEE计算）
4. Intel TDX Passthrough: 直接使用 .to(device) 进行数据传输
5. 无 warmup 步骤，直接计时
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
    """性能追踪器 - 三部分计时 + 加密开销"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.timing = {
            "transfer_to_gpu": 0.0,      # CPU -> GPU 传输
            "transfer_to_cpu": 0.0,      # GPU -> CPU 传输
            "gpu_compute": 0.0,          # GPU 计算
            "cpu_compute": 0.0,          # CPU 计算 (TEE)
            "masking": 0.0,              # 掩码生成和应用
            "recovery": 0.0,             # 恢复计算
            "total": 0.0,                # 总时间
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
            "masking_ops": 0,
            "recovery_ops": 0,
        }
    
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
    
    def record_masking(self, elapsed: float):
        """记录掩码时间"""
        self.timing["masking"] += elapsed
        self.operation_counts["masking_ops"] += 1
    
    def record_recovery(self, elapsed: float):
        """记录恢复时间"""
        self.timing["recovery"] += elapsed
        self.operation_counts["recovery_ops"] += 1
    
    def get_summary(self) -> Dict:
        """获取统计摘要"""
        total_transfer = self.timing["transfer_to_gpu"] + self.timing["transfer_to_cpu"]
        total_compute = self.timing["gpu_compute"] + self.timing["cpu_compute"]
        total_crypto = self.timing["masking"] + self.timing["recovery"]
        
        return {
            "timing": {
                "transfer_to_gpu_ms": self.timing["transfer_to_gpu"] * 1000,
                "transfer_to_cpu_ms": self.timing["transfer_to_cpu"] * 1000,
                "total_transfer_ms": total_transfer * 1000,
                "gpu_compute_ms": self.timing["gpu_compute"] * 1000,
                "cpu_compute_ms": self.timing["cpu_compute"] * 1000,
                "total_compute_ms": total_compute * 1000,
                "masking_ms": self.timing["masking"] * 1000,
                "recovery_ms": self.timing["recovery"] * 1000,
                "total_crypto_ms": total_crypto * 1000,
                "total_ms": self.timing["total"] * 1000,
            },
            "timing_percentage": {
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
        print(f"{'Performance Summary (OTP Encryption Scheme)':^80}")
        print(f"{'='*80}")
        
        print(f"\n{'Timing Breakdown':^80}")
        print(f"{'-'*80}")
        timing = summary["timing"]
        pct = summary["timing_percentage"]
        print(f"  Transfer (CPU<->GPU):  {timing['total_transfer_ms']:>10.2f} ms  ({pct['transfer_pct']:>5.1f}%)")
        print(f"    - To GPU:            {timing['transfer_to_gpu_ms']:>10.2f} ms")
        print(f"    - To CPU:            {timing['transfer_to_cpu_ms']:>10.2f} ms")
        print(f"  GPU Compute:           {timing['gpu_compute_ms']:>10.2f} ms  ({pct['gpu_compute_pct']:>5.1f}%)")
        print(f"  CPU Compute (TEE):     {timing['cpu_compute_ms']:>10.2f} ms  ({pct['cpu_compute_pct']:>5.1f}%)")
        print(f"  Crypto (Mask+Recover): {timing['total_crypto_ms']:>10.2f} ms  ({pct['crypto_pct']:>5.1f}%)")
        print(f"    - Masking:           {timing['masking_ms']:>10.2f} ms")
        print(f"    - Recovery:          {timing['recovery_ms']:>10.2f} ms")
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
        print(f"  Crypto Operations:")
        print(f"    - Masking:           {ops['masking_ops']:>8}")
        print(f"    - Recovery:          {ops['recovery_ops']:>8}")
        
        print(f"{'='*80}\n")


class OTPEncryption:
    """OTP 加密方案：使用加法秘密分享"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def mask_linear_input(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """掩码 Linear 输入: 返回 (X-R, R)"""
        R = torch.randn_like(X, device=self.device)
        X_masked = X - R
        return X_masked, R
    
    def recover_linear_output(self, Y_masked: torch.Tensor, RW: torch.Tensor) -> torch.Tensor:
        """恢复 Linear 输出: Y = (X-R)W + RW"""
        return Y_masked + RW
    
    def compute_RW(self, R: torch.Tensor, weight_shape: Tuple[int, int]) -> torch.Tensor:
        """在 TEE 中计算 RW（使用随机权重模拟）"""
        # 使用随机权重模拟计算开销，确保 dtype 匹配
        W_random = torch.randn(weight_shape, device=self.device, dtype=R.dtype) * 0.01
        RW = torch.matmul(R, W_random.T)
        return RW


class EmbeddedAdditiveOutsource:
    """嵌入式加法外包方案用于 Matmul"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def mask_matmul_inputs(self, Q: torch.Tensor, K_T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """掩码 Matmul 输入"""
        R_Q = torch.randn_like(Q, device=self.device) * 0.1
        R_K = torch.randn_like(K_T, device=self.device) * 0.1
        
        Q_masked = Q - R_Q
        K_T_masked = K_T - R_K
        
        return Q_masked, K_T_masked, R_Q, R_K
    
    def recover_matmul_output(self, QK_T_masked: torch.Tensor, R_Q: torch.Tensor, R_K: torch.Tensor, K_T: torch.Tensor) -> torch.Tensor:
        """恢复 Matmul 输出"""
        # QK^T = (Q-R_Q)(K^T-R_K) + R_Q*K^T + Q*R_K - R_Q*R_K
        # 简化版本：QK^T ≈ QK_T_masked + R_Q @ K_T
        correction = torch.matmul(R_Q, K_T)
        return QK_T_masked + correction


class TEELlamaModel(nn.Module):
    """TEE+GPU 混合 LLaMA 模型 - OTP 加密方案"""
    
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
        
        # 加密方案
        self.otp_enc = OTPEncryption(self.cpu_device)
        self.matmul_enc = EmbeddedAdditiveOutsource(self.cpu_device)
        
        print(f"✓ TEE+GPU Hybrid Model initialized (OTP Encryption Scheme)")
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
        """Attention 层 - 使用 OTP 加密"""
        batch_size, seq_len, _ = hidden_states.shape
        attn_layer = self.gpu_layers_attn[layer_idx]
        
        # TEE: 掩码输入
        t0 = time.perf_counter()
        hs_masked, R = self.otp_enc.mask_linear_input(hidden_states)
        self.tracker.record_masking(time.perf_counter() - t0)
        
        # GPU: QKV projections (在掩码数据上)
        hs_gpu = self._to_gpu(hs_masked)
        
        t0 = time.perf_counter()
        q_proj_masked = attn_layer.q_proj(hs_gpu)
        k_proj_masked = attn_layer.k_proj(hs_gpu)
        v_proj_masked = attn_layer.v_proj(hs_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_linear")
        
        # TEE: 恢复 QKV
        q_proj_masked_cpu = self._to_cpu(q_proj_masked)
        k_proj_masked_cpu = self._to_cpu(k_proj_masked)
        v_proj_masked_cpu = self._to_cpu(v_proj_masked)
        
        t0 = time.perf_counter()
        # 计算 RW 并恢复
        RW_q = self.otp_enc.compute_RW(R, (self.hidden_size, self.num_heads * self.head_dim))
        RW_k = self.otp_enc.compute_RW(R, (self.hidden_size, self.num_kv_heads * self.head_dim))
        RW_v = self.otp_enc.compute_RW(R, (self.hidden_size, self.num_kv_heads * self.head_dim))
        
        query_states = self.otp_enc.recover_linear_output(q_proj_masked_cpu, RW_q)
        key_states = self.otp_enc.recover_linear_output(k_proj_masked_cpu, RW_k)
        value_states = self.otp_enc.recover_linear_output(v_proj_masked_cpu, RW_v)
        self.tracker.record_recovery(time.perf_counter() - t0)
        
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
        
        # TEE: 掩码 Q 和 K^T
        key_T = key_states.transpose(2, 3)
        
        t0 = time.perf_counter()
        Q_masked, K_T_masked, R_Q, R_K = self.matmul_enc.mask_matmul_inputs(query_states, key_T)
        self.tracker.record_masking(time.perf_counter() - t0)
        
        # GPU: Q @ K^T (在掩码数据上)
        Q_masked_gpu = self._to_gpu(Q_masked)
        K_T_masked_gpu = self._to_gpu(K_T_masked)
        
        t0 = time.perf_counter()
        attn_weights_masked = torch.matmul(Q_masked_gpu, K_T_masked_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_matmul")
        
        # TEE: 恢复 attention weights
        attn_weights_masked_cpu = self._to_cpu(attn_weights_masked)
        key_T_cpu = key_T  # 已经在 CPU 上
        
        t0 = time.perf_counter()
        attn_weights = self.matmul_enc.recover_matmul_output(attn_weights_masked_cpu, R_Q, R_K, key_T_cpu)
        self.tracker.record_recovery(time.perf_counter() - t0)
        
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
        
        # TEE: 掩码输出
        t0 = time.perf_counter()
        attn_output_masked, R_out = self.otp_enc.mask_linear_input(attn_output)
        self.tracker.record_masking(time.perf_counter() - t0)
        
        # GPU: O projection
        attn_output_gpu = self._to_gpu(attn_output_masked)
        
        t0 = time.perf_counter()
        attn_output_final_masked = attn_layer.o_proj(attn_output_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_linear")
        
        # TEE: 恢复最终输出
        attn_output_final_masked_cpu = self._to_cpu(attn_output_final_masked)
        
        t0 = time.perf_counter()
        RW_o = self.otp_enc.compute_RW(R_out, (self.hidden_size, self.hidden_size))
        attn_output_final = self.otp_enc.recover_linear_output(attn_output_final_masked_cpu, RW_o)
        self.tracker.record_recovery(time.perf_counter() - t0)
        
        return attn_output_final
    
    def mlp(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP 层 - 使用 OTP 加密"""
        mlp_layer = self.gpu_layers_mlp[layer_idx]
        
        # TEE: 掩码输入
        t0 = time.perf_counter()
        hs_masked, R = self.otp_enc.mask_linear_input(hidden_states)
        self.tracker.record_masking(time.perf_counter() - t0)
        
        # GPU: Gate + Up
        hs_gpu = self._to_gpu(hs_masked)
        
        t0 = time.perf_counter()
        gate_masked = mlp_layer.gate_proj(hs_gpu)
        up_masked = mlp_layer.up_proj(hs_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_linear")
        
        # TEE: 恢复
        gate_masked_cpu = self._to_cpu(gate_masked)
        up_masked_cpu = self._to_cpu(up_masked)
        
        t0 = time.perf_counter()
        intermediate_size = self.config.intermediate_size
        RW_gate = self.otp_enc.compute_RW(R, (self.hidden_size, intermediate_size))
        RW_up = self.otp_enc.compute_RW(R, (self.hidden_size, intermediate_size))
        
        gate = self.otp_enc.recover_linear_output(gate_masked_cpu, RW_gate)
        up = self.otp_enc.recover_linear_output(up_masked_cpu, RW_up)
        self.tracker.record_recovery(time.perf_counter() - t0)
        
        # TEE: SiLU + multiply
        t0 = time.perf_counter()
        gate = F.silu(gate)
        intermediate = gate * up
        elapsed = time.perf_counter() - t0
        self.tracker.record_cpu_compute(elapsed, "cpu_silu")
        
        # TEE: 掩码中间结果
        t0 = time.perf_counter()
        intermediate_masked, R_inter = self.otp_enc.mask_linear_input(intermediate)
        self.tracker.record_masking(time.perf_counter() - t0)
        
        # GPU: Down
        intermediate_gpu = self._to_gpu(intermediate_masked)
        
        t0 = time.perf_counter()
        output_masked = mlp_layer.down_proj(intermediate_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_linear")
        
        # TEE: 恢复输出
        output_masked_cpu = self._to_cpu(output_masked)
        
        t0 = time.perf_counter()
        RW_down = self.otp_enc.compute_RW(R_inter, (intermediate_size, self.hidden_size))
        output = self.otp_enc.recover_linear_output(output_masked_cpu, RW_down)
        self.tracker.record_recovery(time.perf_counter() - t0)
        
        return output
    
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
    print(f"{'TEE+GPU Hybrid Inference Benchmark (OTP Encryption)':^80}")
    print(f"{'='*80}")
    print(f"  Token length: {prefill_length}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Encryption: Linear(Additive Secret Sharing), Matmul(Embedded Additive)")
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
        "encryption_scheme": "OTP (Additive Secret Sharing)",
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
    print("Initializing TEE+GPU Hybrid Model (OTP Encryption Scheme)...")
    model = TEELlamaModel(hf_model, gpu_device, CPU_DEVICE)
    
    # 运行测试
    results = run_benchmark(model, tokenizer, PREFILL_TOKEN_LENGTH)
    
    # 保存结果
    output_file = OUTPUT_FILE.replace(".json", "_otp.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
