"""
TEE+GPU 混合推理 - Intel TDX 版本
直接使用 GPU Passthrough，CPU 计算在 TEE 中执行
无 warmup，直接计时
"""
import os
import json
import time
from typing import Dict
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
    """性能追踪器 - 三部分计时"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.timing = {
            "transfer_to_gpu": 0.0,      # CPU -> GPU 传输
            "transfer_to_cpu": 0.0,      # GPU -> CPU 传输
            "gpu_compute": 0.0,          # GPU 计算
            "cpu_compute": 0.0,          # CPU 计算 (TEE)
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
    
    def get_summary(self) -> Dict:
        """获取统计摘要"""
        total_transfer = self.timing["transfer_to_gpu"] + self.timing["transfer_to_cpu"]
        total_compute = self.timing["gpu_compute"] + self.timing["cpu_compute"]
        
        return {
            "timing": {
                "transfer_to_gpu_ms": self.timing["transfer_to_gpu"] * 1000,
                "transfer_to_cpu_ms": self.timing["transfer_to_cpu"] * 1000,
                "total_transfer_ms": total_transfer * 1000,
                "gpu_compute_ms": self.timing["gpu_compute"] * 1000,
                "cpu_compute_ms": self.timing["cpu_compute"] * 1000,
                "total_compute_ms": total_compute * 1000,
                "total_ms": self.timing["total"] * 1000,
            },
            "timing_percentage": {
                "transfer_pct": (total_transfer / self.timing["total"] * 100) if self.timing["total"] > 0 else 0,
                "gpu_compute_pct": (self.timing["gpu_compute"] / self.timing["total"] * 100) if self.timing["total"] > 0 else 0,
                "cpu_compute_pct": (self.timing["cpu_compute"] / self.timing["total"] * 100) if self.timing["total"] > 0 else 0,
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
        print(f"{'Performance Summary':^80}")
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
        
        print(f"{'='*80}\n")


class TEELlamaModel(nn.Module):
    """TEE+GPU 混合 LLaMA 模型"""
    
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
        
        print(f"✓ TEE+GPU Hybrid Model initialized")
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
        """Attention 层"""
        batch_size, seq_len, _ = hidden_states.shape
        attn_layer = self.gpu_layers_attn[layer_idx]
        
        # GPU: QKV projections
        hs_gpu = self._to_gpu(hidden_states)
        
        t0 = time.perf_counter()
        q_proj = attn_layer.q_proj(hs_gpu)
        k_proj = attn_layer.k_proj(hs_gpu)
        v_proj = attn_layer.v_proj(hs_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_linear")
        
        # CPU: Reshape & Rotary
        query_states = self._to_cpu(q_proj).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self._to_cpu(k_proj).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self._to_cpu(v_proj).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        t0 = time.perf_counter()
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        elapsed = time.perf_counter() - t0
        self.tracker.record_cpu_compute(elapsed, "cpu_rotary")
        
        # GPU: Q @ K^T
        query_gpu = self._to_gpu(query_states)
        key_gpu = self._to_gpu(key_states)
        
        t0 = time.perf_counter()
        attn_weights = torch.matmul(query_gpu, key_gpu.transpose(2, 3))
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_matmul")
        
        # CPU: Softmax
        attn_weights = self._to_cpu(attn_weights) * self.scaling
        
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
        
        # CPU: Reshape
        attn_output = self._to_cpu(attn_output).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # GPU: O projection
        attn_output_gpu = self._to_gpu(attn_output)
        
        t0 = time.perf_counter()
        attn_output = attn_layer.o_proj(attn_output_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_linear")
        
        return self._to_cpu(attn_output)
    
    def mlp(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP 层"""
        mlp_layer = self.gpu_layers_mlp[layer_idx]
        
        # GPU: Gate + Up
        hs_gpu = self._to_gpu(hidden_states)
        
        t0 = time.perf_counter()
        gate = mlp_layer.gate_proj(hs_gpu)
        up = mlp_layer.up_proj(hs_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_linear")
        
        # CPU: SiLU + multiply
        gate = self._to_cpu(gate)
        up = self._to_cpu(up)
        
        t0 = time.perf_counter()
        gate = F.silu(gate)
        intermediate = gate * up
        elapsed = time.perf_counter() - t0
        self.tracker.record_cpu_compute(elapsed, "cpu_silu")
        
        # GPU: Down
        intermediate_gpu = self._to_gpu(intermediate)
        
        t0 = time.perf_counter()
        output = mlp_layer.down_proj(intermediate_gpu)
        elapsed = time.perf_counter() - t0
        self.tracker.record_gpu_compute(elapsed, "gpu_linear")
        
        return self._to_cpu(output)
    
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
        
        # CPU: Final norm
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
    print(f"{'TEE+GPU Hybrid Inference Benchmark':^80}")
    print(f"{'='*80}")
    print(f"  Token length: {prefill_length}")
    print(f"  Model: {MODEL_PATH}")
    print(f"{'='*80}\n")
    
    # 直接运行 benchmark (无 warmup)
    print("Running benchmark (no warmup)...")
    model.tracker.reset()
    
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
    print("Initializing TEE+GPU Hybrid Model...")
    model = TEELlamaModel(hf_model, gpu_device, CPU_DEVICE)
    
    # 运行测试
    results = run_benchmark(model, tokenizer, PREFILL_TOKEN_LENGTH)
    
    # 保存结果
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
