"""
Llama 模型实现 - CPU 版本
参考 tee_gpu 架构，但全部在 CPU 上本地运行，无需 server-client 通信
"""
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置
PREFILL_TOKEN_LENGTH = 8
DEFAULT_MODEL_PATH = "/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b"
DEFAULT_DTYPE = "float32"

TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class LlamaRMSNorm(nn.Module):
    """LlamaRMSNorm is equivalent to T5LayerNorm"""
    
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


class LlamaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    
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
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for grouped-query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class CPUCompute:
    """CPU 计算服务 - 模拟 GPU 服务端的接口，但在本地 CPU 执行"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 提取模块
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.lm_head = model.lm_head
        
        print(f"✓ CPU compute initialized on device: {device}")
    
    @torch.no_grad()
    def embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embedding 层"""
        return self.embed_tokens(input_ids.to(self.device))
    
    @torch.no_grad()
    def batch_linear(self, layer_idx: int, module_names: List[str], hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """批量 Linear 操作"""
        layer = self.layers[layer_idx]
        outputs = []
        
        for module_name in module_names:
            if module_name == "q_proj":
                output = layer.self_attn.q_proj(hidden_states)
            elif module_name == "k_proj":
                output = layer.self_attn.k_proj(hidden_states)
            elif module_name == "v_proj":
                output = layer.self_attn.v_proj(hidden_states)
            elif module_name == "o_proj":
                output = layer.self_attn.o_proj(hidden_states)
            elif module_name == "gate_proj":
                output = layer.mlp.gate_proj(hidden_states)
            elif module_name == "up_proj":
                output = layer.mlp.up_proj(hidden_states)
            elif module_name == "down_proj":
                output = layer.mlp.down_proj(hidden_states)
            else:
                raise ValueError(f"Unknown module: {module_name}")
            outputs.append(output)
        
        return outputs
    
    @torch.no_grad()
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """矩阵乘法"""
        return torch.matmul(a, b)
    
    @torch.no_grad()
    def lm_head_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """LM Head"""
        return self.lm_head(hidden_states)
    
    def get_init_data(self) -> Dict:
        """获取初始化数据"""
        rotary_emb = self.model.model.rotary_emb
        
        # RMSNorm 权重
        norm_weights = {}
        for i, layer in enumerate(self.layers):
            weight = layer.input_layernorm.weight.detach().cpu().numpy()
            norm_weights[f"layer_{i}_input_layernorm"] = {
                "weight": weight.tobytes(),
                "shape": list(weight.shape),
                "dtype": str(weight.dtype),
                "eps": layer.input_layernorm.variance_epsilon,
            }
            weight = layer.post_attention_layernorm.weight.detach().cpu().numpy()
            norm_weights[f"layer_{i}_post_attention_layernorm"] = {
                "weight": weight.tobytes(),
                "shape": list(weight.shape),
                "dtype": str(weight.dtype),
                "eps": layer.post_attention_layernorm.variance_epsilon,
            }
        
        weight = self.model.model.norm.weight.detach().cpu().numpy()
        norm_weights["final_norm"] = {
            "weight": weight.tobytes(),
            "shape": list(weight.shape),
            "dtype": str(weight.dtype),
            "eps": self.model.model.norm.variance_epsilon,
        }
        
        # RotaryEmbedding 参数
        inv_freq = rotary_emb.inv_freq.cpu().numpy()
        
        # 确保 attention_scaling 是可序列化的
        attention_scaling = rotary_emb.attention_scaling
        if isinstance(attention_scaling, (np.ndarray, torch.Tensor)):
            attention_scaling = float(attention_scaling)
        elif attention_scaling is None:
            attention_scaling = 1.0
        
        return {
            "config": {
                "num_layers": self.model.config.num_hidden_layers,
                "hidden_size": self.model.config.hidden_size,
                "num_heads": self.model.config.num_attention_heads,
                "num_kv_heads": self.model.config.num_key_value_heads,
                "head_dim": getattr(self.model.config, "head_dim", self.model.config.hidden_size // self.model.config.num_attention_heads),
            },
            "rotary_emb_params": {
                "inv_freq": inv_freq.tobytes(),
                "inv_freq_shape": list(inv_freq.shape),
                "inv_freq_dtype": str(inv_freq.dtype),
                "attention_scaling": attention_scaling,
            },
            "norm_weights": norm_weights,
        }


class LlamaModel:
    """Llama 模型 - 直接调用本地 CPU 计算"""
    
    def __init__(self, cpu_compute: CPUCompute, config: Dict, rotary_params: Dict, norm_weights: Dict):
        self.cpu = cpu_compute
        self.config = config
        
        # 配置
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = config["head_dim"]
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        
        # RotaryEmbedding
        inv_freq_dtype = np.dtype(rotary_params.get("inv_freq_dtype", "float32"))
        inv_freq = np.frombuffer(rotary_params["inv_freq"], dtype=inv_freq_dtype).reshape(rotary_params["inv_freq_shape"])
        inv_freq = torch.from_numpy(inv_freq.copy()).float()
        self.rotary_emb = LlamaRotaryEmbedding(inv_freq, rotary_params["attention_scaling"])
        
        # RMSNorm 层
        self.input_layernorms = []
        self.post_attention_layernorms = []
        
        for i in range(self.num_layers):
            input_norm = norm_weights[f"layer_{i}_input_layernorm"]
            input_dtype = np.dtype(input_norm.get("dtype", "float32"))
            weight = np.frombuffer(input_norm["weight"], dtype=input_dtype).reshape(input_norm["shape"])
            weight = torch.from_numpy(weight.copy()).float()
            self.input_layernorms.append(LlamaRMSNorm(weight, input_norm["eps"]))
            
            post_norm = norm_weights[f"layer_{i}_post_attention_layernorm"]
            post_dtype = np.dtype(post_norm.get("dtype", "float32"))
            weight = np.frombuffer(post_norm["weight"], dtype=post_dtype).reshape(post_norm["shape"])
            weight = torch.from_numpy(weight.copy()).float()
            self.post_attention_layernorms.append(LlamaRMSNorm(weight, post_norm["eps"]))
        
        final_norm = norm_weights["final_norm"]
        final_dtype = np.dtype(final_norm.get("dtype", "float32"))
        weight = np.frombuffer(final_norm["weight"], dtype=final_dtype).reshape(final_norm["shape"])
        weight = torch.from_numpy(weight.copy()).float()
        self.final_norm = LlamaRMSNorm(weight, final_norm["eps"])
        
        print(f"✓ Model initialized: {self.num_layers} layers")
    
    def attention(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Self-attention layer"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projections
        qkv = self.cpu.batch_linear(layer_idx, ["q_proj", "k_proj", "v_proj"], hidden_states)
        query_states, key_states, value_states = qkv
        
        # Reshape
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Rotary embeddings
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Attention weights
        attn_weights = self.cpu.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Attention output
        attn_output = self.cpu.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        attn_output = self.cpu.batch_linear(layer_idx, ["o_proj"], attn_output)[0]
        
        return attn_output
    
    def mlp(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP layer"""
        # Gate + Up projections
        gate_up = self.cpu.batch_linear(layer_idx, ["gate_proj", "up_proj"], hidden_states)
        gate, up = gate_up
        
        # SiLU activation and multiply
        gate = F.silu(gate)
        intermediate = gate * up
        
        # Down projection
        output = self.cpu.batch_linear(layer_idx, ["down_proj"], intermediate)[0]
        
        return output
    
    def decoder_layer(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Decoder layer"""
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorms[layer_idx](hidden_states)
        hidden_states = self.attention(layer_idx, hidden_states, position_ids)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorms[layer_idx](hidden_states)
        hidden_states = self.mlp(layer_idx, hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Embedding
        hidden_states = self.cpu.embedding(input_ids)
        
        # Position IDs
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Decoder layers
        for layer_idx in range(self.num_layers):
            hidden_states = self.decoder_layer(layer_idx, hidden_states, position_ids)
        
        # Final norm
        hidden_states = self.final_norm(hidden_states)
        
        # LM Head
        logits = self.cpu.lm_head_forward(hidden_states)
        
        return logits


def load_model(model_path: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """Load model from pretrained weights"""
    is_local = os.path.exists(model_path)
    
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}, Dtype: {dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=is_local,
        trust_remote_code=True
    )
    
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def run_benchmark(model: LlamaModel, tokenizer: AutoTokenizer, prefill_length: int) -> None:
    """Run benchmark"""
    # 创建输入
    input_ids = torch.full((1, prefill_length), tokenizer.pad_token_id, dtype=torch.long)
    
    print(f"\nWarming up...")
    _ = model.forward(input_ids)
    
    print(f"Running benchmark with {prefill_length} tokens...")
    
    # Benchmark
    start = time.perf_counter()
    _ = model.forward(input_ids)
    total_time = time.perf_counter() - start
    
    print(f"\n✓ Prefill completed in {total_time:.4f} seconds")
    print(f"  Throughput: {prefill_length / total_time:.2f} tokens/sec")


def main() -> None:
    """主函数"""
    # 配置
    model_path = os.environ.get("LLAMA_MODEL_PATH", DEFAULT_MODEL_PATH)
    device = torch.device("cpu")
    dtype = TORCH_DTYPE_MAP.get(os.environ.get("LLAMA_DTYPE", DEFAULT_DTYPE), torch.float32)
    
    # 加载模型
    print("="*60)
    print("Llama Model (CPU Version)")
    print("="*60)
    
    base_model = load_model(model_path, device, dtype)
    
    # 加载 tokenizer
    is_local = os.path.exists(model_path)
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=is_local,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建 CPU 计算服务
    cpu_compute = CPUCompute(base_model, device)
    
    # 获取初始化数据
    print("Initializing model...")
    init_data = cpu_compute.get_init_data()
    
    # 创建模型
    model = LlamaModel(
        cpu_compute,
        init_data["config"],
        init_data["rotary_emb_params"],
        init_data["norm_weights"]
    )
    
    # 运行 benchmark
    run_benchmark(model, tokenizer, PREFILL_TOKEN_LENGTH)


if __name__ == "__main__":
    main()
