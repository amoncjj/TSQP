"""
TEE 客户端 - Prefill 阶段性能测试
TEE 端执行: Softmax, RMSNorm, RotaryEmbedding, 激活函数 (SiLU)
GPU 端执行: Linear, Embedding, Matmul 等其他所有计算
"""
import os
import time
from typing import Dict, List, Optional

import zmq
import msgpack
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# 配置
PREFILL_TOKEN_LENGTH = 1024
DEFAULT_MODEL_PATH = "/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b"
DEFAULT_GPU_ENDPOINT = "localhost:50051"

# 数据类型映射
TORCH_DTYPE_TO_STR: Dict[torch.dtype, str] = {
    torch.float32: "torch.float32",
    torch.float16: "torch.float16",
    torch.bfloat16: "torch.bfloat16",
    torch.int64: "torch.int64",
}

STR_TO_NUMPY: Dict[str, np.dtype] = {
    "torch.float32": np.float32,
    "torch.float16": np.float16,
    "torch.bfloat16": np.float32,
    "torch.int64": np.int64,
}

RESPONSE_DTYPE = "torch.float32"


class TEERMSNorm(nn.Module):
    """TEE 端的 RMSNorm 实现"""
    
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
    """TEE 端的 RotaryEmbedding 实现"""
    
    def __init__(self, inv_freq: torch.Tensor, attention_scaling: float):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = attention_scaling
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # 计算频率
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
    cos = cos.unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复 key/value 以匹配 query 的头数"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GPUClient:
    """GPU 服务器客户端"""
    
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{endpoint}")
        print(f"✓ Connected to GPU server at {endpoint}")
        
        # 通信统计
        self.comm_time = 0.0
        self.serial_time = 0.0
        self.rpc_count = 0
    
    def _send_request(self, method: str, request: Dict) -> Dict:
        """发送请求并接收响应"""
        # 序列化
        t_serial = time.perf_counter()
        message = {"method": method, "request": request}
        message_bytes = msgpack.packb(message, use_bin_type=True)
        self.serial_time += time.perf_counter() - t_serial
        
        # 通信
        t_comm = time.perf_counter()
        self.socket.send(message_bytes)
        response_bytes = self.socket.recv()
        self.comm_time += time.perf_counter() - t_comm
        self.rpc_count += 1
        
        # 反序列化
        t_serial = time.perf_counter()
        response = msgpack.unpackb(response_bytes, raw=False)
        self.serial_time += time.perf_counter() - t_serial
        
        if response["status"] == "error":
            print(f"Server error: {response['error']}")
            if "traceback" in response:
                print(response["traceback"])
            raise RuntimeError(f"Server error: {response['error']}")
        
        return response["response"]
    
    def init(self) -> Dict:
        """初始化 - 获取模型配置和参数"""
        return self._send_request("Init", {})
    
    def embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embedding 层"""
        tensor_cpu = input_ids.detach().to(torch.int64).cpu().contiguous()
        request = {
            "input_ids": tensor_cpu.numpy().tobytes(),
            "input_shape": list(tensor_cpu.shape),
            "dtype": TORCH_DTYPE_TO_STR[torch.int64],
        }
        
        response = self._send_request("Embedding", request)
        output_array = np.frombuffer(response["output"], dtype=STR_TO_NUMPY[RESPONSE_DTYPE])
        return torch.from_numpy(output_array).view(*response["shape"])
    
    def linear(self, layer_idx: int, module_name: str, hidden_states: torch.Tensor) -> torch.Tensor:
        """Linear 层"""
        tensor_cpu = hidden_states.detach().to(torch.float32).cpu().contiguous()
        request = {
            "layer_idx": layer_idx,
            "module_name": module_name,
            "hidden_states": tensor_cpu.numpy().tobytes(),
            "shape": list(tensor_cpu.shape),
            "dtype": TORCH_DTYPE_TO_STR[torch.float32],
        }
        
        response = self._send_request("Linear", request)
        output_array = np.frombuffer(response["output"], dtype=STR_TO_NUMPY[RESPONSE_DTYPE])
        return torch.from_numpy(output_array).view(*response["shape"])
    
    def batch_linear(self, layer_idx: int, module_names: List[str], hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """批量 Linear 层 - 一次 RPC 调用多个 Linear"""
        tensor_cpu = hidden_states.detach().to(torch.float32).cpu().contiguous()
        request = {
            "layer_idx": layer_idx,
            "module_names": module_names,
            "hidden_states": tensor_cpu.numpy().tobytes(),
            "shape": list(tensor_cpu.shape),
            "dtype": TORCH_DTYPE_TO_STR[torch.float32],
        }
        
        response = self._send_request("BatchLinear", request)
        outputs = []
        for output_data in response["outputs"]:
            output_array = np.frombuffer(output_data["output"], dtype=STR_TO_NUMPY[RESPONSE_DTYPE])
            outputs.append(torch.from_numpy(output_array).view(*output_data["shape"]))
        return outputs
    
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """矩阵乘法"""
        a_cpu = a.detach().to(torch.float32).cpu().contiguous()
        b_cpu = b.detach().to(torch.float32).cpu().contiguous()
        
        request = {
            "a_buffer": a_cpu.numpy().tobytes(),
            "a_shape": list(a_cpu.shape),
            "b_buffer": b_cpu.numpy().tobytes(),
            "b_shape": list(b_cpu.shape),
            "dtype": TORCH_DTYPE_TO_STR[torch.float32],
        }
        
        response = self._send_request("Matmul", request)
        output_array = np.frombuffer(response["output"], dtype=STR_TO_NUMPY[RESPONSE_DTYPE])
        return torch.from_numpy(output_array).view(*response["shape"])
    
    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """LM Head"""
        tensor_cpu = hidden_states.detach().to(torch.float32).cpu().contiguous()
        request = {
            "hidden_states": tensor_cpu.numpy().tobytes(),
            "shape": list(tensor_cpu.shape),
            "dtype": TORCH_DTYPE_TO_STR[torch.float32],
        }
        
        response = self._send_request("LMHead", request)
        output_array = np.frombuffer(response["output"], dtype=STR_TO_NUMPY[RESPONSE_DTYPE])
        return torch.from_numpy(output_array).view(*response["shape"])
    
    def close(self) -> None:
        """关闭连接"""
        self.socket.close()
        self.context.term()


class TEELlamaModel:
    """TEE 端的 LLaMA 模型实现"""
    
    def __init__(self, gpu_client: GPUClient, config: Dict, rotary_params: Dict, norm_weights: Dict):
        self.gpu = gpu_client
        self.config = config
        
        # 模型配置
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = config["head_dim"]
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        
        # 计时统计
        self.timing_stats = {
            "gpu_embedding": 0.0,
            "gpu_linear": 0.0,
            "gpu_matmul": 0.0,
            "gpu_lm_head": 0.0,
            "tee_rmsnorm": 0.0,
            "tee_rotary": 0.0,
            "tee_softmax": 0.0,
            "tee_silu": 0.0,
            "communication": 0.0,  # 通信时间
            "serialization": 0.0,  # 序列化时间
        }
        self.operation_counts = {
            "gpu_embedding": 0,
            "gpu_linear": 0,
            "gpu_matmul": 0,
            "gpu_lm_head": 0,
            "tee_rmsnorm": 0,
            "tee_rotary": 0,
            "tee_softmax": 0,
            "tee_silu": 0,
            "rpc_calls": 0,  # RPC 调用次数
        }
        
        # 初始化 RotaryEmbedding
        inv_freq = torch.frombuffer(
            rotary_params["inv_freq"],
            dtype=torch.float32
        ).view(*rotary_params["inv_freq_shape"])
        self.rotary_emb = TEERotaryEmbedding(inv_freq, rotary_params["attention_scaling"])
        
        # 初始化所有 RMSNorm 层
        self.input_layernorms = []
        self.post_attention_layernorms = []
        
        for i in range(self.num_layers):
            # Input LayerNorm
            input_norm_data = norm_weights[f"layer_{i}_input_layernorm"]
            weight = torch.frombuffer(input_norm_data["weight"], dtype=torch.float32).view(*input_norm_data["shape"])
            self.input_layernorms.append(TEERMSNorm(weight, input_norm_data["eps"]))
            
            # Post Attention LayerNorm
            post_norm_data = norm_weights[f"layer_{i}_post_attention_layernorm"]
            weight = torch.frombuffer(post_norm_data["weight"], dtype=torch.float32).view(*post_norm_data["shape"])
            self.post_attention_layernorms.append(TEERMSNorm(weight, post_norm_data["eps"]))
        
        # Final Norm
        final_norm_data = norm_weights["final_norm"]
        weight = torch.frombuffer(final_norm_data["weight"], dtype=torch.float32).view(*final_norm_data["shape"])
        self.final_norm = TEERMSNorm(weight, final_norm_data["eps"])
        
        print(f"✓ TEE model initialized: {self.num_layers} layers")
    
    def attention(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Attention 层 - TEE 端执行 Softmax 和 RoPE，GPU 端执行 Linear 和 Matmul"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # GPU: QKV projections (批量调用，减少 RPC 次数)
        t0 = time.perf_counter()
        qkv_outputs = self.gpu.batch_linear(layer_idx, ["q_proj", "k_proj", "v_proj"], hidden_states)
        query_states, key_states, value_states = qkv_outputs
        self.timing_stats["gpu_linear"] += time.perf_counter() - t0
        self.operation_counts["gpu_linear"] += 3
        
        # Reshape
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # TEE: Apply rotary embeddings
        t0 = time.perf_counter()
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        self.timing_stats["tee_rotary"] += time.perf_counter() - t0
        self.operation_counts["tee_rotary"] += 1
        
        # TEE: Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # GPU: Q @ K^T
        t0 = time.perf_counter()
        attn_weights = self.gpu.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights * self.scaling
        self.timing_stats["gpu_matmul"] += time.perf_counter() - t0
        self.operation_counts["gpu_matmul"] += 1
        
        # TEE: Softmax
        t0 = time.perf_counter()
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self.timing_stats["tee_softmax"] += time.perf_counter() - t0
        self.operation_counts["tee_softmax"] += 1
        
        # GPU: Attention @ V
        t0 = time.perf_counter()
        attn_output = self.gpu.matmul(attn_weights, value_states)
        self.timing_stats["gpu_matmul"] += time.perf_counter() - t0
        self.operation_counts["gpu_matmul"] += 1
        
        # Reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # GPU: Output projection
        t0 = time.perf_counter()
        attn_output = self.gpu.linear(layer_idx, "o_proj", attn_output)
        self.timing_stats["gpu_linear"] += time.perf_counter() - t0
        self.operation_counts["gpu_linear"] += 1
        
        return attn_output
    
    def mlp(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP 层 - TEE 端执行 SiLU，GPU 端执行 Linear"""
        # GPU: gate_proj and up_proj (批量调用)
        t0 = time.perf_counter()
        gate_up_outputs = self.gpu.batch_linear(layer_idx, ["gate_proj", "up_proj"], hidden_states)
        gate, up = gate_up_outputs
        self.timing_stats["gpu_linear"] += time.perf_counter() - t0
        self.operation_counts["gpu_linear"] += 2
        
        # TEE: SiLU activation
        t0 = time.perf_counter()
        gate = F.silu(gate)
        self.timing_stats["tee_silu"] += time.perf_counter() - t0
        self.operation_counts["tee_silu"] += 1
        
        # TEE: Element-wise multiplication
        intermediate = gate * up
        
        # GPU: down_proj
        t0 = time.perf_counter()
        output = self.gpu.linear(layer_idx, "down_proj", intermediate)
        self.timing_stats["gpu_linear"] += time.perf_counter() - t0
        self.operation_counts["gpu_linear"] += 1
        
        return output
    
    def decoder_layer(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Decoder 层"""
        # Self Attention
        residual = hidden_states
        t0 = time.perf_counter()
        hidden_states = self.input_layernorms[layer_idx](hidden_states)  # TEE: RMSNorm
        self.timing_stats["tee_rmsnorm"] += time.perf_counter() - t0
        self.operation_counts["tee_rmsnorm"] += 1
        
        hidden_states = self.attention(layer_idx, hidden_states, position_ids)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        t0 = time.perf_counter()
        hidden_states = self.post_attention_layernorms[layer_idx](hidden_states)  # TEE: RMSNorm
        self.timing_stats["tee_rmsnorm"] += time.perf_counter() - t0
        self.operation_counts["tee_rmsnorm"] += 1
        
        hidden_states = self.mlp(layer_idx, hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播 - Prefill 阶段"""
        batch_size, seq_len = input_ids.shape
        
        # GPU: Embedding
        t0 = time.perf_counter()
        hidden_states = self.gpu.embedding(input_ids)
        self.timing_stats["gpu_embedding"] += time.perf_counter() - t0
        self.operation_counts["gpu_embedding"] += 1
        
        # Position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        
        # Decoder layers
        for layer_idx in range(self.num_layers):
            hidden_states = self.decoder_layer(layer_idx, hidden_states, position_ids)
        
        # TEE: Final norm
        t0 = time.perf_counter()
        hidden_states = self.final_norm(hidden_states)
        self.timing_stats["tee_rmsnorm"] += time.perf_counter() - t0
        self.operation_counts["tee_rmsnorm"] += 1
        
        # GPU: LM head (只计算最后一个 token)
        t0 = time.perf_counter()
        logits = self.gpu.lm_head(hidden_states[:, -1:, :])
        self.timing_stats["gpu_lm_head"] += time.perf_counter() - t0
        self.operation_counts["gpu_lm_head"] += 1
        
        return logits
    
    def print_timing_stats(self):
        """打印计时统计信息"""
        # 从 GPU client 获取通信统计
        self.timing_stats["communication"] = self.gpu.comm_time
        self.timing_stats["serialization"] = self.gpu.serial_time
        self.operation_counts["rpc_calls"] = self.gpu.rpc_count
        
        print(f"\n{'='*70}")
        print(f"{'Operation Timing Statistics':^70}")
        print(f"{'='*70}")
        print(f"{'Operation':<25} {'Count':>10} {'Total (s)':>12} {'Avg (ms)':>12} {'%':>8}")
        print(f"{'-'*70}")
        
        total_time = sum(self.timing_stats.values())
        
        # 通信开销
        print(f"\n{'Communication Overhead':^70}")
        print(f"{'-'*70}")
        comm_time = self.timing_stats["communication"]
        serial_time = self.timing_stats["serialization"]
        rpc_count = self.operation_counts["rpc_calls"]
        
        print(f"{'RPC Calls':<25} {rpc_count:>10} {comm_time:>12.4f} {comm_time/rpc_count*1000:>12.4f} {comm_time/total_time*100:>7.2f}%")
        print(f"{'Serialization':<25} {'':<10} {serial_time:>12.4f} {serial_time/rpc_count*1000:>12.4f} {serial_time/total_time*100:>7.2f}%")
        comm_overhead = comm_time + serial_time
        print(f"{'Total Comm Overhead':<25} {'':<10} {comm_overhead:>12.4f} {'':<12} {comm_overhead/total_time*100:>7.2f}%")
        
        # GPU 操作
        print(f"\n{'GPU Operations':^70}")
        print(f"{'-'*70}")
        for op in ["gpu_embedding", "gpu_linear", "gpu_matmul", "gpu_lm_head"]:
            count = self.operation_counts[op]
            total = self.timing_stats[op]
            avg = (total / count * 1000) if count > 0 else 0
            pct = (total / total_time * 100) if total_time > 0 else 0
            op_name = op.replace("gpu_", "").upper()
            print(f"{op_name:<25} {count:>10} {total:>12.4f} {avg:>12.4f} {pct:>7.2f}%")
        
        # TEE 操作
        print(f"\n{'TEE Operations':^70}")
        print(f"{'-'*70}")
        for op in ["tee_rmsnorm", "tee_rotary", "tee_softmax", "tee_silu"]:
            count = self.operation_counts[op]
            total = self.timing_stats[op]
            avg = (total / count * 1000) if count > 0 else 0
            pct = (total / total_time * 100) if total_time > 0 else 0
            op_name = op.replace("tee_", "").upper()
            print(f"{op_name:<25} {count:>10} {total:>12.4f} {avg:>12.4f} {pct:>7.2f}%")
        
        # 总计
        print(f"{'-'*70}")
        gpu_total = sum(self.timing_stats[k] for k in self.timing_stats if k.startswith("gpu_"))
        tee_total = sum(self.timing_stats[k] for k in self.timing_stats if k.startswith("tee_"))
        print(f"{'GPU Compute':<25} {'':<10} {gpu_total:>12.4f} {'':<12} {gpu_total/total_time*100:>7.2f}%")
        print(f"{'TEE Compute':<25} {'':<10} {tee_total:>12.4f} {'':<12} {tee_total/total_time*100:>7.2f}%")
        print(f"{'Communication':<25} {'':<10} {comm_overhead:>12.4f} {'':<12} {comm_overhead/total_time*100:>7.2f}%")
        print(f"{'TOTAL':<25} {'':<10} {total_time:>12.4f} {'':<12} {'100.00':>7}%")
        print(f"{'='*70}")
        
        # 优化建议
        if comm_overhead / total_time > 0.5:
            print(f"\n⚠️  Communication overhead is {comm_overhead/total_time*100:.1f}% of total time!")
            print(f"   Suggestions:")
            print(f"   - Reduce RPC calls: {rpc_count} calls, avg {comm_time/rpc_count*1000:.2f}ms per call")
            print(f"   - Consider batching more operations")
            print(f"   - Use faster serialization or compression")
        print()


def run_prefill_benchmark(model: TEELlamaModel, tokenizer, prefill_length: int) -> float:
    """运行 prefill 阶段性能测试"""
    # 创建固定长度的输入
    input_ids = torch.full((1, prefill_length), tokenizer.pad_token_id, dtype=torch.long)
    
    print(f"\n{'='*60}")
    print(f"Running Prefill Benchmark")
    print(f"{'='*60}")
    print(f"Token length: {prefill_length}")
    print(f"TEE operations: Softmax, RMSNorm, RotaryEmbedding, SiLU")
    print(f"GPU operations: Linear, Embedding, Matmul")
    
    # 执行 prefill 并计时
    start_time = time.perf_counter()
    
    logits = model.forward(input_ids)
    
    elapsed_time = time.perf_counter() - start_time
    
    print(f"{'='*60}")
    print(f"Prefill time: {elapsed_time:.4f} seconds")
    print(f"Throughput: {prefill_length / elapsed_time:.2f} tokens/sec")
    print(f"Logits shape: {logits.shape}")
    print(f"{'='*60}\n")
    
    # 打印详细的操作计时统计
    model.print_timing_stats()
    
    return elapsed_time


def main() -> None:
    """主函数"""
    # 读取配置
    model_path = os.environ.get("LLAMA_MODEL_PATH", DEFAULT_MODEL_PATH)
    endpoint = os.environ.get("LLAMA_GPU_ENDPOINT", DEFAULT_GPU_ENDPOINT)
    is_local = os.path.exists(model_path)
    
    # 加载 tokenizer
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=is_local,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 连接 GPU 服务器
    gpu_client = GPUClient(endpoint)
    
    try:
        # 初始化模型
        print("Initializing model from GPU server...")
        init_data = gpu_client.init()
        
        model = TEELlamaModel(
            gpu_client,
            init_data["config"],
            init_data["rotary_emb_params"],
            init_data["norm_weights"]
        )
        
        # 运行 prefill 测试
        run_prefill_benchmark(model, tokenizer, PREFILL_TOKEN_LENGTH)
        
    finally:
        gpu_client.close()
        print("✓ Connection closed")


if __name__ == "__main__":
    main()
