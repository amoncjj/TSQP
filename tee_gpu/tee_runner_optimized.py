"""
高性能 TEE 客户端 - 优化版
关键优化：
1. 使用 IPC 而不是 TCP
2. 最小化数据拷贝
3. 批量操作
4. 详细的性能分析
"""
import os
import time
from typing import Dict, List

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
DEFAULT_IPC_PATH = "ipc:///tmp/tsqp_gpu_server.ipc"


class TEERMSNorm(nn.Module):
    """TEE 端的 RMSNorm"""
    
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
    """TEE 端的 RotaryEmbedding"""
    
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


class GPUClient:
    """高性能 GPU 客户端"""
    
    def __init__(self, ipc_path: str) -> None:
        self.ipc_path = ipc_path
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        
        # 优化 ZeroMQ 性能
        self.socket.setsockopt(zmq.SNDHWM, 1000)
        self.socket.setsockopt(zmq.RCVHWM, 1000)
        self.socket.setsockopt(zmq.LINGER, 0)
        
        self.socket.connect(ipc_path)
        print(f"✓ Connected to GPU server at {ipc_path}")
        
        # 性能统计
        self.stats = {
            "rpc_count": 0,
            "rpc_time": 0.0,
            "serialize_time": 0.0,
            "deserialize_time": 0.0,
            "total_bytes_sent": 0,
            "total_bytes_recv": 0,
        }
    
    def _send_request(self, method: str, request: Dict) -> Dict:
        """发送请求"""
        self.stats["rpc_count"] += 1
        
        # 序列化
        t0 = time.perf_counter()
        message = {"method": method, "request": request}
        message_bytes = msgpack.packb(message, use_bin_type=True)
        self.stats["serialize_time"] += time.perf_counter() - t0
        self.stats["total_bytes_sent"] += len(message_bytes)
        
        # RPC 调用
        t0 = time.perf_counter()
        self.socket.send(message_bytes)
        response_bytes = self.socket.recv()
        self.stats["rpc_time"] += time.perf_counter() - t0
        self.stats["total_bytes_recv"] += len(response_bytes)
        
        # 反序列化
        t0 = time.perf_counter()
        response = msgpack.unpackb(response_bytes, raw=False)
        self.stats["deserialize_time"] += time.perf_counter() - t0
        
        if response["status"] == "error":
            print(f"Server error: {response['error']}")
            if "traceback" in response:
                print(response["traceback"])
            raise RuntimeError(f"Server error: {response['error']}")
        
        return response["response"]
    
    def init(self) -> Dict:
        """初始化"""
        return self._send_request("Init", {})
    
    def embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embedding"""
        # 最小化拷贝
        tensor_cpu = input_ids.cpu().contiguous() if input_ids.is_cuda else input_ids.contiguous()
        request = {
            "buffer": tensor_cpu.numpy().tobytes(),
            "shape": list(tensor_cpu.shape),
        }
        
        response = self._send_request("Embedding", request)
        
        # 零拷贝反序列化
        array = np.frombuffer(response["buffer"], dtype=np.float32).reshape(response["shape"])
        return torch.from_numpy(array.copy())  # copy 是必须的，因为 frombuffer 返回只读
    
    def batch_linear(self, layer_idx: int, module_names: List[str], hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """批量 Linear"""
        tensor_cpu = hidden_states.cpu().contiguous() if hidden_states.is_cuda else hidden_states.contiguous()
        request = {
            "layer_idx": layer_idx,
            "module_names": module_names,
            "hidden_states": {
                "buffer": tensor_cpu.numpy().tobytes(),
                "shape": list(tensor_cpu.shape),
            }
        }
        
        response = self._send_request("BatchLinear", request)
        
        outputs = []
        for output_data in response["outputs"]:
            array = np.frombuffer(output_data["buffer"], dtype=np.float32).reshape(output_data["shape"])
            outputs.append(torch.from_numpy(array.copy()))
        return outputs
    
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """矩阵乘法"""
        a_cpu = a.cpu().contiguous() if a.is_cuda else a.contiguous()
        b_cpu = b.cpu().contiguous() if b.is_cuda else b.contiguous()
        
        request = {
            "a": {
                "buffer": a_cpu.numpy().tobytes(),
                "shape": list(a_cpu.shape),
            },
            "b": {
                "buffer": b_cpu.numpy().tobytes(),
                "shape": list(b_cpu.shape),
            }
        }
        
        response = self._send_request("Matmul", request)
        array = np.frombuffer(response["buffer"], dtype=np.float32).reshape(response["shape"])
        return torch.from_numpy(array.copy())
    
    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """LM Head"""
        tensor_cpu = hidden_states.cpu().contiguous() if hidden_states.is_cuda else hidden_states.contiguous()
        request = {
            "hidden_states": {
                "buffer": tensor_cpu.numpy().tobytes(),
                "shape": list(tensor_cpu.shape),
            }
        }
        
        response = self._send_request("LMHead", request)
        array = np.frombuffer(response["buffer"], dtype=np.float32).reshape(response["shape"])
        return torch.from_numpy(array.copy())
    
    def print_stats(self):
        """打印统计信息"""
        print(f"\n{'='*70}")
        print(f"{'GPU Client Statistics':^70}")
        print(f"{'='*70}")
        print(f"RPC Calls:        {self.stats['rpc_count']}")
        print(f"RPC Time:         {self.stats['rpc_time']:.4f}s ({self.stats['rpc_time']/self.stats['rpc_count']*1000:.2f}ms/call)")
        print(f"Serialize Time:   {self.stats['serialize_time']:.4f}s ({self.stats['serialize_time']/self.stats['rpc_count']*1000:.2f}ms/call)")
        print(f"Deserialize Time: {self.stats['deserialize_time']:.4f}s ({self.stats['deserialize_time']/self.stats['rpc_count']*1000:.2f}ms/call)")
        print(f"Data Sent:        {self.stats['total_bytes_sent']/1024/1024:.2f} MB")
        print(f"Data Received:    {self.stats['total_bytes_recv']/1024/1024:.2f} MB")
        print(f"{'='*70}\n")
    
    def close(self) -> None:
        """关闭连接"""
        self.socket.close()
        self.context.term()


class TEELlamaModel:
    """TEE 端的 LLaMA 模型"""
    
    def __init__(self, gpu_client: GPUClient, config: Dict, rotary_params: Dict, norm_weights: Dict):
        self.gpu = gpu_client
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
        inv_freq = np.frombuffer(rotary_params["inv_freq"], dtype=np.float32).reshape(rotary_params["inv_freq_shape"])
        inv_freq = torch.from_numpy(inv_freq.copy()).float()
        self.rotary_emb = TEERotaryEmbedding(inv_freq, rotary_params["attention_scaling"])
        
        # RMSNorm 层
        self.input_layernorms = []
        self.post_attention_layernorms = []
        
        for i in range(self.num_layers):
            input_norm = norm_weights[f"layer_{i}_input_layernorm"]
            weight = np.frombuffer(input_norm["weight"], dtype=np.float32).reshape(input_norm["shape"])
            weight = torch.from_numpy(weight.copy()).float()
            self.input_layernorms.append(TEERMSNorm(weight, input_norm["eps"]))
            
            post_norm = norm_weights[f"layer_{i}_post_attention_layernorm"]
            weight = np.frombuffer(post_norm["weight"], dtype=np.float32).reshape(post_norm["shape"])
            weight = torch.from_numpy(weight.copy()).float()
            self.post_attention_layernorms.append(TEERMSNorm(weight, post_norm["eps"]))
        
        final_norm = norm_weights["final_norm"]
        weight = np.frombuffer(final_norm["weight"], dtype=np.float32).reshape(final_norm["shape"])
        weight = torch.from_numpy(weight.copy()).float()
        self.final_norm = TEERMSNorm(weight, final_norm["eps"])
        
        # 性能统计
        self.timing = {
            "gpu_embedding": 0.0,
            "gpu_linear": 0.0,
            "gpu_matmul": 0.0,
            "gpu_lm_head": 0.0,
            "tee_rmsnorm": 0.0,
            "tee_rotary": 0.0,
            "tee_softmax": 0.0,
            "tee_silu": 0.0,
            "tee_other": 0.0,
        }
        self.counts = {k: 0 for k in self.timing.keys()}
        
        print(f"✓ TEE model initialized: {self.num_layers} layers")
    
    def attention(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Attention 层"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # GPU: QKV projections (批量)
        t0 = time.perf_counter()
        qkv = self.gpu.batch_linear(layer_idx, ["q_proj", "k_proj", "v_proj"], hidden_states)
        self.timing["gpu_linear"] += time.perf_counter() - t0
        self.counts["gpu_linear"] += 3
        
        query_states, key_states, value_states = qkv
        
        # TEE: Reshape
        t0 = time.perf_counter()
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        self.timing["tee_other"] += time.perf_counter() - t0
        
        # TEE: Rotary embeddings
        t0 = time.perf_counter()
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        self.timing["tee_rotary"] += time.perf_counter() - t0
        self.counts["tee_rotary"] += 1
        
        # GPU: Q @ K^T
        t0 = time.perf_counter()
        attn_weights = self.gpu.matmul(query_states, key_states.transpose(2, 3))
        self.timing["gpu_matmul"] += time.perf_counter() - t0
        self.counts["gpu_matmul"] += 1
        
        # TEE: Scale + Softmax
        t0 = time.perf_counter()
        attn_weights = attn_weights * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self.timing["tee_softmax"] += time.perf_counter() - t0
        self.counts["tee_softmax"] += 1
        
        # GPU: Attn @ V
        t0 = time.perf_counter()
        attn_output = self.gpu.matmul(attn_weights, value_states)
        self.timing["gpu_matmul"] += time.perf_counter() - t0
        self.counts["gpu_matmul"] += 1
        
        # TEE: Reshape
        t0 = time.perf_counter()
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        self.timing["tee_other"] += time.perf_counter() - t0
        
        # GPU: O projection
        t0 = time.perf_counter()
        attn_output = self.gpu.batch_linear(layer_idx, ["o_proj"], attn_output)[0]
        self.timing["gpu_linear"] += time.perf_counter() - t0
        self.counts["gpu_linear"] += 1
        
        return attn_output
    
    def mlp(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP 层"""
        # GPU: Gate + Up (批量)
        t0 = time.perf_counter()
        gate_up = self.gpu.batch_linear(layer_idx, ["gate_proj", "up_proj"], hidden_states)
        self.timing["gpu_linear"] += time.perf_counter() - t0
        self.counts["gpu_linear"] += 2
        
        gate, up = gate_up
        
        # TEE: SiLU + multiply
        t0 = time.perf_counter()
        gate = F.silu(gate)
        intermediate = gate * up
        self.timing["tee_silu"] += time.perf_counter() - t0
        self.counts["tee_silu"] += 1
        
        # GPU: Down
        t0 = time.perf_counter()
        output = self.gpu.batch_linear(layer_idx, ["down_proj"], intermediate)[0]
        self.timing["gpu_linear"] += time.perf_counter() - t0
        self.counts["gpu_linear"] += 1
        
        return output
    
    def decoder_layer(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Decoder 层"""
        # Attention
        residual = hidden_states
        t0 = time.perf_counter()
        hidden_states = self.input_layernorms[layer_idx](hidden_states)
        self.timing["tee_rmsnorm"] += time.perf_counter() - t0
        self.counts["tee_rmsnorm"] += 1
        
        hidden_states = self.attention(layer_idx, hidden_states, position_ids)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        t0 = time.perf_counter()
        hidden_states = self.post_attention_layernorms[layer_idx](hidden_states)
        self.timing["tee_rmsnorm"] += time.perf_counter() - t0
        self.counts["tee_rmsnorm"] += 1
        
        hidden_states = self.mlp(layer_idx, hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len = input_ids.shape
        
        # GPU: Embedding
        t0 = time.perf_counter()
        hidden_states = self.gpu.embedding(input_ids)
        self.timing["gpu_embedding"] += time.perf_counter() - t0
        self.counts["gpu_embedding"] += 1
        
        # Position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        
        # Decoder layers
        for layer_idx in range(self.num_layers):
            hidden_states = self.decoder_layer(layer_idx, hidden_states, position_ids)
        
        # TEE: Final norm
        t0 = time.perf_counter()
        hidden_states = self.final_norm(hidden_states)
        self.timing["tee_rmsnorm"] += time.perf_counter() - t0
        self.counts["tee_rmsnorm"] += 1
        
        # GPU: LM head
        t0 = time.perf_counter()
        logits = self.gpu.lm_head(hidden_states[:, -1:, :])
        self.timing["gpu_lm_head"] += time.perf_counter() - t0
        self.counts["gpu_lm_head"] += 1
        
        return logits
    
    def print_timing_stats(self):
        """打印性能统计"""
        total_time = sum(self.timing.values())
        
        print(f"\n{'='*70}")
        print(f"{'TEE Model Timing Statistics':^70}")
        print(f"{'='*70}")
        print(f"{'Operation':<20} {'Count':>8} {'Total(s)':>12} {'Avg(ms)':>12} {'%':>8}")
        print(f"{'-'*70}")
        
        # GPU 操作
        print(f"\n{'GPU Operations':^70}")
        print(f"{'-'*70}")
        for op in ["gpu_embedding", "gpu_linear", "gpu_matmul", "gpu_lm_head"]:
            count = self.counts[op]
            t = self.timing[op]
            avg = (t / count * 1000) if count > 0 else 0
            pct = (t / total_time * 100) if total_time > 0 else 0
            name = op.replace("gpu_", "").upper()
            print(f"{name:<20} {count:>8} {t:>12.4f} {avg:>12.4f} {pct:>7.2f}%")
        
        # TEE 操作
        print(f"\n{'TEE Operations':^70}")
        print(f"{'-'*70}")
        for op in ["tee_rmsnorm", "tee_rotary", "tee_softmax", "tee_silu", "tee_other"]:
            count = self.counts.get(op, 0)
            t = self.timing[op]
            avg = (t / count * 1000) if count > 0 else 0
            pct = (t / total_time * 100) if total_time > 0 else 0
            name = op.replace("tee_", "").upper()
            print(f"{name:<20} {count:>8} {t:>12.4f} {avg:>12.4f} {pct:>7.2f}%")
        
        # 总计
        print(f"{'-'*70}")
        gpu_total = sum(self.timing[k] for k in self.timing if k.startswith("gpu_"))
        tee_total = sum(self.timing[k] for k in self.timing if k.startswith("tee_"))
        print(f"{'GPU Total':<20} {'':<8} {gpu_total:>12.4f} {'':<12} {gpu_total/total_time*100:>7.2f}%")
        print(f"{'TEE Total':<20} {'':<8} {tee_total:>12.4f} {'':<12} {tee_total/total_time*100:>7.2f}%")
        print(f"{'TOTAL':<20} {'':<8} {total_time:>12.4f} {'':<12} {'100.00':>7}%")
        print(f"{'='*70}\n")


def run_benchmark(model: TEELlamaModel, tokenizer, prefill_length: int) -> float:
    """运行性能测试"""
    input_ids = torch.full((1, prefill_length), tokenizer.pad_token_id, dtype=torch.long)
    
    print(f"\n{'='*70}")
    print(f"{'Prefill Benchmark':^70}")
    print(f"{'='*70}")
    print(f"Token length: {prefill_length}")
    print(f"TEE: Softmax, RMSNorm, RotaryEmbedding, SiLU")
    print(f"GPU: Linear, Embedding, Matmul, LM Head")
    print(f"{'='*70}\n")
    
    # Warmup
    print("Warming up...")
    _ = model.forward(input_ids)
    
    # Reset stats
    model.timing = {k: 0.0 for k in model.timing}
    model.counts = {k: 0 for k in model.counts}
    model.gpu.stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in model.gpu.stats.items()}
    
    # Benchmark
    print("Running benchmark...")
    start_time = time.perf_counter()
    logits = model.forward(input_ids)
    elapsed_time = time.perf_counter() - start_time
    
    print(f"\n{'='*70}")
    print(f"Prefill time: {elapsed_time:.4f}s")
    print(f"Throughput: {prefill_length / elapsed_time:.2f} tokens/sec")
    print(f"Logits shape: {logits.shape}")
    print(f"{'='*70}")
    
    # 打印详细统计
    model.print_timing_stats()
    model.gpu.print_stats()
    
    return elapsed_time


def main() -> None:
    """主函数"""
    model_path = os.environ.get("LLAMA_MODEL_PATH", DEFAULT_MODEL_PATH)
    ipc_path = os.environ.get("LLAMA_IPC_PATH", DEFAULT_IPC_PATH)
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
    gpu_client = GPUClient(ipc_path)
    
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
        
        # 运行测试
        run_benchmark(model, tokenizer, PREFILL_TOKEN_LENGTH)
        
    finally:
        gpu_client.close()
        print("✓ Connection closed")


if __name__ == "__main__":
    main()
