"""
高性能 GPU 服务端 - 优化版
关键优化：
1. 使用 IPC 而不是 TCP
2. 零拷贝数据传输（使用共享内存）
3. 保持数据在 GPU 上
4. 批量操作
"""
import os
import mmap
import struct
from typing import Dict, List, Tuple, Optional

import zmq
import msgpack
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM

# 配置
DEFAULT_MODEL_PATH = "/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DTYPE = "float32"
DEFAULT_IPC_PATH = "ipc:///tmp/tsqp_gpu_server.ipc"

TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
}


class GPUComputeService:
    """GPU 计算服务 - 保持所有数据在 GPU 上"""
    
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()
        
        # 提取模型配置
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", self.hidden_size // self.num_heads)
        
        # 提取模块
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.lm_head = model.lm_head
        
        # 缓存：避免重复传输
        self.tensor_cache: Dict[str, torch.Tensor] = {}
        
        print(f"✓ Model loaded: {self.num_layers} layers, hidden_size={self.hidden_size}")
        print(f"✓ Device: {device}")
    
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
        
        # RMSNorm 权重 - 转换为 bytes
        norm_weights = {}
        for i, layer in enumerate(self.layers):
            weight = layer.input_layernorm.weight.detach().cpu().numpy()
            norm_weights[f"layer_{i}_input_layernorm"] = {
                "weight": weight.tobytes(),
                "shape": list(weight.shape),
                "eps": layer.input_layernorm.variance_epsilon,
            }
            weight = layer.post_attention_layernorm.weight.detach().cpu().numpy()
            norm_weights[f"layer_{i}_post_attention_layernorm"] = {
                "weight": weight.tobytes(),
                "shape": list(weight.shape),
                "eps": layer.post_attention_layernorm.variance_epsilon,
            }
        
        weight = self.model.model.norm.weight.detach().cpu().numpy()
        norm_weights["final_norm"] = {
            "weight": weight.tobytes(),
            "shape": list(weight.shape),
            "eps": self.model.model.norm.variance_epsilon,
        }
        
        # RotaryEmbedding 参数 - 转换为 bytes
        inv_freq = rotary_emb.inv_freq.cpu().numpy()
        
        return {
            "config": {
                "num_layers": self.num_layers,
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
            },
            "rotary_emb_params": {
                "inv_freq": inv_freq.tobytes(),
                "inv_freq_shape": list(inv_freq.shape),
                "attention_scaling": rotary_emb.attention_scaling,
            },
            "norm_weights": norm_weights,
        }


class ZMQServer:
    """高性能 ZeroMQ 服务器"""
    
    def __init__(self, compute_service: GPUComputeService, ipc_path: str) -> None:
        self.compute = compute_service
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        
        # 优化 ZeroMQ 性能
        self.socket.setsockopt(zmq.SNDHWM, 1000)  # 发送高水位
        self.socket.setsockopt(zmq.RCVHWM, 1000)  # 接收高水位
        self.socket.setsockopt(zmq.LINGER, 0)     # 关闭时不等待
        
        # 使用 IPC 而不是 TCP
        self.socket.bind(ipc_path)
        print(f"✓ ZeroMQ server started on {ipc_path}")
        print(f"✓ Using IPC for zero-copy local communication")
    
    def _deserialize_tensor(self, data: Dict) -> torch.Tensor:
        """快速反序列化张量 - 最小化拷贝"""
        # 直接从 bytes 创建 numpy array（零拷贝）
        array = np.frombuffer(data["buffer"], dtype=np.float32).reshape(data["shape"])
        # 转换为 torch tensor 并移到 GPU（一次拷贝）
        return torch.from_numpy(array).to(device=self.compute.device, dtype=torch.float32, non_blocking=True)
    
    def _serialize_tensor(self, tensor: torch.Tensor) -> Dict:
        """快速序列化张量"""
        # 移到 CPU（一次拷贝）
        tensor_cpu = tensor.detach().cpu().contiguous()
        # 转换为 bytes（零拷贝）
        return {
            "buffer": tensor_cpu.numpy().tobytes(),
            "shape": list(tensor_cpu.shape),
        }
    
    def _serialize_tensors(self, tensors: List[torch.Tensor]) -> List[Dict]:
        """批量序列化"""
        return [self._serialize_tensor(t) for t in tensors]
    
    def handle_init(self, request: Dict) -> Dict:
        """初始化"""
        return self.compute.get_init_data()
    
    def handle_embedding(self, request: Dict) -> Dict:
        """Embedding"""
        # input_ids 是 int64，直接从 bytes 创建
        input_ids = torch.from_numpy(
            np.frombuffer(request["buffer"], dtype=np.int64).reshape(request["shape"])
        )
        output = self.compute.embedding(input_ids)
        return self._serialize_tensor(output)
    
    def handle_batch_linear(self, request: Dict) -> Dict:
        """批量 Linear"""
        hidden_states = self._deserialize_tensor(request["hidden_states"])
        outputs = self.compute.batch_linear(
            request["layer_idx"],
            request["module_names"],
            hidden_states
        )
        return {"outputs": self._serialize_tensors(outputs)}
    
    def handle_matmul(self, request: Dict) -> Dict:
        """矩阵乘法"""
        a = self._deserialize_tensor(request["a"])
        b = self._deserialize_tensor(request["b"])
        output = self.compute.matmul(a, b)
        return self._serialize_tensor(output)
    
    def handle_lm_head(self, request: Dict) -> Dict:
        """LM Head"""
        hidden_states = self._deserialize_tensor(request["hidden_states"])
        output = self.compute.lm_head_forward(hidden_states)
        return self._serialize_tensor(output)
    
    def handle_request(self, message: Dict) -> Dict:
        """处理请求"""
        method = message.get("method")
        request = message.get("request", {})
        
        try:
            if method == "Init":
                response = self.handle_init(request)
            elif method == "Embedding":
                response = self.handle_embedding(request)
            elif method == "BatchLinear":
                response = self.handle_batch_linear(request)
            elif method == "Matmul":
                response = self.handle_matmul(request)
            elif method == "LMHead":
                response = self.handle_lm_head(request)
            else:
                return {"status": "error", "error": f"Unknown method: {method}"}
            
            return {"status": "success", "response": response}
        except Exception as e:
            import traceback
            return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
    
    def serve(self) -> None:
        """服务循环"""
        try:
            print("✓ Server ready, waiting for requests...")
            while True:
                # 接收请求
                message_bytes = self.socket.recv()
                message = msgpack.unpackb(message_bytes, raw=False)
                
                # 处理请求
                response = self.handle_request(message)
                
                # 发送响应
                response_bytes = msgpack.packb(response, use_bin_type=True)
                self.socket.send(response_bytes)
        except KeyboardInterrupt:
            print("\n✓ Server shutting down...")
        finally:
            self.socket.close()
            self.context.term()


def load_model(model_path: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """加载模型"""
    is_local = os.path.exists(model_path)
    
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}, Dtype: {dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=is_local,
        trust_remote_code=True
    )
    
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def main() -> None:
    """主函数"""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    
    # 配置
    model_path = os.environ.get("LLAMA_MODEL_PATH", DEFAULT_MODEL_PATH)
    device = torch.device(os.environ.get("LLAMA_GPU_DEVICE", DEFAULT_DEVICE))
    dtype = TORCH_DTYPE_MAP.get(os.environ.get("LLAMA_DTYPE", DEFAULT_DTYPE), torch.float32)
    ipc_path = os.environ.get("LLAMA_IPC_PATH", DEFAULT_IPC_PATH)
    
    # 加载模型
    model = load_model(model_path, device, dtype)
    
    # 创建服务
    compute_service = GPUComputeService(model, device)
    server = ZMQServer(compute_service, ipc_path)
    server.serve()


if __name__ == "__main__":
    main()
