"""
GPU 服务端 - 执行所有 GPU 密集型计算
包括: Linear, Embedding, Matmul, 以及除 Softmax/RMSNorm/RotaryEmbedding/激活函数外的所有操作
"""
import os
from typing import Dict, List, Tuple

import zmq
import msgpack
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig

# 默认配置
DEFAULT_MODEL_PATH = "/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DTYPE = "float32"
DEFAULT_PORT = "50051"

# 数据类型映射
TORCH_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "torch.float32": torch.float32,
    "float16": torch.float16,
    "torch.float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "torch.int64": torch.int64,
}

NUMPY_DTYPE_MAP: Dict[str, np.dtype] = {
    "float32": np.float32,
    "torch.float32": np.float32,
    "float16": np.float16,
    "torch.float16": np.float16,
    "bfloat16": np.float32,
    "torch.bfloat16": np.float32,
    "int64": np.int64,
    "torch.int64": np.int64,
}

RESPONSE_DTYPE = "torch.float32"
RESPONSE_TORCH_DTYPE = torch.float32


class GPUComputeService:
    """GPU 计算服务 - 执行所有 GPU 端计算"""
    
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
        
        # 提取各层的模块
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.lm_head = model.lm_head
        
        print(f"✓ Model loaded: {self.num_layers} layers, hidden_size={self.hidden_size}")
    
    def embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embedding 层"""
        with torch.no_grad():
            return self.embed_tokens(input_ids.to(self.device))
    
    def linear(self, layer_idx: int, module_name: str, hidden_states: torch.Tensor) -> torch.Tensor:
        """执行指定层的 Linear 操作"""
        layer = self.layers[layer_idx]
        
        with torch.no_grad():
            if module_name == "q_proj":
                return layer.self_attn.q_proj(hidden_states)
            elif module_name == "k_proj":
                return layer.self_attn.k_proj(hidden_states)
            elif module_name == "v_proj":
                return layer.self_attn.v_proj(hidden_states)
            elif module_name == "o_proj":
                return layer.self_attn.o_proj(hidden_states)
            elif module_name == "gate_proj":
                return layer.mlp.gate_proj(hidden_states)
            elif module_name == "up_proj":
                return layer.mlp.up_proj(hidden_states)
            elif module_name == "down_proj":
                return layer.mlp.down_proj(hidden_states)
            else:
                raise ValueError(f"Unknown module: {module_name}")
    
    def lm_head_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """LM Head 前向传播"""
        with torch.no_grad():
            return self.lm_head(hidden_states)
    
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """矩阵乘法"""
        with torch.no_grad():
            return torch.matmul(a, b)
    
    def get_rotary_emb_params(self) -> Dict:
        """获取 RotaryEmbedding 的参数（inv_freq）"""
        rotary_emb = self.model.model.rotary_emb
        return {
            "inv_freq": rotary_emb.inv_freq.cpu().numpy().tobytes(),
            "inv_freq_shape": list(rotary_emb.inv_freq.shape),
            "attention_scaling": rotary_emb.attention_scaling,
        }
    
    def get_norm_weights(self) -> Dict:
        """获取所有 RMSNorm 的权重"""
        weights = {}
        
        # 每层的 input_layernorm 和 post_attention_layernorm
        for i, layer in enumerate(self.layers):
            weights[f"layer_{i}_input_layernorm"] = {
                "weight": layer.input_layernorm.weight.detach().cpu().numpy().tobytes(),
                "shape": list(layer.input_layernorm.weight.shape),
                "eps": layer.input_layernorm.variance_epsilon,
            }
            weights[f"layer_{i}_post_attention_layernorm"] = {
                "weight": layer.post_attention_layernorm.weight.detach().cpu().numpy().tobytes(),
                "shape": list(layer.post_attention_layernorm.weight.shape),
                "eps": layer.post_attention_layernorm.variance_epsilon,
            }
        
        # 最后的 norm
        weights["final_norm"] = {
            "weight": self.model.model.norm.weight.detach().cpu().numpy().tobytes(),
            "shape": list(self.model.model.norm.weight.shape),
            "eps": self.model.model.norm.variance_epsilon,
        }
        
        return weights


class ZMQServer:
    """ZeroMQ 服务器"""
    
    def __init__(self, compute_service: GPUComputeService, port: str) -> None:
        self.compute = compute_service
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        print(f"✓ ZeroMQ server started on port {port}")
    
    def _tensor_from_bytes(self, buffer: bytes, shape: List[int], dtype_str: str) -> torch.Tensor:
        """从字节流重建张量"""
        numpy_dtype = NUMPY_DTYPE_MAP.get(dtype_str, np.float32)
        torch_dtype = TORCH_DTYPE_MAP.get(dtype_str, torch.float32)
        
        array = np.frombuffer(buffer, dtype=numpy_dtype).reshape(shape)
        return torch.from_numpy(array).to(device=self.compute.device, dtype=torch_dtype)
    
    def _tensor_to_bytes(self, tensor: torch.Tensor) -> Tuple[bytes, List[int], str]:
        """将张量转换为字节流"""
        tensor_cpu = tensor.detach().to(dtype=RESPONSE_TORCH_DTYPE, device="cpu").contiguous()
        return (
            tensor_cpu.numpy().tobytes(),
            list(tensor_cpu.shape),
            RESPONSE_DTYPE
        )
    
    def handle_init(self, request: Dict) -> Dict:
        """初始化 - 返回模型配置和参数"""
        return {
            "config": {
                "num_layers": self.compute.num_layers,
                "hidden_size": self.compute.hidden_size,
                "num_heads": self.compute.num_heads,
                "num_kv_heads": self.compute.num_kv_heads,
                "head_dim": self.compute.head_dim,
            },
            "rotary_emb_params": self.compute.get_rotary_emb_params(),
            "norm_weights": self.compute.get_norm_weights(),
        }
    
    def handle_embedding(self, request: Dict) -> Dict:
        """处理 Embedding 请求"""
        input_ids = self._tensor_from_bytes(
            request["input_ids"],
            request["input_shape"],
            request["dtype"]
        )
        
        output = self.compute.embedding(input_ids)
        buffer, shape, dtype = self._tensor_to_bytes(output)
        
        return {
            "output": buffer,
            "shape": shape,
            "dtype": dtype,
        }
    
    def handle_linear(self, request: Dict) -> Dict:
        """处理 Linear 请求"""
        hidden_states = self._tensor_from_bytes(
            request["hidden_states"],
            request["shape"],
            request["dtype"]
        )
        
        output = self.compute.linear(
            request["layer_idx"],
            request["module_name"],
            hidden_states
        )
        
        buffer, shape, dtype = self._tensor_to_bytes(output)
        
        return {
            "output": buffer,
            "shape": shape,
            "dtype": dtype,
        }
    
    def handle_matmul(self, request: Dict) -> Dict:
        """处理矩阵乘法请求"""
        a = self._tensor_from_bytes(
            request["a_buffer"],
            request["a_shape"],
            request["dtype"]
        )
        b = self._tensor_from_bytes(
            request["b_buffer"],
            request["b_shape"],
            request["dtype"]
        )
        
        output = self.compute.matmul(a, b)
        buffer, shape, dtype = self._tensor_to_bytes(output)
        
        return {
            "output": buffer,
            "shape": shape,
            "dtype": dtype,
        }
    
    def handle_lm_head(self, request: Dict) -> Dict:
        """处理 LM Head 请求"""
        hidden_states = self._tensor_from_bytes(
            request["hidden_states"],
            request["shape"],
            request["dtype"]
        )
        
        output = self.compute.lm_head_forward(hidden_states)
        buffer, shape, dtype = self._tensor_to_bytes(output)
        
        return {
            "output": buffer,
            "shape": shape,
            "dtype": dtype,
        }
    
    def handle_request(self, message: Dict) -> Dict:
        """处理请求"""
        method = message.get("method")
        request = message.get("request", {})
        
        try:
            if method == "Init":
                response = self.handle_init(request)
            elif method == "Embedding":
                response = self.handle_embedding(request)
            elif method == "Linear":
                response = self.handle_linear(request)
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
        """启动服务循环"""
        try:
            while True:
                message_bytes = self.socket.recv()
                message = msgpack.unpackb(message_bytes, raw=False)
                response = self.handle_request(message)
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
        torch_dtype=dtype,
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
    
    # 读取配置
    model_path = os.environ.get("LLAMA_MODEL_PATH", DEFAULT_MODEL_PATH)
    device = torch.device(os.environ.get("LLAMA_GPU_DEVICE", DEFAULT_DEVICE))
    dtype = TORCH_DTYPE_MAP.get(os.environ.get("LLAMA_DTYPE", DEFAULT_DTYPE), torch.float32)
    port = os.environ.get("LLAMA_GPU_PORT", DEFAULT_PORT)
    
    # 加载模型
    model = load_model(model_path, device, dtype)
    
    # 创建计算服务和服务器
    compute_service = GPUComputeService(model, device)
    server = ZMQServer(compute_service, port)
    server.serve()


if __name__ == "__main__":
    main()
