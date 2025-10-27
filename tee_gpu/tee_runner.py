"""
TEE 客户端 - Prefill 阶段性能测试
连接到 GPU 服务器，执行模型 prefill 阶段推理
"""
import os
import time
from typing import Dict, List

import zmq
import msgpack
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 配置
PREFILL_TOKEN_LENGTH = 128
DEFAULT_MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
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


class RemoteLinearProxy(nn.Module):
    """远程 Linear 层代理"""
    
    def __init__(self, module_name: str, client: 'ZMQClient') -> None:
        super().__init__()
        self.module_name = module_name
        self.client = client
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        tensor_cpu = hidden_states.detach().to(torch.float32).cpu().contiguous()
        
        request = {
            "module_name": self.module_name,
            "input_buffer": tensor_cpu.numpy().tobytes(),
            "input_shape": list(tensor_cpu.shape),
            "dtype": TORCH_DTYPE_TO_STR[torch.float32],
        }
        
        response = self.client.forward(request)
        output_array = np.frombuffer(response["output_buffer"], dtype=STR_TO_NUMPY[RESPONSE_DTYPE])
        output_tensor = torch.from_numpy(output_array).view(*response["output_shape"])
        
        return output_tensor.to(dtype=original_dtype)


class RemoteEmbeddingProxy(nn.Module):
    """远程 Embedding 层代理"""
    
    def __init__(self, module_name: str, client: 'ZMQClient') -> None:
        super().__init__()
        self.module_name = module_name
        self.client = client
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        tensor_cpu = input_ids.detach().to(torch.int64).cpu().contiguous()
        
        request = {
            "module_name": self.module_name,
            "input_buffer": tensor_cpu.numpy().tobytes(),
            "input_shape": list(tensor_cpu.shape),
            "dtype": TORCH_DTYPE_TO_STR[torch.int64],
        }
        
        response = self.client.forward(request)
        output_array = np.frombuffer(response["output_buffer"], dtype=STR_TO_NUMPY[RESPONSE_DTYPE])
        output_tensor = torch.from_numpy(output_array).view(*response["output_shape"])
        
        return output_tensor


class RemoteMatmul:
    """远程矩阵乘法（用于 Attention）"""
    
    def __init__(self, client: 'ZMQClient') -> None:
        self.client = client
    
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_cpu = a.detach().to(torch.float32).cpu().contiguous()
        b_cpu = b.detach().to(torch.float32).cpu().contiguous()
        
        request = {
            "a_buffer": a_cpu.numpy().tobytes(),
            "a_shape": list(a_cpu.shape),
            "b_buffer": b_cpu.numpy().tobytes(),
            "b_shape": list(b_cpu.shape),
            "dtype": TORCH_DTYPE_TO_STR[torch.float32],
        }
        
        response = self.client.matmul(request)
        output_array = np.frombuffer(response["output_buffer"], dtype=STR_TO_NUMPY[RESPONSE_DTYPE])
        output_tensor = torch.from_numpy(output_array).view(*response["output_shape"])
        
        return output_tensor.to(dtype=a.dtype)


class ZMQClient:
    """ZeroMQ 客户端"""
    
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{endpoint}")
        print(f"✓ Connected to server at {endpoint}")
    
    def _send_request(self, method: str, request: Dict) -> Dict:
        """发送请求并接收响应"""
        message = {"method": method, "request": request}
        message_bytes = msgpack.packb(message, use_bin_type=True)
        self.socket.send(message_bytes)
        
        response_bytes = self.socket.recv()
        response = msgpack.unpackb(response_bytes, raw=False)
        
        if response["status"] == "error":
            raise RuntimeError(f"Server error: {response['error']}")
        
        return response["response"]
    
    def register(self, module_names: List[str]) -> Dict:
        """注册模块"""
        response = self._send_request("RegisterClient", {"module_names": module_names})
        if not response["ok"]:
            raise RuntimeError(f"Missing modules: {response['missing_modules']}")
        return response
    
    def forward(self, request: Dict) -> Dict:
        """前向传播"""
        return self._send_request("Forward", request)
    
    def matmul(self, request: Dict) -> Dict:
        """矩阵乘法"""
        return self._send_request("Matmul", request)
    
    def fetch_state(self, param_names: List[str], buffer_names: List[str]) -> Dict:
        """获取非线性层状态"""
        return self._send_request("FetchNonLinearTensors", {
            "parameter_names": param_names,
            "buffer_names": buffer_names,
        })
    
    def close(self) -> None:
        """关闭连接"""
        self.socket.close()
        self.context.term()


def load_model_config(model_path: str) -> AutoModelForCausalLM:
    """加载模型配置（不加载权重）"""
    is_local = os.path.exists(model_path)
    
    print(f"Loading model config from: {model_path}")
    
    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=is_local,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(torch.device("cpu"))
    model.eval()
    
    return model


def get_linear_modules(model: nn.Module) -> List[str]:
    """获取所有 Linear 和 Embedding 模块名称"""
    return [
        name for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Embedding))
    ]


def apply_nonlinear_state(model: nn.Module, tensors: Dict) -> None:
    """应用非线性层的参数和 buffer"""
    param_map = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())
    
    for tensor_data in tensors["parameters"]:
        if tensor_data["name"] in param_map:
            tensor = torch.frombuffer(tensor_data["tensor_buffer"], dtype=torch.float32).clone()
            tensor = tensor.view(*tensor_data["shape"])
            param_map[tensor_data["name"]].data.copy_(tensor)
    
    for tensor_data in tensors["buffers"]:
        if tensor_data["name"] in buffer_map:
            tensor = torch.frombuffer(tensor_data["tensor_buffer"], dtype=torch.float32).clone()
            tensor = tensor.view(*tensor_data["shape"])
            buffer_map[tensor_data["name"]].copy_(tensor)


def replace_with_remote_modules(model: nn.Module, client: ZMQClient) -> None:
    """将 Linear 和 Embedding 替换为远程代理"""
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            
            if isinstance(child, nn.Linear):
                proxy = RemoteLinearProxy(full_name, client)
                setattr(parent, child_name, proxy)
            elif isinstance(child, nn.Embedding):
                proxy = RemoteEmbeddingProxy(full_name, client)
                setattr(parent, child_name, proxy)


def inject_remote_matmul(client: ZMQClient) -> None:
    """注入远程矩阵乘法（用于 Attention）"""
    remote_matmul = RemoteMatmul(client)
    original_matmul = torch.matmul
    
    def wrapped_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # 只对多维矩阵乘法使用远程调用
        if a.dim() >= 3 and b.dim() >= 3:
            return remote_matmul(a, b)
        return original_matmul(a, b)
    
    torch.matmul = wrapped_matmul  # type: ignore
    
    original_tensor_matmul = torch.Tensor.matmul
    
    def wrapped_tensor_matmul(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        if self.dim() >= 3 and other.dim() >= 3:
            return remote_matmul(self, other)
        return original_tensor_matmul(self, other)
    
    torch.Tensor.matmul = wrapped_tensor_matmul  # type: ignore


def run_prefill_benchmark(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prefill_length: int) -> float:
    """运行 prefill 阶段性能测试"""
    # 创建固定长度的输入
    input_ids = torch.full((1, prefill_length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"\n{'='*60}")
    print(f"Running Prefill Benchmark")
    print(f"{'='*60}")
    print(f"Token length: {prefill_length}")
    
    # 执行 prefill 并计时
    start_time = time.perf_counter()
    
    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    
    elapsed_time = time.perf_counter() - start_time
    
    print(f"{'='*60}")
    print(f"Prefill time: {elapsed_time:.4f} seconds")
    print(f"Throughput: {prefill_length / elapsed_time:.2f} tokens/sec")
    print(f"{'='*60}\n")
    
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
    
    # 加载模型配置
    model = load_model_config(model_path)
    linear_modules = get_linear_modules(model)
    print(f"✓ Found {len(linear_modules)} linear/embedding modules")
    
    # 连接服务器
    client = ZMQClient(endpoint)
    
    try:
        # 注册模块并获取非线性层状态
        print("Registering modules with server...")
        register_response = client.register(linear_modules)
        
        print("Fetching non-linear layer states...")
        tensors = client.fetch_state(
            register_response["nonlinear_parameter_names"],
            register_response["nonlinear_buffer_names"]
        )
        
        # 应用状态并替换模块
        print("Applying non-linear states...")
        apply_nonlinear_state(model, tensors)
        
        print("Replacing linear modules with remote proxies...")
        replace_with_remote_modules(model, client)
        
        print("Injecting remote matmul...")
        inject_remote_matmul(client)
        
        print("✓ Model setup complete")
        
        # 运行 prefill 测试
        run_prefill_benchmark(model, tokenizer, PREFILL_TOKEN_LENGTH)
        
    finally:
        client.close()
        print("✓ Connection closed")


if __name__ == "__main__":
    main()
