import json
import os
import time
from typing import Dict, List, Sequence

import grpc
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import msg_pb2
import msg_pb2_grpc

# 配置：在代码中直接指定
PREFILL_TOKEN_LENGTH = 128  # 直接在这里修改 token 数量
DEFAULT_MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_GPU_ENDPOINT = "localhost:50051"

# 环境变量
MODEL_PATH_ENV = "LLAMA_MODEL_PATH"
GPU_ENDPOINT_ENV = "LLAMA_GPU_ENDPOINT"

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
    def __init__(self, module_name: str, stub: msg_pb2_grpc.RemoteModuleServiceStub) -> None:
        super().__init__()
        self.module_name = module_name
        self.stub = stub

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        tensor_cpu = hidden_states.detach().to(torch.float32).cpu().contiguous()
        request = msg_pb2.ForwardRequest(
            module_name=self.module_name,
            input_buffer=tensor_cpu.numpy().tobytes(),
            input_shape=list(tensor_cpu.shape),
            dtype=TORCH_DTYPE_TO_STR[torch.float32],
        )
        response = self.stub.Forward(request)
        output_array = np.frombuffer(response.output_buffer, dtype=STR_TO_NUMPY[RESPONSE_DTYPE])
        output_tensor = torch.from_numpy(output_array).view(*response.output_shape)
        return output_tensor.to(dtype=original_dtype)


class RemoteEmbeddingProxy(nn.Module):
    """远程 Embedding 层代理"""
    def __init__(self, module_name: str, stub: msg_pb2_grpc.RemoteModuleServiceStub) -> None:
        super().__init__()
        self.module_name = module_name
        self.stub = stub

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        tensor_cpu = input_ids.detach().to(torch.int64).cpu().contiguous()
        request = msg_pb2.ForwardRequest(
            module_name=self.module_name,
            input_buffer=tensor_cpu.numpy().tobytes(),
            input_shape=list(tensor_cpu.shape),
            dtype=TORCH_DTYPE_TO_STR[torch.int64],
        )
        response = self.stub.Forward(request)
        output_array = np.frombuffer(response.output_buffer, dtype=STR_TO_NUMPY[RESPONSE_DTYPE])
        output_tensor = torch.from_numpy(output_array).view(*response.output_shape)
        return output_tensor


class RemoteMatmul:
    """远程矩阵乘法"""
    def __init__(self, stub: msg_pb2_grpc.RemoteModuleServiceStub) -> None:
        self.stub = stub

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_cpu = a.detach().to(torch.float32).cpu().contiguous()
        b_cpu = b.detach().to(torch.float32).cpu().contiguous()
        request = msg_pb2.MatmulRequest(
            a_buffer=a_cpu.numpy().tobytes(),
            a_shape=list(a_cpu.shape),
            b_buffer=b_cpu.numpy().tobytes(),
            b_shape=list(b_cpu.shape),
            dtype=TORCH_DTYPE_TO_STR[torch.float32],
        )
        response = self.stub.Matmul(request)
        output_array = np.frombuffer(response.output_buffer, dtype=STR_TO_NUMPY[RESPONSE_DTYPE])
        output_tensor = torch.from_numpy(output_array).view(*response.output_shape)
        return output_tensor.to(dtype=a.dtype)


class RemoteModuleClient:
    """远程模块客户端"""
    def __init__(self, endpoint: str) -> None:
        self.channel = grpc.insecure_channel(endpoint)
        self.stub = msg_pb2_grpc.RemoteModuleServiceStub(self.channel)

    def register(self, module_names: Sequence[str]) -> msg_pb2.ModuleListResponse:
        response = self.stub.RegisterClient(msg_pb2.ModuleListRequest(module_names=list(module_names)))
        if not response.ok:
            raise RuntimeError(f"Missing remote modules: {response.missing_modules}")
        return response

    def fetch_state(
        self,
        parameter_names: Sequence[str],
        buffer_names: Sequence[str],
    ) -> msg_pb2.NonLinearTensorResponse:
        request = msg_pb2.NonLinearTensorRequest(
            parameter_names=list(parameter_names),
            buffer_names=list(buffer_names),
        )
        return self.stub.FetchNonLinearTensors(request)


def load_split_model(model_path: str) -> AutoModelForCausalLM:
    """加载分离模型（只加载配置，不加载权重）"""
    is_local_path = os.path.exists(model_path)
    
    print(f"Loading model config from: {model_path}")
    print(f"Is local path: {is_local_path}")
    
    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=is_local_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(torch.device("cpu"))
    model.eval()
    return model


def list_linear_modules(model: nn.Module) -> List[str]:
    """列出所有 Linear 和 Embedding 模块"""
    return [name for name, module in model.named_modules() if isinstance(module, (nn.Linear, nn.Embedding))]


def apply_state_to_model(
    model: nn.Module,
    tensors: msg_pb2.NonLinearTensorResponse,
) -> None:
    """将非线性层的参数和 buffer 应用到模型"""
    parameter_map = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())

    for tensor_proto in tensors.parameters:
        destination = parameter_map.get(tensor_proto.name)
        if destination is None:
            continue
        tensor = torch.frombuffer(tensor_proto.tensor_buffer, dtype=torch.float32).clone()
        tensor = tensor.view(*tensor_proto.shape)
        destination.data.copy_(tensor)

    for tensor_proto in tensors.buffers:
        destination = buffer_map.get(tensor_proto.name)
        if destination is None:
            continue
        tensor = torch.frombuffer(tensor_proto.tensor_buffer, dtype=torch.float32).clone()
        tensor = tensor.view(*tensor_proto.shape)
        destination.copy_(tensor)


def replace_linear_modules(model: nn.Module, stub: msg_pb2_grpc.RemoteModuleServiceStub) -> None:
    """将 Linear 和 Embedding 模块替换为远程代理"""
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear):
                proxy = RemoteLinearProxy(full_name, stub)
                setattr(parent, child_name, proxy)
            elif isinstance(child, nn.Embedding):
                proxy = RemoteEmbeddingProxy(full_name, stub)
                setattr(parent, child_name, proxy)


def inject_remote_matmul(model: nn.Module, stub: msg_pb2_grpc.RemoteModuleServiceStub) -> None:
    """注入远程矩阵乘法（用于 Attention）"""
    remote_matmul = RemoteMatmul(stub)
    original_matmul = torch.matmul
    
    def wrapped_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # 只对多维矩阵乘法使用远程调用（Attention 相关）
        if a.dim() >= 3 and b.dim() >= 3:
            return remote_matmul(a, b)
        else:
            return original_matmul(a, b)
    
    torch.matmul = wrapped_matmul  # type: ignore
    
    original_tensor_matmul = torch.Tensor.matmul
    def wrapped_tensor_matmul(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        if self.dim() >= 3 and other.dim() >= 3:
            return remote_matmul(self, other)
        else:
            return original_tensor_matmul(self, other)
    
    torch.Tensor.matmul = wrapped_tensor_matmul  # type: ignore


def run_prefill(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefill_length: int,
) -> float:
    """
    执行 prefill 阶段，返回 prefill 时间（秒）
    """
    # 创建固定长度的 token 序列
    input_ids = torch.full((1, prefill_length), tokenizer.pad_token_id, dtype=torch.long).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"Prefill token length: {prefill_length}")
    
    # 前向传播（prefill）并计时
    start = time.perf_counter()
    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    elapsed = time.perf_counter() - start
    
    return elapsed


def main() -> None:
    """主函数"""
    model_path = os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
    endpoint = os.environ.get(GPU_ENDPOINT_ENV, DEFAULT_GPU_ENDPOINT)
    
    is_local_path = os.path.exists(model_path)
    
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=is_local_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_split_model(model_path)
    linear_module_names = list_linear_modules(model)

    # 连接 GPU 服务器并注册模块
    print(f"Connecting to GPU server at {endpoint}")
    client = RemoteModuleClient(endpoint)
    register_response = client.register(linear_module_names)
    tensors = client.fetch_state(register_response.nonlinear_parameter_names, register_response.nonlinear_buffer_names)
    apply_state_to_model(model, tensors)
    replace_linear_modules(model, client.stub)
    inject_remote_matmul(model, client.stub)

    print(f"Running prefill with {PREFILL_TOKEN_LENGTH} tokens...")
    prefill_time = run_prefill(model, tokenizer, PREFILL_TOKEN_LENGTH)
    
    print(f"\nPrefill time: {prefill_time:.4f} seconds")


if __name__ == "__main__":
    main()
