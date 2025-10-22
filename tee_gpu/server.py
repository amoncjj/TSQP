import concurrent.futures
import dataclasses
import json
import os
import time
from typing import Dict, Iterable, List, Sequence

import grpc
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

import msg_pb2
import msg_pb2_grpc

GRPC_MAX_MESSAGE_LENGTH = 512 * 1024 * 1024
DEFAULT_MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DTYPE = "float32"

TORCH_DTYPE_FROM_STRING: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "torch.float32": torch.float32,
    "float16": torch.float16,
    "torch.float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "torch.int64": torch.int64,
}

NUMPY_DTYPE_FROM_STRING: Dict[str, np.dtype] = {
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
RESPONSE_NUMPY_DTYPE = np.float32
RESPONSE_TORCH_DTYPE = torch.float32


@dataclasses.dataclass
class RemoteModuleRecord:
    module_name: str
    module_ref: nn.Module
    input_dtype: torch.dtype

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.module_ref(input_tensor)


class LlamaModuleRegistry:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()

        self.remote_modules: Dict[str, RemoteModuleRecord] = {}
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.to(device)
                module.eval()
                self.remote_modules[module_name] = RemoteModuleRecord(
                    module_name=module_name,
                    module_ref=module,
                    input_dtype=torch.float32,
                )
            elif isinstance(module, nn.Embedding):
                module.to(device)
                module.eval()
                self.remote_modules[module_name] = RemoteModuleRecord(
                    module_name=module_name,
                    module_ref=module,
                    input_dtype=torch.int64,
                )

        self.parameter_map: Dict[str, torch.nn.Parameter] = dict(model.named_parameters())
        self.buffer_map: Dict[str, torch.Tensor] = dict(model.named_buffers())

        remote_prefixes = set(self.remote_modules.keys())
        self.nonlinear_parameter_names: List[str] = []
        for parameter_name in self.parameter_map:
            module_prefix, _, leaf_name = parameter_name.rpartition(".")
            if module_prefix in remote_prefixes and leaf_name in {"weight", "bias"}:
                continue
            self.nonlinear_parameter_names.append(parameter_name)

        self.nonlinear_buffer_names: List[str] = []
        for buffer_name in self.buffer_map:
            module_prefix, _, _ = buffer_name.rpartition(".")
            if module_prefix in remote_prefixes:
                continue
            self.nonlinear_buffer_names.append(buffer_name)

    def list_remote_module_names(self) -> Iterable[str]:
        return self.remote_modules.keys()

    def get_remote_module(self, module_name: str) -> RemoteModuleRecord:
        if module_name not in self.remote_modules:
            raise KeyError(f"Module {module_name} is not registered for remote execution")
        return self.remote_modules[module_name]

    def collect_named_parameters(self, parameter_names: Sequence[str]) -> List[msg_pb2.NamedTensor]:
        tensors: List[msg_pb2.NamedTensor] = []
        for name in parameter_names:
            tensor = self.parameter_map.get(name)
            if tensor is None:
                continue
            tensor_cpu = tensor.detach().to(dtype=RESPONSE_TORCH_DTYPE, device="cpu").contiguous()
            tensors.append(
                msg_pb2.NamedTensor(
                    name=name,
                    tensor_buffer=tensor_cpu.numpy().tobytes(),
                    shape=list(tensor_cpu.shape),
                    dtype=RESPONSE_DTYPE,
                )
            )
        return tensors

    def collect_named_buffers(self, buffer_names: Sequence[str]) -> List[msg_pb2.NamedTensor]:
        tensors: List[msg_pb2.NamedTensor] = []
        for name in buffer_names:
            tensor = self.buffer_map.get(name)
            if tensor is None:
                continue
            tensor_cpu = tensor.detach().to(dtype=RESPONSE_TORCH_DTYPE, device="cpu").contiguous()
            tensors.append(
                msg_pb2.NamedTensor(
                    name=name,
                    tensor_buffer=tensor_cpu.numpy().tobytes(),
                    shape=list(tensor_cpu.shape),
                    dtype=RESPONSE_DTYPE,
                )
            )
        return tensors


class RemoteModuleService(msg_pb2_grpc.RemoteModuleServiceServicer):
    def __init__(self, module_registry: LlamaModuleRegistry, device: torch.device) -> None:
        super().__init__()
        self.module_registry = module_registry
        self.device = device

    def RegisterClient(self, request: msg_pb2.ModuleListRequest, context: grpc.ServicerContext) -> msg_pb2.ModuleListResponse:
        requested = set(request.module_names)
        available = set(self.module_registry.list_remote_module_names())
        missing_modules = sorted(requested - available)
        return msg_pb2.ModuleListResponse(
            ok=len(missing_modules) == 0,
            missing_modules=missing_modules,
            nonlinear_parameter_names=list(self.module_registry.nonlinear_parameter_names),
            nonlinear_buffer_names=list(self.module_registry.nonlinear_buffer_names),
        )

    def Forward(self, request: msg_pb2.ForwardRequest, context: grpc.ServicerContext) -> msg_pb2.ForwardResponse:
        module_record = self.module_registry.get_remote_module(request.module_name)
        input_dtype = TORCH_DTYPE_FROM_STRING.get(request.dtype, module_record.input_dtype)
        numpy_dtype = NUMPY_DTYPE_FROM_STRING.get(request.dtype, np.float32)
        input_array = np.frombuffer(request.input_buffer, dtype=numpy_dtype).reshape(tuple(request.input_shape))
        input_tensor = torch.from_numpy(input_array).to(device=self.device, dtype=input_dtype)

        output_tensor = module_record.run(input_tensor)
        output_cpu = output_tensor.detach().to(dtype=RESPONSE_TORCH_DTYPE, device="cpu").contiguous()
        return msg_pb2.ForwardResponse(
            output_buffer=output_cpu.numpy().tobytes(),
            output_shape=list(output_cpu.shape),
            dtype=RESPONSE_DTYPE,
        )

    def FetchNonLinearTensors(self, request: msg_pb2.NonLinearTensorRequest, context: grpc.ServicerContext) -> msg_pb2.NonLinearTensorResponse:
        parameter_tensors = self.module_registry.collect_named_parameters(request.parameter_names)
        buffer_tensors = self.module_registry.collect_named_buffers(request.buffer_names)
        return msg_pb2.NonLinearTensorResponse(parameters=parameter_tensors, buffers=buffer_tensors)


def resolve_model_path() -> str:
    return os.environ.get("LLAMA_MODEL_PATH", DEFAULT_MODEL_PATH)


def resolve_device() -> torch.device:
    device_str = os.environ.get("LLAMA_GPU_DEVICE", DEFAULT_DEVICE)
    return torch.device(device_str)


def resolve_dtype() -> torch.dtype:
    dtype_str = os.environ.get("LLAMA_DTYPE", DEFAULT_DTYPE)
    return TORCH_DTYPE_FROM_STRING.get(dtype_str, torch.float32)


def load_model(device: torch.device, dtype: torch.dtype) -> nn.Module:
    model_path = resolve_model_path()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    except (OSError, ValueError):
        config = AutoConfig.from_pretrained(model_path)
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(state_dict_path):
            raise RuntimeError(
                f"Unable to locate pre-trained weights at {state_dict_path}. Set LLAMA_MODEL_PATH to a valid directory."
            )
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model = AutoModelForCausalLM.from_config(config)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        metadata = {"missing_keys": missing, "unexpected_keys": unexpected}
        print(json.dumps({"event": "load_state_dict", "metadata": metadata}))
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def build_grpc_server(module_registry: LlamaModuleRegistry) -> grpc.Server:
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=8),
        options=[
            ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
        ],
    )
    msg_pb2_grpc.add_RemoteModuleServiceServicer_to_server(RemoteModuleService(module_registry, module_registry.device), server)
    server.add_insecure_port('[::]:50051')
    return server


def serve() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU device is required to host linear and embedding layers")

    device = resolve_device()
    dtype = resolve_dtype()
    model = load_model(device, dtype)
    module_registry = LlamaModuleRegistry(model, device)

    server = build_grpc_server(module_registry)
    server.start()
    print("GPU remote module service started on port 50051")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop(grace=None)


if __name__ == "__main__":
    serve()
