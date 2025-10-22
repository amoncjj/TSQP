import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import grpc
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import msg_pb2
import msg_pb2_grpc

DEFAULT_PROMPT = "Hello, world!"
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_LENGTH = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_RESULT_PATH = "tee_gpu_benchmark.json"
DEFAULT_MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_GPU_ENDPOINT = "localhost:50051"

PROMPT_LIST_ENV = "LLAMA_PROMPT_LIST"
PROMPT_PATH_ENV = "LLAMA_PROMPT_PATH"
BATCH_SIZE_ENV = "LLAMA_TEE_BATCH_SIZE"
MAX_LENGTH_ENV = "LLAMA_MAX_LENGTH"
TEMPERATURE_ENV = "LLAMA_TEMPERATURE"
TOP_P_ENV = "LLAMA_TOP_P"
MODEL_PATH_ENV = "LLAMA_MODEL_PATH"
RESULT_PATH_ENV = "LLAMA_GPU_RESULT_PATH"
GPU_ENDPOINT_ENV = "LLAMA_GPU_ENDPOINT"

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


def read_prompts() -> List[str]:
    prompt_list_env = os.environ.get(PROMPT_LIST_ENV)
    prompt_path_env = os.environ.get(PROMPT_PATH_ENV)

    if prompt_list_env:
        try:
            return json.loads(prompt_list_env)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse LLAMA_PROMPT_LIST: {exc}")

    if prompt_path_env and os.path.exists(prompt_path_env):
        with open(prompt_path_env, "r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
            if prompts:
                return prompts

    return [DEFAULT_PROMPT]


def resolve_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for {name}: {exc}")


def resolve_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError as exc:
        raise RuntimeError(f"Invalid float for {name}: {exc}")


@dataclass
class RemoteModuleDescriptor:
    name: str
    input_dtype: str


class RemoteLinearProxy(nn.Module):
    def __init__(self, module_name: str, stub: msg_pb2_grpc.RemoteModuleServiceStub) -> None:
        super().__init__()
        self.module_name = module_name
        self.stub = stub

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        original_dtype = hidden_states.dtype
        tensor_cpu = hidden_states.detach().to(torch.float32).cpu().contiguous()
        request = msg_pb2.ForwardRequest(
            module_name=self.module_name,
            input_buffer=tensor_cpu.numpy().tobytes(),
            input_shape=list(tensor_cpu.shape),
            dtype="torch.float32",
        )
        response = self.stub.Forward(request)
        output_array = np.frombuffer(response.output_buffer, dtype=STR_TO_NUMPY[RESPONSE_DTYPE])
        output_tensor = torch.from_numpy(output_array).view(*response.output_shape)
        return output_tensor.to(dtype=original_dtype)


class RemoteModuleClient:
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
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(torch.device("cpu"))
    model.eval()
    return model


def list_linear_modules(model: nn.Module) -> List[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def apply_state_to_model(
    model: nn.Module,
    tensors: msg_pb2.NonLinearTensorResponse,
) -> None:
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
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                proxy = RemoteLinearProxy(full_name, stub)
                setattr(parent, child_name, proxy)


def run_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_length: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Iterable[str]]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return {"prompts": prompts, "completions": completions}


def benchmark_split_inference(
    model_path: str,
    endpoint: str,
    prompts: List[str],
    max_length: int,
    temperature: float,
    top_p: float,
    result_path: str,
) -> Dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = load_split_model(model_path)
    linear_module_names = list_linear_modules(model)

    client = RemoteModuleClient(endpoint)
    register_response = client.register(linear_module_names)
    tensors = client.fetch_state(register_response.nonlinear_parameter_names, register_response.nonlinear_buffer_names)
    apply_state_to_model(model, tensors)
    replace_linear_modules(model, client.stub)

    start = time.perf_counter()
    generation = run_generation(model, tokenizer, prompts, max_length, temperature, top_p)
    elapsed = time.perf_counter() - start

    results = {
        "mode": "tee-gpu-split",
        "prompt_count": len(prompts),
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "time_seconds": elapsed,
        "outputs": generation["completions"],
    }

    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    return results


def main() -> None:
    prompts = read_prompts()
    batch_size = resolve_int_env(BATCH_SIZE_ENV, DEFAULT_BATCH_SIZE)
    prompts = (prompts * ((batch_size + len(prompts) - 1) // len(prompts)))[:batch_size]

    model_path = os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
    endpoint = os.environ.get(GPU_ENDPOINT_ENV, DEFAULT_GPU_ENDPOINT)
    result_path = os.environ.get(RESULT_PATH_ENV, DEFAULT_RESULT_PATH)
    max_length = resolve_int_env(MAX_LENGTH_ENV, DEFAULT_MAX_LENGTH)
    temperature = resolve_float_env(TEMPERATURE_ENV, DEFAULT_TEMPERATURE)
    top_p = resolve_float_env(TOP_P_ENV, DEFAULT_TOP_P)

    results = benchmark_split_inference(
        model_path=model_path,
        endpoint=endpoint,
        prompts=prompts,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        result_path=result_path,
    )
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
