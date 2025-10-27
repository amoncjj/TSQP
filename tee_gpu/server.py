"""
GPU 服务端 - 使用 ZeroMQ 提供远程模块推理服务
托管 LLaMA 模型的 Linear 和 Embedding 层
"""
import os
import time
from typing import Dict, List

import zmq
import msgpack
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM

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


class ModuleRegistry:
    """模块注册表 - 管理可远程调用的模块"""
    
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()
        
        # 注册所有 Linear 和 Embedding 模块
        self.modules: Dict[str, nn.Module] = {}
        self.module_dtypes: Dict[str, torch.dtype] = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.to(device).eval()
                self.modules[name] = module
                self.module_dtypes[name] = torch.float32
            elif isinstance(module, nn.Embedding):
                module.to(device).eval()
                self.modules[name] = module
                self.module_dtypes[name] = torch.int64
        
        # 收集非线性层的参数和 buffer
        self.parameters = dict(model.named_parameters())
        self.buffers = dict(model.named_buffers())
        
        # 过滤出非线性层的参数和 buffer
        linear_prefixes = set(self.modules.keys())
        self.nonlinear_param_names = [
            name for name in self.parameters.keys()
            if not any(name.startswith(f"{prefix}.") and name.split(".")[-1] in {"weight", "bias"} 
                      for prefix in linear_prefixes)
        ]
        self.nonlinear_buffer_names = [
            name for name in self.buffers.keys()
            if not any(name.startswith(f"{prefix}.") for prefix in linear_prefixes)
        ]
    
    def forward(self, module_name: str, input_tensor: torch.Tensor) -> torch.Tensor:
        """执行模块前向传播"""
        if module_name not in self.modules:
            raise KeyError(f"Module {module_name} not found")
        
        with torch.no_grad():
            return self.modules[module_name](input_tensor)
    
    def get_nonlinear_tensors(self, param_names: List[str], buffer_names: List[str]) -> Dict:
        """获取非线性层的参数和 buffer"""
        params = []
        for name in param_names:
            if name in self.parameters:
                tensor = self.parameters[name].detach().to(dtype=RESPONSE_TORCH_DTYPE, device="cpu").contiguous()
                params.append({
                    "name": name,
                    "tensor_buffer": tensor.numpy().tobytes(),
                    "shape": list(tensor.shape),
                    "dtype": RESPONSE_DTYPE,
                })
        
        buffers = []
        for name in buffer_names:
            if name in self.buffers:
                tensor = self.buffers[name].detach().to(dtype=RESPONSE_TORCH_DTYPE, device="cpu").contiguous()
                buffers.append({
                    "name": name,
                    "tensor_buffer": tensor.numpy().tobytes(),
                    "shape": list(tensor.shape),
                    "dtype": RESPONSE_DTYPE,
                })
        
        return {"parameters": params, "buffers": buffers}


class ZMQServer:
    """ZeroMQ 服务器"""
    
    def __init__(self, registry: ModuleRegistry, port: str) -> None:
        self.registry = registry
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        print(f"✓ ZeroMQ server started on port {port}")
    
    def handle_register(self, request: Dict) -> Dict:
        """处理客户端注册请求"""
        requested = set(request["module_names"])
        available = set(self.registry.modules.keys())
        missing = sorted(requested - available)
        
        return {
            "ok": len(missing) == 0,
            "missing_modules": missing,
            "nonlinear_parameter_names": self.registry.nonlinear_param_names,
            "nonlinear_buffer_names": self.registry.nonlinear_buffer_names,
        }
    
    def handle_forward(self, request: Dict) -> Dict:
        """处理前向传播请求"""
        module_name = request["module_name"]
        dtype_str = request["dtype"]
        
        # 解析输入
        input_dtype = TORCH_DTYPE_MAP.get(dtype_str, self.registry.module_dtypes[module_name])
        numpy_dtype = NUMPY_DTYPE_MAP.get(dtype_str, np.float32)
        
        input_array = np.frombuffer(request["input_buffer"], dtype=numpy_dtype)
        input_array = input_array.reshape(tuple(request["input_shape"]))
        input_tensor = torch.from_numpy(input_array).to(device=self.registry.device, dtype=input_dtype)
        
        # 执行前向传播
        output_tensor = self.registry.forward(module_name, input_tensor)
        
        # 返回结果
        output_cpu = output_tensor.detach().to(dtype=RESPONSE_TORCH_DTYPE, device="cpu").contiguous()
        return {
            "output_buffer": output_cpu.numpy().tobytes(),
            "output_shape": list(output_cpu.shape),
            "dtype": RESPONSE_DTYPE,
        }
    
    def handle_fetch_tensors(self, request: Dict) -> Dict:
        """处理获取非线性层张量请求"""
        return self.registry.get_nonlinear_tensors(
            request["parameter_names"],
            request["buffer_names"]
        )
    
    def handle_matmul(self, request: Dict) -> Dict:
        """处理矩阵乘法请求"""
        dtype = TORCH_DTYPE_MAP.get(request["dtype"], torch.float32)
        numpy_dtype = NUMPY_DTYPE_MAP.get(request["dtype"], np.float32)
        
        a_array = np.frombuffer(request["a_buffer"], dtype=numpy_dtype).reshape(tuple(request["a_shape"]))
        b_array = np.frombuffer(request["b_buffer"], dtype=numpy_dtype).reshape(tuple(request["b_shape"]))
        
        a_tensor = torch.from_numpy(a_array).to(device=self.registry.device, dtype=dtype)
        b_tensor = torch.from_numpy(b_array).to(device=self.registry.device, dtype=dtype)
        
        with torch.no_grad():
            output_tensor = torch.matmul(a_tensor, b_tensor)
        
        output_cpu = output_tensor.detach().to(dtype=RESPONSE_TORCH_DTYPE, device="cpu").contiguous()
        return {
            "output_buffer": output_cpu.numpy().tobytes(),
            "output_shape": list(output_cpu.shape),
            "dtype": RESPONSE_DTYPE,
        }
    
    def handle_request(self, message: Dict) -> Dict:
        """处理请求"""
        method = message.get("method")
        request = message.get("request", {})
        
        try:
            if method == "RegisterClient":
                response = self.handle_register(request)
            elif method == "Forward":
                response = self.handle_forward(request)
            elif method == "FetchNonLinearTensors":
                response = self.handle_fetch_tensors(request)
            elif method == "Matmul":
                response = self.handle_matmul(request)
            else:
                return {"status": "error", "error": f"Unknown method: {method}"}
            
            return {"status": "success", "response": response}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
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
    
    # 创建注册表和服务器
    registry = ModuleRegistry(model, device)
    print(f"✓ Registered {len(registry.modules)} remote modules")
    
    server = ZMQServer(registry, port)
    server.serve()


if __name__ == "__main__":
    main()
