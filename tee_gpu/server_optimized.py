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
        
        # 确保 attention_scaling 是可序列化的
        attention_scaling = rotary_emb.attention_scaling
        if isinstance(attention_scaling, (np.ndarray, torch.Tensor)):
            attention_scaling = float(attention_scaling)
        elif attention_scaling is None:
            attention_scaling = 1.0
        
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
                "attention_scaling": attention_scaling,
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
        
        # 共享内存句柄（在 Init 时建立）
        self.shm_rx = None  # 从客户端 TX 读
        self.shm_tx = None  # 向客户端 RX 写
        self.shm_rx_size = 0
        self.shm_tx_size = 0
        self.wire_dtype = "float32"
    
    def _from_wire_dtype(self, array: np.ndarray) -> np.ndarray:
        if self.wire_dtype == "bfloat16":
            # numpy 原生不支持 bfloat16 算子，存储为 np.uint16，转换到 float32
            # 这里假设 array 为 np.uint16 视图
            import torch
            t = torch.from_numpy(array.view(np.uint16)).to(torch.bfloat16)
            return t.to(torch.float32).cpu().numpy()
        return array
    
    def _to_wire_dtype(self, tensor_cpu: torch.Tensor) -> Tuple[memoryview, int, str]:
        if self.wire_dtype == "bfloat16":
            data = tensor_cpu.to(torch.bfloat16).contiguous().view(torch.uint8)
            buf = memoryview(data.numpy())
            return buf, len(buf), "bfloat16"
        else:
            buf = memoryview(tensor_cpu.numpy().view(dtype=np.uint8))
            return buf, len(buf), "float32"
    
    def _read_tensor_from_shm(self, offset: int, nbytes: int, shape: List[int], dtype: str) -> torch.Tensor:
        mv = self.shm_rx.buf[offset:offset + nbytes]
        if dtype == "int64":
            arr = np.frombuffer(mv, dtype=np.int64).reshape(shape)
            return torch.from_numpy(arr.copy())
        elif dtype == "bfloat16":
            arr = np.frombuffer(mv, dtype=np.uint16).reshape(shape)
            arr32 = self._from_wire_dtype(arr)
            return torch.from_numpy(arr32.copy()).to(device=self.compute.device, dtype=torch.float32)
        else:
            arr = np.frombuffer(mv, dtype=np.float32).reshape(shape)
            return torch.from_numpy(arr.copy()).to(device=self.compute.device, dtype=torch.float32)
    
    def _write_tensor_to_shm(self, tensor: torch.Tensor, offset: int) -> int:
        tensor_cpu = tensor.detach().cpu().contiguous()
        buf, nbytes, dtype = self._to_wire_dtype(tensor_cpu)
        self.shm_tx.buf[offset:offset + nbytes] = buf[:nbytes]
        return nbytes, dtype
    
    def _serialize_tensors(self, tensors: List[torch.Tensor]) -> List[Dict]:
        """批量序列化"""
        return [self._serialize_tensor(t) for t in tensors]
    
    def handle_init(self, request: Dict) -> Dict:
        """初始化：映射客户端共享内存，并返回模型元数据"""
        from multiprocessing import shared_memory
        self.shm_rx = shared_memory.SharedMemory(name=request["shm_tx_name"], create=False)
        self.shm_tx = shared_memory.SharedMemory(name=request["shm_rx_name"], create=False)
        self.shm_rx_size = int(request["shm_tx_size"])  # 客户端 TX -> 服务端 RX
        self.shm_tx_size = int(request["shm_rx_size"])  # 服务端 TX -> 客户端 RX
        self.wire_dtype = request.get("wire_dtype", "float32")
        return self.compute.get_init_data()
    
    def handle_embedding(self, request: Dict) -> Dict:
        """Embedding（共享内存传输）"""
        offset = request["offset"]
        nbytes = request["nbytes"]
        shape = request["shape"]
        input_ids = self._read_tensor_from_shm(offset, nbytes, shape, dtype="int64")
        output = self.compute.embedding(input_ids)
        out_bytes, out_dtype = int(np.prod(output.shape) * (2 if self.wire_dtype == "bfloat16" else 4)), self.wire_dtype
        out_offset = request.get("out_offset", 0)
        written_bytes, used_dtype = self._write_tensor_to_shm(output, out_offset)
        return {"offset": out_offset, "nbytes": written_bytes, "shape": list(output.shape), "dtype": used_dtype}
    
    def handle_batch_linear(self, request: Dict) -> Dict:
        """批量 Linear（共享内存传输）"""
        in_offset = request["offset"]
        in_nbytes = request["nbytes"]
        shape = request["shape"]
        hidden_states = self._read_tensor_from_shm(in_offset, in_nbytes, shape, dtype=request.get("dtype", self.wire_dtype))
        outputs = self.compute.batch_linear(
            request["layer_idx"],
            request["module_names"],
            hidden_states
        )
        out_offset = request.get("out_offset", 0)
        out_descs = []
        cur = out_offset
        for t in outputs:
            nbytes, used_dtype = self._write_tensor_to_shm(t, cur)
            out_descs.append({"offset": cur, "nbytes": nbytes, "shape": list(t.shape), "dtype": used_dtype})
            cur += nbytes
        return {"outputs": out_descs}
    
    def handle_matmul(self, request: Dict) -> Dict:
        """矩阵乘法（共享内存传输）"""
        a = self._read_tensor_from_shm(request["a_offset"], request["a_nbytes"], request["a_shape"], request.get("a_dtype", self.wire_dtype))
        b = self._read_tensor_from_shm(request["b_offset"], request["b_nbytes"], request["b_shape"], request.get("b_dtype", self.wire_dtype))
        output = self.compute.matmul(a, b)
        out_offset = request.get("out_offset", 0)
        written_bytes, used_dtype = self._write_tensor_to_shm(output, out_offset)
        return {"offset": out_offset, "nbytes": written_bytes, "shape": list(output.shape), "dtype": used_dtype}
    
    def handle_lm_head(self, request: Dict) -> Dict:
        """LM Head（共享内存传输）"""
        hs = self._read_tensor_from_shm(request["offset"], request["nbytes"], request["shape"], request.get("dtype", self.wire_dtype))
        output = self.compute.lm_head_forward(hs)
        out_offset = request.get("out_offset", 0)
        written_bytes, used_dtype = self._write_tensor_to_shm(output, out_offset)
        return {"offset": out_offset, "nbytes": written_bytes, "shape": list(output.shape), "dtype": used_dtype}
    
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
                message_bytes = self.socket.recv()
                message = msgpack.unpackb(message_bytes, raw=False)
                response = self.handle_request(message)
                response_bytes = msgpack.packb(response, use_bin_type=True)
                self.socket.send(response_bytes)
        except KeyboardInterrupt:
            print("\n✓ Server shutting down...")
        finally:
            try:
                if self.shm_rx is not None:
                    self.shm_rx.close()
                if self.shm_tx is not None:
                    self.shm_tx.close()
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
