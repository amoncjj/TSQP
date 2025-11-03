"""
高性能 GPU 服务端 - 优化版
关键优化：
1. 使用 IPC 而不是 TCP
2. 零拷贝数据传输（使用共享内存）
3. 保持数据在 GPU 上
4. 批量操作
5. 小数据(<10MB)使用共享内存环形缓冲区，大数据使用ZeroMQ
"""
import os
import mmap
import struct
import time
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager
from multiprocessing import shared_memory

import zmq
import msgpack
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM

# 配置
DEFAULT_MODEL_PATH = "/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DTYPE = "float16"
DEFAULT_IPC_PATH = "ipc:///tmp/tsqp_gpu_server.ipc"
MAX_SHM_CHUNK_BYTES = 10 * 1024 * 1024  # 10MB 阈值

TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
}


class ShmRingBuffer:
    """共享内存环形缓冲区 - 用于小数据传输"""
    
    def __init__(self, max_chunk_bytes: int, max_chunks: int, name: Optional[str] = None):
        """
        初始化环形缓冲区
        
        Buffer memory layout:
                  data                                 metadata
                    |                                      |
                    | (current_idx)                        | (current_idx)
                    v                                      v
        +-------------------------------+----------------------------------------+
        | chunk0 | chunk1 | ... | chunk | metadata0 | metadata1 | ... | metadata |
        +-------------------------------+----------------------------------------+
        | max_chunks x max_chunk_bytes  | max_chunks x 2 bytes (writer+reader)   |
        
        metadata: [written_flag, read_flag]
        """
        self.max_chunk_bytes = max_chunk_bytes
        self.max_chunks = max_chunks
        self.metadata_size = 2  # written_flag + read_flag
        self.total_bytes = (max_chunk_bytes + self.metadata_size) * max_chunks
        self.data_offset = 0
        self.metadata_offset = max_chunk_bytes * max_chunks
        
        if name is None:
            # 创建新缓冲区
            self.is_creator = True
            self.shared_memory = shared_memory.SharedMemory(create=True, size=self.total_bytes)
            # 初始化元数据为0
            with memoryview(self.shared_memory.buf[self.metadata_offset:]) as metadata_buffer:
                torch.frombuffer(metadata_buffer, dtype=torch.uint8).fill_(0)
        else:
            # 打开已存在的缓冲区
            self.is_creator = False
            self.shared_memory = shared_memory.SharedMemory(name=name)
            assert self.shared_memory.size == self.total_bytes
        
        self.current_idx = 0
    
    def handle(self):
        return (self.max_chunk_bytes, self.max_chunks, self.shared_memory.name)
    
    def __del__(self):
        if hasattr(self, "shared_memory"):
            try:
                self.shared_memory.close()
            except Exception:
                pass  # 忽略关闭时的错误
            if self.is_creator:
                try:
                    self.shared_memory.unlink()
                except FileNotFoundError:
                    pass  # 共享内存已经被清理，忽略
                except Exception:
                    pass  # 忽略其他清理错误
    
    @contextmanager
    def get_data(self, current_idx: int):
        start = self.data_offset + current_idx * self.max_chunk_bytes
        end = start + self.max_chunk_bytes
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf
    
    @contextmanager
    def get_metadata(self, current_idx: int):
        start = self.metadata_offset + current_idx * self.metadata_size
        end = start + self.metadata_size
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf
    
    @contextmanager
    def acquire_write(self, timeout: Optional[float] = None):
        """获取写权限"""
        start_time = time.monotonic()
        while True:
            with self.get_metadata(self.current_idx) as metadata_buffer:
                written_flag = metadata_buffer[0]
                read_flag = metadata_buffer[1]
                
                if written_flag and not read_flag:
                    # 已写入但未读取，等待
                    time.sleep(0)
                    if timeout is not None and time.monotonic() - start_time > timeout:
                        raise TimeoutError("Write timeout")
                    continue
                
                # 可以写入
                metadata_buffer[0] = 0  # 标记为未写入
                with self.get_data(self.current_idx) as buf:
                    yield buf
                
                # 写入完成
                metadata_buffer[1] = 0  # 标记为未读取
                metadata_buffer[0] = 1  # 标记为已写入
                self.current_idx = (self.current_idx + 1) % self.max_chunks
                break
    
    @contextmanager
    def acquire_read(self, timeout: Optional[float] = None):
        """获取读权限"""
        start_time = time.monotonic()
        while True:
            with self.get_metadata(self.current_idx) as metadata_buffer:
                written_flag = metadata_buffer[0]
                read_flag = metadata_buffer[1]
                
                if not written_flag or read_flag:
                    # 未写入或已读取，等待
                    time.sleep(0)
                    if timeout is not None and time.monotonic() - start_time > timeout:
                        raise TimeoutError("Read timeout")
                    continue
                
                # 可以读取
                with self.get_data(self.current_idx) as buf:
                    yield buf
                
                # 读取完成
                metadata_buffer[1] = 1  # 标记为已读取
                self.current_idx = (self.current_idx + 1) % self.max_chunks
                break


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
    """高性能 ZeroMQ 服务器 - 支持共享内存环形缓冲区"""
    
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
        
        # 共享内存环形缓冲区（在 Init 时建立）
        self.shm_ring_tx = None  # 服务端->客户端的环形缓冲区
        self.shm_ring_rx = None  # 客户端->服务端的环形缓冲区
        self.wire_dtype = "float32"
        
        # 统计信息
        self.stats = {
            "shm_transfers": 0,
            "zmq_transfers": 0,
            "shm_bytes": 0,
            "zmq_bytes": 0,
        }
    
    def _serialize_tensor_to_bytes(self, tensor: torch.Tensor) -> Tuple[bytes, List[int], str]:
        """将张量序列化为字节"""
        tensor_cpu = tensor.detach().cpu().contiguous()
        if self.wire_dtype == "bfloat16":
            data = tensor_cpu.to(torch.bfloat16).contiguous().view(torch.uint8).numpy()
            return data.tobytes(), list(tensor.shape), "bfloat16"
        else:
            data = tensor_cpu.to(torch.float32).numpy()
            return data.tobytes(), list(tensor.shape), "float32"
    
    def _deserialize_tensor_from_bytes(self, data: bytes, shape: List[int], dtype: str) -> torch.Tensor:
        """从字节反序列化张量"""
        if dtype == "int64":
            arr = np.frombuffer(data, dtype=np.int64).reshape(shape)
            return torch.from_numpy(arr.copy()).to(device=self.compute.device)
        elif dtype == "bfloat16":
            arr = np.frombuffer(data, dtype=np.uint16).reshape(shape)
            t = torch.from_numpy(arr.copy()).view(torch.bfloat16).to(torch.float32)
            return t.to(device=self.compute.device)
        else:
            arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
            return torch.from_numpy(arr.copy()).to(device=self.compute.device)
    
    def handle_init(self, request: Dict) -> Dict:
        """初始化：创建共享内存环形缓冲区，并返回模型元数据"""
        self.wire_dtype = request.get("wire_dtype", "float32")
        
        # 创建环形缓冲区
        max_chunks = request.get("max_chunks", 10)
        self.shm_ring_tx = ShmRingBuffer(MAX_SHM_CHUNK_BYTES, max_chunks)  # 服务端->客户端
        self.shm_ring_rx = ShmRingBuffer(MAX_SHM_CHUNK_BYTES, max_chunks)  # 客户端->服务端
        
        init_data = self.compute.get_init_data()
        init_data["shm_ring_tx_handle"] = self.shm_ring_tx.handle()
        init_data["shm_ring_rx_handle"] = self.shm_ring_rx.handle()
        init_data["max_shm_chunk_bytes"] = MAX_SHM_CHUNK_BYTES
        
        print(f"✓ Shared memory ring buffers created (max_chunk={MAX_SHM_CHUNK_BYTES/1024/1024:.1f}MB, chunks={max_chunks})")
        return init_data
    
    def _receive_tensor(self, tensor_desc: Dict) -> torch.Tensor:
        """接收张量（自动选择共享内存或ZeroMQ）"""
        use_shm = tensor_desc.get("use_shm", False)
        
        if use_shm:
            # 从共享内存读取
            with self.shm_ring_rx.acquire_read(timeout=5.0) as buf:
                # 读取实际数据大小
                actual_size = int.from_bytes(buf[:4], byteorder='little')
                data = bytes(buf[4:4+actual_size])
            
            self.stats["shm_transfers"] += 1
            self.stats["shm_bytes"] += actual_size
        else:
            # 从ZeroMQ读取（已经在request中）
            data = tensor_desc["data"]
            self.stats["zmq_transfers"] += 1
            self.stats["zmq_bytes"] += len(data)
        
        return self._deserialize_tensor_from_bytes(data, tensor_desc["shape"], tensor_desc["dtype"])
    
    def _send_tensor(self, tensor: torch.Tensor) -> Dict:
        """发送张量（自动选择共享内存或ZeroMQ）"""
        data, shape, dtype = self._serialize_tensor_to_bytes(tensor)
        data_size = len(data)
        
        if data_size < MAX_SHM_CHUNK_BYTES:
            # 使用共享内存
            with self.shm_ring_tx.acquire_write(timeout=5.0) as buf:
                # 写入数据大小（4字节）+ 数据
                buf[:4] = data_size.to_bytes(4, byteorder='little')
                buf[4:4+data_size] = data
            
            self.stats["shm_transfers"] += 1
            self.stats["shm_bytes"] += data_size
            return {"use_shm": True, "shape": shape, "dtype": dtype}
        else:
            # 使用ZeroMQ
            self.stats["zmq_transfers"] += 1
            self.stats["zmq_bytes"] += data_size
            return {"use_shm": False, "data": data, "shape": shape, "dtype": dtype}
    
    def handle_embedding(self, request: Dict) -> Dict:
        """Embedding"""
        input_ids = self._receive_tensor(request["input_ids"])
        output = self.compute.embedding(input_ids)
        return {"output": self._send_tensor(output)}
    
    def handle_batch_linear(self, request: Dict) -> Dict:
        """批量 Linear"""
        hidden_states = self._receive_tensor(request["hidden_states"])
        outputs = self.compute.batch_linear(
            request["layer_idx"],
            request["module_names"],
            hidden_states
        )
        return {"outputs": [self._send_tensor(t) for t in outputs]}
    
    def handle_matmul(self, request: Dict) -> Dict:
        """矩阵乘法"""
        a = self._receive_tensor(request["a"])
        b = self._receive_tensor(request["b"])
        output = self.compute.matmul(a, b)
        return {"output": self._send_tensor(output)}
    
    def handle_lm_head(self, request: Dict) -> Dict:
        """LM Head"""
        hidden_states = self._receive_tensor(request["hidden_states"])
        output = self.compute.lm_head_forward(hidden_states)
        return {"output": self._send_tensor(output)}
    
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
            self._print_stats()
        finally:
            self.socket.close()
            self.context.term()
    
    def _print_stats(self):
        """打印统计信息"""
        total_transfers = self.stats["shm_transfers"] + self.stats["zmq_transfers"]
        total_bytes = self.stats["shm_bytes"] + self.stats["zmq_bytes"]
        
        if total_transfers == 0:
            return
        
        print(f"\n{'='*70}")
        print(f"{'Server Transfer Statistics':^70}")
        print(f"{'='*70}")
        print(f"Shared Memory Transfers: {self.stats['shm_transfers']:>8} ({self.stats['shm_transfers']/total_transfers*100:>5.1f}%)")
        print(f"  Data transferred:      {self.stats['shm_bytes']/1024/1024:>8.2f} MB")
        print(f"ZeroMQ Transfers:        {self.stats['zmq_transfers']:>8} ({self.stats['zmq_transfers']/total_transfers*100:>5.1f}%)")
        print(f"  Data transferred:      {self.stats['zmq_bytes']/1024/1024:>8.2f} MB")
        print(f"{'─'*70}")
        print(f"Total Transfers:         {total_transfers:>8}")
        print(f"Total Data:              {total_bytes/1024/1024:>8.2f} MB")
        print(f"{'='*70}\n")


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
