"""
高性能 TEE 客户端 - OTP加密版
实现方案：
1. Linear层: 使用加法秘密分享 (X-R)W + RW
   - 在线计算: 每次生成随机掩码R
   - TEE内部用随机权重代替真实权重，计算RW（模拟计算开销）
   - 发送(X-R)到GPU计算(X-R)W（使用真实权重）
   - 恢复: TEE中计算 Y = (X-R)W + RW
2. Matmul层: 使用嵌入式加法外包 (Embedded Additive Outsource)
3. 性能统计: 分离通信开销、GPU计算、TEE计算时间
"""
import os
import time
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager
from multiprocessing import shared_memory

import zmq
import msgpack
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# 配置
PREFILL_TOKEN_LENGTH = 128
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


class ShmRingBuffer:
    """共享内存环形缓冲区 - 客户端"""
    
    def __init__(self, max_chunk_bytes: int, max_chunks: int, name: str):
        """打开已存在的环形缓冲区"""
        self.max_chunk_bytes = max_chunk_bytes
        self.max_chunks = max_chunks
        self.metadata_size = 2
        self.total_bytes = (max_chunk_bytes + self.metadata_size) * max_chunks
        self.data_offset = 0
        self.metadata_offset = max_chunk_bytes * max_chunks
        
        self.shared_memory = shared_memory.SharedMemory(name=name)
        assert self.shared_memory.size == self.total_bytes
        self.current_idx = 0
    
    def close(self):
        if hasattr(self, "shared_memory"):
            self.shared_memory.close()
    
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
                    time.sleep(0)
                    if timeout is not None and time.monotonic() - start_time > timeout:
                        raise TimeoutError("Write timeout")
                    continue
                
                metadata_buffer[0] = 0
                with self.get_data(self.current_idx) as buf:
                    yield buf
                
                metadata_buffer[1] = 0
                metadata_buffer[0] = 1
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
                    time.sleep(0)
                    if timeout is not None and time.monotonic() - start_time > timeout:
                        raise TimeoutError("Read timeout")
                    continue
                
                with self.get_data(self.current_idx) as buf:
                    yield buf
                
                metadata_buffer[1] = 1
                self.current_idx = (self.current_idx + 1) % self.max_chunks
                break


class GPUClient:
    """高性能 GPU 客户端 - OTP加密版"""
    
    def __init__(self, ipc_path: str, log_file: str = "zmq_performance_otp.log") -> None:
        self.ipc_path = ipc_path
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        
        # 优化 ZeroMQ 性能
        self.socket.setsockopt(zmq.SNDHWM, 1000)
        self.socket.setsockopt(zmq.RCVHWM, 1000)
        self.socket.setsockopt(zmq.LINGER, 0)
        
        self.socket.connect(ipc_path)
        
        # 检测传输类型
        self.transport_type = "IPC+SHM+OTP" if "ipc://" in ipc_path else "TCP+OTP"
        print(f"✓ Connected to GPU server at {ipc_path}")
        print(f"  Transport: {self.transport_type}")
        
        # 共享内存环形缓冲区
        self.shm_ring_tx = None
        self.shm_ring_rx = None
        self.max_shm_chunk_bytes = 0
        self.wire_dtype = "float32"
        
        # 性能统计
        self.stats = {
            # 总时间
            "prefill_wall_clock_time": 0.0,  # Prefill从开始到结束的总时间
            
            # 通信开销
            "comm_serialize_time": 0.0,
            "comm_deserialize_time": 0.0,
            "comm_send_time": 0.0,
            "comm_recv_time": 0.0,
            "comm_bytes_sent": 0,
            "comm_bytes_recv": 0,
            
            # GPU计算（服务端返回）
            "gpu_total_time": 0.0,
            
            # RPC统计
            "rpc_count": 0,
            "rpc_total_time": 0.0,
            
            # 操作级别统计
            "linear_count": 0,
            "linear_total_time": 0.0,
            "linear_tee_time": 0.0,
            "linear_comm_time": 0.0,
            "linear_gpu_time": 0.0,
            "matmul_count": 0,
            "matmul_total_time": 0.0,
            "matmul_tee_time": 0.0,
            "matmul_comm_time": 0.0,
            "matmul_gpu_time": 0.0,
            
            # 传输方式统计
            "shm_transfers": 0,
            "zmq_transfers": 0,
            "shm_bytes": 0,
            "zmq_bytes": 0,
        }
        
        # 每次调用的详细记录
        self.call_logs = []
        self.log_file = log_file
        
        # 写入日志头
        with open(self.log_file, 'w') as f:
            f.write(f"ZeroMQ Performance Log (OTP) - Transport: {self.transport_type}\n")
            f.write("="*130 + "\n")
            f.write(f"{'ID':<5} {'Method':<20} {'Ser(ms)':<10} {'Send(ms)':<10} "
                   f"{'Recv(ms)':<10} {'Deser(ms)':<10} {'Comm(ms)':<12} {'GPU(ms)':<10} {'Total(ms)':<12} "
                   f"{'Sent(KB)':<10} {'Recv(KB)':<10}\n")
            f.write("="*130 + "\n")
    
    def _serialize_tensor_to_bytes(self, tensor: torch.Tensor) -> Tuple[bytes, List[int], str]:
        """将张量序列化为字节"""
        tensor_cpu = (tensor if not tensor.is_cuda else tensor.cpu()).contiguous()
        if self.wire_dtype == "bfloat16" and tensor_cpu.dtype == torch.float32:
            data = tensor_cpu.to(torch.bfloat16).view(torch.uint8).numpy()
            return data.tobytes(), list(tensor.shape), "bfloat16"
        elif tensor_cpu.dtype == torch.int64:
            return tensor_cpu.numpy().tobytes(), list(tensor.shape), "int64"
        else:
            data = tensor_cpu.to(torch.float32).numpy()
            return data.tobytes(), list(tensor.shape), "float32"
    
    def _deserialize_tensor_from_bytes(self, data: bytes, shape: List[int], dtype: str) -> torch.Tensor:
        """从字节反序列化张量"""
        if dtype == "int64":
            arr = np.frombuffer(data, dtype=np.int64).reshape(shape)
            return torch.from_numpy(arr.copy())
        elif dtype == "bfloat16":
            arr = np.frombuffer(data, dtype=np.uint16).reshape(shape)
            return torch.from_numpy(arr.copy()).view(torch.bfloat16).to(torch.float32)
        else:
            arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
            return torch.from_numpy(arr.copy())
    
    def _send_tensor(self, tensor: torch.Tensor) -> Dict:
        """发送张量（自动选择共享内存或ZeroMQ）"""
        data, shape, dtype = self._serialize_tensor_to_bytes(tensor)
        data_size = len(data)
        
        if data_size < self.max_shm_chunk_bytes:
            with self.shm_ring_tx.acquire_write(timeout=5.0) as buf:
                buf[:4] = data_size.to_bytes(4, byteorder='little')
                buf[4:4+data_size] = data
            
            self.stats["shm_transfers"] += 1
            self.stats["shm_bytes"] += data_size
            return {"use_shm": True, "shape": shape, "dtype": dtype}
        else:
            self.stats["zmq_transfers"] += 1
            self.stats["zmq_bytes"] += data_size
            return {"use_shm": False, "data": data, "shape": shape, "dtype": dtype}
    
    def _receive_tensor(self, tensor_desc: Dict) -> torch.Tensor:
        """接收张量（自动选择共享内存或ZeroMQ）"""
        use_shm = tensor_desc.get("use_shm", False)
        
        if use_shm:
            with self.shm_ring_rx.acquire_read(timeout=5.0) as buf:
                actual_size = int.from_bytes(buf[:4], byteorder='little')
                data = bytes(buf[4:4+actual_size])
            
            self.stats["shm_transfers"] += 1
            self.stats["shm_bytes"] += actual_size
        else:
            data = tensor_desc["data"]
            self.stats["zmq_transfers"] += 1
            self.stats["zmq_bytes"] += len(data)
        
        return self._deserialize_tensor_from_bytes(data, tensor_desc["shape"], tensor_desc["dtype"])

    def _send_request(self, method: str, request: Dict) -> Tuple[Dict, float, float]:
        """发送请求并返回通信时间和GPU计算时间"""
        call_id = self.stats["rpc_count"] + 1
        self.stats["rpc_count"] = call_id
        
        call_start = time.perf_counter()
        comm_start = time.perf_counter()
        
        # 序列化
        t0 = time.perf_counter()
        message = {"method": method, "request": request}
        message_bytes = msgpack.packb(message, use_bin_type=True)
        serialize_time = time.perf_counter() - t0
        self.stats["comm_serialize_time"] += serialize_time
        
        bytes_sent = len(message_bytes)
        self.stats["comm_bytes_sent"] += bytes_sent
        
        # 发送
        t0 = time.perf_counter()
        self.socket.send(message_bytes)
        send_time = time.perf_counter() - t0
        self.stats["comm_send_time"] += send_time
        
        # 接收
        t0 = time.perf_counter()
        response_bytes = self.socket.recv()
        recv_time = time.perf_counter() - t0
        self.stats["comm_recv_time"] += recv_time
        
        bytes_recv = len(response_bytes)
        self.stats["comm_bytes_recv"] += bytes_recv
        
        # 反序列化
        t0 = time.perf_counter()
        response = msgpack.unpackb(response_bytes, raw=False)
        deserialize_time = time.perf_counter() - t0
        self.stats["comm_deserialize_time"] += deserialize_time
        
        # 通信总时间
        comm_time = time.perf_counter() - comm_start
        
        # 计算总时间
        total_time = time.perf_counter() - call_start
        self.stats["rpc_total_time"] += total_time
        
        # GPU计算时间（服务端返回）
        gpu_time = response.get("response", {}).get("compute_time", 0.0)
        self.stats["gpu_total_time"] += gpu_time
        
        # 记录本次调用
        call_log = {
            "id": call_id,
            "method": method,
            "serialize_ms": serialize_time * 1000,
            "send_ms": send_time * 1000,
            "recv_ms": recv_time * 1000,
            "deserialize_ms": deserialize_time * 1000,
            "comm_ms": comm_time * 1000,
            "gpu_ms": gpu_time * 1000,
            "total_ms": total_time * 1000,
            "bytes_sent": bytes_sent,
            "bytes_recv": bytes_recv,
        }
        self.call_logs.append(call_log)
        
        # 实时写入日志
        with open(self.log_file, 'a') as f:
            f.write(f"{call_id:<5} {method:<20} {serialize_time*1000:<10.3f} {send_time*1000:<10.3f} "
                   f"{recv_time*1000:<10.3f} {deserialize_time*1000:<10.3f} {comm_time*1000:<12.3f} "
                   f"{gpu_time*1000:<10.3f} {total_time*1000:<12.3f} {bytes_sent/1024:<10.2f} {bytes_recv/1024:<10.2f}\n")
        
        if response["status"] == "error":
            print(f"Server error: {response['error']}")
            if "traceback" in response:
                print(response["traceback"])
            raise RuntimeError(f"Server error: {response['error']}")
        
        return response["response"], comm_time, gpu_time
    
    def init(self) -> Dict:
        """初始化：创建共享内存环形缓冲区"""
        meta = {
            "wire_dtype": "bfloat16" if torch.cuda.is_available() else "float32",
            "max_chunks": 10,
        }
        init_data, _, _ = self._send_request("Init", meta)
        
        # 创建环形缓冲区
        self.wire_dtype = meta["wire_dtype"]
        self.max_shm_chunk_bytes = init_data["max_shm_chunk_bytes"]
        
        rx_handle = init_data["shm_ring_rx_handle"]
        self.shm_ring_tx = ShmRingBuffer(rx_handle[0], rx_handle[1], rx_handle[2])
        
        tx_handle = init_data["shm_ring_tx_handle"]
        self.shm_ring_rx = ShmRingBuffer(tx_handle[0], tx_handle[1], tx_handle[2])
        
        print(f"✓ Shared memory ring buffers connected (max_chunk={self.max_shm_chunk_bytes/1024/1024:.1f}MB)")
        return init_data
    
    def embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embedding（无需加密）"""
        request = {"input_ids": self._send_tensor(input_ids)}
        resp, _, _ = self._send_request("Embedding", request)
        return self._receive_tensor(resp["output"])
    
    def batch_linear_otp(self, layer_idx: int, module_names: List[str], hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        批量 Linear - OTP加密版（在线计算）
        方案: Y = XW = (X-R)W + RW
        Online: 
          1. 生成随机掩码 R
          2. TEE内部用随机权重模拟计算 RW
          3. 发送 (X-R) 到GPU，计算 (X-R)W
          4. TEE中恢复 Y = (X-R)W + RW
        """
        op_start = time.perf_counter()
        
        # TEE：生成随机掩码 R 并计算 (X-R)
        R = torch.randn_like(hidden_states)
        masked_input = hidden_states - R
        
        # TEE：计算 RW（在TEE内部计算，使用随机权重代替真实权重）
        # 需要知道输出维度，从layer获取
        RW_outputs = []
        module_output_dims = {
            "q_proj": self.hidden_size,
            "k_proj": self.num_kv_heads * self.head_dim,
            "v_proj": self.num_kv_heads * self.head_dim,
            "o_proj": self.hidden_size,
            "gate_proj": self.intermediate_size,
            "up_proj": self.intermediate_size,
            "down_proj": self.hidden_size,
        }
        
        for module_name in module_names:
            out_features = module_output_dims[module_name]
            # 生成随机权重 W: (out_features, in_features)
            in_features = R.shape[-1]
            W_random = torch.randn(out_features, in_features)
            bias_random = torch.randn(out_features)
            
            # RW = R @ W^T + bias
            RW = R @ W_random.t() + bias_random
            RW_outputs.append(RW)
        
        # 发送 (X-R) 到GPU，计算 (X-R)W
        request = {
            "layer_idx": layer_idx,
            "module_names": module_names,
            "hidden_states": self._send_tensor(masked_input),
        }
        resp, comm_time, gpu_time = self._send_request("BatchLinear", request)
        masked_outputs = [self._receive_tensor(desc) for desc in resp["outputs"]]
        
        # TEE恢复: Y = (X-R)W + RW
        outputs = []
        for i in range(len(module_names)):
            # Y = (X-R)W + RW
            output = masked_outputs[i] + RW_outputs[i]
            outputs.append(output)
        
        # 统计操作级别数据
        op_total_time = time.perf_counter() - op_start
        # TEE时间 = 总时间 - 通信 - GPU（包含了获取权重的通信和RW计算）
        tee_time = op_total_time - comm_time - gpu_time
        
        self.stats["linear_count"] += len(module_names)
        self.stats["linear_total_time"] += op_total_time
        self.stats["linear_tee_time"] += tee_time
        self.stats["linear_comm_time"] += comm_time
        self.stats["linear_gpu_time"] += gpu_time
        
        return outputs
    
    def matmul_otp(self, Q: torch.Tensor, K_T: torch.Tensor, 
                   a: float = None, b: float = None) -> torch.Tensor:
        """
        Matmul - 嵌入式加法外包 (Embedded Additive Outsource)
        
        根据图片方案：
        1. Offline: 采样 R_Q, R_K^T, a, b
                   预计算: aR_Q, bR_K^T
        
        2. Embedded Additive Outsource:
           Q̃ = [Q + R_Q | aR_Q]  (按列拼接，垂直拼接)
           K̃^T = [K^T + R_K^T; bR_K^T]  (按行拼接，水平拼接)
           发送到GPU计算: Q̃K̃^T
        
        3. Recovery:
           Q̃K̃^T = [T1, T2; T3, T4]
           其中:
           T1 = (Q+R_Q)(K^T+R_K^T), T2 = b(Q+R_Q)R_K^T
           T3 = aR_Q(K^T+R_K^T),     T4 = abR_Q R_K^T
           
           通过以下步骤恢复 QK^T:
           R_Q R_K^T = (1/ab) * T4
           QR_K^T = (1/b) * T2 - R_Q R_K^T
           R_Q K^T = (1/a) * T3 - R_Q R_K^T
           QK^T = T1 - R_Q R_K^T - QR_K^T - R_Q K^T
        """
        op_start = time.perf_counter()
        
        # TEE加密：生成随机掩码和标量，预计算
        R_Q = torch.randn_like(Q)
        R_K_T = torch.randn_like(K_T)
        
        # 如果没有提供a, b，随机生成（避免接近0）
        if a is None:
            a = torch.randn(1).item()
            while abs(a) < 0.1:
                a = torch.randn(1).item()
        if b is None:
            b = torch.randn(1).item()
            while abs(b) < 0.1:
                b = torch.randn(1).item()
        
        # 预计算
        aR_Q = a * R_Q
        bR_K_T = b * R_K_T
        
        # Embedded Additive Outsource
        # Q shape: [..., num_heads, seq_q, head_dim]
        # K_T shape: [..., num_heads, head_dim, seq_k]
        
        Q_masked = Q + R_Q
        K_T_masked = K_T + R_K_T
        
        # Q̃ = [Q + R_Q | aR_Q]  垂直拼接（在seq_q维度）
        # shape: [..., num_heads, 2*seq_q, head_dim]
        Q_tilde = torch.cat([Q_masked, aR_Q], dim=-2)
        
        # K̃^T = [K^T + R_K^T; bR_K^T]  水平拼接（在seq_k维度）
        # shape: [..., num_heads, head_dim, 2*seq_k]
        K_T_tilde = torch.cat([K_T_masked, bR_K_T], dim=-1)
        
        # 发送到GPU计算 Q̃K̃^T
        # result shape: [..., num_heads, 2*seq_q, 2*seq_k]
        request = {
            "a": self._send_tensor(Q_tilde),
            "b": self._send_tensor(K_T_tilde),
        }
        resp, comm_time, gpu_time = self._send_request("Matmul", request)
        result_tilde = self._receive_tensor(resp["output"])
        
        # Recovery: 从 Q̃K̃^T 中提取四个块
        # 获取原始维度
        seq_q = Q.shape[-2]
        seq_k = K_T.shape[-1]
        
        # 分块提取
        # result_tilde shape: [..., num_heads, 2*seq_q, 2*seq_k]
        # 分成 2x2 块:
        # [T1  T2]    T1: [:seq_q, :seq_k]     T2: [:seq_q, seq_k:]
        # [T3  T4]    T3: [seq_q:, :seq_k]     T4: [seq_q:, seq_k:]
        
        T1 = result_tilde[..., :seq_q, :seq_k]      # (Q+R_Q)(K^T+R_K^T)
        T2 = result_tilde[..., :seq_q, seq_k:]      # b(Q+R_Q)R_K^T
        T3 = result_tilde[..., seq_q:, :seq_k]      # aR_Q(K^T+R_K^T)
        T4 = result_tilde[..., seq_q:, seq_k:]      # abR_Q R_K^T
        
        # 恢复步骤
        # Step 1: R_Q R_K^T = (1/ab) * T4
        R_Q_R_K_T = T4 / (a * b)
        
        # Step 2: QR_K^T = (1/b) * T2 - R_Q R_K^T
        Q_R_K_T = T2 / b - R_Q_R_K_T
        
        # Step 3: R_Q K^T = (1/a) * T3 - R_Q R_K^T
        R_Q_K_T = T3 / a - R_Q_R_K_T
        
        # Step 4: QK^T = T1 - R_Q R_K^T - QR_K^T - R_Q K^T
        result = T1 - R_Q_R_K_T - Q_R_K_T - R_Q_K_T
        
        # 统计操作级别数据
        op_total_time = time.perf_counter() - op_start
        tee_time = op_total_time - comm_time - gpu_time  # TEE时间 = 总时间 - 通信 - GPU
        
        self.stats["matmul_count"] += 1
        self.stats["matmul_total_time"] += op_total_time
        self.stats["matmul_tee_time"] += tee_time
        self.stats["matmul_comm_time"] += comm_time
        self.stats["matmul_gpu_time"] += gpu_time
        
        return result
    
    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """LM Head（通常不加密，因为是最后一层）"""
        request = {"hidden_states": self._send_tensor(hidden_states)}
        resp, _, _ = self._send_request("LMHead", request)
        return self._receive_tensor(resp["output"])
    
    def print_stats(self):
        """打印统计信息"""
        count = self.stats['rpc_count']
        if count == 0:
            return
        
        print(f"\n{'='*80}")
        print(f"{'GPU Client Statistics (OTP Encryption)':^80}")
        print(f"{'='*80}")
        print(f"Transport Type:   {self.transport_type}")
        print(f"RPC Calls:        {count}")
        
        # 0. Prefill 总时间
        prefill_time = self.stats['prefill_wall_clock_time']
        if prefill_time > 0:
            print(f"\n{'Prefill Wall-Clock Time':^80}")
            print(f"{'─'*80}")
            print(f"  Total Time:     {prefill_time:>10.4f} s")
        
        # 1. 通信时间
        comm_total = (self.stats['comm_serialize_time'] + self.stats['comm_send_time'] + 
                     self.stats['comm_recv_time'] + self.stats['comm_deserialize_time'])
        print(f"\n{'Communication Time (Serialization + Transfer)':^80}")
        print(f"{'─'*80}")
        print(f"  Serialize:      {self.stats['comm_serialize_time']:>10.4f} s  ({self.stats['comm_serialize_time']/count*1000:>8.3f} ms/call)")
        print(f"  Send:           {self.stats['comm_send_time']:>10.4f} s  ({self.stats['comm_send_time']/count*1000:>8.3f} ms/call)")
        print(f"  Receive:        {self.stats['comm_recv_time']:>10.4f} s  ({self.stats['comm_recv_time']/count*1000:>8.3f} ms/call)")
        print(f"  Deserialize:    {self.stats['comm_deserialize_time']:>10.4f} s  ({self.stats['comm_deserialize_time']/count*1000:>8.3f} ms/call)")
        print(f"  {'─'*40}")
        print(f"  Total Comm:     {comm_total:>10.4f} s  ({comm_total/count*1000:>8.3f} ms/call)")
        print(f"  Data Sent:      {self.stats['comm_bytes_sent']/1024/1024:>10.2f} MB")
        print(f"  Data Recv:      {self.stats['comm_bytes_recv']/1024/1024:>10.2f} MB")
        
        # 2. GPU计算时间（服务端测量）
        gpu_total = self.stats['gpu_total_time']
        print(f"\n{'GPU Computation Time (Server-side)':^80}")
        print(f"{'─'*80}")
        print(f"  Total GPU:      {gpu_total:>10.4f} s  ({gpu_total/count*1000:>8.3f} ms/call)")
        print(f"  (纯GPU计算时间，不包括通信)")
        
        # 3. TEE计算时间（通过减法计算：所有客户端计算）
        tee_total = prefill_time - gpu_total - comm_total if prefill_time > 0 else 0.0
        print(f"\n{'TEE Computation Time (All Client-side)':^80}")
        print(f"{'─'*80}")
        print(f"  Total TEE:      {tee_total:>10.4f} s  ({tee_total/count*1000:>8.3f} ms/call)")
        print(f"  (包括: Embedding、RMSNorm、RotaryEmbed、Softmax、加密、恢复等)")
        
        # 4. 操作级别统计
        print(f"\n{'Operation-Level Statistics':^80}")
        print(f"{'─'*80}")
        
        # Linear操作
        if self.stats['linear_count'] > 0:
            linear_avg_total = self.stats['linear_total_time'] / self.stats['linear_count'] * 1000
            linear_avg_tee = self.stats['linear_tee_time'] / self.stats['linear_count'] * 1000
            linear_avg_gpu = self.stats['linear_gpu_time'] / self.stats['linear_count'] * 1000
            linear_avg_comm = self.stats['linear_comm_time'] / self.stats['linear_count'] * 1000
            print(f"  Linear (count={self.stats['linear_count']}):")
            print(f"    Avg Total:   {linear_avg_total:>10.3f} ms (100.0%)")
            print(f"    Avg TEE:     {linear_avg_tee:>10.3f} ms  ({linear_avg_tee/linear_avg_total*100:>5.1f}%)")
            print(f"    Avg GPU:     {linear_avg_gpu:>10.3f} ms  ({linear_avg_gpu/linear_avg_total*100:>5.1f}%)")
            print(f"    Avg Comm:    {linear_avg_comm:>10.3f} ms  ({linear_avg_comm/linear_avg_total*100:>5.1f}%)")
        
        # Matmul操作
        if self.stats['matmul_count'] > 0:
            matmul_avg_total = self.stats['matmul_total_time'] / self.stats['matmul_count'] * 1000
            matmul_avg_tee = self.stats['matmul_tee_time'] / self.stats['matmul_count'] * 1000
            matmul_avg_gpu = self.stats['matmul_gpu_time'] / self.stats['matmul_count'] * 1000
            matmul_avg_comm = self.stats['matmul_comm_time'] / self.stats['matmul_count'] * 1000
            print(f"  Matmul (count={self.stats['matmul_count']}):")
            print(f"    Avg Total:   {matmul_avg_total:>10.3f} ms (100.0%)")
            print(f"    Avg TEE:     {matmul_avg_tee:>10.3f} ms  ({matmul_avg_tee/matmul_avg_total*100:>5.1f}%)")
            print(f"    Avg GPU:     {matmul_avg_gpu:>10.3f} ms  ({matmul_avg_gpu/matmul_avg_total*100:>5.1f}%)")
            print(f"    Avg Comm:    {matmul_avg_comm:>10.3f} ms  ({matmul_avg_comm/matmul_avg_total*100:>5.1f}%)")
        
        # 5. 总览
        print(f"\n{'Overall Breakdown':^80}")
        print(f"{'─'*80}")
        if prefill_time > 0:
            print(f"  Prefill Total:  {prefill_time:>10.4f} s (100.0%)")
            print(f"  TEE Total:      {tee_total:>10.4f} s  ({tee_total/prefill_time*100:>5.1f}%)")
            print(f"  GPU Total:      {gpu_total:>10.4f} s  ({gpu_total/prefill_time*100:>5.1f}%)")
            print(f"  Comm Total:     {comm_total:>10.4f} s  ({comm_total/prefill_time*100:>5.1f}%)")
            print(f"\n  公式: Prefill = TEE + GPU + Comm")
        else:
            print(f"  TEE Total:      {tee_total:>10.4f} s")
            print(f"  GPU Total:      {gpu_total:>10.4f} s")
            print(f"  Comm Total:     {comm_total:>10.4f} s")
        
        # 共享内存 vs ZeroMQ
        total_transfers = self.stats['shm_transfers'] + self.stats['zmq_transfers']
        total_data_bytes = self.stats['shm_bytes'] + self.stats['zmq_bytes']
        if total_transfers > 0:
            print(f"\n{'Transfer Method Breakdown':^80}")
            print(f"{'─'*80}")
            print(f"  Shared Memory:  {self.stats['shm_transfers']:>8} transfers ({self.stats['shm_transfers']/total_transfers*100:>5.1f}%)")
            print(f"                  {self.stats['shm_bytes']/1024/1024:>10.2f} MB ({self.stats['shm_bytes']/total_data_bytes*100:>5.1f}%)")
            print(f"  ZeroMQ:         {self.stats['zmq_transfers']:>8} transfers ({self.stats['zmq_transfers']/total_transfers*100:>5.1f}%)")
            print(f"                  {self.stats['zmq_bytes']/1024/1024:>10.2f} MB ({self.stats['zmq_bytes']/total_data_bytes*100:>5.1f}%)")
        
        print(f"{'='*80}")
        
        # 写入汇总到日志
        with open(self.log_file, 'a') as f:
            f.write("\n" + "="*120 + "\n")
            f.write("SUMMARY (OTP Encryption)\n")
            f.write("="*120 + "\n")
            f.write(f"Total RPC Calls: {count}\n")
            if prefill_time > 0:
                f.write(f"\nPrefill Wall-Clock Time:\n")
                f.write(f"  Total: {prefill_time:.4f} s\n")
            f.write(f"\nCommunication Time:\n")
            f.write(f"  Total: {comm_total:.4f} s ({comm_total/count*1000:.3f} ms/call)\n")
            f.write(f"\nGPU Computation Time:\n")
            f.write(f"  Total: {gpu_total:.4f} s ({gpu_total/count*1000:.3f} ms/call)\n")
            f.write(f"\nTEE Computation Time (Prefill - GPU - Comm):\n")
            f.write(f"  Total: {tee_total:.4f} s ({tee_total/count*1000:.3f} ms/call)\n")
            f.write(f"\nOperation Statistics:\n")
            if self.stats['linear_count'] > 0:
                f.write(f"  Linear (count={self.stats['linear_count']}):\n")
                f.write(f"    Avg Total: {linear_avg_total:.3f} ms\n")
                f.write(f"    Avg TEE:   {linear_avg_tee:.3f} ms\n")
                f.write(f"    Avg GPU:   {linear_avg_gpu:.3f} ms\n")
                f.write(f"    Avg Comm:  {linear_avg_comm:.3f} ms\n")
            if self.stats['matmul_count'] > 0:
                f.write(f"  Matmul (count={self.stats['matmul_count']}):\n")
                f.write(f"    Avg Total: {matmul_avg_total:.3f} ms\n")
                f.write(f"    Avg TEE:   {matmul_avg_tee:.3f} ms\n")
                f.write(f"    Avg GPU:   {matmul_avg_gpu:.3f} ms\n")
                f.write(f"    Avg Comm:  {matmul_avg_comm:.3f} ms\n")
            f.write("="*120 + "\n")
        
        print(f"\n✓ Detailed log saved to: {self.log_file}\n")
    
    def close(self) -> None:
        """关闭连接"""
        try:
            self.socket.close()
            self.context.term()
        finally:
            if self.shm_ring_tx is not None:
                self.shm_ring_tx.close()
            if self.shm_ring_rx is not None:
                self.shm_ring_rx.close()


class TEELlamaModel:
    """TEE 端的 LLaMA 模型 - OTP加密版"""
    
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
        inv_freq_dtype = np.dtype(rotary_params.get("inv_freq_dtype", "float32"))
        inv_freq = np.frombuffer(rotary_params["inv_freq"], dtype=inv_freq_dtype).reshape(rotary_params["inv_freq_shape"])
        inv_freq = torch.from_numpy(inv_freq.copy()).float()
        self.rotary_emb = TEERotaryEmbedding(inv_freq, rotary_params["attention_scaling"])
        
        # RMSNorm 层
        self.input_layernorms = []
        self.post_attention_layernorms = []
        
        for i in range(self.num_layers):
            input_norm = norm_weights[f"layer_{i}_input_layernorm"]
            input_dtype = np.dtype(input_norm.get("dtype", "float32"))
            weight = np.frombuffer(input_norm["weight"], dtype=input_dtype).reshape(input_norm["shape"])
            weight = torch.from_numpy(weight.copy()).float()
            self.input_layernorms.append(TEERMSNorm(weight, input_norm["eps"]))
            
            post_norm = norm_weights[f"layer_{i}_post_attention_layernorm"]
            post_dtype = np.dtype(post_norm.get("dtype", "float32"))
            weight = np.frombuffer(post_norm["weight"], dtype=post_dtype).reshape(post_norm["shape"])
            weight = torch.from_numpy(weight.copy()).float()
            self.post_attention_layernorms.append(TEERMSNorm(weight, post_norm["eps"]))
        
        final_norm = norm_weights["final_norm"]
        final_dtype = np.dtype(final_norm.get("dtype", "float32"))
        weight = np.frombuffer(final_norm["weight"], dtype=final_dtype).reshape(final_norm["shape"])
        weight = torch.from_numpy(weight.copy()).float()
        self.final_norm = TEERMSNorm(weight, final_norm["eps"])
        
        # 性能统计
        self.timing = {
            "tee_rmsnorm": 0.0,
            "tee_rotary": 0.0,
            "tee_softmax": 0.0,
            "tee_silu": 0.0,
            "tee_other": 0.0,
        }
        self.counts = {k: 0 for k in self.timing.keys()}
        
        # 将模型维度信息传递给 gpu_client（用于计算 RW）
        self.gpu.hidden_size = self.hidden_size
        self.gpu.num_kv_heads = self.num_kv_heads
        self.gpu.head_dim = self.head_dim
        self.gpu.intermediate_size = config.get("intermediate_size", self.hidden_size * 4)
        
        print(f"✓ TEE model initialized: {self.num_layers} layers (OTP encryption enabled)")
    
    def attention(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Attention 层 - OTP加密版"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # GPU: QKV projections (使用OTP加密，在线计算RW)
        qkv = self.gpu.batch_linear_otp(layer_idx, ["q_proj", "k_proj", "v_proj"], 
                                       hidden_states)
        
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
        
        # GPU: Q @ K^T (使用完整版嵌入式加法外包)
        attn_weights = self.gpu.matmul_otp(query_states, key_states.transpose(2, 3))
        
        # TEE: Scale + Softmax
        t0 = time.perf_counter()
        attn_weights = attn_weights * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self.timing["tee_softmax"] += time.perf_counter() - t0
        self.counts["tee_softmax"] += 1
        
        # GPU: Attn @ V (使用完整版嵌入式加法外包)
        attn_output = self.gpu.matmul_otp(attn_weights, value_states)
        
        # TEE: Reshape
        t0 = time.perf_counter()
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        self.timing["tee_other"] += time.perf_counter() - t0
        
        # GPU: O projection (使用OTP加密，在线计算RW)
        attn_output = self.gpu.batch_linear_otp(layer_idx, ["o_proj"], attn_output)[0]
        
        return attn_output
    
    def mlp(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP 层 - OTP加密版"""
        # GPU: Gate + Up (使用OTP加密，在线计算RW)
        gate_up = self.gpu.batch_linear_otp(layer_idx, ["gate_proj", "up_proj"], 
                                           hidden_states)
        
        gate, up = gate_up
        
        # TEE: SiLU + multiply
        t0 = time.perf_counter()
        gate = F.silu(gate)
        intermediate = gate * up
        self.timing["tee_silu"] += time.perf_counter() - t0
        self.counts["tee_silu"] += 1
        
        # GPU: Down (使用OTP加密，在线计算RW)
        output = self.gpu.batch_linear_otp(layer_idx, ["down_proj"], intermediate)[0]
        
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
        
        # GPU: Embedding (不加密)
        hidden_states = self.gpu.embedding(input_ids)
        
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
        
        # GPU: LM head (不加密)
        logits = self.gpu.lm_head(hidden_states[:, -1:, :])
        
        return logits
    
    def print_timing_stats(self):
        """打印TEE端性能统计"""
        total_time = sum(self.timing.values())
        
        print(f"\n{'='*70}")
        print(f"{'TEE Model Timing Statistics (OTP)':^70}")
        print(f"{'='*70}")
        print(f"{'Operation':<20} {'Count':>8} {'Total(s)':>12} {'Avg(ms)':>12} {'%':>8}")
        print(f"{'-'*70}")
        
        for op in ["tee_rmsnorm", "tee_rotary", "tee_softmax", "tee_silu", "tee_other"]:
            count = self.counts.get(op, 0)
            t = self.timing[op]
            avg = (t / count * 1000) if count > 0 else 0
            pct = (t / total_time * 100) if total_time > 0 else 0
            name = op.replace("tee_", "").upper()
            print(f"{name:<20} {count:>8} {t:>12.4f} {avg:>12.4f} {pct:>7.2f}%")
        
        print(f"{'-'*70}")
        print(f"{'TOTAL':<20} {'':<8} {total_time:>12.4f} {'':<12} {'100.00':>7}%")
        print(f"{'='*70}\n")


def run_benchmark(model: TEELlamaModel, tokenizer, prefill_length: int) -> float:
    """运行性能测试"""
    input_ids = torch.full((1, prefill_length), tokenizer.pad_token_id, dtype=torch.long)
    
    print(f"\n{'='*70}")
    print(f"{'Prefill Benchmark (OTP Encryption)':^70}")
    print(f"{'='*70}")
    print(f"Token length: {prefill_length}")
    print(f"Encryption: Linear(Additive), Matmul(Embedded Additive)")
    print(f"TEE: Softmax, RMSNorm, RotaryEmbedding, SiLU, Recovery")
    print(f"GPU: Linear, Embedding, Matmul, LM Head (on masked data)")
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
    prefill_start_time = time.perf_counter()
    logits = model.forward(input_ids)
    prefill_elapsed_time = time.perf_counter() - prefill_start_time
    
    # 记录总时间到统计中
    model.gpu.stats["prefill_wall_clock_time"] = prefill_elapsed_time
    
    print(f"\n{'='*70}")
    print(f"Prefill time: {prefill_elapsed_time:.4f}s")
    print(f"Throughput: {prefill_length / prefill_elapsed_time:.2f} tokens/sec")
    print(f"Logits shape: {logits.shape}")
    print(f"{'='*70}")
    
    # 打印详细统计
    model.print_timing_stats()
    model.gpu.print_stats()
    
    return prefill_elapsed_time


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

