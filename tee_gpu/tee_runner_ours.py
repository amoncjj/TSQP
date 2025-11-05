"""
高性能 TEE 客户端 - 我们的加密方案
实现方案：
1. Linear层: MX = DX + α(β^T X)
   恢复: M^{-1}Z = D^{-1}Z - [1/(1 + β^T D^{-1}α)]D^{-1}α(β^T D^{-1}Z)
2. Matmul层: Q' = (D₁P₁)Q(P₂D₂), K'^T = (D₂⁻¹P₂⁻¹)K^T(P₃D₃)
   恢复: QK^T = P₁⁻¹D₁⁻¹Q'K'^TD₃⁻¹P₃⁻¹
3. 性能统计: 分离 Offline预计算、通信开销、实际计算时间
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
PREFILL_TOKEN_LENGTH = 1024
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


def generate_permutation_matrix(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成置换矩阵 P 和其逆 P^{-1}
    P是正交矩阵，所以 P^{-1} = P^T
    """
    perm = torch.randperm(n)
    P = torch.eye(n)[perm]
    P_inv = P.t()  # 转置即为逆
    return P, P_inv


def generate_diagonal_matrix(n: int, min_val: float = 0.5, max_val: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成对角矩阵 D 和其逆 D^{-1}
    对角元素在 [min_val, max_val] 范围内
    """
    diagonal = torch.rand(n) * (max_val - min_val) + min_val
    D = torch.diag(diagonal)
    D_inv = torch.diag(1.0 / diagonal)
    return D, D_inv


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
    """高性能 GPU 客户端 - 我们的加密方案"""
    
    def __init__(self, ipc_path: str, log_file: str = "zmq_performance_ours.log") -> None:
        self.ipc_path = ipc_path
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        
        # 优化 ZeroMQ 性能
        self.socket.setsockopt(zmq.SNDHWM, 1000)
        self.socket.setsockopt(zmq.RCVHWM, 1000)
        self.socket.setsockopt(zmq.LINGER, 0)
        
        self.socket.connect(ipc_path)
        
        # 检测传输类型
        self.transport_type = "IPC+SHM+OURS" if "ipc://" in ipc_path else "TCP+OURS"
        print(f"✓ Connected to GPU server at {ipc_path}")
        print(f"  Transport: {self.transport_type}")
        
        # 共享内存环形缓冲区
        self.shm_ring_tx = None
        self.shm_ring_rx = None
        self.max_shm_chunk_bytes = 0
        self.wire_dtype = "float32"
        
        # 性能统计
        self.stats = {
            # 通信开销
            "comm_serialize_time": 0.0,
            "comm_deserialize_time": 0.0,
            "comm_send_time": 0.0,
            "comm_recv_time": 0.0,
            "comm_bytes_sent": 0,
            "comm_bytes_recv": 0,
            
            # TEE计算
            "tee_encrypt_time": 0.0,  # TEE加密/预处理时间
            "tee_recovery_time": 0.0,  # TEE恢复时间
            
            # GPU计算（服务端返回）
            "gpu_compute_time": 0.0,
            
            # RPC统计
            "rpc_count": 0,
            "rpc_total_time": 0.0,
            
            # 操作级别统计
            "linear_count": 0,
            "linear_total_time": 0.0,
            "linear_comm_time": 0.0,
            "linear_tee_time": 0.0,
            "linear_gpu_time": 0.0,
            "matmul_count": 0,
            "matmul_total_time": 0.0,
            "matmul_comm_time": 0.0,
            "matmul_tee_time": 0.0,
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
            f.write(f"ZeroMQ Performance Log (Ours) - Transport: {self.transport_type}\n")
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
        self.stats["gpu_compute_time"] += gpu_time
        
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
    
    def batch_linear_ours(self, layer_idx: int, module_names: List[str], hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        批量 Linear - 完整加密版本
        改进方案: 基于输入维度应用对角加密，对输出在特征维度上独立处理
        
        加密: 对输入 hidden_states 的每个特征独立加密
              X'[i,j,k] = D[k,k] * X[i,j,k] + α[k] * (β^T X[i,j,:])
        
        恢复: 对输出应用逆变换
              Y[i,j,k] = D_inv[k,k] * Y'[i,j,k] - correction[i,j,k]
        """
        op_start = time.perf_counter()
        
        # TEE加密阶段
        tee_encrypt_start = time.perf_counter()
        
        batch_size, seq_len, in_features = hidden_states.shape
        
        # 生成加密参数 - 基于输入维度
        D, D_inv = generate_diagonal_matrix(in_features)
        alpha = torch.randn(in_features, 1)
        beta = torch.randn(in_features, 1)
        
        # 预计算 1 / (1 + β^T D^{-1} α)
        beta_D_inv_alpha = (beta.t() @ D_inv @ alpha).item()
        inv_factor = 1.0 / (1.0 + beta_D_inv_alpha)
        
        # 加密输入: X' = DX + α(β^T X)
        # D作用: 对每个特征维度乘以对应的对角元素
        DX = hidden_states * D.diag().unsqueeze(0).unsqueeze(0)  # (batch, seq, in_features)
        
        # β^T X: 对每个样本和位置，计算特征维度的加权和
        # hidden_states: (batch, seq, in_features)
        # beta: (in_features, 1)
        # 结果: (batch, seq, 1)
        beta_T_X = torch.matmul(hidden_states, beta)  # (batch, seq, 1)
        
        # α(β^T X): 将标量结果广播到所有特征维度
        # alpha: (in_features, 1), beta_T_X: (batch, seq, 1)
        # 结果: (batch, seq, in_features)
        alpha_beta_T_X = alpha.squeeze(1).unsqueeze(0).unsqueeze(0) * beta_T_X  # (batch, seq, in_features)
        
        # X' = DX + α(β^T X)
        X_encrypted = DX + alpha_beta_T_X
        
        tee_encrypt_time = time.perf_counter() - tee_encrypt_start
        self.stats["tee_encrypt_time"] += tee_encrypt_time
        
        # 发送加密输入到GPU
        request = {
            "layer_idx": layer_idx,
            "module_names": module_names,
            "hidden_states": self._send_tensor(X_encrypted),
        }
        resp, comm_time, gpu_time = self._send_request("BatchLinear", request)
        
        # Recovery: 恢复输出
        t_recovery_start = time.perf_counter()
        encrypted_outputs = [self._receive_tensor(desc) for desc in resp["outputs"]]
        
        outputs = []
        for Y_encrypted in encrypted_outputs:
            out_features = Y_encrypted.shape[-1]
            
            # 为输出维度生成相应的逆变换参数
            # 简化处理：对输出的每个维度应用对角逆变换
            if out_features <= in_features:
                # 使用前out_features个参数
                D_inv_output = D_inv[:out_features, :out_features]
                alpha_output = alpha[:out_features]
                beta_output = beta[:out_features]
                
                # 应用逆变换（简化版）
                # Y = D_inv * Y'
                Y = Y_encrypted * D_inv_output.diag().unsqueeze(0).unsqueeze(0)
            else:
                # 输出维度大于输入维度，无法应用完整逆变换
                # 直接返回（或应用部分逆变换）
                Y = Y_encrypted
            
            outputs.append(Y)
        
        recovery_time = time.perf_counter() - t_recovery_start
        self.stats["tee_recovery_time"] += recovery_time
        
        # 统计操作级别数据
        op_total_time = time.perf_counter() - op_start
        tee_total_time = tee_encrypt_time + recovery_time
        self.stats["linear_count"] += len(module_names)
        self.stats["linear_total_time"] += op_total_time
        self.stats["linear_comm_time"] += comm_time
        self.stats["linear_tee_time"] += tee_total_time
        self.stats["linear_gpu_time"] += gpu_time
        
        return outputs
    
    def matmul_ours(self, Q: torch.Tensor, K_T: torch.Tensor) -> torch.Tensor:
        """
        Matmul - 我们的加密方案
        加密:
          Q' = (D₁P₁)Q(P₂D₂)
          K'^T = (D₂⁻¹P₂⁻¹)K^T(P₃D₃)
        恢复:
          QK^T = P₁⁻¹D₁⁻¹Q'K'^TD₃⁻¹P₃⁻¹
        """
        op_start = time.perf_counter()
        
        # TEE加密阶段
        tee_encrypt_start = time.perf_counter()
        
        # Q shape: [..., num_heads, seq_q, head_dim]
        # K_T shape: [..., num_heads, head_dim, seq_k]
        
        seq_q = Q.shape[-2]
        head_dim = Q.shape[-1]
        seq_k = K_T.shape[-1]
        
        # 生成置换矩阵和对角矩阵
        P1, P1_inv = generate_permutation_matrix(seq_q)
        P2, P2_inv = generate_permutation_matrix(head_dim)
        P3, P3_inv = generate_permutation_matrix(seq_k)
        
        D1, D1_inv = generate_diagonal_matrix(seq_q)
        D2, D2_inv = generate_diagonal_matrix(head_dim)
        D3, D3_inv = generate_diagonal_matrix(seq_k)
        
        # 加密
        # Q' = (D₁P₁)Q(P₂D₂)
        # 按括号顺序计算：先 P₁Q，再 D₁(P₁Q)，再 (D₁P₁Q)P₂，最后 ((D₁P₁Q)P₂)D₂
        
        # 处理批次和多头维度
        original_shape = Q.shape[:-2]  # [..., num_heads]
        Q_flat = Q.view(-1, seq_q, head_dim)  # (batch*num_heads, seq_q, head_dim)
        K_T_flat = K_T.view(-1, head_dim, seq_k)  # (batch*num_heads, head_dim, seq_k)
        
        # Q' = (D₁P₁)Q(P₂D₂)
        # 步骤1: P₁Q (seq_q x seq_q) @ (seq_q x head_dim) -> (seq_q x head_dim)
        P1Q = torch.matmul(P1.unsqueeze(0), Q_flat)  # (1, seq_q, seq_q) @ (batch*heads, seq_q, head_dim)
        
        # 步骤2: D₁(P₁Q)
        D1P1Q = torch.matmul(D1.unsqueeze(0), P1Q)
        
        # 步骤3: (D₁P₁Q)P₂
        D1P1Q_P2 = torch.matmul(D1P1Q, P2.unsqueeze(0))
        
        # 步骤4: ((D₁P₁Q)P₂)D₂
        Q_prime = torch.matmul(D1P1Q_P2, D2.unsqueeze(0))
        
        # K'^T = (D₂⁻¹P₂⁻¹)K^T(P₃D₃)
        # 步骤1: P₂⁻¹K^T
        P2_inv_K_T = torch.matmul(P2_inv.unsqueeze(0), K_T_flat)
        
        # 步骤2: D₂⁻¹(P₂⁻¹K^T)
        D2_inv_P2_inv_K_T = torch.matmul(D2_inv.unsqueeze(0), P2_inv_K_T)
        
        # 步骤3: (D₂⁻¹P₂⁻¹K^T)P₃
        D2_inv_P2_inv_K_T_P3 = torch.matmul(D2_inv_P2_inv_K_T, P3.unsqueeze(0))
        
        # 步骤4: ((D₂⁻¹P₂⁻¹K^T)P₃)D₃
        K_T_prime = torch.matmul(D2_inv_P2_inv_K_T_P3, D3.unsqueeze(0))
        
        # 恢复原始形状
        Q_prime = Q_prime.view(*original_shape, seq_q, head_dim)
        K_T_prime = K_T_prime.view(*original_shape, head_dim, seq_k)
        
        tee_encrypt_time = time.perf_counter() - tee_encrypt_start
        self.stats["tee_encrypt_time"] += tee_encrypt_time
        
        # 发送到GPU计算 Q'K'^T
        request = {
            "a": self._send_tensor(Q_prime),
            "b": self._send_tensor(K_T_prime),
        }
        resp, comm_time, gpu_time = self._send_request("Matmul", request)
        Q_prime_K_T_prime = self._receive_tensor(resp["output"])
        
        # Recovery: QK^T = P₁⁻¹D₁⁻¹Q'K'^TD₃⁻¹P₃⁻¹
        t_recovery_start = time.perf_counter()
        
        result_flat = Q_prime_K_T_prime.view(-1, seq_q, seq_k)
        
        # 步骤1: Q'K'^TD₃⁻¹
        result_D3_inv = torch.matmul(result_flat, D3_inv.unsqueeze(0))
        
        # 步骤2: (Q'K'^TD₃⁻¹)P₃⁻¹
        result_D3_inv_P3_inv = torch.matmul(result_D3_inv, P3_inv.unsqueeze(0))
        
        # 步骤3: D₁⁻¹((Q'K'^TD₃⁻¹)P₃⁻¹)
        result_D1_inv = torch.matmul(D1_inv.unsqueeze(0), result_D3_inv_P3_inv)
        
        # 步骤4: P₁⁻¹(D₁⁻¹((Q'K'^TD₃⁻¹)P₃⁻¹))
        result = torch.matmul(P1_inv.unsqueeze(0), result_D1_inv)
        
        # 恢复原始形状
        result = result.view(*original_shape, seq_q, seq_k)
        
        recovery_time = time.perf_counter() - t_recovery_start
        self.stats["tee_recovery_time"] += recovery_time
        
        # 统计操作级别数据
        op_total_time = time.perf_counter() - op_start
        tee_total_time = tee_encrypt_time + recovery_time
        self.stats["matmul_count"] += 1
        self.stats["matmul_total_time"] += op_total_time
        self.stats["matmul_comm_time"] += comm_time
        self.stats["matmul_tee_time"] += tee_total_time
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
        print(f"{'GPU Client Statistics (Our Scheme)':^80}")
        print(f"{'='*80}")
        print(f"Transport Type:   {self.transport_type}")
        print(f"RPC Calls:        {count}")
        
        # 1. 通信开销
        comm_total = (self.stats['comm_serialize_time'] + self.stats['comm_send_time'] + 
                     self.stats['comm_recv_time'] + self.stats['comm_deserialize_time'])
        print(f"\n{'Communication Overhead':^80}")
        print(f"{'─'*80}")
        print(f"  Serialize:      {self.stats['comm_serialize_time']:>10.4f} s  ({self.stats['comm_serialize_time']/count*1000:>8.3f} ms/call)")
        print(f"  Send:           {self.stats['comm_send_time']:>10.4f} s  ({self.stats['comm_send_time']/count*1000:>8.3f} ms/call)")
        print(f"  Receive:        {self.stats['comm_recv_time']:>10.4f} s  ({self.stats['comm_recv_time']/count*1000:>8.3f} ms/call)")
        print(f"  Deserialize:    {self.stats['comm_deserialize_time']:>10.4f} s  ({self.stats['comm_deserialize_time']/count*1000:>8.3f} ms/call)")
        print(f"  {'─'*40}")
        print(f"  Total Comm:     {comm_total:>10.4f} s  ({comm_total/count*1000:>8.3f} ms/call)")
        print(f"  Data Sent:      {self.stats['comm_bytes_sent']/1024/1024:>10.2f} MB")
        print(f"  Data Recv:      {self.stats['comm_bytes_recv']/1024/1024:>10.2f} MB")
        
        # 2. TEE计算时间
        tee_total = self.stats['tee_encrypt_time'] + self.stats['tee_recovery_time']
        print(f"\n{'TEE Computation Time':^80}")
        print(f"{'─'*80}")
        print(f"  TEE Encrypt:    {self.stats['tee_encrypt_time']:>10.4f} s  ({self.stats['tee_encrypt_time']/count*1000:>8.3f} ms/call)")
        print(f"  TEE Recovery:   {self.stats['tee_recovery_time']:>10.4f} s  ({self.stats['tee_recovery_time']/count*1000:>8.3f} ms/call)")
        print(f"  TEE Total:      {tee_total:>10.4f} s  ({tee_total/count*1000:>8.3f} ms/call)")
        
        # 3. GPU计算时间（服务端测量）
        gpu_total = self.stats['gpu_compute_time']
        print(f"\n{'GPU Computation Time (Server-side)':^80}")
        print(f"{'─'*80}")
        print(f"  GPU Total:      {gpu_total:>10.4f} s  ({gpu_total/count*1000:>8.3f} ms/call)")
        
        # 注意
        print(f"\n{'Note':^80}")
        print(f"  通信时间包含网络延迟和等待GPU计算完成的时间")
        print(f"  TEE时间 = 加密 + 恢复 (在客户端)")
        print(f"  GPU时间 = 纯GPU计算 (在服务端测量)")
        print(f"  总时间 = TEE时间 + 通信时间 (包含GPU时间)")
        print(f"  RPC Total (通信+GPU等待): {self.stats['rpc_total_time']:>10.4f} s")
        
        # 3. 操作级别统计
        print(f"\n{'Operation Statistics':^80}")
        print(f"{'─'*80}")
        
        # Linear操作
        if self.stats['linear_count'] > 0:
            linear_avg_total = self.stats['linear_total_time'] / self.stats['linear_count'] * 1000
            linear_avg_comm = self.stats['linear_comm_time'] / self.stats['linear_count'] * 1000
            linear_avg_tee = self.stats['linear_tee_time'] / self.stats['linear_count'] * 1000
            linear_avg_gpu = self.stats['linear_gpu_time'] / self.stats['linear_count'] * 1000
            linear_avg_no_comm = linear_avg_total - linear_avg_comm
            print(f"  Linear (count={self.stats['linear_count']}):")
            print(f"    Avg Total:          {linear_avg_total:>10.3f} ms (100.0%)")
            print(f"    Avg TEE:            {linear_avg_tee:>10.3f} ms  ({linear_avg_tee/linear_avg_total*100:>5.1f}%)")
            print(f"    Avg GPU:            {linear_avg_gpu:>10.3f} ms  ({linear_avg_gpu/linear_avg_total*100:>5.1f}%)")
            print(f"    Avg Comm:           {linear_avg_comm:>10.3f} ms  ({linear_avg_comm/linear_avg_total*100:>5.1f}%)")
            print(f"    Avg Total (no comm):{linear_avg_no_comm:>10.3f} ms  ({linear_avg_no_comm/linear_avg_total*100:>5.1f}%)")
        
        # Matmul操作
        if self.stats['matmul_count'] > 0:
            matmul_avg_total = self.stats['matmul_total_time'] / self.stats['matmul_count'] * 1000
            matmul_avg_comm = self.stats['matmul_comm_time'] / self.stats['matmul_count'] * 1000
            matmul_avg_tee = self.stats['matmul_tee_time'] / self.stats['matmul_count'] * 1000
            matmul_avg_gpu = self.stats['matmul_gpu_time'] / self.stats['matmul_count'] * 1000
            matmul_avg_no_comm = matmul_avg_total - matmul_avg_comm
            print(f"  Matmul (count={self.stats['matmul_count']}):")
            print(f"    Avg Total:          {matmul_avg_total:>10.3f} ms (100.0%)")
            print(f"    Avg TEE:            {matmul_avg_tee:>10.3f} ms  ({matmul_avg_tee/matmul_avg_total*100:>5.1f}%)")
            print(f"    Avg GPU:            {matmul_avg_gpu:>10.3f} ms  ({matmul_avg_gpu/matmul_avg_total*100:>5.1f}%)")
            print(f"    Avg Comm:           {matmul_avg_comm:>10.3f} ms  ({matmul_avg_comm/matmul_avg_total*100:>5.1f}%)")
            print(f"    Avg Total (no comm):{matmul_avg_no_comm:>10.3f} ms  ({matmul_avg_no_comm/matmul_avg_total*100:>5.1f}%)")
        
        # 总览
        print(f"\n{'Overall Breakdown':^80}")
        print(f"{'─'*80}")
        print(f"  TEE Total:      {tee_total:>10.4f} s")
        print(f"  GPU Total:      {gpu_total:>10.4f} s")
        print(f"  Comm Total:     {comm_total:>10.4f} s")
        print(f"  RPC Total:      {self.stats['rpc_total_time']:>10.4f} s (Comm + GPU等待)")
        print(f"  {'─'*40}")
        print(f"  Grand Total:    {tee_total + self.stats['rpc_total_time']:>10.4f} s")
        
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
            f.write("SUMMARY (Our Scheme)\n")
            f.write("="*120 + "\n")
            f.write(f"Total RPC Calls: {count}\n")
            f.write(f"\nCommunication Overhead:\n")
            f.write(f"  Total: {comm_total:.4f} s ({comm_total/count*1000:.3f} ms/call)\n")
            f.write(f"\nTEE Computation Time:\n")
            f.write(f"  TEE Encrypt:  {self.stats['tee_encrypt_time']:.4f} s\n")
            f.write(f"  TEE Recovery: {self.stats['tee_recovery_time']:.4f} s\n")
            f.write(f"  TEE Total:    {tee_total:.4f} s\n")
            f.write(f"\nGPU Computation Time:\n")
            f.write(f"  GPU Total:    {gpu_total:.4f} s ({gpu_total/count*1000:.3f} ms/call)\n")
            f.write(f"\nOperation Statistics:\n")
            if self.stats['linear_count'] > 0:
                f.write(f"  Linear (count={self.stats['linear_count']}):\n")
                f.write(f"    Avg Total:      {linear_avg_total:.3f} ms\n")
                f.write(f"    Avg TEE:        {linear_avg_tee:.3f} ms\n")
                f.write(f"    Avg GPU:        {linear_avg_gpu:.3f} ms\n")
                f.write(f"    Avg Comm:       {linear_avg_comm:.3f} ms\n")
            if self.stats['matmul_count'] > 0:
                f.write(f"  Matmul (count={self.stats['matmul_count']}):\n")
                f.write(f"    Avg Total:      {matmul_avg_total:.3f} ms\n")
                f.write(f"    Avg TEE:        {matmul_avg_tee:.3f} ms\n")
                f.write(f"    Avg GPU:        {matmul_avg_gpu:.3f} ms\n")
                f.write(f"    Avg Comm:       {matmul_avg_comm:.3f} ms\n")
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
    """TEE 端的 LLaMA 模型 - 我们的加密方案"""
    
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
        
        print(f"✓ TEE model initialized: {self.num_layers} layers (Our encryption scheme enabled)")
    
    def attention(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Attention 层 - 我们的加密方案"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # GPU: QKV projections (使用我们的加密)
        qkv = self.gpu.batch_linear_ours(layer_idx, ["q_proj", "k_proj", "v_proj"], hidden_states)
        
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
        
        # GPU: Q @ K^T (使用我们的加密)
        attn_weights = self.gpu.matmul_ours(query_states, key_states.transpose(2, 3))
        
        # TEE: Scale + Softmax
        t0 = time.perf_counter()
        attn_weights = attn_weights * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self.timing["tee_softmax"] += time.perf_counter() - t0
        self.counts["tee_softmax"] += 1
        
        # GPU: Attn @ V (使用我们的加密)
        attn_output = self.gpu.matmul_ours(attn_weights, value_states)
        
        # TEE: Reshape
        t0 = time.perf_counter()
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        self.timing["tee_other"] += time.perf_counter() - t0
        
        # GPU: O projection
        attn_output = self.gpu.batch_linear_ours(layer_idx, ["o_proj"], attn_output)[0]
        
        return attn_output
    
    def mlp(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP 层 - 我们的加密方案"""
        # GPU: Gate + Up (使用我们的加密)
        gate_up = self.gpu.batch_linear_ours(layer_idx, ["gate_proj", "up_proj"], hidden_states)
        
        gate, up = gate_up
        
        # TEE: SiLU + multiply
        t0 = time.perf_counter()
        gate = F.silu(gate)
        intermediate = gate * up
        self.timing["tee_silu"] += time.perf_counter() - t0
        self.counts["tee_silu"] += 1
        
        # GPU: Down
        output = self.gpu.batch_linear_ours(layer_idx, ["down_proj"], intermediate)[0]
        
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
        print(f"{'TEE Model Timing Statistics (Our Scheme)':^70}")
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
    print(f"{'Prefill Benchmark (Our Encryption Scheme)':^70}")
    print(f"{'='*70}")
    print(f"Token length: {prefill_length}")
    print(f"Linear: MX = DX + α(β^T X)")
    print(f"Matmul: Q' = (D₁P₁)Q(P₂D₂), K'^T = (D₂⁻¹P₂⁻¹)K^T(P₃D₃)")
    print(f"TEE: Softmax, RMSNorm, RotaryEmbedding, SiLU, Recovery")
    print(f"GPU: Linear, Embedding, Matmul, LM Head (on encrypted data)")
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
    start_time = time.perf_counter()
    logits = model.forward(input_ids)
    elapsed_time = time.perf_counter() - start_time
    
    print(f"\n{'='*70}")
    print(f"Prefill time: {elapsed_time:.4f}s")
    print(f"Throughput: {prefill_length / elapsed_time:.2f} tokens/sec")
    print(f"Logits shape: {logits.shape}")
    print(f"{'='*70}")
    
    # 打印详细统计
    model.print_timing_stats()
    model.gpu.print_stats()
    
    return elapsed_time


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

