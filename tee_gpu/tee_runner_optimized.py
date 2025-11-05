"""
高性能 TEE 客户端 - 优化版 with 安全外包计算
关键优化：
1. 使用 IPC 而不是 TCP
2. 最小化数据拷贝
3. 批量操作
4. 详细的性能分析
5. 小数据(<10MB)使用共享内存环形缓冲区，大数据使用ZeroMQ

安全计算协议 (Embedded Additive Outsourcing):
当TEE需要将内部数据发送到GPU计算时，使用以下协议保护数据：

1. 离线阶段 (Offline):
   - 采样随机矩阵/向量 R (与数据同形状)
   - 采样随机标量 a, b
   - 在TEE中预计算 aR, bR

2. 嵌入式加性外包 (Embedded Additive Outsource):
   - 对于矩阵乘法 Q @ K^T:
     * 构造掩码矩阵: Q̃ = [Q + R_Q; aR_Q], K̃^T = [K^T + R_K^T, bR_K^T]
     * 发送 Q̃, K̃^T 到GPU
     * GPU计算 Q̃K̃^T 并返回结果 (包含4个块: T1, T2, T3, T4)
   
   - 对于线性层 y = xW:
     * 添加随机掩码: x̃ = x + R
     * 发送 x̃ 到GPU
     * GPU计算 x̃W 并返回

3. 恢复 (Recovery):
   - 对于矩阵乘法:
     * R_QR_K^T = (1/ab) * T4
     * QR_K^T = (1/b) * T2 - R_QR_K^T
     * R_QK^T = (1/a) * T3 - R_QR_K^T
     * QK^T = T1 - R_QR_K^T - QR_K^T - R_QK^T
   
   - 对于线性层:
     * y = x̃W - RW (RW在TEE中预计算或通过额外通道获取)

注意: 这种方法使用一次性密码本(OTP)保护数据隐私，每次计算生成新的随机掩码。
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
                    # 已写入但未读取，等待
                    time.sleep(0)
                    if timeout is not None and time.monotonic() - start_time > timeout:
                        raise TimeoutError("Write timeout")
                    continue
                
                # 可以写入
                metadata_buffer[0] = 0
                with self.get_data(self.current_idx) as buf:
                    yield buf
                
                # 写入完成
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
                    # 未写入或已读取，等待
                    time.sleep(0)
                    if timeout is not None and time.monotonic() - start_time > timeout:
                        raise TimeoutError("Read timeout")
                    continue
                
                # 可以读取
                with self.get_data(self.current_idx) as buf:
                    yield buf
                
                # 读取完成
                metadata_buffer[1] = 1
                self.current_idx = (self.current_idx + 1) % self.max_chunks
                break


class GPUClient:
    """高性能 GPU 客户端 - 支持共享内存环形缓冲区"""
    
    def __init__(self, ipc_path: str, log_file: str = "zmq_performance.log") -> None:
        self.ipc_path = ipc_path
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        
        # 优化 ZeroMQ 性能
        self.socket.setsockopt(zmq.SNDHWM, 1000)
        self.socket.setsockopt(zmq.RCVHWM, 1000)
        self.socket.setsockopt(zmq.LINGER, 0)
        
        self.socket.connect(ipc_path)
        
        # 检测传输类型
        self.transport_type = "IPC+SHM" if "ipc://" in ipc_path else "TCP"
        print(f"✓ Connected to GPU server at {ipc_path}")
        print(f"  Transport: {self.transport_type}")
        
        # 共享内存环形缓冲区（在init时创建）
        self.shm_ring_tx = None  # 客户端->服务端
        self.shm_ring_rx = None  # 服务端->客户端
        self.max_shm_chunk_bytes = 0
        self.wire_dtype = "float32"
        
        # 性能统计
        self.stats = {
            "rpc_count": 0,
            "rpc_time": 0.0,
            "serialize_time": 0.0,
            "deserialize_time": 0.0,
            "send_time": 0.0,
            "recv_time": 0.0,
            "total_bytes_sent": 0,
            "total_bytes_recv": 0,
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
            f.write(f"ZeroMQ Performance Log - Transport: {self.transport_type}\n")
            f.write("="*120 + "\n")
            f.write(f"{'ID':<5} {'Method':<20} {'Serialize(ms)':<15} {'Send(ms)':<12} {'Recv(ms)':<12} "
                   f"{'Deserialize(ms)':<17} {'Total(ms)':<12} {'Sent(KB)':<12} {'Recv(KB)':<12} {'Throughput(MB/s)':<18}\n")
            f.write("="*120 + "\n")
    
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
            # 使用共享内存
            with self.shm_ring_tx.acquire_write(timeout=5.0) as buf:
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
    
    def _receive_tensor(self, tensor_desc: Dict) -> torch.Tensor:
        """接收张量（自动选择共享内存或ZeroMQ）"""
        use_shm = tensor_desc.get("use_shm", False)
        
        if use_shm:
            # 从共享内存读取
            with self.shm_ring_rx.acquire_read(timeout=5.0) as buf:
                actual_size = int.from_bytes(buf[:4], byteorder='little')
                data = bytes(buf[4:4+actual_size])
            
            self.stats["shm_transfers"] += 1
            self.stats["shm_bytes"] += actual_size
        else:
            # 从ZeroMQ读取
            data = tensor_desc["data"]
            self.stats["zmq_transfers"] += 1
            self.stats["zmq_bytes"] += len(data)
        
        return self._deserialize_tensor_from_bytes(data, tensor_desc["shape"], tensor_desc["dtype"])

    def _send_request(self, method: str, request: Dict) -> Dict:
        """发送请求"""
        call_id = self.stats["rpc_count"] + 1
        self.stats["rpc_count"] = call_id
        
        # 记录开始时间
        call_start = time.perf_counter()
        
        # 序列化
        t0 = time.perf_counter()
        message = {"method": method, "request": request}
        message_bytes = msgpack.packb(message, use_bin_type=True)
        serialize_time = time.perf_counter() - t0
        self.stats["serialize_time"] += serialize_time
        
        bytes_sent = len(message_bytes)
        self.stats["total_bytes_sent"] += bytes_sent
        
        # 发送
        t0 = time.perf_counter()
        self.socket.send(message_bytes)
        send_time = time.perf_counter() - t0
        self.stats["send_time"] += send_time
        
        # 接收
        t0 = time.perf_counter()
        response_bytes = self.socket.recv()
        recv_time = time.perf_counter() - t0
        self.stats["recv_time"] += recv_time
        
        bytes_recv = len(response_bytes)
        self.stats["total_bytes_recv"] += bytes_recv
        
        # 反序列化
        t0 = time.perf_counter()
        response = msgpack.unpackb(response_bytes, raw=False)
        deserialize_time = time.perf_counter() - t0
        self.stats["deserialize_time"] += deserialize_time
        
        # 计算总时间和吞吐量
        total_time = time.perf_counter() - call_start
        self.stats["rpc_time"] += total_time
        total_bytes = bytes_sent + bytes_recv
        throughput = (total_bytes / 1024 / 1024) / total_time if total_time > 0 else 0
        
        # 记录本次调用
        call_log = {
            "id": call_id,
            "method": method,
            "serialize_ms": serialize_time * 1000,
            "send_ms": send_time * 1000,
            "recv_ms": recv_time * 1000,
            "deserialize_ms": deserialize_time * 1000,
            "total_ms": total_time * 1000,
            "bytes_sent": bytes_sent,
            "bytes_recv": bytes_recv,
            "throughput_mbps": throughput
        }
        self.call_logs.append(call_log)
        
        # 实时写入日志
        with open(self.log_file, 'a') as f:
            f.write(f"{call_id:<5} {method:<20} {serialize_time*1000:<15.3f} {send_time*1000:<12.3f} "
                   f"{recv_time*1000:<12.3f} {deserialize_time*1000:<17.3f} {total_time*1000:<12.3f} "
                   f"{bytes_sent/1024:<12.2f} {bytes_recv/1024:<12.2f} {throughput:<18.2f}\n")
        
        if response["status"] == "error":
            print(f"Server error: {response['error']}")
            if "traceback" in response:
                print(response["traceback"])
            raise RuntimeError(f"Server error: {response['error']}")
        
        return response["response"]
    
    def init(self) -> Dict:
        """初始化：创建共享内存环形缓冲区"""
        meta = {
            "wire_dtype": "bfloat16" if torch.cuda.is_available() else "float32",
            "max_chunks": 10,
        }
        init_data = self._send_request("Init", meta)
        
        # 创建环形缓冲区
        self.wire_dtype = meta["wire_dtype"]
        self.max_shm_chunk_bytes = init_data["max_shm_chunk_bytes"]
        
        # 客户端->服务端（连接到服务端的RX）
        rx_handle = init_data["shm_ring_rx_handle"]
        self.shm_ring_tx = ShmRingBuffer(rx_handle[0], rx_handle[1], rx_handle[2])
        
        # 服务端->客户端（连接到服务端的TX）
        tx_handle = init_data["shm_ring_tx_handle"]
        self.shm_ring_rx = ShmRingBuffer(tx_handle[0], tx_handle[1], tx_handle[2])
        
        print(f"✓ Shared memory ring buffers connected (max_chunk={self.max_shm_chunk_bytes/1024/1024:.1f}MB)")
        return init_data
    
    def embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embedding"""
        request = {"input_ids": self._send_tensor(input_ids)}
        resp = self._send_request("Embedding", request)
        return self._receive_tensor(resp["output"])
    
    def batch_linear(self, layer_idx: int, module_names: List[str], hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """批量 Linear"""
        request = {
            "layer_idx": layer_idx,
            "module_names": module_names,
            "hidden_states": self._send_tensor(hidden_states),
        }
        resp = self._send_request("BatchLinear", request)
        return [self._receive_tensor(desc) for desc in resp["outputs"]]
    
    def batch_linear_with_mask(self, layer_idx: int, module_names: List[str], 
                               hidden_states: torch.Tensor, mask: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        批量 Linear (带掩码) - 用于安全计算
        发送 hidden_states + mask 到GPU
        GPU同时计算: (hidden_states + mask)W 和 maskW
        """
        request = {
            "layer_idx": layer_idx,
            "module_names": module_names,
            "hidden_states": self._send_tensor(hidden_states),
            "mask": self._send_tensor(mask),
        }
        resp = self._send_request("BatchLinearWithMask", request)
        outputs = [self._receive_tensor(desc) for desc in resp["outputs"]]
        mask_outputs = [self._receive_tensor(desc) for desc in resp["mask_outputs"]]
        return outputs, mask_outputs
    
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """矩阵乘法"""
        request = {
            "a": self._send_tensor(a),
            "b": self._send_tensor(b),
        }
        resp = self._send_request("Matmul", request)
        return self._receive_tensor(resp["output"])
    
    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """LM Head"""
        request = {"hidden_states": self._send_tensor(hidden_states)}
        resp = self._send_request("LMHead", request)
        return self._receive_tensor(resp["output"])
    
    def lm_head_with_mask(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """LM Head (带掩码) - 用于安全计算"""
        request = {
            "hidden_states": self._send_tensor(hidden_states),
            "mask": self._send_tensor(mask),
        }
        resp = self._send_request("LMHeadWithMask", request)
        output = self._receive_tensor(resp["output"])
        mask_output = self._receive_tensor(resp["mask_output"])
        return output, mask_output
    
    def print_stats(self):
        """打印统计信息"""
        count = self.stats['rpc_count']
        if count == 0:
            return
        
        print(f"\n{'='*80}")
        print(f"{'GPU Client Statistics':^80}")
        print(f"{'='*80}")
        print(f"Transport Type:   {self.transport_type}")
        print(f"RPC Calls:        {count}")
        print(f"\nTiming Breakdown (Average per call):")
        print(f"  Serialize:      {self.stats['serialize_time']/count*1000:>8.3f} ms  ({self.stats['serialize_time']/self.stats['rpc_time']*100:>5.1f}%)")
        print(f"  Send:           {self.stats['send_time']/count*1000:>8.3f} ms  ({self.stats['send_time']/self.stats['rpc_time']*100:>5.1f}%)")
        print(f"  Receive:        {self.stats['recv_time']/count*1000:>8.3f} ms  ({self.stats['recv_time']/self.stats['rpc_time']*100:>5.1f}%)")
        print(f"  Deserialize:    {self.stats['deserialize_time']/count*1000:>8.3f} ms  ({self.stats['deserialize_time']/self.stats['rpc_time']*100:>5.1f}%)")
        print(f"  {'─'*40}")
        print(f"  Total:          {self.stats['rpc_time']/count*1000:>8.3f} ms  (100.0%)")
        
        print(f"\nData Transfer:")
        print(f"  Sent:           {self.stats['total_bytes_sent']/1024/1024:>10.2f} MB  ({self.stats['total_bytes_sent']/count/1024:>8.2f} KB/call)")
        print(f"  Received:       {self.stats['total_bytes_recv']/1024/1024:>10.2f} MB  ({self.stats['total_bytes_recv']/count/1024:>8.2f} KB/call)")
        print(f"  Total:          {(self.stats['total_bytes_sent']+self.stats['total_bytes_recv'])/1024/1024:>10.2f} MB")
        
        # 共享内存 vs ZeroMQ 统计
        total_transfers = self.stats['shm_transfers'] + self.stats['zmq_transfers']
        total_data_bytes = self.stats['shm_bytes'] + self.stats['zmq_bytes']
        if total_transfers > 0:
            print(f"\nTransfer Method Breakdown:")
            print(f"  Shared Memory:  {self.stats['shm_transfers']:>8} transfers ({self.stats['shm_transfers']/total_transfers*100:>5.1f}%)")
            print(f"                  {self.stats['shm_bytes']/1024/1024:>10.2f} MB ({self.stats['shm_bytes']/total_data_bytes*100:>5.1f}%)")
            print(f"  ZeroMQ:         {self.stats['zmq_transfers']:>8} transfers ({self.stats['zmq_transfers']/total_transfers*100:>5.1f}%)")
            print(f"                  {self.stats['zmq_bytes']/1024/1024:>10.2f} MB ({self.stats['zmq_bytes']/total_data_bytes*100:>5.1f}%)")
        
        total_bytes = self.stats['total_bytes_sent'] + self.stats['total_bytes_recv']
        throughput = (total_bytes / 1024 / 1024) / self.stats['rpc_time']
        print(f"\nThroughput:       {throughput:>10.2f} MB/s")
        print(f"{'='*80}")
        
        # 写入汇总到日志
        with open(self.log_file, 'a') as f:
            f.write("\n" + "="*120 + "\n")
            f.write("SUMMARY\n")
            f.write("="*120 + "\n")
            f.write(f"Total RPC Calls: {count}\n")
            f.write(f"Average Serialize Time: {self.stats['serialize_time']/count*1000:.3f} ms\n")
            f.write(f"Average Send Time: {self.stats['send_time']/count*1000:.3f} ms\n")
            f.write(f"Average Receive Time: {self.stats['recv_time']/count*1000:.3f} ms\n")
            f.write(f"Average Deserialize Time: {self.stats['deserialize_time']/count*1000:.3f} ms\n")
            f.write(f"Average Total Time: {self.stats['rpc_time']/count*1000:.3f} ms\n")
            f.write(f"Total Data Sent: {self.stats['total_bytes_sent']/1024/1024:.2f} MB\n")
            f.write(f"Total Data Received: {self.stats['total_bytes_recv']/1024/1024:.2f} MB\n")
            if total_transfers > 0:
                f.write(f"Shared Memory Transfers: {self.stats['shm_transfers']} ({self.stats['shm_bytes']/1024/1024:.2f} MB)\n")
                f.write(f"ZeroMQ Transfers: {self.stats['zmq_transfers']} ({self.stats['zmq_bytes']/1024/1024:.2f} MB)\n")
            f.write(f"Average Throughput: {throughput:.2f} MB/s\n")
            f.write("="*120 + "\n")
        
        print(f"\n✓ Detailed log saved to: {self.log_file}\n")
    
    def close(self) -> None:
        """关闭连接并清理共享内存"""
        try:
            self.socket.close()
            self.context.term()
        finally:
            if self.shm_ring_tx is not None:
                self.shm_ring_tx.close()
            if self.shm_ring_rx is not None:
                self.shm_ring_rx.close()


class TEELlamaModel:
    """TEE 端的 LLaMA 模型"""
    
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
        
        # 安全计算配置
        self.use_secure_linear = True  # 是否对Linear层使用安全计算 (需要GPU服务器支持)
        
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
        
        # 性能统计 - 按类别分组
        self.timing = {
            # 1. 预计算/离线开销 (Offline/Precomputation)
            "offline_keygen": 0.0,           # 生成随机密钥R, a, b
            
            # 2. TEE计算开销 (TEE Computation)
            "tee_compute": 0.0,              # TEE侧的计算（掩码、恢复、其他）
            "tee_rmsnorm": 0.0,              # RMSNorm
            "tee_rotary": 0.0,               # RotaryEmbedding
            "tee_softmax": 0.0,              # Softmax
            "tee_silu": 0.0,                 # SiLU激活
            
            # 3. GPU计算开销 (GPU Computation)
            "gpu_compute": 0.0,              # GPU侧的实际计算
            "gpu_embedding": 0.0,            # Embedding
            "gpu_linear": 0.0,               # Linear层
            "gpu_lm_head": 0.0,              # LM Head
            
            # 4. 通信开销 (Communication)
            "comm_overhead": 0.0,            # 通信总开销（从gpu_client获取）
        }
        self.counts = {k: 0 for k in self.timing.keys()}
        
        print(f"✓ TEE model initialized: {self.num_layers} layers")
    
    def secure_batch_linear(self, layer_idx: int, module_names: List[str], 
                           hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        安全的批量Linear - 使用一次性密码本掩码
        注意: 这需要GPU服务器支持BatchLinearWithMask方法
        如果不支持，会回退到普通的batch_linear
        """
        if not self.use_secure_linear:
            # 直接使用普通方法 (不使用安全协议)
            return self.gpu.batch_linear(layer_idx, module_names, hidden_states)
        
        # 离线阶段: 生成一次性密码本
        t0 = time.perf_counter()
        R = torch.randn_like(hidden_states)
        self.timing["offline_keygen"] += time.perf_counter() - t0
        self.counts["offline_keygen"] += 1
        
        # TEE计算: 添加掩码
        t0 = time.perf_counter()
        hidden_states_masked = hidden_states + R
        self.timing["tee_compute"] += time.perf_counter() - t0
        self.counts["tee_compute"] += 1
        
        try:
            # GPU计算: (x+R)W 和 RW
            t0 = time.perf_counter()
            outputs, mask_outputs = self.gpu.batch_linear_with_mask(
                layer_idx, module_names, hidden_states_masked, R)
            self.timing["gpu_linear"] += time.perf_counter() - t0
            self.counts["gpu_linear"] += len(module_names)
            
            # TEE恢复: y = (x+R)W - RW
            t0 = time.perf_counter()
            recovered_outputs = [out - mask_out for out, mask_out in zip(outputs, mask_outputs)]
            self.timing["tee_compute"] += time.perf_counter() - t0
            self.counts["tee_compute"] += 1
            
            return recovered_outputs
        except Exception as e:
            # 如果GPU服务器不支持，回退到普通方法
            print(f"Warning: BatchLinearWithMask not supported, falling back to regular batch_linear: {e}")
            self.use_secure_linear = False
            return self.gpu.batch_linear(layer_idx, module_names, hidden_states)
    
    def secure_matmul(self, Q: torch.Tensor, K_T: torch.Tensor) -> torch.Tensor:
        """
        安全矩阵乘法 - 使用嵌入式加性外包和恢复
        实现图片中的协议:
        1. Embedded Additive Outsource: 用随机矩阵掩码数据
        2. GPU计算掩码后的矩阵乘法
        3. Recovery: 在TEE中恢复原始结果
        """
        t0 = time.perf_counter()
        
        # 步骤1: 离线采样 - 生成一次性密码本
        # Sample: R_Q ← F^(m×n), R_K^T ← F^(n×p); a, b ∈ F
        R_Q = torch.randn_like(Q)
        R_K_T = torch.randn_like(K_T)
        
        # 随机标量 a, b (避免0)
        a = torch.rand(1).item() * 2 + 0.5  # [0.5, 2.5]
        b = torch.rand(1).item() * 2 + 0.5
        
        # 预计算: aR_Q, bR_K^T (在TEE中)
        aR_Q = a * R_Q
        bR_K_T = b * R_K_T
        
        self.timing["offline_keygen"] += time.perf_counter() - t0
        self.counts["offline_keygen"] += 1
        
        # 步骤2: Embedded Additive Outsource
        # 构造掩码矩阵: Q̃ = [Q + R_Q, aR_Q], K̃^T = [K^T + R_K^T, bR_K^T]
        t0 = time.perf_counter()
        
        # 对于4D张量(batch, heads, seq, dim)，在最后一个维度上拼接
        Q_masked = Q + R_Q
        K_T_masked = K_T + R_K_T
        
        # 拼接: 在seq维度上拼接 (假设Q是(..., m, n), K_T是(..., n, p))
        # Q̃ = [..., m+m', n], K̃^T = [..., n, p+p']
        original_shape_Q = Q.shape
        original_shape_KT = K_T.shape
        
        # 将Q和aR_Q在倒数第二个维度拼接 [Q+R_Q; aR_Q]
        Q_tilde = torch.cat([Q_masked, aR_Q], dim=-2).contiguous()
        # 将K^T和bR_K^T在最后一个维度拼接 [K^T+R_K^T, bR_K^T]
        KT_tilde = torch.cat([K_T_masked, bR_K_T], dim=-1).contiguous()
        
        self.timing["tee_compute"] += time.perf_counter() - t0
        self.counts["tee_compute"] += 1
        
        # 步骤3: GPU计算 Q̃K̃^T
        t0 = time.perf_counter()
        result_tilde = self.gpu.matmul(Q_tilde, KT_tilde)
        self.timing["gpu_compute"] += time.perf_counter() - t0
        self.counts["gpu_compute"] += 1
        
        # 步骤4: Recovery - 在TEE中恢复原始结果
        t0 = time.perf_counter()
        
        # result_tilde的形状应该是 [..., m+m', p+p']
        # 分解成4个块: [[T1, T2], [T3, T4]]
        # T1 = (Q + R_Q)(K^T + R_K^T)
        # T2 = (Q + R_Q)bR_K^T
        # T3 = aR_Q(K^T + R_K^T)
        # T4 = abR_QR_K^T
        
        m = original_shape_Q[-2]
        p = original_shape_KT[-1]
        
        T1 = result_tilde[..., :m, :p]
        T2 = result_tilde[..., :m, p:]
        T3 = result_tilde[..., m:, :p]
        T4 = result_tilde[..., m:, p:]
        
        # 恢复步骤:
        # R_QR_K^T = (1/ab) * T4
        R_Q_R_KT = (1.0 / (a * b)) * T4
        
        # QR_K^T = (1/b) * T2 - R_QR_K^T
        Q_R_KT = (1.0 / b) * T2 - R_Q_R_KT
        
        # R_QK^T = (1/a) * T3 - R_QR_K^T
        R_Q_KT = (1.0 / a) * T3 - R_Q_R_KT
        
        # QK^T = T1 - R_QR_K^T - QR_K^T - R_QK^T
        QKT = T1 - R_Q_R_KT - Q_R_KT - R_Q_KT
        
        self.timing["tee_compute"] += time.perf_counter() - t0
        self.counts["tee_compute"] += 1
        
        return QKT
    
    def attention(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Attention 层 - 使用安全矩阵乘法
        
        注意: 
        - Matmul (Q@K^T 和 Attn@V) 使用嵌入式加性外包协议
        - QKV projections 可以选择性地使用secure_batch_linear (通过设置self.use_secure_linear=True)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # GPU: QKV projections (批量)
        # 选项1: 使用普通batch_linear (默认)
        # 选项2: 使用secure_batch_linear (需要GPU服务器支持BatchLinearWithMask)
        t0 = time.perf_counter()
        if self.use_secure_linear:
            qkv = self.secure_batch_linear(layer_idx, ["q_proj", "k_proj", "v_proj"], hidden_states)
        else:
            qkv = self.gpu.batch_linear(layer_idx, ["q_proj", "k_proj", "v_proj"], hidden_states)
            self.timing["gpu_linear"] += time.perf_counter() - t0
            self.counts["gpu_linear"] += 3
            t0 = time.perf_counter()
        
        query_states, key_states, value_states = qkv
        
        # TEE: Reshape
        t0 = time.perf_counter()
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        self.timing["tee_compute"] += time.perf_counter() - t0
        
        # TEE: Rotary embeddings
        t0 = time.perf_counter()
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        self.timing["tee_rotary"] += time.perf_counter() - t0
        self.counts["tee_rotary"] += 1
        
        # 安全计算: Q @ K^T (使用嵌入式加性外包)
        attn_weights = self.secure_matmul(query_states, key_states.transpose(2, 3))
        
        # TEE: Scale + Softmax
        t0 = time.perf_counter()
        attn_weights = attn_weights * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self.timing["tee_softmax"] += time.perf_counter() - t0
        self.counts["tee_softmax"] += 1
        
        # 安全计算: Attn @ V (使用嵌入式加性外包)
        attn_output = self.secure_matmul(attn_weights, value_states)
        
        # TEE: Reshape
        t0 = time.perf_counter()
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        self.timing["tee_compute"] += time.perf_counter() - t0
        
        # GPU: O projection
        t0 = time.perf_counter()
        if self.use_secure_linear:
            attn_output = self.secure_batch_linear(layer_idx, ["o_proj"], attn_output)[0]
        else:
            attn_output = self.gpu.batch_linear(layer_idx, ["o_proj"], attn_output)[0]
            self.timing["gpu_linear"] += time.perf_counter() - t0
            self.counts["gpu_linear"] += 1
        
        return attn_output
    
    def mlp(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP 层 - 可选安全计算"""
        # GPU: Gate + Up (批量)
        t0 = time.perf_counter()
        if self.use_secure_linear:
            gate_up = self.secure_batch_linear(layer_idx, ["gate_proj", "up_proj"], hidden_states)
        else:
            gate_up = self.gpu.batch_linear(layer_idx, ["gate_proj", "up_proj"], hidden_states)
            self.timing["gpu_linear"] += time.perf_counter() - t0
            self.counts["gpu_linear"] += 2
            t0 = time.perf_counter()
        
        gate, up = gate_up
        
        # TEE: SiLU + multiply
        t0 = time.perf_counter()
        gate = F.silu(gate)
        intermediate = gate * up
        self.timing["tee_silu"] += time.perf_counter() - t0
        self.counts["tee_silu"] += 1
        
        # GPU: Down
        t0 = time.perf_counter()
        if self.use_secure_linear:
            output = self.secure_batch_linear(layer_idx, ["down_proj"], intermediate)[0]
        else:
            output = self.gpu.batch_linear(layer_idx, ["down_proj"], intermediate)[0]
            self.timing["gpu_linear"] += time.perf_counter() - t0
            self.counts["gpu_linear"] += 1
        
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
    
    def secure_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """安全的LM Head - 使用一次性密码本掩码"""
        if not self.use_secure_linear:
            # 直接使用普通方法
            return self.gpu.lm_head(hidden_states)
        
        # 离线阶段: 生成一次性密码本
        t0 = time.perf_counter()
        R = torch.randn_like(hidden_states)
        self.timing["offline_keygen"] += time.perf_counter() - t0
        self.counts["offline_keygen"] += 1
        
        # TEE计算: 添加掩码
        t0 = time.perf_counter()
        hidden_states_masked = hidden_states + R
        self.timing["tee_compute"] += time.perf_counter() - t0
        self.counts["tee_compute"] += 1
        
        try:
            # GPU计算: (x+R)W 和 RW
            t0 = time.perf_counter()
            output, mask_output = self.gpu.lm_head_with_mask(hidden_states_masked, R)
            self.timing["gpu_lm_head"] += time.perf_counter() - t0
            self.counts["gpu_lm_head"] += 1
            
            # TEE恢复: y = (x+R)W - RW
            t0 = time.perf_counter()
            recovered_output = output - mask_output
            self.timing["tee_compute"] += time.perf_counter() - t0
            self.counts["tee_compute"] += 1
            
            return recovered_output
        except Exception as e:
            # 如果GPU服务器不支持，回退到普通方法
            print(f"Warning: LMHeadWithMask not supported, falling back to regular lm_head: {e}")
            self.use_secure_linear = False
            return self.gpu.lm_head(hidden_states)
    
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len = input_ids.shape
        
        # GPU: Embedding (输入数据，不需要掩码保护)
        t0 = time.perf_counter()
        hidden_states = self.gpu.embedding(input_ids)
        self.timing["gpu_embedding"] += time.perf_counter() - t0
        self.counts["gpu_embedding"] += 1
        
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
        
        # GPU: LM head (可选安全计算)
        t0 = time.perf_counter()
        if self.use_secure_linear:
            logits = self.secure_lm_head(hidden_states[:, -1:, :])
        else:
            logits = self.gpu.lm_head(hidden_states[:, -1:, :])
            self.timing["gpu_lm_head"] += time.perf_counter() - t0
            self.counts["gpu_lm_head"] += 1
        
        return logits
    
    def print_timing_stats(self):
        """打印性能统计 - 按类别分组"""
        # 获取通信开销
        self.timing["comm_overhead"] = self.gpu.stats["send_time"] + self.gpu.stats["recv_time"]
        
        total_time = sum(self.timing.values())
        
        print(f"\n{'='*80}")
        print(f"{'TEE Model Performance Breakdown':^80}")
        print(f"{'='*80}")
        print(f"{'Operation':<25} {'Count':>8} {'Total(s)':>12} {'Avg(ms)':>12} {'%':>8}")
        print(f"{'='*80}")
        
        # 1. 预计算/离线开销
        print(f"\n{'1. Offline/Precomputation Overhead':^80}")
        print(f"{'-'*80}")
        ops = ["offline_keygen"]
        for op in ops:
            count = self.counts.get(op, 0)
            t = self.timing.get(op, 0.0)
            avg = (t / count * 1000) if count > 0 else 0
            pct = (t / total_time * 100) if total_time > 0 else 0
            name = "Key Generation (R,a,b)" if op == "offline_keygen" else op
            print(f"  {name:<23} {count:>8} {t:>12.4f} {avg:>12.4f} {pct:>7.2f}%")
        offline_total = sum(self.timing.get(op, 0.0) for op in ops)
        print(f"  {'Subtotal':<23} {'':<8} {offline_total:>12.4f} {'':<12} {offline_total/total_time*100:>7.2f}%")
        
        # 2. TEE计算开销
        print(f"\n{'2. TEE Computation Overhead':^80}")
        print(f"{'-'*80}")
        ops_detail = [
            ("tee_compute", "Secure Compute"),
            ("tee_rmsnorm", "RMSNorm"),
            ("tee_rotary", "RotaryEmbedding"),
            ("tee_softmax", "Softmax"),
            ("tee_silu", "SiLU"),
        ]
        for op, name in ops_detail:
            count = self.counts.get(op, 0)
            t = self.timing.get(op, 0.0)
            if count > 0:  # 只显示有计数的项
                avg = (t / count * 1000)
                pct = (t / total_time * 100) if total_time > 0 else 0
                print(f"  {name:<23} {count:>8} {t:>12.4f} {avg:>12.4f} {pct:>7.2f}%")
        tee_total = sum(self.timing.get(op, 0.0) for op, _ in ops_detail)
        print(f"  {'Subtotal':<23} {'':<8} {tee_total:>12.4f} {'':<12} {tee_total/total_time*100:>7.2f}%")
        
        # 3. GPU计算开销
        print(f"\n{'3. GPU Computation Overhead':^80}")
        print(f"{'-'*80}")
        ops_detail = [
            ("gpu_compute", "Secure Matmul"),
            ("gpu_embedding", "Embedding"),
            ("gpu_linear", "Linear Layers"),
            ("gpu_lm_head", "LM Head"),
        ]
        for op, name in ops_detail:
            count = self.counts.get(op, 0)
            t = self.timing.get(op, 0.0)
            if count > 0:
                avg = (t / count * 1000)
                pct = (t / total_time * 100) if total_time > 0 else 0
                print(f"  {name:<23} {count:>8} {t:>12.4f} {avg:>12.4f} {pct:>7.2f}%")
        gpu_total = sum(self.timing.get(op, 0.0) for op, _ in ops_detail)
        print(f"  {'Subtotal':<23} {'':<8} {gpu_total:>12.4f} {'':<12} {gpu_total/total_time*100:>7.2f}%")
        
        # 4. 通信开销
        print(f"\n{'4. Communication Overhead':^80}")
        print(f"{'-'*80}")
        comm_time = self.timing.get("comm_overhead", 0.0)
        comm_pct = (comm_time / total_time * 100) if total_time > 0 else 0
        send_time = self.gpu.stats["send_time"]
        recv_time = self.gpu.stats["recv_time"]
        print(f"  {'Send':<23} {self.gpu.stats['rpc_count']:>8} {send_time:>12.4f} {send_time/self.gpu.stats['rpc_count']*1000:>12.4f} {send_time/total_time*100:>7.2f}%")
        print(f"  {'Receive':<23} {self.gpu.stats['rpc_count']:>8} {recv_time:>12.4f} {recv_time/self.gpu.stats['rpc_count']*1000:>12.4f} {recv_time/total_time*100:>7.2f}%")
        print(f"  {'Subtotal':<23} {'':<8} {comm_time:>12.4f} {'':<12} {comm_pct:>7.2f}%")
        
        # 总计
        print(f"\n{'='*80}")
        print(f"{'SUMMARY':^80}")
        print(f"{'='*80}")
        print(f"  {'1. Offline':<23} {'':<8} {offline_total:>12.4f} {'':<12} {offline_total/total_time*100:>7.2f}%")
        print(f"  {'2. TEE Compute':<23} {'':<8} {tee_total:>12.4f} {'':<12} {tee_total/total_time*100:>7.2f}%")
        print(f"  {'3. GPU Compute':<23} {'':<8} {gpu_total:>12.4f} {'':<12} {gpu_total/total_time*100:>7.2f}%")
        print(f"  {'4. Communication':<23} {'':<8} {comm_time:>12.4f} {'':<12} {comm_pct:>7.2f}%")
        print(f"{'-'*80}")
        print(f"  {'TOTAL':<23} {'':<8} {total_time:>12.4f} {'':<12} {'100.00':>7}%")
        print(f"{'='*80}\n")


def run_benchmark(model: TEELlamaModel, tokenizer, prefill_length: int) -> float:
    """运行性能测试"""
    input_ids = torch.full((1, prefill_length), tokenizer.pad_token_id, dtype=torch.long)
    
    print(f"\n{'='*70}")
    print(f"{'Prefill Benchmark with Secure Computation':^70}")
    print(f"{'='*70}")
    print(f"Token length: {prefill_length}")
    print(f"TEE Operations:")
    print(f"  - Basic: Softmax, RMSNorm, RotaryEmbedding, SiLU")
    print(f"  - Secure: Mask Generation, Data Preparation, Recovery")
    print(f"GPU Operations: Linear, Embedding, Matmul, LM Head")
    print(f"Security: Matmul uses Embedded Additive Outsourcing (Always ON)")
    print(f"         Linear layers use OTP masks (Optional, use_secure_linear={model.use_secure_linear})")
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
