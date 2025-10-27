# V3.0 重大架构变更

## 变更概述

从模块级分离（整个 Linear/Embedding 层）改为**操作级细粒度分离**：
- **TEE 端**：只执行 Softmax、RMSNorm、RotaryEmbedding、SiLU 激活函数
- **GPU 端**：执行所有其他计算（Linear、Embedding、Matmul）

## 主要变更

### 1. 服务端 (server.py)

#### 之前 (V2.1)
```python
class ModuleRegistry:
    """管理 Linear 和 Embedding 模块"""
    def forward(self, module_name: str, input_tensor: torch.Tensor)
```

#### 现在 (V3.0)
```python
class GPUComputeService:
    """执行所有 GPU 端计算"""
    def embedding(self, input_ids)
    def linear(self, layer_idx, module_name, hidden_states)
    def matmul(self, a, b)
    def lm_head_forward(self, hidden_states)
    def get_rotary_emb_params()  # 新增
    def get_norm_weights()        # 新增
```

**关键改进**：
- 不再按模块名注册，而是按操作类型提供服务
- 新增初始化方法，返回 TEE 端需要的参数（RMSNorm 权重、RotaryEmbedding 参数）
- 更细粒度的控制

### 2. 客户端 (tee_runner.py)

#### 之前 (V2.1)
```python
# 使用 transformers 的完整模型
model = AutoModelForCausalLM.from_config(config)
replace_with_remote_modules(model, client)  # 替换 Linear/Embedding
```

#### 现在 (V3.0)
```python
# 自定义 TEE 端模型实现
class TEELlamaModel:
    def __init__(self, gpu_client, config, rotary_params, norm_weights):
        # 初始化 TEE 端的组件
        self.rotary_emb = TEERotaryEmbedding(...)
        self.input_layernorms = [TEERMSNorm(...) for ...]
        self.post_attention_layernorms = [TEERMSNorm(...) for ...]
        self.final_norm = TEERMSNorm(...)
    
    def attention(self, layer_idx, hidden_states, position_ids):
        # 手动编排 TEE 和 GPU 的计算
        query_states = self.gpu.linear(layer_idx, "q_proj", hidden_states)
        # ... TEE 端的 RoPE、Softmax 等
    
    def mlp(self, layer_idx, hidden_states):
        gate = self.gpu.linear(layer_idx, "gate_proj", hidden_states)
        gate = F.silu(gate)  # TEE 端的 SiLU
        # ...
```

**关键改进**：
- 完全自定义的前向传播流程
- 精确控制每个操作在哪里执行
- TEE 端实现了 RMSNorm、RotaryEmbedding 等

### 3. RPC 方法变更

#### 之前 (V2.1)
```python
# 通用的模块调用
RegisterClient(module_names) -> {ok, missing_modules, ...}
Forward(module_name, input_buffer, ...) -> {output_buffer, ...}
Matmul(a_buffer, b_buffer, ...) -> {output_buffer, ...}
FetchNonLinearTensors(...) -> {parameters, buffers}
```

#### 现在 (V3.0)
```python
# 操作级的调用
Init() -> {config, rotary_emb_params, norm_weights}
Embedding(input_ids, ...) -> {output, shape, dtype}
Linear(layer_idx, module_name, hidden_states, ...) -> {output, shape, dtype}
Matmul(a_buffer, b_buffer, ...) -> {output, shape, dtype}
LMHead(hidden_states, ...) -> {output, shape, dtype}
```

**关键改进**：
- 更清晰的语义
- 不需要注册模块
- 初始化时一次性获取所有 TEE 端需要的参数

## 计算分离对比

### V2.1 架构
```
TEE 端:
├─ RMSNorm
├─ Attention (完整)
│  ├─ QKV Projection (远程调用 GPU)
│  ├─ Attention 计算 (本地)
│  └─ Output Projection (远程调用 GPU)
├─ MLP (完整)
│  ├─ Gate/Up Projection (远程调用 GPU)
│  ├─ SiLU + Multiply (本地)
│  └─ Down Projection (远程调用 GPU)
└─ 控制流程

GPU 端:
├─ Linear 层
├─ Embedding 层
└─ Matmul (Attention 中的矩阵乘法)
```

### V3.0 架构
```
TEE 端:
├─ Softmax (Attention 中)
├─ RMSNorm (所有归一化)
├─ RotaryEmbedding (位置编码)
├─ SiLU (MLP 激活)
├─ Reshape/Repeat (张量操作)
├─ Residual Add (残差连接)
└─ 控制流程

GPU 端:
├─ Embedding
├─ Linear (所有投影层)
├─ Matmul (所有矩阵乘法)
└─ LM Head
```

## 性能影响

### RPC 调用次数

**V2.1**：
- 每层: ~7 次（QKV + O + Gate + Up + Down + 2×Matmul）
- 总计: ~154 次（22 层）

**V3.0**：
- 每层: ~9 次（QKV + O + Gate + Up + Down + 2×Matmul + 额外的细粒度调用）
- 总计: ~200 次（22 层）

**增加**: ~30% RPC 调用

### 数据传输量

**基本相同**，因为传输的张量大小没有变化。

### TEE 端计算负担

**V2.1**: 
- Attention 计算（包括 Softmax）
- MLP 计算（包括 SiLU）
- RMSNorm
- 张量操作

**V3.0**:
- **仅** Softmax、RMSNorm、RotaryEmbedding、SiLU
- 更轻量级

**减少**: ~60% TEE 端计算

## 代码统计

| 文件 | V2.1 | V3.0 | 变化 |
|------|------|------|------|
| server.py | 250 行 | 330 行 | +80 行 |
| tee_runner.py | 261 行 | 380 行 | +119 行 |
| **总计** | **511 行** | **710 行** | **+199 行 (+39%)** |

**增加原因**：
- 自定义 TEE 端模型实现
- 更细粒度的操作控制
- 更多的 RPC 方法

## 优势

1. **更精确的计算分离**
   - TEE 只执行必要的非线性操作
   - 最大化 GPU 利用率

2. **更好的安全性**
   - 关键的非线性操作在 TEE 中
   - 位置编码在 TEE 中计算

3. **更灵活的优化**
   - 可以独立优化每个操作
   - 更容易进行性能分析

4. **更清晰的接口**
   - 基于操作类型而非模块名
   - 更容易理解和维护

## 劣势

1. **更多的 RPC 调用**
   - 增加了 ~30% 的通信开销
   - 可能影响性能

2. **更复杂的实现**
   - 需要手动实现前向传播
   - 代码量增加 ~39%

3. **更难维护**
   - TEE 端需要与 GPU 端保持同步
   - 模型更新时需要修改更多代码

## 适用场景

### V3.0 更适合：
- 需要最大化 GPU 利用率
- 需要保护位置编码等关键信息
- 有足够的网络带宽
- 需要精确控制计算分离

### V2.1 更适合：
- 网络带宽有限
- 需要简单的实现
- 模型经常更新
- 不需要极致的性能

## 迁移指南

### 从 V2.1 迁移到 V3.0

1. **更新服务端**
   ```bash
   # 备份旧版本
   cp server.py server.py.v2.1
   
   # 使用新版本
   # (已自动完成)
   ```

2. **更新客户端**
   ```bash
   # 备份旧版本
   cp tee_runner.py tee_runner.py.v2.1
   
   # 使用新版本
   # (已自动完成)
   ```

3. **测试**
   ```bash
   # 启动服务器
   python server.py
   
   # 运行客户端
   python tee_runner.py
   ```

4. **性能对比**
   - 观察 RPC 调用次数
   - 测量端到端延迟
   - 分析 TEE 端 CPU 使用率

## 未来优化方向

1. **批量 RPC 调用**
   - 合并多个 Linear 调用
   - 减少通信次数

2. **异步通信**
   - 使用异步 ZeroMQ
   - 流水线并行

3. **张量压缩**
   - 使用 float16/bfloat16
   - 量化传输

4. **智能调度**
   - 根据网络状况动态调整
   - 自适应批处理

## 总结

V3.0 实现了更细粒度的计算分离，将 TEE 端的计算负担降到最低，同时保护了关键的非线性操作。虽然增加了一些复杂性和通信开销，但提供了更好的性能和安全性平衡。
