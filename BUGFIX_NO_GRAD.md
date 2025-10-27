# 梯度追踪错误修复

## 问题描述

运行 `tee_runner_optimized.py` 时遇到错误：

```
RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
```

## 根本原因

在推理（inference）时，PyTorch 默认会追踪梯度，导致张量的 `requires_grad=True`。当尝试将这样的张量转换为 numpy 数组时，PyTorch 会报错，因为 numpy 不支持自动微分。

## 解决方案

### 方案 1: 使用 `.detach()` ❌ (不推荐)

```python
tensor_cpu = hidden_states.detach().cpu().contiguous()
```

**缺点**: 需要在每个转换点都添加 `.detach()`，代码冗余。

### 方案 2: 使用 `@torch.no_grad()` ✅ (推荐)

```python
@torch.no_grad()
def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    """前向传播"""
    # ... 所有操作都不会追踪梯度
```

**优点**:
1. **更清晰**: 明确表示这是推理代码，不需要梯度
2. **更高效**: 不追踪梯度可以节省内存和计算
3. **更简洁**: 只需要一个装饰器，不需要到处添加 `.detach()`
4. **更符合最佳实践**: PyTorch 官方推荐的推理方式

## 实际修复

### 修改文件: `tee_gpu/tee_runner_optimized.py`

#### 修改点 1: 添加 `@torch.no_grad()` 装饰器

```python
# 修改前
def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    """前向传播"""
    batch_size, seq_len = input_ids.shape

# 修改后
@torch.no_grad()
def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    """前向传播"""
    batch_size, seq_len = input_ids.shape
```

#### 修改点 2: 保持原有的简洁代码

由于添加了 `@torch.no_grad()`，所有在 `forward()` 中调用的方法（`embedding`, `batch_linear`, `matmul`, `lm_head`）都不会追踪梯度，因此不需要添加 `.detach()`。

```python
# GPUClient 方法保持简洁
def batch_linear(self, layer_idx: int, module_names: List[str], hidden_states: torch.Tensor):
    tensor_cpu = hidden_states.cpu().contiguous() if hidden_states.is_cuda else hidden_states.contiguous()
    # 不需要 .detach()，因为在 @torch.no_grad() 上下文中
```

## 性能影响

使用 `@torch.no_grad()` 的性能优势：

1. **内存节省**: 不保存中间结果用于反向传播
2. **计算加速**: 跳过梯度计算相关的开销
3. **更少的 GPU↔CPU 传输**: 不需要传输梯度信息

## 验证

### 测试 1: 检查梯度状态

```python
@torch.no_grad()
def test():
    x = torch.randn(10, 10)
    print(f"requires_grad: {x.requires_grad}")  # False
    y = x.cpu().numpy()  # ✓ 不会报错

test()
```

### 测试 2: 运行完整推理

```bash
cd /home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu

# 启动服务器
python server_optimized.py &

# 运行客户端
python tee_runner_optimized.py
```

**预期结果**: 不再出现 `RuntimeError: Can't call numpy() on Tensor that requires grad`

## 最佳实践

### 推理代码模板

```python
class InferenceModel:
    def __init__(self):
        self.model.eval()  # 设置为评估模式
    
    @torch.no_grad()  # 禁用梯度追踪
    def forward(self, input_ids):
        # 所有推理代码
        output = self.model(input_ids)
        return output
```

### 训练 vs 推理

| 场景 | 梯度追踪 | 装饰器 | 模式 |
|------|---------|--------|------|
| 训练 | ✓ 需要 | 无 | `model.train()` |
| 推理 | ✗ 不需要 | `@torch.no_grad()` | `model.eval()` |
| 验证 | ✗ 不需要 | `@torch.no_grad()` | `model.eval()` |

## 相关文档

- PyTorch 官方文档: [torch.no_grad()](https://pytorch.org/docs/stable/generated/torch.no_grad.html)
- PyTorch 推理优化: [Inference Mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html)

## 总结

- ✅ 使用 `@torch.no_grad()` 装饰器是推理代码的最佳实践
- ✅ 比 `.detach()` 更清晰、更高效、更简洁
- ✅ 节省内存和计算资源
- ✅ 代码更易维护

**修改文件**: `tee_gpu/tee_runner_optimized.py` (仅添加 1 行装饰器)  
**性能影响**: 正面（减少内存和计算开销）  
**代码复杂度**: 降低（不需要到处添加 `.detach()`）
