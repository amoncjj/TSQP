# 模型配置说明

## 模型路径配置

项目已配置为使用本地模型路径：
```
/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b
```

## 必需的模型文件

确保模型目录包含以下文件：

### 1. 配置文件（必需）
- `config.json` - 模型配置
- `generation_config.json` - 生成配置（可选）

### 2. Tokenizer 文件（必需）
- `tokenizer.json` - Tokenizer 配置
- `tokenizer_config.json` - Tokenizer 元数据
- `special_tokens_map.json` - 特殊 token 映射

### 3. 模型权重文件（必需其一）
- `pytorch_model.bin` - PyTorch 格式权重
- `model.safetensors` - SafeTensors 格式权重
- `model.bin` - 通用二进制格式

### 4. 其他文件（可选）
- `vocab.json` - 词汇表
- `merges.txt` - BPE merges

## 检查模型文件

运行以下命令检查模型目录：

```bash
ls -la /home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b/
```

应该看到类似输出：
```
config.json
generation_config.json
pytorch_model.bin  (或 model.safetensors)
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

## 从 Hugging Face 下载模型

如果需要下载模型，可以使用以下方法：

### 方法 1: 使用 huggingface-cli

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载模型
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b \
  --local-dir-use-symlinks False
```

### 方法 2: 使用 Python 脚本

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    local_dir="/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b",
    local_dir_use_symlinks=False
)
```

### 方法 3: 使用 transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B-Instruct"
save_path = "/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b"

# 下载并保存模型
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(save_path)

# 下载并保存 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)
```

## 环境变量配置

设置模型路径环境变量：

```bash
export LLAMA_MODEL_PATH=/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b
```

或在运行时指定：

```bash
LLAMA_MODEL_PATH=/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b \
  python tee_only_llama/tee_runner.py
```

## 代码修复说明

已修复以下文件以支持本地模型：

1. **tee_only_llama/tee_runner.py**
   - 自动检测本地路径
   - 仅在本地路径存在时使用 `local_files_only=True`

2. **tee_gpu/tee_runner.py**
   - 支持本地模型配置加载
   - 添加调试输出

3. **tee_gpu/server.py**
   - 支持多种权重文件格式
   - 改进错误处理和日志输出

## 测试模型加载

### 测试纯 TEE 模式

```bash
cd /home/junjie_chen@idm.teecertlabs.com/TSQP/tee_only_llama
export LLAMA_MODEL_PATH=/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b
python tee_runner.py
```

### 测试 TEE+GPU 模式

```bash
# 启动 GPU 服务器
cd /home/junjie_chen@idm.teecertlabs.com/TSQP/tee_gpu
export LLAMA_MODEL_PATH=/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b
python server.py &

# 启动 TEE 客户端
python tee_runner.py
```

## 常见问题

### Q1: 报错 "Cannot find the requested files"

**原因**: 模型目录缺少必需的配置文件

**解决**: 确保 `config.json` 和 `tokenizer.json` 存在

### Q2: 报错 "Unable to locate pre-trained weights"

**原因**: 缺少模型权重文件

**解决**: 确保存在 `pytorch_model.bin` 或 `model.safetensors`

### Q3: 模型加载很慢

**原因**: 模型文件较大（约 5GB）

**解决**: 这是正常现象，首次加载需要时间

### Q4: 内存不足

**原因**: LLaMA 3.2-1B 需要约 4-8GB 内存

**解决**: 
- 使用更小的模型
- 增加系统内存
- 使用量化版本

## 权限检查

确保文件权限正确：

```bash
# 检查目录权限
ls -ld /home/junjie_chen@idm.teecertlabs.com/TSQP/weights/

# 检查文件权限
ls -l /home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b/

# 如果需要，修改权限
chmod -R 755 /home/junjie_chen@idm.teecertlabs.com/TSQP/weights/
```

## 验证模型完整性

运行以下 Python 脚本验证模型：

```python
import os
from transformers import AutoConfig, AutoTokenizer

model_path = "/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b"

# 检查目录
print(f"Model directory exists: {os.path.exists(model_path)}")
print(f"Files in directory: {os.listdir(model_path)}")

# 尝试加载配置
try:
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    print(f"✓ Config loaded successfully")
    print(f"  Model type: {config.model_type}")
    print(f"  Hidden size: {config.hidden_size}")
except Exception as e:
    print(f"✗ Failed to load config: {e}")

# 尝试加载 tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print(f"✓ Tokenizer loaded successfully")
    print(f"  Vocab size: {len(tokenizer)}")
except Exception as e:
    print(f"✗ Failed to load tokenizer: {e}")
```

---

**注意**: 确保模型文件完整且未损坏，否则可能导致加载失败或推理错误。
