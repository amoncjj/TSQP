"""
配置文件 - TEE-Only 推理
"""

# 模型配置
MODEL_PATH = "/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b"

# 推理配置
PREFILL_TOKEN_LENGTH = 8  # Prefill 阶段的 token 数量

# 输出配置
OUTPUT_FILE = "tee_only_results.json"

# 设备配置
DEVICE = "cpu"  # TEE-Only 模式，全部在 CPU 上运行
