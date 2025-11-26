"""
配置文件 - TEE+GPU 混合推理
"""

# 模型配置
MODEL_PATH = "/root/weights/llama-2-7b"

# 推理配置
PREFILL_TOKEN_LENGTH = 512  # Prefill 阶段的 token 数量

# 输出配置
OUTPUT_FILE = "tee_gpu_results.json"

# 设备配置
GPU_DEVICE = "cuda:0"  # GPU 设备
CPU_DEVICE = "cpu"     # CPU 设备 (TEE)

