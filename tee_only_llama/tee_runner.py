import json
import os
import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置：在代码中直接指定
PREFILL_TOKEN_LENGTH = 128  # 直接在这里修改 token 数量
DEFAULT_MODEL_PATH = "/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b"
DEFAULT_PROMPT = "Hello, world!"

# 环境变量
MODEL_PATH_ENV = "LLAMA_MODEL_PATH"





def load_model_and_tokenizer(model_path: str) -> tuple:
    """加载模型和 tokenizer"""
    is_local_path = os.path.exists(model_path)
    
    print(f"Loading model from: {model_path}")
    print(f"Is local path: {is_local_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=is_local_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=is_local_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()

    return model, tokenizer


def run_prefill(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefill_length: int,
) -> float:
    """
    执行 prefill 阶段，返回 prefill 时间（秒）
    """
    # 创建固定长度的 token 序列
    input_ids = torch.full((1, prefill_length), tokenizer.pad_token_id, dtype=torch.long).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"Prefill token length: {prefill_length}")
    
    # 前向传播（prefill）并计时
    start = time.perf_counter()
    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    elapsed = time.perf_counter() - start
    
    return elapsed


def main() -> None:
    """主函数"""
    model_path = os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
    
    print(f"Loading model from: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    print(f"Running prefill with {PREFILL_TOKEN_LENGTH} tokens...")
    prefill_time = run_prefill(model, tokenizer, PREFILL_TOKEN_LENGTH)
    
    print(f"\nPrefill time: {prefill_time:.4f} seconds")


if __name__ == "__main__":
    main()
