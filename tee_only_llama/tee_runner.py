"""
TEE-Only 推理 - Intel TDX 版本
全部计算在 CPU (TEE) 中执行
无 warmup 步骤
"""
import json
import os
import time
from typing import Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入配置
from config import (
    MODEL_PATH,
    PREFILL_TOKEN_LENGTH,
    OUTPUT_FILE,
    DEVICE
)


def load_model_and_tokenizer(model_path: str, device: str) -> tuple:
    """加载模型和 tokenizer"""
    is_local_path = os.path.exists(model_path)
    
    print(f"Loading model from: {model_path}")
    print(f"Is local path: {is_local_path}")
    print(f"Device: {device}")
    
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
        device_map=device
    )
    model.eval()

    return model, tokenizer


def run_prefill(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefill_length: int,
) -> Dict:
    """
    执行 prefill 阶段，返回性能统计 (无 warmup)
    """
    # 创建固定长度的 token 序列
    input_ids = torch.full((1, prefill_length), tokenizer.pad_token_id, dtype=torch.long).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"\n{'='*80}")
    print(f"{'TEE-Only Inference Benchmark':^80}")
    print(f"{'='*80}")
    print(f"  Token length: {prefill_length}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Device: {model.device}")
    print(f"{'='*80}\n")
    
    # 直接运行 Benchmark (无 warmup)
    print("Running benchmark...")
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    elapsed = time.perf_counter() - start
    
    logits = outputs.logits
    
    print(f"\n{'='*80}")
    print(f"Benchmark completed!")
    print(f"  Total time: {elapsed:.4f}s")
    print(f"  Throughput: {prefill_length / elapsed:.2f} tokens/sec")
    print(f"  Logits shape: {logits.shape}")
    print(f"{'='*80}\n")
    
    # 统计结果
    results = {
        "benchmark_info": {
            "model_path": MODEL_PATH,
            "prefill_length": prefill_length,
            "device": str(model.device),
            "logits_shape": list(logits.shape),
        },
        "timing": {
            "total_ms": elapsed * 1000,
            "throughput_tokens_per_sec": prefill_length / elapsed,
        }
    }
    
    return results


def main() -> None:
    """主函数"""
    print(f"TEE-Only Mode: All computation on CPU (TEE)")
    
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, DEVICE)
    
    results = run_prefill(model, tokenizer, PREFILL_TOKEN_LENGTH)
    
    # 保存结果
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {OUTPUT_FILE}\n")
    
    # 打印结果
    print(f"{'='*80}")
    print(f"{'Results Summary':^80}")
    print(f"{'='*80}")
    print(f"  Prefill time: {results['timing']['total_ms']:.2f} ms")
    print(f"  Throughput: {results['timing']['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
