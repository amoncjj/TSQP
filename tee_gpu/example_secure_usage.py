"""
安全计算使用示例

演示如何在TEE-GPU协同计算中使用安全外包计算协议
"""
import os
import sys
import torch
from tee_runner_optimized import GPUClient, TEELlamaModel
from transformers import AutoTokenizer

# 配置
MODEL_PATH = "/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b"
IPC_PATH = "ipc:///tmp/tsqp_gpu_server.ipc"
PREFILL_LENGTH = 8


def example_1_secure_matmul_only():
    """
    示例1: 仅使用安全矩阵乘法 (默认配置)
    
    特点:
    - Attention中的 Q@K^T 和 Attn@V 使用嵌入式加性外包
    - Linear层使用普通计算 (无掩码)
    - 性能开销较小
    """
    print("\n" + "="*80)
    print("示例1: 仅使用安全矩阵乘法 (默认)")
    print("="*80)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 连接GPU服务器
    gpu_client = GPUClient(IPC_PATH)
    
    try:
        # 初始化模型
        init_data = gpu_client.init()
        model = TEELlamaModel(
            gpu_client,
            init_data["config"],
            init_data["rotary_emb_params"],
            init_data["norm_weights"]
        )
        
        # 默认配置: use_secure_linear = False
        print(f"✓ 模型初始化完成")
        print(f"  - 安全矩阵乘法: 启用 (自动)")
        print(f"  - 安全线性层: {model.use_secure_linear}")
        
        # 运行推理
        input_ids = torch.full((1, PREFILL_LENGTH), tokenizer.pad_token_id, dtype=torch.long)
        logits = model.forward(input_ids)
        
        print(f"\n✓ 推理完成")
        print(f"  - 输出形状: {logits.shape}")
        
        # 打印统计
        model.print_timing_stats()
        
    finally:
        gpu_client.close()


def example_2_full_secure_computation():
    """
    示例2: 完整安全计算 (所有层使用掩码)
    
    特点:
    - Attention中的 Q@K^T 和 Attn@V 使用嵌入式加性外包
    - 所有Linear层使用OTP掩码保护
    - 最高安全性，但性能开销较大
    
    注意: 需要GPU服务器支持 BatchLinearWithMask 和 LMHeadWithMask
    """
    print("\n" + "="*80)
    print("示例2: 完整安全计算 (所有层使用掩码)")
    print("="*80)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 连接GPU服务器
    gpu_client = GPUClient(IPC_PATH)
    
    try:
        # 初始化模型
        init_data = gpu_client.init()
        model = TEELlamaModel(
            gpu_client,
            init_data["config"],
            init_data["rotary_emb_params"],
            init_data["norm_weights"]
        )
        
        # 启用完整安全计算
        model.use_secure_linear = True
        
        print(f"✓ 模型初始化完成")
        print(f"  - 安全矩阵乘法: 启用 (自动)")
        print(f"  - 安全线性层: {model.use_secure_linear}")
        print(f"\n⚠️  注意: 如果GPU服务器不支持掩码版本的Linear/LMHead,")
        print(f"   将自动回退到普通方法\n")
        
        # 运行推理
        input_ids = torch.full((1, PREFILL_LENGTH), tokenizer.pad_token_id, dtype=torch.long)
        logits = model.forward(input_ids)
        
        print(f"\n✓ 推理完成")
        print(f"  - 输出形状: {logits.shape}")
        print(f"  - 实际使用安全线性层: {model.use_secure_linear}")
        
        # 打印统计
        model.print_timing_stats()
        
    finally:
        gpu_client.close()


def example_3_custom_secure_matmul():
    """
    示例3: 单独测试安全矩阵乘法
    
    演示如何单独使用 secure_matmul 方法
    """
    print("\n" + "="*80)
    print("示例3: 单独测试安全矩阵乘法")
    print("="*80)
    
    # 连接GPU服务器
    gpu_client = GPUClient(IPC_PATH)
    
    try:
        # 初始化模型
        init_data = gpu_client.init()
        model = TEELlamaModel(
            gpu_client,
            init_data["config"],
            init_data["rotary_emb_params"],
            init_data["norm_weights"]
        )
        
        # 创建测试数据
        batch_size, num_heads, seq_len, head_dim = 1, 8, 4, 64
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        print(f"✓ 测试数据:")
        print(f"  - Q shape: {Q.shape}")
        print(f"  - K shape: {K.shape}")
        
        # 使用安全矩阵乘法
        print(f"\n执行安全矩阵乘法: Q @ K^T")
        result = model.secure_matmul(Q, K.transpose(-2, -1))
        
        print(f"\n✓ 结果:")
        print(f"  - 输出形状: {result.shape}")
        print(f"  - 期望形状: ({batch_size}, {num_heads}, {seq_len}, {seq_len})")
        
        # 验证正确性 (与直接计算比较)
        print(f"\n验证正确性...")
        direct_result = torch.matmul(Q, K.transpose(-2, -1))
        error = torch.abs(result - direct_result).max().item()
        print(f"  - 最大误差: {error:.6e}")
        print(f"  - 验证: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")
        
    finally:
        gpu_client.close()


def main():
    """主函数 - 选择要运行的示例"""
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
    else:
        print("\n可用示例:")
        print("  1 - 仅使用安全矩阵乘法 (默认，推荐)")
        print("  2 - 完整安全计算 (需要GPU服务器支持)")
        print("  3 - 单独测试安全矩阵乘法")
        print("\n用法: python example_secure_usage.py [1|2|3]")
        print("默认运行示例1\n")
        example_num = "1"
    
    if example_num == "1":
        example_1_secure_matmul_only()
    elif example_num == "2":
        example_2_full_secure_computation()
    elif example_num == "3":
        example_3_custom_secure_matmul()
    else:
        print(f"错误: 未知的示例编号 '{example_num}'")
        sys.exit(1)


if __name__ == "__main__":
    main()

