#!/usr/bin/env python3
"""
分析 zmq_performance.log 中的传输量
"""
import re
from collections import defaultdict

def parse_log(log_file):
    """解析日志文件"""
    operations = defaultdict(lambda: {"count": 0, "sent": 0, "recv": 0, "total_time": 0})
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # 跳过头部
    data_lines = [l for l in lines if l.strip() and not l.startswith('=') and not l.startswith('ID') and not l.startswith('ZeroMQ') and not l.startswith('SUMMARY') and not l.startswith('Total') and not l.startswith('Average')]
    
    for line in data_lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        
        try:
            method = parts[1]
            sent_kb = float(parts[7])
            recv_kb = float(parts[8])
            total_ms = float(parts[6])
            
            operations[method]["count"] += 1
            operations[method]["sent"] += sent_kb
            operations[method]["recv"] += recv_kb
            operations[method]["total_time"] += total_ms
        except (ValueError, IndexError):
            continue
    
    return operations

def analyze_operations(operations):
    """分析操作统计"""
    print("="*80)
    print("传输量分析 - LLaMA 3.2-1B, 1024 tokens")
    print("="*80)
    
    # 按传输量排序
    sorted_ops = sorted(operations.items(), key=lambda x: x[1]["sent"] + x[1]["recv"], reverse=True)
    
    total_sent = sum(op["sent"] for op in operations.values())
    total_recv = sum(op["recv"] for op in operations.values())
    total_transfer = total_sent + total_recv
    
    print(f"\n{'操作':<15} {'次数':>6} {'发送(MB)':>12} {'接收(MB)':>12} {'总量(MB)':>12} {'占比':>8} {'平均/次(MB)':>14}")
    print("-"*80)
    
    for method, stats in sorted_ops:
        count = stats["count"]
        sent_mb = stats["sent"] / 1024
        recv_mb = stats["recv"] / 1024
        total_mb = (stats["sent"] + stats["recv"]) / 1024
        percentage = (total_mb / (total_transfer / 1024)) * 100
        avg_mb = total_mb / count if count > 0 else 0
        
        print(f"{method:<15} {count:>6} {sent_mb:>12.2f} {recv_mb:>12.2f} {total_mb:>12.2f} {percentage:>7.1f}% {avg_mb:>14.2f}")
    
    print("-"*80)
    print(f"{'总计':<15} {sum(op['count'] for op in operations.values()):>6} {total_sent/1024:>12.2f} {total_recv/1024:>12.2f} {total_transfer/1024:>12.2f} {'100.0%':>8}")
    print("="*80)
    
    # 详细分析 Matmul
    print("\n" + "="*80)
    print("Matmul 详细分析")
    print("="*80)
    
    if "Matmul" in operations:
        matmul_stats = operations["Matmul"]
        count = matmul_stats["count"]
        avg_sent = matmul_stats["sent"] / count / 1024
        avg_recv = matmul_stats["recv"] / count / 1024
        avg_total = (matmul_stats["sent"] + matmul_stats["recv"]) / count / 1024
        
        print(f"总调用次数: {count}")
        print(f"平均发送量: {avg_sent:.2f} MB/次")
        print(f"平均接收量: {avg_recv:.2f} MB/次")
        print(f"平均总量:   {avg_total:.2f} MB/次")
        print(f"\n预计模式:")
        print(f"  - Q @ K^T:    发送 ~16 MB,  接收 ~128 MB  (Attention Scores)")
        print(f"  - Scores @ V: 发送 ~136 MB, 接收 ~8 MB   (Attention Output)")
        print(f"\n问题: Attention Scores [1, 32, 1024, 1024] = 128 MB 太大！")
    
    # 优化建议
    print("\n" + "="*80)
    print("优化建议")
    print("="*80)
    
    matmul_total = operations.get("Matmul", {}).get("sent", 0) + operations.get("Matmul", {}).get("recv", 0)
    matmul_mb = matmul_total / 1024
    
    print(f"\n1. 启用 bfloat16 (立即可行)")
    print(f"   当前传输量: {total_transfer/1024:.2f} MB")
    print(f"   优化后:     {total_transfer/1024/2:.2f} MB  (减少 50%)")
    
    print(f"\n2. Fused Attention (推荐)")
    print(f"   Matmul 传输量: {matmul_mb:.2f} MB  (占 {matmul_mb/(total_transfer/1024)*100:.1f}%)")
    print(f"   优化后:        {matmul_mb*0.3:.2f} MB  (减少 70%)")
    print(f"   总传输量:      {(total_transfer/1024 - matmul_mb*0.7):.2f} MB")
    
    print(f"\n3. 组合优化 (bfloat16 + Fused Attention)")
    print(f"   预期传输量: {(total_transfer/1024 - matmul_mb*0.7)/2:.2f} MB  (减少 ~80%)")
    print(f"   性能提升:   3-5x")
    
    print("\n" + "="*80)

def main():
    log_file = "tee_gpu/zmq_performance.log"
    
    try:
        operations = parse_log(log_file)
        analyze_operations(operations)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {log_file}")
        print("请先运行 tee_runner_optimized.py 生成日志")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
