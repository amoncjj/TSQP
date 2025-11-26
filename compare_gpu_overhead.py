"""
对比 TEE+GPU 混合模式的实际开销
分析数据传输、同步等隐藏开销
"""
import json
import sys
from pathlib import Path

def load_results(file_path):
    """加载性能测试结果"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_overhead(results):
    """分析各项开销"""
    timing = results.get('timing', {})
    
    # 提取各项指标
    total_time = timing.get('total_ms', 0)
    gpu_compute = timing.get('gpu_compute_ms', 0)
    cpu_compute = timing.get('cpu_compute_ms', 0)
    transfer_to_gpu = timing.get('transfer_to_gpu_ms', 0)
    transfer_to_cpu = timing.get('transfer_to_cpu_ms', 0)
    
    # 加密相关（如果存在）
    encryption = timing.get('encryption_ms', 0)
    decryption = timing.get('decryption_ms', 0)
    masking = timing.get('masking_ms', 0)
    
    # 计算隐藏开销
    accounted_time = (gpu_compute + cpu_compute + transfer_to_gpu + 
                      transfer_to_cpu + encryption + decryption + masking)
    hidden_overhead = total_time - accounted_time
    
    return {
        'total_time': total_time,
        'gpu_compute': gpu_compute,
        'cpu_compute': cpu_compute,
        'transfer_to_gpu': transfer_to_gpu,
        'transfer_to_cpu': transfer_to_cpu,
        'total_transfer': transfer_to_gpu + transfer_to_cpu,
        'encryption': encryption,
        'decryption': decryption,
        'masking': masking,
        'total_crypto': encryption + decryption + masking,
        'accounted_time': accounted_time,
        'hidden_overhead': hidden_overhead,
        'hidden_overhead_pct': (hidden_overhead / total_time * 100) if total_time > 0 else 0
    }

def print_analysis(name, data):
    """打印分析结果"""
    print(f"\n{'='*80}")
    print(f" {name}")
    print(f"{'='*80}")
    
    print(f"\n【总时间】")
    print(f"  总耗时: {data['total_time']:.2f} ms")
    
    print(f"\n【GPU 计算】")
    print(f"  GPU 纯计算: {data['gpu_compute']:.2f} ms ({data['gpu_compute']/data['total_time']*100:.1f}%)")
    
    print(f"\n【CPU 计算】")
    print(f"  CPU 计算: {data['cpu_compute']:.2f} ms ({data['cpu_compute']/data['total_time']*100:.1f}%)")
    
    print(f"\n【数据传输】")
    print(f"  CPU → GPU: {data['transfer_to_gpu']:.2f} ms")
    print(f"  GPU → CPU: {data['transfer_to_cpu']:.2f} ms")
    print(f"  传输总计: {data['total_transfer']:.2f} ms ({data['total_transfer']/data['total_time']*100:.1f}%)")
    
    if data['total_crypto'] > 0:
        print(f"\n【加密/解密】")
        if data['encryption'] > 0:
            print(f"  加密: {data['encryption']:.2f} ms")
        if data['decryption'] > 0:
            print(f"  解密: {data['decryption']:.2f} ms")
        if data['masking'] > 0:
            print(f"  掩码: {data['masking']:.2f} ms")
        print(f"  加密总计: {data['total_crypto']:.2f} ms ({data['total_crypto']/data['total_time']*100:.1f}%)")
    
    print(f"\n【开销分析】")
    print(f"  已统计时间: {data['accounted_time']:.2f} ms ({data['accounted_time']/data['total_time']*100:.1f}%)")
    print(f"  隐藏开销: {data['hidden_overhead']:.2f} ms ({data['hidden_overhead_pct']:.1f}%)")
    print(f"    ↑ 包括：GPU 同步等待、Kernel 启动、内存分配等")
    
    # GPU 有效利用率
    if data['total_time'] > 0:
        gpu_utilization = data['gpu_compute'] / data['total_time'] * 100
        print(f"\n【GPU 利用率】")
        print(f"  有效利用率: {gpu_utilization:.1f}%")
        print(f"  空闲时间: {data['total_time'] - data['gpu_compute']:.2f} ms ({100-gpu_utilization:.1f}%)")

def compare_schemes(ours_data, otp_data):
    """对比两种加密方案"""
    print(f"\n{'='*80}")
    print(f" 方案对比")
    print(f"{'='*80}")
    
    print(f"\n{'指标':<30} {'Our Scheme':>20} {'OTP Scheme':>20}")
    print(f"{'-'*80}")
    
    metrics = [
        ('总时间', 'total_time'),
        ('GPU 计算', 'gpu_compute'),
        ('CPU 计算', 'cpu_compute'),
        ('数据传输', 'total_transfer'),
        ('加密开销', 'total_crypto'),
        ('隐藏开销', 'hidden_overhead'),
    ]
    
    for label, key in metrics:
        ours_val = ours_data.get(key, 0)
        otp_val = otp_data.get(key, 0)
        diff = ours_val - otp_val
        diff_pct = (diff / otp_val * 100) if otp_val > 0 else 0
        
        print(f"{label:<30} {ours_val:>17.2f} ms {otp_val:>17.2f} ms", end="")
        if abs(diff_pct) > 5:
            print(f"  ({diff_pct:+.1f}%)", end="")
        print()
    
    # 分析差异
    print(f"\n【关键发现】")
    
    crypto_diff = ours_data['total_crypto'] - otp_data['total_crypto']
    if crypto_diff > 10:
        print(f"  • Our Scheme 的加密开销比 OTP 高 {crypto_diff:.2f} ms")
        print(f"    原因：矩阵变换（einsum）比加法掩码更复杂")
    
    transfer_diff = ours_data['total_transfer'] - otp_data['total_transfer']
    if abs(transfer_diff) > 10:
        print(f"  • 数据传输差异：{transfer_diff:+.2f} ms")
        print(f"    原因：两种方案的数据传输模式基本相同")
    
    gpu_diff = ours_data['gpu_compute'] - otp_data['gpu_compute']
    if abs(gpu_diff) > 10:
        print(f"  • GPU 计算差异：{gpu_diff:+.2f} ms")
        print(f"    原因：在加密数据上执行相同的计算，理论上应该相同")

def estimate_pure_gpu(data):
    """估算纯 GPU 计算的理论时间"""
    print(f"\n{'='*80}")
    print(f" 纯 GPU 计算理论估算")
    print(f"{'='*80}")
    
    # 假设：纯 GPU = GPU 计算时间 + 少量同步开销（~5%）
    pure_gpu_time = data['gpu_compute'] * 1.05
    
    print(f"\n  当前混合模式总时间: {data['total_time']:.2f} ms")
    print(f"  纯 GPU 计算时间: {data['gpu_compute']:.2f} ms")
    print(f"  估算纯 GPU 总时间: {pure_gpu_time:.2f} ms (含 5% 同步开销)")
    print(f"\n  混合模式 vs 纯 GPU:")
    print(f"    慢了: {data['total_time'] - pure_gpu_time:.2f} ms")
    print(f"    慢了: {(data['total_time'] / pure_gpu_time - 1) * 100:.1f}%")
    
    print(f"\n  【开销构成】")
    overhead_items = [
        ('数据传输', data['total_transfer']),
        ('加密/解密', data['total_crypto']),
        ('CPU 计算', data['cpu_compute']),
        ('隐藏开销（同步等待）', data['hidden_overhead']),
    ]
    
    total_overhead = data['total_time'] - pure_gpu_time
    for label, value in overhead_items:
        pct = (value / total_overhead * 100) if total_overhead > 0 else 0
        print(f"    {label}: {value:.2f} ms ({pct:.1f}%)")

def main():
    """主函数"""
    print(f"\n{'='*80}")
    print(f"{'GPU 开销对比分析':^80}")
    print(f"{'='*80}")
    
    # 查找结果文件
    ours_file = Path("tee_gpu_results_ours.json")
    otp_file = Path("tee_gpu_results_otp.json")
    
    if not ours_file.exists() or not otp_file.exists():
        print("\n❌ 错误：找不到结果文件")
        print(f"  需要的文件:")
        print(f"    - {ours_file}")
        print(f"    - {otp_file}")
        print(f"\n  请先运行以下命令生成结果：")
        print(f"    python tee_gpu/tee_runner_ours.py")
        print(f"    python tee_gpu/tee_runner_otp.py")
        return
    
    # 加载并分析结果
    ours_results = load_results(ours_file)
    otp_results = load_results(otp_file)
    
    ours_data = analyze_overhead(ours_results)
    otp_data = analyze_overhead(otp_results)
    
    # 打印详细分析
    print_analysis("Our Encryption Scheme", ours_data)
    print_analysis("OTP Scheme", otp_data)
    
    # 对比两种方案
    compare_schemes(ours_data, otp_data)
    
    # 估算纯 GPU 性能
    estimate_pure_gpu(ours_data)
    
    print(f"\n{'='*80}")
    print(f"分析完成！详细解释请查看: gpu_overhead_analysis.md")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

