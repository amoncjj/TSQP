#!/usr/bin/env python3
"""
ZeroMQ性能日志分析工具
分析zmq_performance.log,生成详细的性能报告
"""
import re
from typing import List, Dict
from collections import defaultdict


def parse_log(log_file: str) -> List[Dict]:
    """解析日志文件"""
    calls = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # 跳过头部
    data_start = False
    for line in lines:
        if line.startswith("=====") and data_start:
            break
        if data_start and line.strip() and not line.startswith("SUMMARY"):
            parts = line.split()
            if len(parts) >= 10 and parts[0].isdigit():
                try:
                    call = {
                        'id': int(parts[0]),
                        'method': parts[1],
                        'serialize_ms': float(parts[2]),
                        'send_ms': float(parts[3]),
                        'recv_ms': float(parts[4]),
                        'deserialize_ms': float(parts[5]),
                        'total_ms': float(parts[6]),
                        'sent_kb': float(parts[7]),
                        'recv_kb': float(parts[8]),
                        'throughput_mbps': float(parts[9])
                    }
                    calls.append(call)
                except (ValueError, IndexError):
                    continue
        if "ID" in line and "Method" in line:
            data_start = True
    
    return calls


def analyze_by_method(calls: List[Dict]) -> Dict:
    """按方法分组分析"""
    method_stats = defaultdict(lambda: {
        'count': 0,
        'total_time': 0.0,
        'serialize_time': 0.0,
        'send_time': 0.0,
        'recv_time': 0.0,
        'deserialize_time': 0.0,
        'total_sent': 0.0,
        'total_recv': 0.0,
    })
    
    for call in calls:
        method = call['method']
        stats = method_stats[method]
        stats['count'] += 1
        stats['total_time'] += call['total_ms']
        stats['serialize_time'] += call['serialize_ms']
        stats['send_time'] += call['send_ms']
        stats['recv_time'] += call['recv_ms']
        stats['deserialize_time'] += call['deserialize_ms']
        stats['total_sent'] += call['sent_kb']
        stats['total_recv'] += call['recv_kb']
    
    return dict(method_stats)


def print_report(calls: List[Dict], method_stats: Dict):
    """打印分析报告"""
    if not calls:
        print("No data found in log file!")
        return
    
    print("\n" + "="*100)
    print(f"{'ZeroMQ Performance Analysis Report':^100}")
    print("="*100)
    
    # 总体统计
    total_calls = len(calls)
    total_time = sum(c['total_ms'] for c in calls)
    total_serialize = sum(c['serialize_ms'] for c in calls)
    total_send = sum(c['send_ms'] for c in calls)
    total_recv = sum(c['recv_ms'] for c in calls)
    total_deserialize = sum(c['deserialize_ms'] for c in calls)
    total_sent = sum(c['sent_kb'] for c in calls)
    total_recv_data = sum(c['recv_kb'] for c in calls)
    
    print(f"\n{'OVERALL STATISTICS':^100}")
    print("-"*100)
    print(f"Total RPC Calls:      {total_calls}")
    print(f"Total Time:           {total_time:.2f} ms")
    print(f"Average Time/Call:    {total_time/total_calls:.3f} ms")
    print(f"\nTime Breakdown (Average per call):")
    print(f"  Serialize:          {total_serialize/total_calls:>8.3f} ms  ({total_serialize/total_time*100:>5.1f}%)")
    print(f"  Send:               {total_send/total_calls:>8.3f} ms  ({total_send/total_time*100:>5.1f}%)")
    print(f"  Receive:            {total_recv/total_calls:>8.3f} ms  ({total_recv/total_time*100:>5.1f}%)")
    print(f"  Deserialize:        {total_deserialize/total_calls:>8.3f} ms  ({total_deserialize/total_time*100:>5.1f}%)")
    print(f"\nData Transfer:")
    print(f"  Total Sent:         {total_sent/1024:.2f} MB  ({total_sent/total_calls:.2f} KB/call)")
    print(f"  Total Received:     {total_recv_data/1024:.2f} MB  ({total_recv_data/total_calls:.2f} KB/call)")
    print(f"  Total:              {(total_sent+total_recv_data)/1024:.2f} MB")
    print(f"  Throughput:         {(total_sent+total_recv_data)/1024/(total_time/1000):.2f} MB/s")
    
    # 按方法统计
    print(f"\n{'STATISTICS BY METHOD':^100}")
    print("-"*100)
    print(f"{'Method':<20} {'Calls':>8} {'Avg Time(ms)':>15} {'Total Time(ms)':>17} "
          f"{'Avg Sent(KB)':>15} {'Avg Recv(KB)':>15} {'% of Total':>12}")
    print("-"*100)
    
    # 按总时间排序
    sorted_methods = sorted(method_stats.items(), 
                           key=lambda x: x[1]['total_time'], 
                           reverse=True)
    
    for method, stats in sorted_methods:
        count = stats['count']
        avg_time = stats['total_time'] / count
        total_method_time = stats['total_time']
        avg_sent = stats['total_sent'] / count
        avg_recv = stats['total_recv'] / count
        pct = total_method_time / total_time * 100
        
        print(f"{method:<20} {count:>8} {avg_time:>15.3f} {total_method_time:>17.2f} "
              f"{avg_sent:>15.2f} {avg_recv:>15.2f} {pct:>11.1f}%")
    
    # 瓶颈分析
    print(f"\n{'BOTTLENECK ANALYSIS':^100}")
    print("-"*100)
    
    # 找出最慢的调用
    slowest_calls = sorted(calls, key=lambda x: x['total_ms'], reverse=True)[:5]
    print("\nTop 5 Slowest Calls:")
    print(f"{'ID':<8} {'Method':<20} {'Total(ms)':>12} {'Serialize':>12} {'Send':>10} {'Recv':>10} {'Deserialize':>14}")
    print("-"*100)
    for call in slowest_calls:
        print(f"{call['id']:<8} {call['method']:<20} {call['total_ms']:>12.3f} "
              f"{call['serialize_ms']:>12.3f} {call['send_ms']:>10.3f} "
              f"{call['recv_ms']:>10.3f} {call['deserialize_ms']:>14.3f}")
    
    # 找出数据量最大的调用
    largest_calls = sorted(calls, key=lambda x: x['sent_kb'] + x['recv_kb'], reverse=True)[:5]
    print("\nTop 5 Largest Data Transfers:")
    print(f"{'ID':<8} {'Method':<20} {'Sent(KB)':>12} {'Recv(KB)':>12} {'Total(KB)':>12} {'Throughput(MB/s)':>18}")
    print("-"*100)
    for call in largest_calls:
        total_kb = call['sent_kb'] + call['recv_kb']
        print(f"{call['id']:<8} {call['method']:<20} {call['sent_kb']:>12.2f} "
              f"{call['recv_kb']:>12.2f} {total_kb:>12.2f} {call['throughput_mbps']:>18.2f}")
    
    # 优化建议
    print(f"\n{'OPTIMIZATION RECOMMENDATIONS':^100}")
    print("-"*100)
    
    serialize_pct = total_serialize / total_time * 100
    send_recv_pct = (total_send + total_recv) / total_time * 100
    
    if serialize_pct > 30:
        print(f"⚠️  Serialization占{serialize_pct:.1f}%的时间 - 建议:")
        print("   1. 使用共享内存替代msgpack序列化")
        print("   2. 使用更高效的序列化格式(如protobuf)")
        print("   3. 减少传输的数据量(如使用bfloat16)")
    
    if send_recv_pct > 30:
        print(f"⚠️  网络传输占{send_recv_pct:.1f}%的时间 - 建议:")
        print("   1. 确认使用IPC而非TCP")
        print("   2. 增大ZeroMQ缓冲区")
        print("   3. 考虑批量传输")
    
    # 计算RPC调用频率
    method_counts = {m: s['count'] for m, s in method_stats.items()}
    if max(method_counts.values()) > 10:
        print(f"⚠️  RPC调用次数过多({total_calls}次) - 建议:")
        print("   1. 算子融合,减少RPC调用次数")
        print("   2. 批量处理多个操作")
        print("   3. 缓存重复计算的结果")
    
    print("="*100 + "\n")


def main():
    import sys
    
    log_file = sys.argv[1] if len(sys.argv) > 1 else "zmq_performance.log"
    
    try:
        calls = parse_log(log_file)
        if not calls:
            print(f"Error: No valid data found in {log_file}")
            return
        
        method_stats = analyze_by_method(calls)
        print_report(calls, method_stats)
        
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found!")
        print("Usage: python analyze_zmq_performance.py [log_file]")
    except Exception as e:
        print(f"Error analyzing log: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
