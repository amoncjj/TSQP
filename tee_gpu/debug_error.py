#!/usr/bin/env python3
"""
调试错误脚本
"""
import sys
import traceback

# 读取 error.md
with open("/Users/junjiechen/IdeaProjects/TSQP/TSQP/error.md", "r") as f:
    error_text = f.read()

print("=" * 80)
print("完整错误信息:")
print("=" * 80)
print(error_text)
print("=" * 80)

# 分析错误
lines = error_text.strip().split("\n")
print(f"\n总共 {len(lines)} 行")

# 查找关键信息
print("\n关键信息:")
for i, line in enumerate(lines, 1):
    if "File" in line or "Error" in line or "Exception" in line:
        print(f"  {i}: {line}")

# 检查是否被截断
if not any("Error" in line or "Exception" in line for line in lines):
    print("\n⚠️  错误信息可能被截断，没有找到具体的错误类型")
    print("   最后一行:", lines[-1] if lines else "无")
