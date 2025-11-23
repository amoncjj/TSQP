#!/usr/bin/env bash
# TEE+GPU 混合推理基准测试
# 直接运行 Python，不使用 Gramine (Intel TDX 模式)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEE_GPU_DIR="${PROJECT_ROOT}/tee_gpu"

cd "${TEE_GPU_DIR}"

echo "Running TEE+GPU Hybrid Inference Benchmark..."
python tee_runner_optimized.py

echo ""
echo "Benchmark completed. Results:"
cat tee_gpu_results.json
