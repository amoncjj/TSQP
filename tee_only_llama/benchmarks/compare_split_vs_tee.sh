#!/usr/bin/env bash
# 比较 TEE+GPU 混合推理 vs TEE-Only 推理
# 直接运行 Python，不使用 Gramine (Intel TDX 模式)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEE_GPU_DIR="${PROJECT_ROOT}/tee_gpu"
TEE_ONLY_DIR="${PROJECT_ROOT}/tee_only_llama"

echo "=========================================="
echo "Running TEE+GPU Hybrid Benchmark"
echo "=========================================="
cd "${TEE_GPU_DIR}"
python tee_runner_optimized.py

echo ""
echo "=========================================="
echo "Running TEE-Only Benchmark"
echo "=========================================="
cd "${TEE_ONLY_DIR}"
python tee_runner.py

echo ""
echo "=========================================="
echo "Comparison Results"
echo "=========================================="
echo ""
echo "TEE+GPU Hybrid Results:"
cat "${TEE_GPU_DIR}/tee_gpu_results.json"
echo ""
echo "TEE-Only Results:"
cat "${TEE_ONLY_DIR}/tee_only_results.json"
