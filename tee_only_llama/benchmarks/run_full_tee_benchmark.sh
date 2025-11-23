#!/usr/bin/env bash
# TEE-Only 推理基准测试
# 直接运行 Python，不使用 Gramine (Intel TDX 模式)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEE_ONLY_DIR="${PROJECT_ROOT}/tee_only_llama"

cd "${TEE_ONLY_DIR}"

echo "Running TEE-Only Inference Benchmark..."
python tee_runner.py

echo ""
echo "Benchmark completed. Results:"
cat tee_only_results.json
