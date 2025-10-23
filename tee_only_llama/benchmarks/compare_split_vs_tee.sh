#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEE_GPU_DIR="${PROJECT_ROOT}/tee_gpu"
TEE_ONLY_DIR="${PROJECT_ROOT}/tee_only_llama"

# Shared configuration
export LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-${PROJECT_ROOT}/weights/llama3.2-1b}"
export LLAMA_PROMPT_PATH="${LLAMA_PROMPT_PATH:-${TEE_ONLY_DIR}/prompts.txt}"
export LLAMA_TEE_BATCH_SIZE="${LLAMA_TEE_BATCH_SIZE:-4}"
export LLAMA_MAX_LENGTH="${LLAMA_MAX_LENGTH:-256}"
export LLAMA_TEMPERATURE="${LLAMA_TEMPERATURE:-0.7}"
export LLAMA_TOP_P="${LLAMA_TOP_P:-0.9}"
export LLAMA_GPU_ENDPOINT="${LLAMA_GPU_ENDPOINT:-localhost:50051}"

[[ -f "${LLAMA_PROMPT_PATH}" ]] || cat <<'EOF' > "${LLAMA_PROMPT_PATH}"
Describe the benefits of running large language model inference inside a TEE.
Summarize the security challenges of heterogeneous inference pipelines.
Provide five potential attack vectors against GPU-assisted inference and their mitigations.
EOF

# Run split benchmark
pushd "${TEE_GPU_DIR}" > /dev/null
make clean
make SGX=1 server.manifest.sgx server.sig tee_runner.manifest.sgx tee_runner.sig
python3 server.py &
SERVER_PID=$!
trap "kill ${SERVER_PID}" EXIT
sleep 5
gramine-sgx ./tee_runner.manifest.sgx
kill ${SERVER_PID}
trap - EXIT
popd > /dev/null

# Run TEE-only benchmark
pushd "${TEE_ONLY_DIR}" > /dev/null
make clean
make SGX=1 tee_only.manifest.sgx tee_only.sig
gramine-sgx tee_only tee_runner.py
popd > /dev/null

jq -n \
  --arg split "$(cat "${TEE_GPU_DIR}/tee_gpu_benchmark.json")" \
  --arg teeonly "$(cat "${TEE_ONLY_DIR}/tee_only_results.json")" \
  '{split: ($split | fromjson), tee_only: ($teeonly | fromjson)}'
