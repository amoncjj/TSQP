#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEE_GPU_DIR="${PROJECT_ROOT}/tee_gpu"
TEE_ONLY_DIR="${PROJECT_ROOT}/tee_only_llama"

export LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-${PROJECT_ROOT}/weights/llama3.2-1b}" \
       LLAMA_TEE_RESULT_PATH="${LLAMA_TEE_RESULT_PATH:-${TEE_ONLY_DIR}/tee_only_results.json}" \
       LLAMA_PROMPT_PATH="${LLAMA_PROMPT_PATH:-${TEE_ONLY_DIR}/prompts.txt}" \
       LLAMA_TEE_BATCH_SIZE="${LLAMA_TEE_BATCH_SIZE:-4}" \
       LLAMA_MAX_LENGTH="${LLAMA_MAX_LENGTH:-256}" \
       LLAMA_TEMPERATURE="${LLAMA_TEMPERATURE:-0.7}" \
       LLAMA_TOP_P="${LLAMA_TOP_P:-0.9}"

if [[ ! -f "${LLAMA_PROMPT_PATH}" ]]; then
  cat <<'EOF' > "${LLAMA_PROMPT_PATH}"
Describe the benefits of running large language model inference inside a TEE.
Summarize the security challenges of heterogeneous inference pipelines.
Provide five potential attack vectors against GPU-assisted inference and their mitigations.
EOF
fi

pushd "${TEE_GPU_DIR}" > /dev/null
make clean
make SGX=1 tee_runner.manifest.sgx tee_runner.sig
popd > /dev/null

pushd "${TEE_GPU_DIR}" > /dev/null
gramine-sgx ./tee_runner.manifest.sgx
popd > /dev/null

cat "${LLAMA_TEE_RESULT_PATH}"
