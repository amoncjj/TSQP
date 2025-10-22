#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEE_GPU_DIR="${PROJECT_ROOT}/tee_gpu"

export LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-${PROJECT_ROOT}/weights/llama3.2-1b}" \
       LLAMA_GPU_RESULT_PATH="${LLAMA_GPU_RESULT_PATH:-${TEE_GPU_DIR}/tee_gpu_benchmark.json}" \
       LLAMA_PROMPT_PATH="${LLAMA_PROMPT_PATH:-${TEE_GPU_DIR}/prompts.txt}" \
       LLAMA_TEE_BATCH_SIZE="${LLAMA_TEE_BATCH_SIZE:-4}" \
       LLAMA_MAX_LENGTH="${LLAMA_MAX_LENGTH:-256}" \
       LLAMA_TEMPERATURE="${LLAMA_TEMPERATURE:-0.7}" \
       LLAMA_TOP_P="${LLAMA_TOP_P:-0.9}" \
       LLAMA_GPU_ENDPOINT="${LLAMA_GPU_ENDPOINT:-localhost:50051}"

if [[ ! -f "${LLAMA_PROMPT_PATH}" ]]; then
  cat <<'EOF' > "${LLAMA_PROMPT_PATH}"
Describe the benefits of running large language model inference inside a TEE.
Summarize the security challenges of heterogeneous inference pipelines.
Provide five potential attack vectors against GPU-assisted inference and their mitigations.
EOF
fi

pushd "${TEE_GPU_DIR}" > /dev/null
make clean
make GRAMINE_INSTALL_DIR=${GRAMINE_INSTALL_DIR:-} SGX=1 server.manifest.sgx server.sig tee_runner.manifest.sgx tee_runner.sig
popd > /dev/null

pushd "${TEE_GPU_DIR}" > /dev/null
# Start the GPU-hosted linear service outside Gramine (host environment assumed)
python3 server.py &
SERVER_PID=$!
trap "kill ${SERVER_PID}" EXIT
sleep 5

# Run the split benchmark inside Gramine
gramine-sgx ./tee_runner.manifest.sgx

kill ${SERVER_PID}
trap - EXIT
popd > /dev/null

cat "${LLAMA_GPU_RESULT_PATH}"
