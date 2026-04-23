#!/usr/bin/env bash
# Start the FP16 reference replica on the specified GPU in the background.
#
# Usage:
#   bash scripts/start_reference.sh <model> <port> <cuda-device>
#
# Example:
#   bash scripts/start_reference.sh Qwen/Qwen3-8B 8001 1
#
# Health-checks the replica every 5s up to 10 minutes; exits 0 when ready.
# The vllm process stays running in the background with output in
# ./runs/reference.log; stop it with `pkill -f "vllm serve"`.

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-8B}"
PORT="${2:-8001}"
CUDA_DEVICE="${3:-1}"
LOG_DIR="${LOG_DIR:-./runs}"
TIMEOUT_S="${TIMEOUT_S:-600}"
MEM_UTIL="${MEM_UTIL:-0.85}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/reference.log"

echo "starting reference replica: model=${MODEL} port=${PORT} cuda=${CUDA_DEVICE}"
echo "logs -> ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    nohup vllm serve "${MODEL}" \
        --port "${PORT}" \
        --dtype auto \
        --gpu-memory-utilization "${MEM_UTIL}" \
    > "${LOG_FILE}" 2>&1 &

REF_PID=$!
echo "reference pid: ${REF_PID}"

start_time=$(date +%s)
while true; do
    now=$(date +%s)
    elapsed=$((now - start_time))
    if [[ "${elapsed}" -ge "${TIMEOUT_S}" ]]; then
        echo "TIMEOUT: reference not ready after ${TIMEOUT_S}s"
        exit 1
    fi
    if ! kill -0 "${REF_PID}" 2>/dev/null; then
        echo "ERROR: reference exited during startup (pid ${REF_PID})"
        tail -n 50 "${LOG_FILE}" >&2 || true
        exit 1
    fi
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/v1/models" | grep -q 200; then
        echo "READY: reference at http://127.0.0.1:${PORT}  elapsed=${elapsed}s"
        echo "${REF_PID}" > "${LOG_DIR}/reference.pid"
        exit 0
    fi
    sleep 5
done
