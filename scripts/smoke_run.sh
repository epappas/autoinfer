#!/usr/bin/env bash
# Run a 3-trial smoke of the main example config. Expected wall-clock
# on a single H100: roughly 30-60 minutes.
#
# Prereqs (see docs/runbook-iteration-zero.md):
#   - reference replica up on port 8001 (CUDA_VISIBLE_DEVICES=1)
#   - trace and quality prompts prepared via scripts/fetch_sharegpt.py
#   - GPU 0 free for the candidate
#
# Output: ./runs/qwen3-8b-l1-slice/*.json, run.log.

set -euo pipefail

CONFIG="${CONFIG:-examples/qwen3-8b-l1-slice/config.yaml}"
TRIALS="${TRIALS:-3}"
LOG_DIR="${LOG_DIR:-./runs}"
LOG_FILE="${LOG_DIR}/smoke_run.log"

mkdir -p "${LOG_DIR}"
echo "smoke: config=${CONFIG} trials=${TRIALS} log=${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    uv run autoinfer run "${CONFIG}" --max-trials "${TRIALS}" 2>&1 | tee "${LOG_FILE}"
