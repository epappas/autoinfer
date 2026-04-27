#!/usr/bin/env bash
# Launch a joint multi-layer campaign on Basilica.
#
# Two configs supported:
#   - L1xL2 joint (default): engine-config + hardware topology
#   - L1xL2xL3 joint: adds LLM-driven kernel search
#
# Preflight checks required env vars, prints a cost + time estimate,
# and refuses to proceed without --yes.
#
# Modes:
#   full   — config defaults
#   smoke  — small per-layer caps for fast validation (~30-40 min)
#   tiny   — minimum viable (1 L1 + 1 L2 + 2 L3) for sanity checks
#
# Usage:
#   export BASILICA_API_TOKEN="..."
#   export OPENROUTER_API_KEY="..."
#   export HF_TOKEN="..."
#
#   ./scripts/launch_joint_campaign.sh --mode smoke --yes
#   ./scripts/launch_joint_campaign.sh --mode full --yes
#   ./scripts/launch_joint_campaign.sh \
#       --config examples/qwen3-8b-l1-l2-l3-joint/config.yaml \
#       --mode smoke --yes

set -euo pipefail

CONFIG="examples/qwen3-8b-l1-l2-joint/config.yaml"
MODE="full"
YES="no"
ARTIFACTS_DIR=""
BRANCH=""
GPU_MODELS=""
GPUS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)        MODE="$2"; shift 2 ;;
        --mode=*)      MODE="${1#*=}"; shift ;;
        --config)      CONFIG="$2"; shift 2 ;;
        --artifacts)   ARTIFACTS_DIR="$2"; shift 2 ;;
        --branch)      BRANCH="$2"; shift 2 ;;
        --branch=*)    BRANCH="${1#*=}"; shift ;;
        --gpu-models)  GPU_MODELS="$2"; shift 2 ;;
        --gpu-models=*)GPU_MODELS="${1#*=}"; shift ;;
        --gpus)        GPUS="$2"; shift 2 ;;
        --gpus=*)      GPUS="${1#*=}"; shift ;;
        --yes|-y)      YES="yes"; shift ;;
        --help|-h)     sed -n '1,30p' "$0"; exit 0 ;;
        *)             echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -n "${BASILICA_API_TOKEN:-}" ]] || fail "BASILICA_API_TOKEN not set"
[[ -n "${OPENROUTER_API_KEY:-}" ]] || fail "OPENROUTER_API_KEY not set"
[[ -f "$CONFIG" ]] || fail "config not found: $CONFIG"

# Auto-detect which layers the config enables — affects smoke trial caps.
HAS_L1=0; HAS_L2=0; HAS_L3=0
grep -qE "^[[:space:]]+l1_engine:" "$CONFIG" && HAS_L1=1
grep -qE "^[[:space:]]+l2_topology:" "$CONFIG" && HAS_L2=1
grep -qE "^[[:space:]]+l3_kernel:" "$CONFIG" && HAS_L3=1

CONFIG_NAME=$(basename "$(dirname "$CONFIG")")
if [[ -z "$ARTIFACTS_DIR" ]]; then
    ARTIFACTS_DIR="./basilica-artifacts/${CONFIG_NAME}-$(date +%s)"
fi

case "$MODE" in
    full)
        LAYER_ARGS=()
        TRIALS_DESC="per-config defaults (L1=$HAS_L1×N L2=$HAS_L2×N L3=$HAS_L3×N)"
        WALL_EST="~2.5-3 h"
        COST_EST="~\$30-50 Basilica spot + ~\$0.20 OpenRouter"
        ;;
    smoke)
        LAYER_ARGS=()
        SMOKE_DESC=()
        [[ "$HAS_L1" == "1" ]] && { LAYER_ARGS+=(--layer-trials l1_engine=2); SMOKE_DESC+=("L1=2"); }
        [[ "$HAS_L2" == "1" ]] && { LAYER_ARGS+=(--layer-trials l2_topology=1); SMOKE_DESC+=("L2=1"); }
        [[ "$HAS_L3" == "1" ]] && { LAYER_ARGS+=(--layer-trials l3_kernel=3); SMOKE_DESC+=("L3=3"); }
        TRIALS_DESC="${SMOKE_DESC[*]:-(none — config has no enabled layers?)}"
        WALL_EST="~30-40 min"
        COST_EST="~\$5 Basilica spot + ~\$0.05 OpenRouter"
        ;;
    tiny)
        LAYER_ARGS=()
        TINY_DESC=()
        [[ "$HAS_L1" == "1" ]] && { LAYER_ARGS+=(--layer-trials l1_engine=1); TINY_DESC+=("L1=1"); }
        [[ "$HAS_L2" == "1" ]] && { LAYER_ARGS+=(--layer-trials l2_topology=1); TINY_DESC+=("L2=1"); }
        [[ "$HAS_L3" == "1" ]] && { LAYER_ARGS+=(--layer-trials l3_kernel=2); TINY_DESC+=("L3=2"); }
        TRIALS_DESC="${TINY_DESC[*]} (tiny)"
        WALL_EST="~25-35 min"
        COST_EST="~\$3 Basilica spot + ~\$0.03 OpenRouter"
        ;;
    *) fail "unknown --mode: $MODE (expected full|smoke|tiny)" ;;
esac

echo "=== joint campaign launch plan ==="
echo "  config:        $CONFIG"
echo "  layers:        L1=$HAS_L1  L2=$HAS_L2  L3=$HAS_L3"
echo "  mode:          $MODE"
echo "  trials:        $TRIALS_DESC"
echo "  wall-clock:    $WALL_EST"
echo "  cost estimate: $COST_EST"
echo "  artifacts:     $ARTIFACTS_DIR"
echo "  branch:        ${BRANCH:-main (orchestrator default)}"
echo "  gpu-models:    ${GPU_MODELS:-(any matching min-gpu-memory-gb)}"
echo "  gpus:          ${GPUS:-2 (orchestrator default)}"
echo "  creds present: BASILICA_API_TOKEN OPENROUTER_API_KEY${HF_TOKEN:+ HF_TOKEN}"
echo

if [[ "$YES" != "yes" ]]; then
    echo "Pass --yes to actually launch. This will create a paid Basilica deployment."
    exit 0
fi

echo "=== launching ==="
# PYTHONUNBUFFERED so progress lands in stdout/log files in real time
# rather than waiting for Python's 4KB stdout buffer to fill.
export PYTHONUNBUFFERED=1
BRANCH_ARGS=()
[[ -n "$BRANCH" ]] && BRANCH_ARGS=(--branch "$BRANCH")
GPU_MODEL_ARGS=()
[[ -n "$GPU_MODELS" ]] && GPU_MODEL_ARGS=(--gpu-models "$GPU_MODELS")
GPU_COUNT_ARGS=()
[[ -n "$GPUS" ]] && GPU_COUNT_ARGS=(--gpus "$GPUS")
exec uv run python -u scripts/orchestrate_iteration_zero.py \
    --config "$CONFIG" \
    --name "autoinfer-${CONFIG_NAME}-$(date +%s)" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    "${BRANCH_ARGS[@]}" \
    "${GPU_MODEL_ARGS[@]}" \
    "${GPU_COUNT_ARGS[@]}" \
    "${LAYER_ARGS[@]}"
