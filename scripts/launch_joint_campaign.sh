#!/usr/bin/env bash
# Launch the joint L1 x L2 campaign on Basilica.
#
# This is the thesis-grade experiment: one run combines local-GPU L1
# (engine-config) trials with remote-Basilica L2 (topology) trials,
# with cross-layer stale-signal propagation active (P4).
#
# Preflight checks required env vars, prints a cost + time estimate,
# and refuses to proceed without --yes.
#
# Defaults (--mode full):
#   - config:    examples/qwen3-8b-l1-l2-joint/config.yaml
#   - L1 trials: 14 (per config)
#   - L2 trials: 6  (per config)
#   - wall:      ~2.5 h (14 * ~2-3 min + 6 * ~20 min + overhead)
#   - cost:      ~$30-50 on Basilica spot + ~$0.20 OpenRouter
#
# Smoke (--mode smoke): 2 L1 + 1 L2 trials; ~30-40 min; ~$5.
#
# Usage:
#   export BASILICA_API_TOKEN="..."
#   export OPENROUTER_API_KEY="..."
#   export HF_TOKEN="..."              # optional but recommended
#   ./scripts/launch_joint_campaign.sh --mode smoke --yes
#   ./scripts/launch_joint_campaign.sh --mode full  --yes

set -euo pipefail

CONFIG="examples/qwen3-8b-l1-l2-joint/config.yaml"
MODE="full"
YES="no"
ARTIFACTS_DIR="./basilica-artifacts/qwen3-8b-l1-l2-joint-$(date +%s)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)        MODE="$2"; shift 2 ;;
        --mode=*)      MODE="${1#*=}"; shift ;;
        --config)      CONFIG="$2"; shift 2 ;;
        --artifacts)   ARTIFACTS_DIR="$2"; shift 2 ;;
        --yes|-y)      YES="yes"; shift ;;
        --help|-h)     sed -n '1,30p' "$0"; exit 0 ;;
        *)             echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -n "${BASILICA_API_TOKEN:-}" ]] || fail "BASILICA_API_TOKEN not set"
[[ -n "${OPENROUTER_API_KEY:-}" ]] || fail "OPENROUTER_API_KEY not set"
[[ -f "$CONFIG" ]] || fail "config not found: $CONFIG"

case "$MODE" in
    full)
        LAYER_ARGS=()
        TRIALS_DESC="L1=14 L2=6 (per config)"
        WALL_EST="~2.5 h"
        COST_EST="~\$30-50 Basilica spot + ~\$0.20 OpenRouter"
        ;;
    smoke)
        LAYER_ARGS=(--layer-trials l1_engine=2 --layer-trials l2_topology=1)
        TRIALS_DESC="L1=2 L2=1 (smoke)"
        WALL_EST="~30-40 min"
        COST_EST="~\$5 Basilica spot + ~\$0.05 OpenRouter"
        ;;
    *) fail "unknown --mode: $MODE (expected smoke|full)" ;;
esac

echo "=== joint L1 x L2 campaign launch plan ==="
echo "  config:        $CONFIG"
echo "  mode:          $MODE"
echo "  trials:        $TRIALS_DESC"
echo "  wall-clock:    $WALL_EST"
echo "  cost estimate: $COST_EST"
echo "  artifacts:     $ARTIFACTS_DIR"
echo "  creds present: BASILICA_API_TOKEN OPENROUTER_API_KEY${HF_TOKEN:+ HF_TOKEN}"
echo

if [[ "$YES" != "yes" ]]; then
    echo "Pass --yes to actually launch. This will create a paid Basilica deployment."
    exit 0
fi

echo "=== launching ==="
exec uv run python scripts/orchestrate_iteration_zero.py \
    --config "$CONFIG" \
    --name "autoinfer-l1l2-joint-$(date +%s)" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    "${LAYER_ARGS[@]}"
