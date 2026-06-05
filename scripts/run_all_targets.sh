#!/usr/bin/env bash
set -euo pipefail

# Drive run_one_target.sh over all 96 unique ContextBench targets in deterministic order.
# Idempotent: re-running skips targets whose 4 conditions have all succeeded.
# Designed for: tmux new-session -d -s fullrun "./scripts/run_all_targets.sh ..."

usage() {
    cat <<'EOF'
Usage: run_all_targets.sh <execution_root> [extra-orchestrator-args...]

Examples:
  ./scripts/run_all_targets.sh data/contextbench_phase2/execution_full
  ./scripts/run_all_targets.sh data/contextbench_phase2/execution_full --timeout-seconds 1800
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

EXECUTION_ROOT="$1"
shift
EXTRA_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="data/contextbench_phase2/prompt_manifest.jsonl"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST" >&2
    exit 2
fi

mkdir -p "$EXECUTION_ROOT/logs"
DRIVER_LOG="$EXECUTION_ROOT/logs/driver.log"

mapfile -t TARGETS < <(jq -r '.target_instance_id' "$MANIFEST" | sort -u)
TOTAL=${#TARGETS[@]}

echo "[$(date -u +%FT%TZ)] driver start | targets=$TOTAL execution_root=$EXECUTION_ROOT" | tee -a "$DRIVER_LOG"

SUCCESS=0
FAIL=0
START=$(date +%s)

for IDX in "${!TARGETS[@]}"; do
    target="${TARGETS[$IDX]}"
    COUNT=$((IDX+1))
    elapsed=$(( $(date +%s) - START ))
    echo "" | tee -a "$DRIVER_LOG"
    echo "=== [$COUNT/$TOTAL] target=$target | elapsed=${elapsed}s | ok=$SUCCESS fail=$FAIL ===" | tee -a "$DRIVER_LOG"

    set +e
    bash scripts/run_one_target.sh "$target" "$EXECUTION_ROOT" "${EXTRA_ARGS[@]}"
    rc=$?
    set -e

    if [[ $rc -eq 0 ]]; then
        SUCCESS=$((SUCCESS+1))
    else
        FAIL=$((FAIL+1))
        echo "[driver] target=$target failed with exit $rc — continuing" | tee -a "$DRIVER_LOG"
    fi
done

total_elapsed=$(( $(date +%s) - START ))
echo "" | tee -a "$DRIVER_LOG"
echo "[$(date -u +%FT%TZ)] driver done | total=${TOTAL} ok=$SUCCESS fail=$FAIL elapsed=${total_elapsed}s" | tee -a "$DRIVER_LOG"

python3 scripts/aggregate_predictions.py "$EXECUTION_ROOT" | tee -a "$DRIVER_LOG"
