#!/usr/bin/env bash
set -euo pipefail

# Run all 4 conditions for ONE ContextBench target end-to-end.
# Steps:
#   1. Pull the public sandbox image (with retries).
#   2. Build a local rebuild that adds OpenHands inside the image,
#      tagged with the SAME name so the orchestrator picks it up via --no-pull.
#   3. Invoke the orchestrator in smoke mode for this target (runs 4 conditions).
#   4. Remove the rebuilt image and per-run workspace dirs to reclaim disk.
#
# Per-run artifacts are preserved at:
#   <execution_root>/runs/<target>/<condition>/{prediction.json,patch.diff,run_meta.json,stdout.jsonl,stderr.log}

usage() {
    cat <<'EOF'
Usage: run_one_target.sh <target_instance_id> <execution_root> [extra-orchestrator-args...]

Examples:
  ./scripts/run_one_target.sh astropy__astropy-15082 data/contextbench_phase2/execution_smoke
  ./scripts/run_one_target.sh django__django-11776  data/contextbench_phase2/execution_full --timeout-seconds 1800
EOF
}

if [[ $# -lt 2 ]]; then
    usage
    exit 1
fi

TARGET="$1"
EXECUTION_ROOT="$2"
shift 2
EXTRA_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="data/contextbench_phase2/prompt_manifest.jsonl"
DOCKERFILE="scripts/contextbench/openhands_image.Dockerfile"

for cmd in jq docker python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: required command not on PATH: $cmd" >&2
        exit 2
    fi
done

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST (run from repo root)" >&2
    exit 2
fi
if [[ ! -f "$DOCKERFILE" ]]; then
    echo "ERROR: Dockerfile not found at $DOCKERFILE" >&2
    exit 2
fi

IMAGE=$(jq -r --arg t "$TARGET" \
    'select(.target_instance_id == $t) | .sandbox_image' "$MANIFEST" | head -1)

if [[ -z "$IMAGE" || "$IMAGE" == "null" ]]; then
    echo "ERROR: no sandbox image in manifest for target $TARGET" >&2
    exit 2
fi

mkdir -p "$EXECUTION_ROOT/logs"
LOG_PREFIX="$EXECUTION_ROOT/logs/$TARGET"

echo "[$(date -u +%FT%TZ)] target=$TARGET image=$IMAGE"

cleanup_execution_root_containers() {
    python3 scripts/run_openhands_contextbench.py \
        --execution-root "$EXECUTION_ROOT" \
        --cleanup-execution-root-containers
}

# Skip if all 4 conditions already have a successful run_meta.json
ALREADY_DONE=1
for cond in no_memory raw adp memory; do
    meta="$EXECUTION_ROOT/runs/$TARGET/$cond/run_meta.json"
    if [[ ! -f "$meta" ]]; then ALREADY_DONE=0; break; fi
    status=$(jq -r '.status // ""' "$meta" 2>/dev/null || echo "")
    if [[ "$status" != "success" && "$status" != "success_empty_patch" ]]; then
        ALREADY_DONE=0; break
    fi
done

FORCE=0
for arg in "${EXTRA_ARGS[@]:-}"; do
    if [[ "$arg" == "--force" ]]; then FORCE=1; fi
done

if [[ "$ALREADY_DONE" == "1" && "$FORCE" == "0" ]]; then
    echo "[skip] $TARGET — all 4 conditions already succeeded"
    exit 0
fi

# 1. Pull base image with retries.
cleanup_execution_root_containers >> "$LOG_PREFIX.cleanup.log" 2>&1 || true

PULL_RETRIES=3
for i in $(seq 1 $PULL_RETRIES); do
    if docker pull "$IMAGE" 2>&1 | tee -a "$LOG_PREFIX.pull.log"; then
        break
    fi
    if [[ $i -eq $PULL_RETRIES ]]; then
        echo "ERROR: docker pull failed for $IMAGE after $PULL_RETRIES attempts" >&2
        exit 3
    fi
    echo "[retry $i/$PULL_RETRIES] sleeping 30s before pull retry..."
    sleep 30
done

# 2. Build OpenHands-augmented image, overwriting the local tag.
BUILD_RETRIES=2
for i in $(seq 1 $BUILD_RETRIES); do
    if docker build \
        --build-arg BASE_IMAGE="$IMAGE" \
        -t "$IMAGE" \
        -f "$DOCKERFILE" . 2>&1 | tee -a "$LOG_PREFIX.build.log"; then
        break
    fi
    if [[ $i -eq $BUILD_RETRIES ]]; then
        echo "ERROR: docker build failed for $IMAGE after $BUILD_RETRIES attempts" >&2
        exit 4
    fi
    echo "[retry $i/$BUILD_RETRIES] sleeping 30s before build retry..."
    sleep 30
done

# 3. Run the orchestrator for this target (4 conditions, image runtime, no-pull).
set +e
python3 scripts/run_openhands_contextbench.py \
    --manifest "$MANIFEST" \
    --mode smoke \
    --smoke-target "$TARGET" \
    --execution-root "$EXECUTION_ROOT" \
    --openhands-runtime image \
    --no-pull \
    --resume \
    --fail-on-off-workspace-writes \
    "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_PREFIX.run.log"
ORCH_EXIT=$?
set -e

# 4. Reclaim disk. Predictions, patches, logs and run_meta stay on disk.
cleanup_execution_root_containers >> "$LOG_PREFIX.cleanup.log" 2>&1 || true
docker rmi -f "$IMAGE" >> "$LOG_PREFIX.cleanup.log" 2>&1 || true
docker image prune -f >> "$LOG_PREFIX.cleanup.log" 2>&1 || true

for cond in no_memory raw adp memory; do
    rm -rf "$EXECUTION_ROOT/runs/$TARGET/$cond/workspace" 2>/dev/null || true
done

df_h=$(df -h "$EXECUTION_ROOT" 2>/dev/null | tail -1 || echo "")
echo "[$(date -u +%FT%TZ)] target=$TARGET orchestrator_exit=$ORCH_EXIT disk: $df_h"
exit $ORCH_EXIT
