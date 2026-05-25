#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Launch a ContextBench run with:
  1. prompt rendering
  2. ollama serve in one tmux session
  3. the OpenHands orchestrator in another tmux session

Examples:
  export LLM_MODEL="openai/devstral-small-2:24b-32k"
  export LLM_BASE_URL="http://127.0.0.1:11435/v1"
  export LLM_API_KEY="dummy"
  export OLLAMA_HOST="127.0.0.1:11435"
  export OLLAMA_MODELS="/mnt/data/shashwat/ollama-models"
  export CUDA_VISIBLE_DEVICES="1"

  ./scripts/launch_contextbench_tmux.sh \
    --mode full \
    --manifest data/contextbench_phase2/execution_10x4_devstral_24b/prompt_manifest_10x4_local_images.jsonl \
    --execution-root data/contextbench_phase2/execution_10x4_devstral_24b_32k_fresh

  ./scripts/launch_contextbench_tmux.sh \
    --mode smoke \
    --smoke-target astropy__astropy-15082 \
    --execution-root data/contextbench_phase2/execution_smoke_devstral_32k

Options:
  --mode smoke|full                      Required.
  --manifest PATH                        Prompt manifest for orchestrator.
  --run-manifest PATH                    Run manifest used for prompt rendering.
  --execution-root PATH                  Fresh execution root. If omitted, one is generated.
  --conditions "a b c d"                 Space-separated condition list.
  --smoke-target ID                      Required for smoke mode.
  --timeout-seconds N                    Per-run orchestrator timeout.
  --max-runs N                           Limit number of selected runs.
  --no-pull                              Pass through to orchestrator.
  --resume                               Pass through to orchestrator.
  --force                                Pass through to orchestrator.
  --stop-on-error                        Pass through to orchestrator.
  --ollama-session NAME                  tmux session name for Ollama.
  --run-session NAME                     tmux session name for orchestrator.
  --cuda-visible-devices VALUE           Exported into Ollama session.
  --ollama-models PATH                   Exported into Ollama session.
  --pull-model-if-missing                Run `ollama pull` before orchestrator if model is absent.
  --dry-run                              Render and create tmux scripts, but do not start tmux sessions.
  --kill-existing-sessions               Kill matching tmux sessions before starting.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

slugify() {
  printf '%s' "$1" | tr '/: ' '___' | tr -cd 'A-Za-z0-9._-'
}

repo_root_from_script() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/.." && pwd
}

MODE=""
MANIFEST="data/contextbench_phase2/prompt_manifest.jsonl"
RUN_MANIFEST="data/contextbench_phase2/run_manifest.jsonl"
EXECUTION_ROOT=""
CONDITIONS="no_memory raw adp memory"
SMOKE_TARGET="astropy__astropy-15082"
TIMEOUT_SECONDS="7200"
MAX_RUNS=""
NO_PULL=1
RESUME=0
FORCE=0
STOP_ON_ERROR=0
OLLAMA_SESSION="contextbench_ollama"
RUN_SESSION="contextbench_run"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-}"
OLLAMA_MODELS_VALUE="${OLLAMA_MODELS:-}"
PULL_MODEL_IF_MISSING=0
DRY_RUN=0
KILL_EXISTING_SESSIONS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --run-manifest) RUN_MANIFEST="$2"; shift 2 ;;
    --execution-root) EXECUTION_ROOT="$2"; shift 2 ;;
    --conditions) CONDITIONS="$2"; shift 2 ;;
    --smoke-target) SMOKE_TARGET="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --max-runs) MAX_RUNS="$2"; shift 2 ;;
    --no-pull) NO_PULL=1; shift ;;
    --resume) RESUME=1; shift ;;
    --force) FORCE=1; shift ;;
    --stop-on-error) STOP_ON_ERROR=1; shift ;;
    --ollama-session) OLLAMA_SESSION="$2"; shift 2 ;;
    --run-session) RUN_SESSION="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    --ollama-models) OLLAMA_MODELS_VALUE="$2"; shift 2 ;;
    --pull-model-if-missing) PULL_MODEL_IF_MISSING=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --kill-existing-sessions) KILL_EXISTING_SESSIONS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "--mode must be one of: smoke, full" >&2
  exit 1
fi

require_cmd python3
require_cmd tmux
require_cmd curl
require_cmd ollama

REPO_ROOT="$(repo_root_from_script)"
cd "$REPO_ROOT"

LLM_MODEL_VALUE="${LLM_MODEL:-}"
LLM_BASE_URL_VALUE="${LLM_BASE_URL:-http://127.0.0.1:11435/v1}"
LLM_API_KEY_VALUE="${LLM_API_KEY:-dummy}"
OLLAMA_HOST_VALUE="${OLLAMA_HOST:-127.0.0.1:11435}"

if [[ -z "$LLM_MODEL_VALUE" ]]; then
  echo "LLM_MODEL must be set in the environment before launching." >&2
  exit 1
fi

OLLAMA_MODEL_NAME="$LLM_MODEL_VALUE"
if [[ "$OLLAMA_MODEL_NAME" == openai/* ]]; then
  OLLAMA_MODEL_NAME="${OLLAMA_MODEL_NAME#openai/}"
elif [[ "$OLLAMA_MODEL_NAME" == ollama_chat/* ]]; then
  OLLAMA_MODEL_NAME="${OLLAMA_MODEL_NAME#ollama_chat/}"
fi

if [[ -z "$EXECUTION_ROOT" ]]; then
  timestamp="$(date +%Y%m%d_%H%M%S)"
  model_slug="$(slugify "$OLLAMA_MODEL_NAME")"
  EXECUTION_ROOT="data/contextbench_phase2/execution_${MODE}_${model_slug}_${timestamp}"
fi

mkdir -p "$EXECUTION_ROOT"
LAUNCHER_DIR="$EXECUTION_ROOT/launcher"
mkdir -p "$LAUNCHER_DIR"

echo "[1/4] Rendering prompts"
python3 scripts/contextbench/render_contextbench_prompts.py \
  --manifest "$RUN_MANIFEST" \
  --write-run-prompt \
  --fail-on-forbidden

OLLAMA_SCRIPT="$LAUNCHER_DIR/start_ollama.sh"
RUN_SCRIPT="$LAUNCHER_DIR/start_orchestrator.sh"
ENV_SNAPSHOT="$LAUNCHER_DIR/env_snapshot.txt"
OLLAMA_LOG="$LAUNCHER_DIR/ollama.log"
RUN_LOG="$LAUNCHER_DIR/orchestrator.log"

cat > "$ENV_SNAPSHOT" <<EOF
REPO_ROOT=$REPO_ROOT
MODE=$MODE
MANIFEST=$MANIFEST
RUN_MANIFEST=$RUN_MANIFEST
EXECUTION_ROOT=$EXECUTION_ROOT
CONDITIONS=$CONDITIONS
SMOKE_TARGET=$SMOKE_TARGET
TIMEOUT_SECONDS=$TIMEOUT_SECONDS
MAX_RUNS=$MAX_RUNS
OLLAMA_SESSION=$OLLAMA_SESSION
RUN_SESSION=$RUN_SESSION
OLLAMA_HOST=$OLLAMA_HOST_VALUE
OLLAMA_MODELS=${OLLAMA_MODELS_VALUE}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}
LLM_MODEL=$LLM_MODEL_VALUE
LLM_BASE_URL=$LLM_BASE_URL_VALUE
OLLAMA_MODEL_NAME=$OLLAMA_MODEL_NAME
EOF

cat > "$OLLAMA_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
export OLLAMA_HOST="$OLLAMA_HOST_VALUE"
export OLLAMA_KEEP_ALIVE="\${OLLAMA_KEEP_ALIVE:-5m0s}"
if [[ -n "${OLLAMA_MODELS_VALUE}" ]]; then
  export OLLAMA_MODELS="$OLLAMA_MODELS_VALUE"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
fi
echo "[ollama] OLLAMA_HOST=\$OLLAMA_HOST"
echo "[ollama] OLLAMA_MODELS=\${OLLAMA_MODELS:-}"
echo "[ollama] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-}"
ollama serve 2>&1 | tee -a "$OLLAMA_LOG"
EOF

cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
export OLLAMA_HOST="$OLLAMA_HOST_VALUE"
export LLM_MODEL="$LLM_MODEL_VALUE"
export LLM_BASE_URL="$LLM_BASE_URL_VALUE"
export LLM_API_KEY="$LLM_API_KEY_VALUE"
if [[ -n "${OLLAMA_MODELS_VALUE}" ]]; then
  export OLLAMA_MODELS="$OLLAMA_MODELS_VALUE"
fi
echo "[run] waiting for ollama tags endpoint at http://$OLLAMA_HOST_VALUE/api/tags"
until curl -fsS "http://$OLLAMA_HOST_VALUE/api/tags" >/dev/null 2>&1; do
  sleep 2
done
if ! ollama list | awk '{print \$1}' | grep -Fx "$OLLAMA_MODEL_NAME" >/dev/null 2>&1; then
  echo "[run] model $OLLAMA_MODEL_NAME not present in ollama list"
  if [[ "$PULL_MODEL_IF_MISSING" == "1" ]]; then
    echo "[run] pulling model $OLLAMA_MODEL_NAME"
    ollama pull "$OLLAMA_MODEL_NAME"
  else
    echo "[run] continuing without pull; orchestrator may fail if model is unavailable"
  fi
fi
cmd=(
  python3 scripts/run_openhands_contextbench.py
  --manifest "$MANIFEST"
  --mode "$MODE"
  --conditions $CONDITIONS
  --execution-root "$EXECUTION_ROOT"
  --openhands-runtime image
  --fail-on-off-workspace-writes
  --timeout-seconds "$TIMEOUT_SECONDS"
)
if [[ "$MODE" == "smoke" ]]; then
  cmd+=(--smoke-target "$SMOKE_TARGET")
else
  cmd+=(--confirm-full-run)
fi
if [[ "$NO_PULL" == "1" ]]; then
  cmd+=(--no-pull)
fi
if [[ -n "$MAX_RUNS" ]]; then
  cmd+=(--max-runs "$MAX_RUNS")
fi
if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
fi
if [[ "$FORCE" == "1" ]]; then
  cmd+=(--force)
fi
if [[ "$STOP_ON_ERROR" == "1" ]]; then
  cmd+=(--stop-on-error)
fi
echo "[run] \${cmd[*]}"
"\${cmd[@]}" 2>&1 | tee -a "$RUN_LOG"
EOF

chmod +x "$OLLAMA_SCRIPT" "$RUN_SCRIPT"

echo "[2/4] Launcher scripts written"
echo "  Ollama:       $OLLAMA_SCRIPT"
echo "  Orchestrator: $RUN_SCRIPT"
echo "  Logs:         $OLLAMA_LOG, $RUN_LOG"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[3/4] Dry run requested; tmux sessions not started."
  exit 0
fi

if [[ "$KILL_EXISTING_SESSIONS" == "1" ]]; then
  tmux has-session -t "$OLLAMA_SESSION" 2>/dev/null && tmux kill-session -t "$OLLAMA_SESSION"
  tmux has-session -t "$RUN_SESSION" 2>/dev/null && tmux kill-session -t "$RUN_SESSION"
fi

if tmux has-session -t "$OLLAMA_SESSION" 2>/dev/null; then
  echo "tmux session already exists: $OLLAMA_SESSION" >&2
  exit 1
fi
if tmux has-session -t "$RUN_SESSION" 2>/dev/null; then
  echo "tmux session already exists: $RUN_SESSION" >&2
  exit 1
fi

echo "[3/4] Starting tmux sessions"
tmux new-session -d -s "$OLLAMA_SESSION" "$OLLAMA_SCRIPT"
sleep 2
tmux new-session -d -s "$RUN_SESSION" "$RUN_SCRIPT"

echo "[4/4] Started"
echo "  Attach Ollama: tmux attach -t $OLLAMA_SESSION"
echo "  Attach Run:    tmux attach -t $RUN_SESSION"
echo "  Execution root: $EXECUTION_ROOT"
