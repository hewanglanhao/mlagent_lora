#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${MLAGENT_PROJECT_DIR:-$SCRIPT_DIR/mlagent_lora}"
OUTPUT_DIR="${MLAGENT_OUTPUT_DIR:-/workspace}"
FINAL_FILE="optimized_lora.cu"

if [ ! -d "$PROJECT_DIR/agent" ]; then
    if [ -d "$SCRIPT_DIR/agent" ]; then
        PROJECT_DIR="$SCRIPT_DIR"
    else
        echo "project directory not found: $PROJECT_DIR" >&2
        exit 1
    fi
fi

sync_final_file() {
    if [ -s "$PROJECT_DIR/$FINAL_FILE" ]; then
        mkdir -p "$OUTPUT_DIR" || return 1
        tmp_file="$(mktemp "$OUTPUT_DIR/.${FINAL_FILE}.tmp.XXXXXX")" || return 1
        if cp -f "$PROJECT_DIR/$FINAL_FILE" "$tmp_file"; then
            mv -f "$tmp_file" "$OUTPUT_DIR/$FINAL_FILE"
        else
            rm -f "$tmp_file"
            return 1
        fi
    fi
}

SYNC_PID=""
cleanup() {
    if [ -n "${SYNC_PID:-}" ]; then
        kill "$SYNC_PID" 2>/dev/null || true
        wait "$SYNC_PID" 2>/dev/null || true
        SYNC_PID=""
    fi
    sync_final_file || true
}
on_exit() {
    status=$?
    cleanup
    exit "$status"
}
on_signal() {
    cleanup
    exit 143
}
trap on_exit EXIT
trap on_signal INT TERM HUP

sync_final_file || true
while true; do
    sync_final_file || true
    sleep 5
done &
SYNC_PID=$!

cd "$PROJECT_DIR" || exit 1

ENV_FILE="$PROJECT_DIR/doc/环境变量.txt"
if [ -f "$ENV_FILE" ]; then
    set +u
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set -u
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MAX_OPT_TIME="${MAX_OPT_TIME:-1700}"
export LLM_TIMEOUT_SEC="${LLM_TIMEOUT_SEC:-180}"
export ENABLE_ASYNC_LLM="${ENABLE_ASYNC_LLM:-1}"
export ASYNC_LLM_ADVISORY="${ASYNC_LLM_ADVISORY:-0}"
export ASYNC_LLM_IDLE_WAIT_SEC="${ASYNC_LLM_IDLE_WAIT_SEC:-3}"
export ASYNC_LLM_FINAL_WAIT_SEC="${ASYNC_LLM_FINAL_WAIT_SEC:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

if [ -z "${CUDA_HOME:-}" ]; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif [ -x /usr/local/cuda-12.4/bin/nvcc ]; then
        export CUDA_HOME="/usr/local/cuda-12.4"
    fi
fi

if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

python3 -m agent.agent_framework "$@"
status=$?

if [ "$status" -ne 0 ]; then
    echo "agent failed with status ${status}; trying bootstrap fallback" >&2
    python3 -m agent.agent_framework --bootstrap-only || true
fi

sync_final_file || true

if [ ! -s "$OUTPUT_DIR/$FINAL_FILE" ]; then
    echo "$FINAL_FILE was not produced in $OUTPUT_DIR" >&2
    exit 1
fi

exit 0
