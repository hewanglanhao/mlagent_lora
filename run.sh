#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

ENV_FILE="$ROOT_DIR/doc/环境变量.txt"
if [ -f "$ENV_FILE" ]; then
    set +u
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set -u
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MAX_OPT_TIME="${MAX_OPT_TIME:-1700}"
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

if [ ! -s "$ROOT_DIR/optimized_lora.cu" ]; then
    echo "optimized_lora.cu was not produced" >&2
    exit 1
fi

exit 0
