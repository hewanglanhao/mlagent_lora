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

# 让 Python 日志实时输出，便于 tail -f 查看运行进度。
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# 单次优化运行的内部时间上限，单位秒；平台 30 分钟限制下默认提前约 2 分钟收尾。
export MAX_OPT_TIME="${MAX_OPT_TIME:-1680}"

# 单次 LLM 请求的最长等待时间，单位秒；过短会更容易 timeout，过长会拖慢整体搜索。
export LLM_TIMEOUT_SEC="${LLM_TIMEOUT_SEC:-180}"

# 启动新候选前要求至少剩余的时间，单位秒；避免候选跑到一半撞上 30 分钟限制。
export MIN_CANDIDATE_TIME_BUDGET_SEC="${MIN_CANDIDATE_TIME_BUDGET_SEC:-600}"

# 严格闭环模式下启动新候选前的更保守剩余时间，单位秒；LLM 生成+审查+编译通常会很慢。
export CLOSED_LOOP_MIN_CANDIDATE_TIME_BUDGET_SEC="${CLOSED_LOOP_MIN_CANDIDATE_TIME_BUDGET_SEC:-900}"

# 已生成候选进入评估前要求剩余的时间，单位秒；防止 codegen 很慢后继续编译导致平台超时。
export CANDIDATE_EVALUATION_TIME_BUDGET_SEC="${CANDIDATE_EVALUATION_TIME_BUDGET_SEC:-420}"

# 同步 LLM 静态审查前要求剩余的时间，单位秒；不足时跳过该审查，继续依赖本地静态检查。
export LLM_STATIC_REVIEW_TIME_BUDGET_SEC="${LLM_STATIC_REVIEW_TIME_BUDGET_SEC:-120}"

# 编译候选前要求剩余的时间，单位秒；PyTorch CUDA 扩展编译在当前环境常见需要 3 分钟以上。
export COMPILE_TIME_BUDGET_SEC="${COMPILE_TIME_BUDGET_SEC:-300}"

# benchmark/profile 前要求剩余的时间，单位秒；不足时跳过，避免最后阶段被硬杀。
export BENCHMARK_TIME_BUDGET_SEC="${BENCHMARK_TIME_BUDGET_SEC:-120}"

# 同步 LLM 瓶颈诊断前要求剩余的时间，单位秒；不足时改用本地 fallback 诊断。
export DIAGNOSIS_TIME_BUDGET_SEC="${DIAGNOSIS_TIME_BUDGET_SEC:-90}"

# 是否启用严格闭环：LLM 生成代码 -> 编译 -> 正确性 -> benchmark/profile -> LLM 分析 -> 下一轮。
export ENABLE_LLM_CLOSED_LOOP="${ENABLE_LLM_CLOSED_LOOP:-1}"

# 是否启用异步 LLM 流水线；严格闭环开启时会自动禁用异步执行，以保证诊断结果进入下一轮。
export ENABLE_ASYNC_LLM="${ENABLE_ASYNC_LLM:-1}"

# 是否让非关键 LLM 审查/诊断也后台执行；开启会增加 API 调用，默认关闭。
export ASYNC_LLM_ADVISORY="${ASYNC_LLM_ADVISORY:-0}"

# 异步模式下是否后台生成 LLM CUDA 代码；严格闭环模式下前台会直接做 LLM codegen。
export ASYNC_LLM_CODEGEN="${ASYNC_LLM_CODEGEN:-1}"

# LLM 生成代码编译失败后的自动修复次数；次数越多越可能修好，但也越耗时/耗 token。
export LLM_CODEGEN_REPAIR_ATTEMPTS="${LLM_CODEGEN_REPAIR_ATTEMPTS:-3}"

# 异步模式下没有本地候选可跑时，最多等 LLM mutation 建议多久，单位秒。
export ASYNC_LLM_IDLE_WAIT_SEC="${ASYNC_LLM_IDLE_WAIT_SEC:-60}"

# 异步模式下 LLM mutation 返回重复/无效建议后，重新等待建议的时间，单位秒。
export ASYNC_LLM_STALE_RETRY_WAIT_SEC="${ASYNC_LLM_STALE_RETRY_WAIT_SEC:-45}"

# 结束前等待后台 LLM 任务收尾的时间，单位秒；设为 0 可更快退出。
export ASYNC_LLM_FINAL_WAIT_SEC="${ASYNC_LLM_FINAL_WAIT_SEC:-2}"

# 指定 PyTorch CUDA 扩展编译目标架构；8.6 对应常见 Ampere GPU。
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
