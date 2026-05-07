#!/usr/bin/env bash
set -u

# =========================
# Basic config
# =========================
MODEL_PATH="/data/yanghaitao/ckpt/Qwen3.6-27B/"
SERVED_MODEL_NAME="qwen3.6-27b"
PORT=8000

CUDA_VISIBLE_DEVICES_VALUE="0,1,2,3,4,5"

# 128K = 131072
MAX_MODEL_LEN=100000

# For 128K long-context stability, start conservatively.
TP_SIZE=2
DP_SIZE=3

MAX_NUM_BATCHED_TOKENS=32768
MAX_NUM_SEQS=32
GPU_MEMORY_UTILIZATION=0.9

LOG_DIR="./logs/vllm_logs"
PID_FILE="./logs/vllm_server.pid"

HEALTH_URL="http://127.0.0.1:${PORT}/health"
MODELS_URL="http://127.0.0.1:${PORT}/v1/models"

CHECK_INTERVAL=30          # health check interval after startup
STARTUP_GRACE_SECONDS=900  # allow long model loading / KV allocation
MAX_HEALTH_FAILS=3         # restart after this many consecutive failed health checks
RESTART_SLEEP=20           # wait before restart to release GPU memory

mkdir -p "${LOG_DIR}"

# =========================
# Optional stability envs
# =========================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# If you previously saw NCCL multi-GPU hang, uncomment this:
# export NCCL_P2P_DISABLE=1

# Avoid proxy intercepting localhost health/API calls
export NO_PROXY="127.0.0.1,localhost,0.0.0.0,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,0.0.0.0,${no_proxy:-}"

# =========================
# Helpers
# =========================
timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

is_process_alive() {
  local pid="$1"
  kill -0 "${pid}" 2>/dev/null
}

health_ok() {
  curl -fsS --max-time 5 "${HEALTH_URL}" >/dev/null 2>&1 \
    || curl -fsS --max-time 5 "${MODELS_URL}" >/dev/null 2>&1
}

kill_server() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}" || true)"

    if [[ -n "${pid}" ]] && is_process_alive "${pid}"; then
      echo "[$(timestamp)] Stopping vLLM process group: ${pid}"

      # Kill the whole process group started by setsid.
      kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
      sleep 10

      if is_process_alive "${pid}"; then
        echo "[$(timestamp)] Force killing vLLM process group: ${pid}"
        kill -KILL "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
      fi
    fi

    rm -f "${PID_FILE}"
  fi
}

start_server() {
  local log_file="${LOG_DIR}/vllm_$(date +%Y%m%d_%H%M%S).log"
  ln -sfn "$(basename "${log_file}")" "${LOG_DIR}/latest.log"

  echo "[$(timestamp)] Starting vLLM..."
  echo "[$(timestamp)] Log: ${log_file}"
  echo "[$(timestamp)] Config: TP=${TP_SIZE}, DP=${DP_SIZE}, max_model_len=${MAX_MODEL_LEN}, max_num_seqs=${MAX_NUM_SEQS}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"

  local cmd=(
    vllm serve "${MODEL_PATH}"
    --port "${PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --data-parallel-size "${DP_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --enable-prefix-caching
    --served-model-name "${SERVED_MODEL_NAME}"
    --trust-remote-code
    --reasoning-parser qwen3
  )

  # setsid creates a new process group, so we can kill all child workers reliably.
  setsid "${cmd[@]}" >> "${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"

  echo "[$(timestamp)] vLLM PID: ${pid}"
}

wait_for_startup() {
  local pid
  pid="$(cat "${PID_FILE}")"

  local elapsed=0
  while (( elapsed < STARTUP_GRACE_SECONDS )); do
    if ! is_process_alive "${pid}"; then
      echo "[$(timestamp)] vLLM exited during startup."
      return 1
    fi

    if health_ok; then
      echo "[$(timestamp)] vLLM is healthy."
      return 0
    fi

    sleep 10
    elapsed=$((elapsed + 10))
    echo "[$(timestamp)] Waiting for vLLM startup... ${elapsed}s/${STARTUP_GRACE_SECONDS}s"
  done

  echo "[$(timestamp)] Startup health check timed out."
  return 1
}

cleanup() {
  echo "[$(timestamp)] Watchdog received exit signal."
  kill_server
  exit 0
}

trap cleanup INT TERM

# =========================
# Main watchdog loop
# =========================
while true; do
  kill_server
  start_server

  if ! wait_for_startup; then
    echo "[$(timestamp)] Startup failed. Restarting after ${RESTART_SLEEP}s..."
    kill_server
    sleep "${RESTART_SLEEP}"
    continue
  fi

  fail_count=0

  while true; do
    pid="$(cat "${PID_FILE}")"

    if ! is_process_alive "${pid}"; then
      echo "[$(timestamp)] vLLM process died. Restarting after ${RESTART_SLEEP}s..."
      sleep "${RESTART_SLEEP}"
      break
    fi

    if health_ok; then
      fail_count=0
    else
      fail_count=$((fail_count + 1))
      echo "[$(timestamp)] Health check failed ${fail_count}/${MAX_HEALTH_FAILS}"

      if (( fail_count >= MAX_HEALTH_FAILS )); then
        echo "[$(timestamp)] Health check failed too many times. Restarting..."
        kill_server
        sleep "${RESTART_SLEEP}"
        break
      fi
    fi

    sleep "${CHECK_INTERVAL}"
  done
done
