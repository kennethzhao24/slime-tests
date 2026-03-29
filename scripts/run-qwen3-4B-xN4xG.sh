#!/bin/bash
# X-node, 4X-GPU (4 per node) Qwen3-4B GRPO training.
#
# Set ACTOR_NUM_NODES to the number of GH200 nodes you want to use.
#
# Examples:
#   2 nodes / 8 GPUs:
#     ACTOR_NUM_NODES=2 MASTER_ADDR=<head_ip> LOCAL_NODE_IP=<head_ip> SOCKET_IFNAME=hsn0 ROLE=head \
#       bash scripts/run-qwen3-4B-xN4xG.sh
#
#   4 nodes / 16 GPUs:
#     ACTOR_NUM_NODES=4 MASTER_ADDR=<head_ip> LOCAL_NODE_IP=<head_ip> SOCKET_IFNAME=hsn0 ROLE=head \
#       bash scripts/run-qwen3-4B-xN4xG.sh
#
# Worker nodes use the same script with ROLE=worker and LOCAL_NODE_IP set to the
# worker's address. If ROLE is unset, the script defaults to head mode.

export PATH="/opt/venv/bin:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python3}"
RAY_BIN="${RAY_BIN:-/opt/venv/bin/ray}"
TRAIN_PY="${TRAIN_PY:-/root/slime/train.py}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
TOTAL_NUM_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-$((8 * ACTOR_NUM_NODES))}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((32 * ACTOR_NUM_NODES))}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-6144}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-tests}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-4B-${ACTOR_NUM_NODES}n${TOTAL_NUM_GPUS}g}"
RAY_CLUSTER_WAIT_TIMEOUT_SECS="${RAY_CLUSTER_WAIT_TIMEOUT_SECS:-300}"
RAY_CLUSTER_WAIT_INTERVAL_SECS="${RAY_CLUSTER_WAIT_INTERVAL_SECS:-5}"
ROLE="${ROLE:-}"
if [[ -z "${ROLE}" ]]; then
  if [[ "${NODE_RANK:-0}" == "1" ]]; then
    ROLE="worker"
  else
    ROLE="head"
  fi
fi
export CC="${CC:-/usr/bin/cc}"
if [[ -z "${CXX:-}" || "${CXX}" == "CC" ]]; then
  export CXX="/usr/bin/g++"
fi

# exclusively for GH200
CACHE_USER="${USER_NAME:-${USER:-$(id -un)}}"
CACHE_ROOT="/tmp/${CACHE_USER}-cache"
mkdir -p "$CACHE_ROOT/triton" "$CACHE_ROOT/torchinductor" "$CACHE_ROOT/xdg"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"

# for rerun the task
pkill -9 sglang
sleep 3
"${RAY_BIN}" stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -euo pipefail
set -x

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# unset proxy to avoid distributed startup issues
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

: "${MASTER_ADDR:?MASTER_ADDR must be set to the head node IP}"
SOCKET_IFNAME="${SOCKET_IFNAME:-eth0}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
echo "Using SOCKET_IFNAME: ${SOCKET_IFNAME}"
echo "Using NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "Configured nodes=${ACTOR_NUM_NODES} gpus_per_node=${ACTOR_NUM_GPUS_PER_NODE} total_gpus=${TOTAL_NUM_GPUS}"

LOCAL_NODE_IP="${LOCAL_NODE_IP:-}"
if [[ -z "${LOCAL_NODE_IP}" ]]; then
    LOCAL_NODE_IP=$("${PYTHON_BIN}" - <<'PY'
import fcntl
import os
import socket
import struct
import sys

ifname = os.environ["SOCKET_IFNAME"]
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    ifreq = struct.pack("256s", ifname[:15].encode("utf-8"))
    res = fcntl.ioctl(s.fileno(), 0x8915, ifreq)
    print(socket.inet_ntoa(res[20:24]))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
)
fi
if [[ -z "${LOCAL_NODE_IP}" ]]; then
    echo "Failed to determine local IPv4 address. Set LOCAL_NODE_IP explicitly." >&2
    exit 1
fi
echo "ROLE=${ROLE} MASTER_ADDR=${MASTER_ADDR} LOCAL_NODE_IP=${LOCAL_NODE_IP}"

CKPT_ARGS=(
   --hf-checkpoint /mnt/datasets/Qwen3-4B
   --ref-load /mnt/datasets/gh200/qwen3_4b_torch_dist_tp2
   --load /mnt/datasets/Qwen3-4B_slime/
   --save /mnt/datasets/Qwen3-4B_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/datasets/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len 4096
   --rollout-temperature 1

   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL}"
   --eval-prompt-data aime /mnt/datasets/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
   --skip-eval-before-train
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${WANDB_GROUP}"
   --wandb-mode offline
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

export no_proxy="127.0.0.1,${MASTER_ADDR}"
RAY_STOP_SENTINEL="${RAY_STOP_SENTINEL:-/u/yzhao25/slime-tests/logs/ray-stop-${SLURM_JOB_ID:-manual}.sentinel}"

if [[ "${ROLE}" == "head" ]]; then
  rm -f "${RAY_STOP_SENTINEL}"
fi

if [[ "${ROLE}" == "worker" ]]; then
  worker_cleanup_done=0
  cleanup_worker() {
    if [[ "${worker_cleanup_done}" == "1" ]]; then
      return
    fi
    worker_cleanup_done=1
    set +e
    echo "Stopping Ray worker on ${LOCAL_NODE_IP}"
    "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
  }
  on_worker_signal() {
    cleanup_worker
    exit 0
  }
  trap on_worker_signal TERM INT
  trap cleanup_worker EXIT

  echo "Starting blocking Ray worker against ${MASTER_ADDR}:6379 from ${LOCAL_NODE_IP}"
  set +e
  "${RAY_BIN}" start \
    --address="${MASTER_ADDR}:6379" \
    --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
    --node-ip-address "${LOCAL_NODE_IP}" \
    --disable-usage-stats \
    --block
  worker_status=$?
  cleanup_worker
  if [[ -f "${RAY_STOP_SENTINEL}" ]]; then
    echo "Detected intentional Ray shutdown via ${RAY_STOP_SENTINEL}"
    exit 0
  fi
  if [[ "${worker_status}" -eq 0 || "${worker_status}" -eq 143 ]]; then
    exit 0
  fi
  exit "${worker_status}"
fi

if [[ "${ROLE}" != "head" ]]; then
  echo "Unsupported ROLE=${ROLE}. Use ROLE=head or ROLE=worker." >&2
  exit 1
fi

HOSTFILE="${HOSTFILE:-}"
if [[ -z "${HOSTFILE}" && -f /root/mpi_rack_hostfile ]]; then
  HOSTFILE=/root/mpi_rack_hostfile
fi

"${RAY_BIN}" start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

if [[ -n "${HOSTFILE}" ]]; then
  for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    if [[ "${WORKER_IP}" == "${MASTER_ADDR}" || "${WORKER_IP}" == "${LOCAL_NODE_IP}" ]]; then
      continue
    fi
    echo "Starting Ray worker on ${WORKER_IP}"
    ssh root@"${WORKER_IP}" \
      "export PATH=/opt/venv/bin:\$PATH; /opt/venv/bin/ray stop --force; pkill -9 ray; pkill -9 python; /opt/venv/bin/ray start --address=${MASTER_ADDR}:6379 --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} --node-ip-address ${WORKER_IP} --disable-usage-stats" &
  done
  wait
fi

echo "Waiting for ${ACTOR_NUM_NODES} Ray nodes before submitting the job..."
deadline=$((SECONDS + RAY_CLUSTER_WAIT_TIMEOUT_SECS))
while true; do
  ACTIVE_NODE_COUNT=$("${RAY_BIN}" status --address="${MASTER_ADDR}:6379" 2>/dev/null | awk '
    /^Active:/ { in_active=1; next }
    /^Pending:/ { in_active=0 }
    in_active && $1 ~ /^[0-9]+$/ && $2 ~ /^node_/ { count += $1 }
    END { print count + 0 }
  ')
  if [[ "${ACTIVE_NODE_COUNT}" -ge "${ACTOR_NUM_NODES}" ]]; then
    echo "Ray cluster is ready with ${ACTIVE_NODE_COUNT} active nodes."
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for ${ACTOR_NUM_NODES} Ray nodes. Only saw ${ACTIVE_NODE_COUNT} active nodes." >&2
    "${RAY_BIN}" status --address="${MASTER_ADDR}:6379" || true
    exit 1
  fi
  echo "Ray currently has ${ACTIVE_NODE_COUNT}/${ACTOR_NUM_NODES} active nodes; retrying in ${RAY_CLUSTER_WAIT_INTERVAL_SECS}s..."
  sleep "${RAY_CLUSTER_WAIT_INTERVAL_SECS}"
done

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "NCCL_SOCKET_IFNAME": "${NCCL_SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}"
  }
}
EOF_JSON
)

job_status=0
"${RAY_BIN}" job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${PYTHON_BIN}" "${TRAIN_PY}" \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --colocate \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${EVAL_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]} || job_status=$?

echo "Stopping Ray head on ${MASTER_ADDR}"
touch "${RAY_STOP_SENTINEL}"
"${RAY_BIN}" stop --force || true
exit "${job_status}"
