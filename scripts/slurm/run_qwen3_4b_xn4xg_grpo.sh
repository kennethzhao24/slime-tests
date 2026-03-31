#!/bin/bash
# Generic SLURM wrapper for Qwen3-4B GRPO on X nodes with 4 GPUs per node.
#
# Default script directives request 2 nodes, but you can override them:
#   sbatch --nodes=4 --job-name=slime-rl-qwen3-4b-4n16g scripts/slurm/run_qwen3_4b_xn4xg_grpo.sh
#
# Or interactively:
#   salloc --nodes=4 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=16 --mem=512g --partition=ghx4
#   bash scripts/slurm/run_qwen3_4b_xn4xg_grpo.sh
#
# The generic in-container launcher derives the total GPU count from ACTOR_NUM_NODES.

#SBATCH --mem=512g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=2:00:00
#SBATCH --job-name=slime-rl-qwen3-4b-xn4xg
#SBATCH --account=bcrc-dtai-gh
#SBATCH --gpus-per-node=4
#SBATCH --output=/u/yzhao25/slime-tests/slurm-%j.out

set -euo pipefail

SOCKET_IFNAME="${SOCKET_IFNAME:-hsn0}"
IMAGE="${APPTAINER_IMAGE:-/u/yzhao25/slime-tests/slime-gh200-cu128.sif}"
HF_BIND="${HF_BIND:-/work/nvme/bekz/yzhao25/huggingface:/mnt/huggingface}"
DATASETS_BIND="${DATASETS_BIND:-/work/nvme/bcrc/yzhao25/rl_datasets:/mnt/datasets}"
LAUNCH_SCRIPT="/u/yzhao25/slime-tests/scripts/run-qwen3-4B-xN4xG.sh"
NODELIST="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}}"
NTFY_TOPIC="${NTFY_TOPIC:-ken_alerts}"
NTFY_URL="${NTFY_URL:-https://ntfy.sh/${NTFY_TOPIC}}"

notify_ntfy() {
  local title="$1"
  local message="$2"
  local tags="${3:-bell}"
  local priority="${4:-default}"

  if [[ -z "${NTFY_URL}" ]]; then
    return 0
  fi

  if ! curl -fsS \
    -H "Title: ${title}" \
    -H "Tags: ${tags}" \
    -H "Priority: ${priority}" \
    -d "${message}" \
    "${NTFY_URL}" >/dev/null; then
    echo "Warning: failed to send ntfy alert to ${NTFY_URL}" >&2
  fi
}

notify_finish() {
  local exit_code="$1"
  local status="finished"
  local tags="white_check_mark"
  local priority="default"

  if [[ "${exit_code}" -ne 0 ]]; then
    status="failed"
    tags="x"
    priority="high"
  fi

  notify_ntfy \
    "Qwen3-4B GRPO ${status}" \
    "Job ${SLURM_JOB_ID:-N/A} ${status} on ${NODELIST:-unknown} (exit=${exit_code})" \
    "${tags}" \
    "${priority}"
}

trap 'notify_finish "$?"' EXIT

if [[ -z "${NODELIST}" ]]; then
  cat >&2 <<'EOF'
This script must run inside a SLURM allocation.

Use one of:
  sbatch scripts/slurm/run_qwen3_4b_xn4xg_grpo.sh
  sbatch --nodes=4 scripts/slurm/run_qwen3_4b_xn4xg_grpo.sh
  salloc --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=16 --mem=512g --partition=ghx4
  bash scripts/slurm/run_qwen3_4b_xn4xg_grpo.sh
EOF
  exit 1
fi

if [[ -z "${ACTOR_NUM_NODES}" ]]; then
  echo "Failed to determine ACTOR_NUM_NODES from the SLURM allocation." >&2
  exit 1
fi

HEAD_NODE=$(scontrol show hostnames "${NODELIST}" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" bash -lc \
  "ip -o -4 addr show dev ${SOCKET_IFNAME} | awk '{print \$4}' | cut -d/ -f1 | head -n 1")

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "Failed to resolve MASTER_ADDR on interface ${SOCKET_IFNAME} for head node ${HEAD_NODE}." >&2
  exit 1
fi

echo "SLURM_NODELIST: ${NODELIST}"
echo "ACTOR_NUM_NODES: ${ACTOR_NUM_NODES}"
echo "HEAD_NODE: ${HEAD_NODE}"
echo "MASTER_ADDR(${SOCKET_IFNAME}): ${MASTER_ADDR}"
echo "IMAGE: ${IMAGE}"
echo "LAUNCH_SCRIPT: ${LAUNCH_SCRIPT}"

notify_ntfy \
  "Qwen3-4B GRPO launched" \
  "Job ${SLURM_JOB_ID:-N/A} launched on ${NODELIST} with ${ACTOR_NUM_NODES} node(s); head=${HEAD_NODE}; master=${MASTER_ADDR}" \
  "rocket" \
  "high"

# srun launches one task per node (ntasks-per-node=1).
# Task 0 becomes the Ray head; the remaining task(s) join as workers.
srun --label --kill-on-bad-exit=1 bash -lc '
  set -euo pipefail

  if [[ "${SLURM_PROCID}" == "0" ]]; then
    ROLE=head
  else
    ROLE=worker
  fi

  echo "Launching ${ROLE} on $(hostname) with MASTER_ADDR='"${MASTER_ADDR}"' via '"${SOCKET_IFNAME}"'"

  export MASTER_ADDR='"${MASTER_ADDR}"'
  export SOCKET_IFNAME='"${SOCKET_IFNAME}"'
  export NCCL_SOCKET_IFNAME='"${SOCKET_IFNAME}"'
  export ACTOR_NUM_NODES='"${ACTOR_NUM_NODES}"'
  export ROLE

  exec apptainer exec --nv \
    --bind "'"${HF_BIND}"'" \
    --bind "'"${DATASETS_BIND}"'" \
    "'"${IMAGE}"'" \
    /bin/bash -lc "bash '"${LAUNCH_SCRIPT}"'"
'
