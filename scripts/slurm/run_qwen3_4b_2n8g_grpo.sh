#!/bin/bash
#SBATCH --mem=512g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00
#SBATCH --job-name=slime-rl-qwen3-4b-2node
#SBATCH --account=bcrc-dtai-gh
#SBATCH --gpus-per-node=4
#SBATCH --output=/u/yzhao25/slime-tests/slurm-%j.out

set -euo pipefail

SOCKET_IFNAME="${SOCKET_IFNAME:-hsn0}"
IMAGE="${APPTAINER_IMAGE:-/u/yzhao25/slime-tests/slime-gh200-cu128.sif}"
HF_BIND="${HF_BIND:-/work/nvme/bekz/yzhao25/huggingface:/mnt/huggingface}"
DATASETS_BIND="${DATASETS_BIND:-/work/nvme/bcrc/yzhao25/rl_datasets:/mnt/datasets}"
LAUNCH_SCRIPT="/u/yzhao25/slime-tests/scripts/run-qwen3-4B-2N8G.sh"

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" bash -lc \
  "ip -o -4 addr show dev ${SOCKET_IFNAME} | awk '{print \$4}' | cut -d/ -f1 | head -n 1")

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "Failed to resolve MASTER_ADDR on interface ${SOCKET_IFNAME} for head node ${HEAD_NODE}." >&2
  exit 1
fi

echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "HEAD_NODE: ${HEAD_NODE}"
echo "MASTER_ADDR(${SOCKET_IFNAME}): ${MASTER_ADDR}"
echo "IMAGE: ${IMAGE}"
echo "LAUNCH_SCRIPT: ${LAUNCH_SCRIPT}"

# srun launches one task per node (ntasks-per-node=1).
# Task 0 becomes the Ray head; the other task(s) join as workers.
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
  export ROLE

  exec apptainer exec --nv \
    --bind "'"${HF_BIND}"'" \
    --bind "'"${DATASETS_BIND}"'" \
    "'"${IMAGE}"'" \
    /bin/bash -lc "bash '"${LAUNCH_SCRIPT}"'"
'
