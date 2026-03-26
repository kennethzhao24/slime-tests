#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

NPROC_PER_NODE="${HF_TO_MEGATRON_NPROC_PER_NODE:-4}"
# The current GH200 image ships a Transformer Engine wheel that does not match
# the installed PyTorch ABI.  Conversion does not require TE, so default to a
# shim that makes Megatron take its built-in "TE unavailable" fallback path.

CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH="/root/Megatron-LM" torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  /root/slime/tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 2 \
  --hf-checkpoint /mnt/datasets/Qwen3-4B-Base \
  --make-vocab-size-divisible-by 1 \
  --save /mnt/datasets/gh200/qwen3_4b_torch_dist_tp2
