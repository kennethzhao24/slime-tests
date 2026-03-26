#!/bin/bash

set -euo pipefail

IMAGE="${APPTAINER_IMAGE:-/u/yzhao25/slime-tests/slime-gh200-cu128.sif}"
HF_BIND="${HF_BIND:-/work/nvme/bekz/yzhao25/huggingface:/mnt/huggingface}"
DATASETS_BIND="${DATASETS_BIND:-/work/nvme/bcrc/yzhao25/rl_datasets:/mnt/datasets}"

if [[ "${APPTAINER_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  echo "Running CUDA preflight inside ${IMAGE}..."
  if ! apptainer exec --nv --bind "${HF_BIND}" --bind "${DATASETS_BIND}" "${IMAGE}" python - <<'PY'
import sys
import torch

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")

try:
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
except Exception as exc:
    print(f"CUDA preflight failed: {exc}", file=sys.stderr)
    sys.exit(2)

if not available or count < 1:
    print(
        "CUDA is not available inside the container. "
        "This usually means the host NVIDIA driver is older than the CUDA runtime bundled in the image.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"Visible CUDA devices: {count}")
PY
  then
    cat <<EOF
Container startup aborted because CUDA is not usable inside ${IMAGE}.

The checked-in default image is:
  /projects/bcrc/yzhao25/slime-gh200-cu128.sif

If you see an error like:
  The NVIDIA driver on your system is too old (found version 12080)

then this node is exposing a CUDA 12.8-era driver stack to a container built with a newer CUDA runtime.
Use a CUDA 12.x-compatible Slime image, or move to a node with a newer NVIDIA driver.

You can override the image without editing this script:
  APPTAINER_IMAGE=/path/to/compatible.sif bash scripts/run_apptainer.sh

If you already validated the image manually, you can skip this check:
  APPTAINER_SKIP_PREFLIGHT=1 bash scripts/run_apptainer.sh
EOF
    exit 1
  fi
fi

exec apptainer exec --nv --bind "${HF_BIND}" \
                        --bind "${DATASETS_BIND}" \
                        "${IMAGE}" \
                        /bin/bash --login
