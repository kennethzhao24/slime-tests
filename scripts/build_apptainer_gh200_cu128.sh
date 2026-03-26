#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

DEF_FILE="${DEF_FILE:-${REPO_ROOT}/docker/slime-gh200-cu128.def}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-${REPO_ROOT}/slime-gh200-cu128.sif}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
USE_SUDO="${USE_SUDO:-0}"

if [[ ! -f "${DEF_FILE}" ]]; then
  echo "Definition file not found: ${DEF_FILE}" >&2
  exit 1
fi

BUILD_CMD=("${APPTAINER_BIN}" build "${OUTPUT_IMAGE}" "${DEF_FILE}")

if [[ "${USE_SUDO}" == "1" ]]; then
  BUILD_CMD=(sudo "${BUILD_CMD[@]}")
fi

echo "Building Apptainer image"
echo "  def:    ${DEF_FILE}"
echo "  output: ${OUTPUT_IMAGE}"
echo "  bin:    ${APPTAINER_BIN}"

"${BUILD_CMD[@]}"

echo
echo "Build complete: ${OUTPUT_IMAGE}"
