#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/build/experiments"
BIN="${RUN_DIR}/emnist_letters_visualized"
FALLBACK_BIN="${ROOT_DIR}/build/emnist_letters_visualized"
BUILD_DIR="${ROOT_DIR}/build"
BUILD_RUN_DIR="${ROOT_DIR}/build/experiments"

SNNR_PATH="${1:-}"
SNNW_PATH="${2:-}"
DATA_DIR="${3:-${ROOT_DIR}/data/EMNIST}"
shift 3 || true
EXTRA_ARGS=("$@")

if [[ -z "${SNNR_PATH}" ]]; then
  echo "Usage: $0 <recording.snnr> [network.snnw] [data_dir]" >&2
  exit 1
fi

if [[ -z "${SNNW_PATH}" ]]; then
  SNNW_PATH="${SNNR_PATH%.snnr}.snnw"
fi

if [[ ! -x "${BIN}" && -x "${FALLBACK_BIN}" ]]; then
  BIN="${FALLBACK_BIN}"
fi

if [[ ! -x "${BIN}" ]]; then
  if [[ -f "${BUILD_DIR}/Makefile" ]]; then
    make -C "${BUILD_DIR}" emnist_letters_visualized
  elif [[ -f "${BUILD_RUN_DIR}/Makefile" ]]; then
    make -C "${BUILD_RUN_DIR}" emnist_letters_visualized
  fi
fi

if [[ ! -x "${BIN}" && -x "${FALLBACK_BIN}" ]]; then
  BIN="${FALLBACK_BIN}"
fi

if [[ ! -x "${BIN}" ]]; then
  echo "Visualizer binary not found. Build it with: make -C ${BUILD_DIR} emnist_letters_visualized" >&2
  exit 1
fi

if [[ ! -f "${SNNR_PATH}" ]]; then
  echo "Recording file not found: ${SNNR_PATH}" >&2
  exit 1
fi

if [[ ! -f "${SNNW_PATH}" ]]; then
  echo "Warning: network structure file not found: ${SNNW_PATH}" >&2
fi

TRAIN_IMAGES="${DATA_DIR}/emnist-letters-train-images-idx3-ubyte"
TRAIN_LABELS="${DATA_DIR}/emnist-letters-train-labels-idx1-ubyte"
TEST_IMAGES="${DATA_DIR}/emnist-letters-test-images-idx3-ubyte"
TEST_LABELS="${DATA_DIR}/emnist-letters-test-labels-idx1-ubyte"

if [[ ! -f "${TRAIN_IMAGES}" || ! -f "${TRAIN_LABELS}" || ! -f "${TEST_IMAGES}" || ! -f "${TEST_LABELS}" ]]; then
  echo "Warning: EMNIST data files not found under ${DATA_DIR}" >&2
fi

CONFIG_PATH="${RUN_DIR}/playback_config.json"
cat > "${CONFIG_PATH}" <<EOF
{
  "network_structure": {
    "path": "${SNNW_PATH}"
  },
  "data": {
    "train_images": "${TRAIN_IMAGES}",
    "train_labels": "${TRAIN_LABELS}",
    "test_images": "${TEST_IMAGES}",
    "test_labels": "${TEST_LABELS}"
  }
}
EOF

cd "${ROOT_DIR}"

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
"${BIN}" "${TRAIN_IMAGES}" "${TRAIN_LABELS}" --playback "${SNNR_PATH}" --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
