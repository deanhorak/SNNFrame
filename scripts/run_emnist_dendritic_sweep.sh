#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${BUILD_DIR}/dendritic_sweeps/${RUN_ID}}"

CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/emnist_v1_sonata/circuit_config.json}"
TRAIN_IMAGES="${TRAIN_IMAGES:-${ROOT_DIR}/data/EMNIST/emnist-letters-train-images-idx3-ubyte}"
TRAIN_LABELS="${TRAIN_LABELS:-${ROOT_DIR}/data/EMNIST/emnist-letters-train-labels-idx1-ubyte}"
TEST_IMAGES="${TEST_IMAGES:-${ROOT_DIR}/data/EMNIST/emnist-letters-test-images-idx3-ubyte}"
TEST_LABELS="${TEST_LABELS:-${ROOT_DIR}/data/EMNIST/emnist-letters-test-labels-idx1-ubyte}"

EXAMPLES_PER_CLASS="${EXAMPLES_PER_CLASS:-100}"
TEST_LIMIT="${TEST_LIMIT:-2600}"
MAX_PASSES="${MAX_PASSES:-3}"
THREADS="${THREADS:-8}"
SEED="${SEED:-42}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
VARIANTS="${VARIANTS:-baseline,dend_l4_l5_out_128x96_tol2,dend_l4_l5_out_256x128_tol2,dend_l5_out_256x128_tol3}"

EXE="${BUILD_DIR}/experiments/emnist_sonata_training"
SUMMARY="${OUT_DIR}/summary.tsv"

mkdir -p "${OUT_DIR}"

run_variant() {
  local name="$1"
  shift

  local log_path="${OUT_DIR}/${name}.log"
  local db_path="${OUT_DIR}/${name}_db"
  local cmd=(
    "${EXE}"
    --config "${CONFIG_PATH}"
    --train-images "${TRAIN_IMAGES}"
    --train-labels "${TRAIN_LABELS}"
    --test-images "${TEST_IMAGES}"
    --test-labels "${TEST_LABELS}"
    --datastore "${db_path}"
    --max-passes "${MAX_PASSES}"
    --examples-per-class "${EXAMPLES_PER_CLASS}"
    --test-limit "${TEST_LIMIT}"
    --threads "${THREADS}"
    --seed "${SEED}"
    "$@"
  )

  if [[ "${TIMEOUT_SECONDS}" != "0" ]]; then
    cmd=(timeout "${TIMEOUT_SECONDS}" "${cmd[@]}")
  fi

  echo "=== ${name} ==="
  echo "Command: ${cmd[*]}"
  echo "Log: ${log_path}"

  set +e
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
    "${cmd[@]}" 2>&1 | tee "${log_path}"
  local status="${PIPESTATUS[0]}"
  set -e

  local accuracy
  accuracy="$(awk '/Final accuracy:/ {value=$3} END {gsub("%", "", value); print value}' "${log_path}")"
  local patterns
  patterns="$(awk '/Total patterns learned:/ {value=$4} END {print value}' "${log_path}")"
  printf "%s\t%s\t%s\t%s\t%s\n" "${name}" "${status}" "${accuracy:-NA}" "${patterns:-NA}" "${log_path}" >> "${SUMMARY}"

  if [[ "${status}" != "0" ]]; then
    echo "Variant ${name} failed with status ${status}"
  fi
}

should_run_variant() {
  local name="$1"
  [[ ",${VARIANTS}," == *",${name},"* ]]
}

{
  printf "run_id\t%s\n" "${RUN_ID}"
  printf "config\t%s\n" "${CONFIG_PATH}"
  printf "examples_per_class\t%s\n" "${EXAMPLES_PER_CLASS}"
  printf "test_limit\t%s\n" "${TEST_LIMIT}"
  printf "max_passes\t%s\n" "${MAX_PASSES}"
  printf "threads\t%s\n" "${THREADS}"
  printf "seed\t%s\n" "${SEED}"
  printf "variants\t%s\n" "${VARIANTS}"
  printf "\n"
  printf "variant\tstatus\tfinal_accuracy\tpatterns\tlog\n"
} > "${SUMMARY}"

cd "${ROOT_DIR}"

if should_run_variant baseline; then
  run_variant baseline \
    --no-dendritic-spike-image-memory
fi

if should_run_variant dend_l4_l5_out_128x96_tol2; then
  run_variant dend_l4_l5_out_128x96_tol2 \
    --dendritic-spike-image-memory \
    --dendritic-image-rows 128 \
    --dendritic-image-time-bins 96 \
    --dendritic-image-bin-ms 1.0 \
    --dendritic-image-tolerance-bins 2
fi

if should_run_variant dend_l4_l5_out_256x128_tol2; then
  run_variant dend_l4_l5_out_256x128_tol2 \
    --dendritic-spike-image-memory \
    --dendritic-image-rows 256 \
    --dendritic-image-time-bins 128 \
    --dendritic-image-bin-ms 1.0 \
    --dendritic-image-tolerance-bins 2
fi

if should_run_variant dend_l5_out_256x128_tol3; then
  run_variant dend_l5_out_256x128_tol3 \
    --dendritic-spike-image-memory \
    --no-dendritic-image-l4 \
    --dendritic-image-l5 \
    --dendritic-image-output \
    --dendritic-image-rows 256 \
    --dendritic-image-time-bins 128 \
    --dendritic-image-bin-ms 1.0 \
    --dendritic-image-tolerance-bins 3
fi

echo "Sweep complete. Summary: ${SUMMARY}"
