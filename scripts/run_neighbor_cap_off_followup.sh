#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BIN="${BUILD_DIR}/experiments/emnist_sonata_training"
BASE_CONFIG="${ROOT_DIR}/configs/emnist_v1_sonata/circuit_config.json"

TRAIN_IMAGES="${ROOT_DIR}/data/EMNIST/emnist-letters-train-images-idx3-ubyte"
TRAIN_LABELS="${ROOT_DIR}/data/EMNIST/emnist-letters-train-labels-idx1-ubyte"
TEST_IMAGES="${ROOT_DIR}/data/EMNIST/emnist-letters-test-images-idx3-ubyte"
TEST_LABELS="${ROOT_DIR}/data/EMNIST/emnist-letters-test-labels-idx1-ubyte"

OUT_LOG="${BUILD_DIR}/followup_summary.log"
TMP_CFG_DIR="${BUILD_DIR}/tmp_followup_cfg"
mkdir -p "${TMP_CFG_DIR}"

run_full() {
  local name="$1"
  local db="${BUILD_DIR}/sonata_${name}_db"
  local log="${BUILD_DIR}/${name}.log"
  echo "=== ${name} ===" | tee -a "${OUT_LOG}"
  (
    cd "${BUILD_DIR}"
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH timeout 9000 "${BIN}" \
      --config "${BASE_CONFIG}" \
      --train-images "${TRAIN_IMAGES}" \
      --train-labels "${TRAIN_LABELS}" \
      --test-images "${TEST_IMAGES}" \
      --test-labels "${TEST_LABELS}" \
      --datastore "${db}" \
      --max-passes 1 \
      --examples-per-class 200 \
      --test-limit 0 > "${log}" 2>&1
  )
  local rc=$?
  local acc
  acc=$(rg -n "Final accuracy:" "${log}" | tail -n1 | sed -E 's/.*Final accuracy: ([0-9.]+)%.*/\1/')
  local t
  t=$(rg -n "Total time:" "${log}" | tail -n1 | sed -E 's/.*Total time: ([0-9.]+)s.*/\1/')
  echo "rc=${rc} final=${acc:-NA} total_s=${t:-NA} log=${log}" | tee -a "${OUT_LOG}"
  echo | tee -a "${OUT_LOG}"
}

run_quick_variant() {
  local name="$1"
  local cfg="$2"
  local db="${BUILD_DIR}/sonata_${name}_db"
  local log="${BUILD_DIR}/${name}.log"
  echo "=== ${name} ===" | tee -a "${OUT_LOG}"
  (
    cd "${BUILD_DIR}"
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH timeout 420 "${BIN}" \
      --config "${cfg}" \
      --train-images "${TRAIN_IMAGES}" \
      --train-labels "${TRAIN_LABELS}" \
      --test-images "${TEST_IMAGES}" \
      --test-labels "${TEST_LABELS}" \
      --datastore "${db}" \
      --max-passes 1 \
      --examples-per-class 20 \
      --test-limit 1040 > "${log}" 2>&1
  )
  local rc=$?
  local acc
  acc=$(rg -n "Final accuracy:" "${log}" | tail -n1 | sed -E 's/.*Final accuracy: ([0-9.]+)%.*/\1/')
  local t
  t=$(rg -n "Total time:" "${log}" | tail -n1 | sed -E 's/.*Total time: ([0-9.]+)s.*/\1/')
  echo "rc=${rc} final=${acc:-NA} total_s=${t:-NA} log=${log}" | tee -a "${OUT_LOG}"
  echo | tee -a "${OUT_LOG}"
}

echo "Follow-up run started at $(date -Iseconds)" | tee "${OUT_LOG}"
echo | tee -a "${OUT_LOG}"

# 1) Variance estimate with repeated full runs (same settings).
run_full "full_neighbor_cap_off_rep1"
run_full "full_neighbor_cap_off_rep2"

# 2) Targeted architecture-focused quick sweeps for persistent confusions.
CFG_BASELINE="${TMP_CFG_DIR}/cfg_baseline.json"
cp "${BASE_CONFIG}" "${CFG_BASELINE}"
run_quick_variant "quick_target_baseline" "${CFG_BASELINE}"

# Add higher frequency channel to sharpen small structural differences (I/L, G/Q, F/P).
CFG_HF="${TMP_CFG_DIR}/cfg_high_freq.json"
cp "${BASE_CONFIG}" "${CFG_HF}"
perl -0777 -i -pe 's/"frequencies": \[0\.8, 1\.6, 3\.2\]/"frequencies": [0.8, 1.6, 3.2, 4.8]/g' "${CFG_HF}"
run_quick_variant "quick_target_high_freq" "${CFG_HF}"

# High-frequency plus more output neurons per class for better class vote stability.
CFG_HF_OUT="${TMP_CFG_DIR}/cfg_high_freq_out6.json"
cp "${CFG_HF}" "${CFG_HF_OUT}"
perl -0777 -i -pe 's/"neurons_per_class": 4/"neurons_per_class": 6/g' "${CFG_HF_OUT}"
run_quick_variant "quick_target_high_freq_out6" "${CFG_HF_OUT}"

echo "Follow-up run finished at $(date -Iseconds)" | tee -a "${OUT_LOG}"
