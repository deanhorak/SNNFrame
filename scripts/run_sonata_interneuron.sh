#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

CONFIG_PATH="${CONFIG_PATH:-${BUILD_DIR}/circuit_config_interneuron_tx.sonata.json}"
DATASTORE_PATH="${DATASTORE_PATH:-${BUILD_DIR}/sonata_experiment_db_interneuron}"
LOG_PATH="${LOG_PATH:-${BUILD_DIR}/run_interneuron_emnist.log}"

MAX_PASSES="${MAX_PASSES:-1}"
EXAMPLES_PER_CLASS="${EXAMPLES_PER_CLASS:-200}"
TEST_LIMIT="${TEST_LIMIT:-5200}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"

cd "${BUILD_DIR}"
rm -rf "${DATASTORE_PATH}"

CMD=(
  ./experiments/emnist_sonata_training
  --config "${CONFIG_PATH}"
  --train-images ../data/EMNIST/emnist-letters-train-images-idx3-ubyte
  --train-labels ../data/EMNIST/emnist-letters-train-labels-idx1-ubyte
  --test-images ../data/EMNIST/emnist-letters-test-images-idx3-ubyte
  --test-labels ../data/EMNIST/emnist-letters-test-labels-idx1-ubyte
  --datastore "${DATASTORE_PATH}"
  --max-passes "${MAX_PASSES}"
  --examples-per-class "${EXAMPLES_PER_CLASS}"
  --test-limit "${TEST_LIMIT}"
  --no-output-vote
)

if [[ "${TIMEOUT_SECONDS}" -gt 0 ]]; then
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
    timeout "${TIMEOUT_SECONDS}" "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
else
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
    "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
fi
