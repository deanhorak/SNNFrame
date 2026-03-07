#!/usr/bin/env bash
set -euo pipefail

cd /home/dean/repos/SNNFrame/build
rm -rf ./sonata_experiment_db

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
MAX_PASSES="${MAX_PASSES:-1}"
EXAMPLES_PER_CLASS="${EXAMPLES_PER_CLASS:-200}"
TEST_LIMIT="${TEST_LIMIT:-0}"

CMD=(
  ./experiments/emnist_sonata_training
  --config ../configs/emnist_v1_sonata/circuit_config.json
  --train-images ../data/EMNIST/emnist-letters-train-images-idx3-ubyte
  --train-labels ../data/EMNIST/emnist-letters-train-labels-idx1-ubyte
  --test-images ../data/EMNIST/emnist-letters-test-images-idx3-ubyte
  --test-labels ../data/EMNIST/emnist-letters-test-labels-idx1-ubyte
  --datastore ./sonata_experiment_db
  --max-passes "${MAX_PASSES}"
  --examples-per-class "${EXAMPLES_PER_CLASS}"
  --test-limit "${TEST_LIMIT}"
  --no-output-vote
)

if [[ "${TIMEOUT_SECONDS}" -gt 0 ]]; then
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH timeout "${TIMEOUT_SECONDS}" \
  "${CMD[@]}" 2>&1 | tee run_test.log
else
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
  "${CMD[@]}" 2>&1 | tee run_test.log
fi
   
# | grep -E "PERF|Training:|Test:|accuracy"
