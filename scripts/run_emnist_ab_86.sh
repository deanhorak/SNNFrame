#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/build/experiments"

DB_PATH="${1:-./emnist_training_db_letters_ab_novote_nocomp3}"
MAX_PASSES="${2:-20}"

cd "${RUN_DIR}"

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
./emnist_letters_training \
  --db-path "${DB_PATH}" \
  --max-passes "${MAX_PASSES}" \
  --no-output-vote \
  --no-output-competition
