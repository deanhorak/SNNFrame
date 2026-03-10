#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BIN="${BUILD_DIR}/experiments/interneuron_adapter_demo"
PORT="${PORT:-55000}"
CHANNELS="${CHANNELS:-16}"
FRAMES="${FRAMES:-20}"
INTERVAL_MS="${INTERVAL_MS:-10}"
RX_DURATION_MS="${RX_DURATION_MS:-5000}"

cd "${ROOT_DIR}"
cmake --build "${BUILD_DIR}" --target interneuron_adapter_demo -j8 >/dev/null

RX_LOG="${BUILD_DIR}/interneuron_rx.log"
TX_LOG="${BUILD_DIR}/interneuron_tx.log"
rm -f "${RX_LOG}" "${TX_LOG}"

env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
  "${BIN}" --mode rx --host 0.0.0.0 --port "${PORT}" --channels "${CHANNELS}" \
  --duration-ms "${RX_DURATION_MS}" --poll-ms 5 >"${RX_LOG}" 2>&1 &
RX_PID=$!
trap 'kill ${RX_PID} 2>/dev/null || true' EXIT

sleep 0.5
env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
  "${BIN}" --mode tx --host 127.0.0.1 --port "${PORT}" --channels "${CHANNELS}" \
  --frames "${FRAMES}" --interval-ms "${INTERVAL_MS}" >"${TX_LOG}" 2>&1

wait "${RX_PID}" || true
trap - EXIT

RX_FRAMES="$(grep -c '^RX frame' "${RX_LOG}" || true)"
TX_FRAMES="$(grep -c '^TX frame' "${TX_LOG}" || true)"

echo "TX frames sent: ${TX_FRAMES}"
echo "RX frames observed: ${RX_FRAMES}"
echo "RX log: ${RX_LOG}"
echo "TX log: ${TX_LOG}"

if [[ "${RX_FRAMES}" -gt 0 ]]; then
  echo "Interneuron TCP demo: PASS"
else
  echo "Interneuron TCP demo: FAIL (no received frames)"
  exit 1
fi
