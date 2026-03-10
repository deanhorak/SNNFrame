#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BASE_CONFIG="${BASE_CONFIG:-${ROOT_DIR}/configs/emnist_v1_sonata/circuit_config.json}"

PORT_L2R="${PORT_L2R:-5050}"
PORT_R2L="${PORT_R2L:-5051}"
AUTO_PORT="${AUTO_PORT:-1}"
LEFT_CONFIG="${LEFT_CONFIG:-${BUILD_DIR}/circuit_config_left_tx.sonata.json}"
RIGHT_CONFIG="${RIGHT_CONFIG:-${BUILD_DIR}/circuit_config_right_rx.sonata.json}"

LEFT_DB="${LEFT_DB:-${BUILD_DIR}/sonata_left_db}"
RIGHT_DB="${RIGHT_DB:-${BUILD_DIR}/sonata_right_db}"
LEFT_LOG="${LEFT_LOG:-${BUILD_DIR}/run_left_hemisphere.log}"
RIGHT_LOG="${RIGHT_LOG:-${BUILD_DIR}/run_right_hemisphere.log}"

MAX_PASSES="${MAX_PASSES:-1}"
EXAMPLES_PER_CLASS="${EXAMPLES_PER_CLASS:-100}"
TEST_LIMIT="${TEST_LIMIT:-1000}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"

port_in_use() {
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$1$"
}

if [[ "${AUTO_PORT}" == "1" ]]; then
  CANDIDATE="${PORT_L2R}"
  for _ in $(seq 1 20); do
    if ! port_in_use "${CANDIDATE}" && ! port_in_use "$((CANDIDATE + 1))"; then
      PORT_L2R="${CANDIDATE}"
      PORT_R2L="$((CANDIDATE + 1))"
      break
    fi
    CANDIDATE=$((CANDIDATE + 1))
  done
fi

if port_in_use "${PORT_L2R}" || port_in_use "${PORT_R2L}"; then
  echo "[interhemisphere] ERROR: one of the ports is already in use: ${PORT_L2R}, ${PORT_R2L}"
  echo "Use PORT_L2R/PORT_R2L or stop the process using them."
  exit 1
fi
echo "[interhemisphere] using corpus-callosum ports L->R=${PORT_L2R}, R->L=${PORT_R2L}"

python3 - <<PY
import json
from pathlib import Path

base = Path("${BASE_CONFIG}")
left_out = Path("${LEFT_CONFIG}")
right_out = Path("${RIGHT_CONFIG}")
port_l2r = int("${PORT_L2R}")
port_r2l = int("${PORT_R2L}")

cfg = json.loads(base.read_text())

def with_adapters(obj, adapters_to_add):
    out = json.loads(json.dumps(obj))
    snn = out.setdefault("snnframe", {})
    adapters = snn.setdefault("adapters", [])
    adapters = [a for a in adapters if a.get("name") not in {"corpus_tx_l", "corpus_rx_l", "corpus_tx_r", "corpus_rx_r"}]
    adapters.extend(adapters_to_add)
    snn["adapters"] = adapters
    return out

left = with_adapters(cfg, [{
    "name": "corpus_tx_l",
    "type": "interneuron_tx",
    "role": "motor",
    "bind_to": "output",
    "temporal_window_ms": 10.0,
    "double_params": {"update_interval_ms": 10.0, "startup_grace_ms": 8000.0},
    "int_params": {"remote_port": port_l2r, "connect_timeout_ms": 1000, "send_timeout_ms": 100},
    "string_params": {"remote_host": "127.0.0.1"}
}, {
    "name": "corpus_rx_l",
    "type": "interneuron_rx",
    "role": "sensory",
    "bind_to": "input",
    "temporal_window_ms": 10.0,
    "int_params": {"bind_port": port_r2l, "neuron_count": 784, "receive_timeout_ms": 5},
    "string_params": {"bind_host": "0.0.0.0"}
}])

right = with_adapters(cfg, [{
    "name": "corpus_rx_r",
    "type": "interneuron_rx",
    "role": "sensory",
    "bind_to": "input",
    "temporal_window_ms": 10.0,
    "int_params": {"bind_port": port_l2r, "neuron_count": 784, "receive_timeout_ms": 5},
    "string_params": {"bind_host": "0.0.0.0"}
}, {
    "name": "corpus_tx_r",
    "type": "interneuron_tx",
    "role": "motor",
    "bind_to": "output",
    "temporal_window_ms": 10.0,
    "double_params": {"update_interval_ms": 10.0, "startup_grace_ms": 8000.0},
    "int_params": {"remote_port": port_r2l, "connect_timeout_ms": 1000, "send_timeout_ms": 100},
    "string_params": {"remote_host": "127.0.0.1"}
}])

left_out.write_text(json.dumps(left, indent=2))
right_out.write_text(json.dumps(right, indent=2))
print("wrote", left_out)
print("wrote", right_out)
PY

cd "${BUILD_DIR}"
rm -rf "${LEFT_DB}" "${RIGHT_DB}"

COMMON_ARGS=(
  --train-images ../data/EMNIST/emnist-letters-train-images-idx3-ubyte
  --train-labels ../data/EMNIST/emnist-letters-train-labels-idx1-ubyte
  --test-images ../data/EMNIST/emnist-letters-test-images-idx3-ubyte
  --test-labels ../data/EMNIST/emnist-letters-test-labels-idx1-ubyte
  --max-passes "${MAX_PASSES}"
  --examples-per-class "${EXAMPLES_PER_CLASS}"
  --test-limit "${TEST_LIMIT}"
  --no-output-vote
)

run_side() {
  local config="$1"
  local db="$2"
  local log="$3"
  if [[ "${TIMEOUT_SECONDS}" -gt 0 ]]; then
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
      timeout "${TIMEOUT_SECONDS}" ./experiments/emnist_sonata_training \
      --config "${config}" --datastore "${db}" "${COMMON_ARGS[@]}" 2>&1 | tee "${log}"
  else
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
      ./experiments/emnist_sonata_training \
      --config "${config}" --datastore "${db}" "${COMMON_ARGS[@]}" 2>&1 | tee "${log}"
  fi
}

wait_for_listener() {
  local port="$1"
  local timeout_s="${2:-180}"
  local elapsed=0
  while (( elapsed < timeout_s )); do
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"; then
      return 0
    fi
    if ! kill -0 "${RIGHT_PID}" >/dev/null 2>&1; then
      return 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  return 1
}

echo "[interhemisphere] starting RIGHT (receiver) first..."
run_side "${RIGHT_CONFIG}" "${RIGHT_DB}" "${RIGHT_LOG}" &
RIGHT_PID=$!
if ! wait_for_listener "${PORT_L2R}" 180; then
  echo "[interhemisphere] ERROR: right hemisphere never opened listener on ${PORT_L2R}."
  echo "Check ${RIGHT_LOG} for details."
  wait "${RIGHT_PID}" || true
  exit 1
fi

echo "[interhemisphere] starting LEFT (transmitter)..."
set +e
run_side "${LEFT_CONFIG}" "${LEFT_DB}" "${LEFT_LOG}"
LEFT_RC=$?
set -e

if kill -0 "${RIGHT_PID}" >/dev/null 2>&1; then
  wait "${RIGHT_PID}" || true
fi

exit "${LEFT_RC}"
