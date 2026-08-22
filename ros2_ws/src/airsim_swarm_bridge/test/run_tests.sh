#!/usr/bin/env bash
# All offline checks for the ROS layer. No simulator, no ROS graph needed.
#
#   ./ros2_ws/src/airsim_swarm_bridge/test/run_tests.sh
#
# Two suites:
#   test_frames.py         pure maths -- NED/FRD <-> ENU/FLU, run in the container
#   test_rpc_roundtrip.py  wire compatibility -- needs the reference msgpack-rpc
#                          server, which only the HOST venv (Python 3.10) can run,
#                          because it is the tornado stack the container cannot have
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
TEST="$REPO/ros2_ws/src/airsim_swarm_bridge/test"
PORT="${MOCK_PORT:-41999}"
FAILED=0

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

step "frame conversions (container)"
"$REPO/scripts/ros_enter.sh" python3 "$TEST/test_frames.py" || FAILED=1

step "RPC wire compatibility (host server <-> container client)"
if [[ ! -x "$REPO/airsim_venv/bin/python" ]]; then
  echo "SKIP: no airsim_venv on the host; cannot run the reference server"
else
  "$REPO/airsim_venv/bin/python" "$TEST/mock_airsim_server.py" --port "$PORT" \
      >/tmp/mock_airsim_server.log 2>&1 &
  MOCK=$!
  # The server needs a moment to bind before the client's first connect.
  for _ in $(seq 20); do
    (exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null && break
    sleep 0.25
  done
  MOCK_PORT="$PORT" "$REPO/scripts/ros_enter.sh" python3 "$TEST/test_rpc_roundtrip.py" || FAILED=1
  kill "$MOCK" 2>/dev/null
  wait "$MOCK" 2>/dev/null
fi

step "swarm/ manoeuvre reuse through the airsim shim"
"$REPO/airsim_venv/bin/python" "$TEST/mock_airsim_server.py" --port "$PORT" \
    >/tmp/mock_airsim_server.log 2>&1 &
MOCK=$!
for _ in $(seq 20); do
  (exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null && break
  sleep 0.25
done
MOCK_PORT="$PORT" "$REPO/scripts/ros_enter.sh" python3 "$TEST/test_swarm_reuse.py" || FAILED=1
kill "$MOCK" 2>/dev/null
wait "$MOCK" 2>/dev/null

if (( FAILED )); then
  printf '\n\033[1;31mSOME TESTS FAILED\033[0m\n'; exit 1
fi
printf '\n\033[1;32mall tests passed\033[0m\n'
