#!/usr/bin/env bash
# All offline checks for the ROS layer. No simulator, no ROS graph needed.
#
#   ./ros2_ws/src/airsim_swarm_bridge/test/run_tests.sh
#
# Five suites:
#   test_frames.py              pure maths -- NED/FRD <-> ENU/FLU, run in the container
#   test_rpc_roundtrip.py       wire compatibility -- needs the reference msgpack-rpc
#                               server, which only the HOST venv (Python 3.10) can run,
#                               because it is the tornado stack the container cannot have
#   test_swarm_reuse.py         swarm/ manoeuvres running unmodified through the shim
#   test_rviz_config.py         every RViz display class in rviz/*.rviz actually loads
#   test_cooperative_mapping.py the 4-drone octomap, end to end, under a SKEWED sim clock
#
# The last two exist because of a specific failure: the cooperative-mapping demo
# built nothing at all while this suite passed. Two blind spots did it --
#   * the mock's "sim clock" was the wall clock, so a bridge stamping sensors from
#     one and TF from the other looked consistent (it is not; see
#     bridge_node._stamp). Hence --clock-rate/--clock-skew below.
#   * the map was only ever displayed through an RViz plugin that fails to dlopen
#     on this machine, and the mapping test runs headless. Hence test_rviz_config.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
TEST="$REPO/ros2_ws/src/airsim_swarm_bridge/test"
PORT="${MOCK_PORT:-41999}"
VENV="$REPO/airsim_venv/bin/python"
FAILED=0
MOCK=""

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# Every suite that needs the reference server goes through these two, so the
# venv guard and the bind wait exist in exactly one place. Previously only the
# first of the three checked for airsim_venv, and the other two failed with
# connection errors on a host without it rather than skipping.
start_mock() {   # start_mock [extra args to mock_airsim_server.py...]
  MOCK=""
  [[ -x "$VENV" ]] || { echo "SKIP: no airsim_venv on the host; cannot run the reference server"; return 1; }
  "$VENV" "$TEST/mock_airsim_server.py" --port "$PORT" "$@" \
      >/tmp/mock_airsim_server.log 2>&1 &
  MOCK=$!
  for _ in $(seq 20); do
    (exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null && return 0
    sleep 0.25
  done
  echo "ERROR: mock server never bound on port $PORT; see /tmp/mock_airsim_server.log" >&2
  stop_mock
  return 1
}

stop_mock() {
  [[ -n "$MOCK" ]] || return 0
  kill "$MOCK" 2>/dev/null
  wait "$MOCK" 2>/dev/null
  MOCK=""
}

step "frame conversions (container)"
"$REPO/scripts/ros_enter.sh" python3 "$TEST/test_frames.py" || FAILED=1

step "RViz configs load (container)"
"$REPO/scripts/ros_enter.sh" python3 "$TEST/test_rviz_config.py" || FAILED=1

step "RPC wire compatibility (host server <-> container client)"
if start_mock; then
  MOCK_PORT="$PORT" "$REPO/scripts/ros_enter.sh" python3 "$TEST/test_rpc_roundtrip.py" || FAILED=1
  stop_mock
fi

step "swarm/ manoeuvre reuse through the airsim shim"
if start_mock; then
  MOCK_PORT="$PORT" "$REPO/scripts/ros_enter.sh" python3 "$TEST/test_swarm_reuse.py" || FAILED=1
  stop_mock
fi

# --clock-rate 0.5 and a 30 s skew reproduce what real AirSim does with four
# drones loaded. At the old default of 1.0 this suite passed against a bridge
# that dropped 100% of clouds in the real demo.
step "cooperative mapping (4-drone octomap geometry, skewed clock, ~60s)"
if start_mock --clock-rate 0.5 --clock-skew -30 --points-per-sweep 16000; then
  MOCK_PORT="$PORT" "$REPO/scripts/ros_enter.sh" python3 "$TEST/test_cooperative_mapping.py" || FAILED=1
  stop_mock
fi

if (( FAILED )); then
  printf '\n\033[1;31mSOME TESTS FAILED\033[0m\n'; exit 1
fi
printf '\n\033[1;32mall tests passed\033[0m\n'
