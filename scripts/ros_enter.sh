#!/usr/bin/env bash
# Host-side convenience wrapper: drop into the `ros2` distrobox with ROS 2 and
# this repo's workspace already sourced, sitting in the workspace directory.
#
#   ./scripts/ros_enter.sh                    # interactive shell
#   ./scripts/ros_enter.sh ros2 topic list    # run one command and exit
#
# The container shares the host network namespace, so AirSim's RPC server on
# 127.0.0.1:41451 is reachable from inside exactly as it is from the host.
set -euo pipefail

CONTAINER="${ROS_CONTAINER:-ros2}"
REPO="/ml/airsim_swarm"                 # same path inside, thanks to ros_setup.sh step 1
WS="$REPO/ros2_ws"

if ! distrobox list 2>/dev/null | grep -qE "\|\s*${CONTAINER}\s*\|"; then
  echo "ERROR: distrobox container '${CONTAINER}' not found. Have: " >&2
  distrobox list >&2
  exit 1
fi

# `distrobox enter` starts a stopped container on its own, so no explicit start.
# ros_env.sh sources ROS *and* evicts the host's pyenv shims / airsim_venv from
# PATH -- without that, `python3` inside the container resolves to an Arch-built
# interpreter and dies on libcrypt.so.2.
PRELUDE=". $REPO/scripts/ros_env.sh
cd $WS 2>/dev/null || cd $REPO"

if (( $# )); then
  # Quote the user's argv so paths with spaces survive the bash -lc hop.
  printf -v CMD '%q ' "$@"
  exec distrobox enter "$CONTAINER" -- bash -lc "$PRELUDE
$CMD"
else
  exec distrobox enter "$CONTAINER" -- bash -lc "$PRELUDE
exec bash -i"
fi
