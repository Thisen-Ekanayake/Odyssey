#!/usr/bin/env bash
# One-time setup of the ROS 2 Jazzy layer INSIDE the `ros2` distrobox container.
#
#   distrobox enter ros2 -- /ml/airsim_swarm/scripts/ros_setup.sh
#
# Safe to re-run: every step is idempotent. Pass --no-build to skip colcon.
#
# Why this script exists at all (see docs/ROS.md for the long version):
#   * the container only mounts $HOME, so the repo is reachable at
#     /run/host/ml/airsim_swarm -- step 1 symlinks /ml so both host and
#     container use the SAME absolute paths and nothing has to branch on it;
#   * the container is Python 3.12, where the `airsim` pip package is dead
#     (it pins tornado<5, and tornado 4.5.3 uses collections.MutableMapping /
#     inspect.getargspec, both removed by 3.10/3.11). We therefore install
#     python3-msgpack and talk raw msgpack-rpc instead -- never `pip install airsim`
#     in here. That would also drag tornado into a ROS environment for no reason.
set -euo pipefail

HOST_REPO="/run/host/ml/airsim_swarm"
REPO="/ml/airsim_swarm"
WS="$REPO/ros2_ws"
ROS_DISTRO_DEFAULT="jazzy"
DO_BUILD=1
[[ "${1:-}" == "--no-build" ]] && DO_BUILD=0

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- guard: this must run inside the container, not on the Arch host ---------
if [[ ! -d "/opt/ros/$ROS_DISTRO_DEFAULT" ]]; then
  die "no /opt/ros/$ROS_DISTRO_DEFAULT here. Run this INSIDE the container:
     distrobox enter ros2 -- $REPO/scripts/ros_setup.sh"
fi

# --- 1. path parity: /ml -> /run/host/ml ------------------------------------
say "1/5  path parity (/ml -> /run/host/ml)"
if [[ -e "$REPO/CLAUDE.md" ]]; then
  echo "     /ml/airsim_swarm already reachable"
elif [[ -d "$HOST_REPO" ]]; then
  sudo ln -sfn /run/host/ml /ml
  echo "     symlinked /ml -> /run/host/ml"
else
  die "$HOST_REPO not visible. Is the repo still at /ml/airsim_swarm on the host?"
fi
[[ -e "$REPO/CLAUDE.md" ]] || die "/ml/airsim_swarm does not look like the repo"

# --- 2. apt dependencies -----------------------------------------------------
say "2/5  apt packages"
PKGS=(
  python3-msgpack                      # raw msgpack-rpc to AirSim (NOT the airsim pip pkg)
  python3-colcon-common-extensions     # colcon build
  mesa-utils                           # glxinfo, for the RViz render check below
  ros-jazzy-rtabmap-ros                # 3D LiDAR SLAM: icp_odometry + loop closure
  ros-jazzy-octomap-server             # 3D occupancy grid
  ros-jazzy-octomap-rviz-plugins       # ...and its RViz display
  ros-jazzy-robot-localization         # EKF fusing IMU + ICP odometry
  ros-jazzy-foxglove-bridge            # web visualisation over websocket
  ros-jazzy-plotjuggler-ros            # time-series plots
  ros-jazzy-pointcloud-to-laserscan    # optional 2D slice for slam_toolbox
  ros-jazzy-slam-toolbox               # optional 2D comparison
  ros-jazzy-rqt-image-view             # quick camera peek
)
MISSING=()
for p in "${PKGS[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p")
done
if (( ${#MISSING[@]} )); then
  echo "     installing: ${MISSING[*]}"
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${MISSING[@]}"
else
  echo "     all present"
fi

# --- 3. render check ---------------------------------------------------------
say "3/5  OpenGL / RViz render path"
GL="$(timeout 20 glxinfo -B 2>/dev/null | sed -n 's/^ *OpenGL renderer string: //p' || true)"
if [[ -z "$GL" ]]; then
  warn "glxinfo produced nothing -- RViz may not open. Is DISPLAY set? (DISPLAY=${DISPLAY:-unset})"
elif [[ "$GL" == *llvmpipe* || "$GL" == *softpipe* ]]; then
  warn "software rendering ($GL). RViz will work but be slow.
       This container was not created with 'distrobox create --nvidia'. To fix later:
         podman commit ros2 ros2-snap    # keeps everything installed
         distrobox rm ros2 && distrobox create --image ros2-snap --name ros2 --nvidia"
else
  echo "     $GL"
fi

# --- 4. build the workspace --------------------------------------------------
say "4/5  colcon build"
if (( DO_BUILD )); then
  [[ -d "$WS/src" ]] || die "no workspace at $WS/src"
  # ros_env.sh both sources ROS and evicts the host's pyenv/airsim_venv from
  # PATH -- colcon must build against the CONTAINER's /usr/bin/python3, or the
  # generated entry points get a shebang pointing at an Arch interpreter.
  # shellcheck disable=SC1091
  source "$REPO/scripts/ros_env.sh"
  # Build artefacts must NOT land on the host ext4 through the symlink with
  # absolute paths baked in -- they are gitignored, but --symlink-install keeps
  # Python edits live without rebuilding, which is what we want during dev.
  ( cd "$WS" && colcon build --symlink-install --event-handlers console_cohesion+ )
else
  echo "     skipped (--no-build)"
fi

# --- 5. shell wiring (container-only ~/.bashrc) ------------------------------
say "5/5  shell wiring"
MARK="# >>> airsim_swarm ros2 >>>"
# NOTE: ~/.bashrc is the SAME FILE the Arch host reads -- distrobox mounts $HOME
# straight through. The block therefore only delegates to ros_env.sh, which
# no-ops outside a container, rather than sourcing ROS unconditionally.
if grep -qF "$MARK" ~/.bashrc 2>/dev/null; then
  echo "     ~/.bashrc already wired"
else
  cat >> ~/.bashrc <<BASHRC

$MARK
# guarded: this file is shared with the Arch host (distrobox mounts \$HOME)
[ -f $REPO/scripts/ros_env.sh ] && . $REPO/scripts/ros_env.sh
# <<< airsim_swarm ros2 <<<
BASHRC
  echo "     appended a guarded source line to ~/.bashrc"
fi

say "done"
cat <<'NEXT'
Next:
  # host, terminal 1
  ./scripts/run_swarm.sh AirSimNH
  # container, terminal 2
  ./scripts/ros_enter.sh
  python3 -m airsim_swarm_bridge.rpc --ping --list-vehicles
NEXT
