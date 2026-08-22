# Shell fragment, meant to be SOURCED (by ros_setup.sh, ros_enter.sh, ~/.bashrc).
# Prepares a shell inside the `ros2` distrobox for ROS 2 + this workspace.
#
# Two container-specific hazards it exists to defuse, both caused by distrobox
# sharing the host's $HOME and $PATH straight into the container:
#
#  1. `python3` resolves to the HOST's interpreters -- the pyenv shims under
#     ~/.pyenv, and (now that ros_setup.sh symlinks /ml) the Arch-built
#     airsim_venv. Running either under Ubuntu 24.04 dies with
#     "libcrypt.so.2: cannot open shared object file". ROS needs the container's
#     own /usr/bin/python3, which is where rclpy and msgpack live.
#  2. ~/.bashrc is the SAME FILE the host reads, so anything added there must
#     no-op outside the container.
#
# Sourcing this on the host is harmless: the guard below returns immediately.

# Only act inside a container (podman/distrobox drops /run/.containerenv).
if [ -f /run/.containerenv ] || [ -f /.dockerenv ]; then

  # 1. drop host-Python path entries, then make sure the container's own bin
  #    directories win. Uses only shell builtins -- no python3, which is
  #    precisely what is broken at this point.
  _asw_clean_path=""
  _asw_ifs_saved=$IFS
  IFS=:
  for _asw_p in $PATH; do
    case "$_asw_p" in
      */.pyenv/*|*/pyenv/*|*/airsim_venv/*) continue ;;
      "") continue ;;
    esac
    case ":$_asw_clean_path:" in
      *":$_asw_p:"*) continue ;;                # de-duplicate
    esac
    _asw_clean_path="${_asw_clean_path:+$_asw_clean_path:}$_asw_p"
  done
  IFS=$_asw_ifs_saved
  PATH="/usr/local/bin:/usr/bin:/bin:$_asw_clean_path"
  export PATH
  unset _asw_clean_path _asw_ifs_saved _asw_p

  # A stale PYTHONPATH/VIRTUAL_ENV pointing at the host venv would re-introduce
  # Arch-built extension modules into the container interpreter.
  case "${PYTHONPATH:-}" in *airsim_venv*|*.pyenv*) unset PYTHONPATH ;; esac
  case "${VIRTUAL_ENV:-}" in *airsim_venv*|*.pyenv*) unset VIRTUAL_ENV ;; esac
  unset PYENV_ROOT PYENV_VERSION 2>/dev/null || true

  # 2. ROS 2 + this workspace
  [ -f /opt/ros/jazzy/setup.bash ] && . /opt/ros/jazzy/setup.bash
  [ -f /ml/airsim_swarm/ros2_ws/install/setup.bash ] && . /ml/airsim_swarm/ros2_ws/install/setup.bash
  export AIRSIM_SWARM_REPO=/ml/airsim_swarm

  # Every drone bridge is its own process on one host-network DDS domain;
  # pin it so a stray ROS node elsewhere on the machine cannot join by accident.
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
  export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
fi
