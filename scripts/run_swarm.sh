#!/usr/bin/env bash
# Launch an AirSim binary environment in the (Vulkan-fixed) container with the swarm.
#   ./run_swarm.sh                 # default env (Africa), GUI window
#   ./run_swarm.sh Blocks          # pick another downloaded env by name
#   HEADLESS=1 ./run_swarm.sh      # off-screen render, control via API only
#   ENV_NAME=AirSimNH ./run_swarm.sh   # env can also be set via the ENV_NAME variable
# Any args after the env name are passed through to the UE4 binary.
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="airsim_swarm:vk"   # base airsim_binary + libvulkan1 (see Dockerfile.vk)
SETTINGS="$WORKDIR/settings.json"

# Pick the environment: first non-flag arg wins, else $ENV_NAME, else Africa.
ENV_NAME="${ENV_NAME:-Africa}"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  ENV_NAME="$1"; shift
fi

# Locate the env's launcher (<Env>.sh) inside the unzipped folder (path varies by release).
ENVS_DIR="$WORKDIR/envs"
ENV_SH="$(find "$ENVS_DIR" -maxdepth 4 -name "${ENV_NAME}.sh" 2>/dev/null | head -1)"
if [[ -z "$ENV_SH" ]]; then
  echo "ERROR: launcher '${ENV_NAME}.sh' not found under $ENVS_DIR." >&2
  echo "Did you download/unzip the '${ENV_NAME}' environment (./fetch_envs.sh ${ENV_NAME})? Launchers present:" >&2
  find "$ENVS_DIR" -maxdepth 4 -name '*.sh' 2>/dev/null | sed 's/^/  /' >&2
  exit 1
fi
ENV_DIR="$(dirname "$ENV_SH")"                # host path of the env (the LinuxNoEditor dir)
CONTAINER_ENV="/home/airsim_user/Blocks"      # mount point inside (name is historical)

# -vulkan is REQUIRED: this UE4 build defaults to OpenGL, which is broken here.
if [[ "${HEADLESS:-0}" == "1" ]]; then
  MODE_ARGS="-vulkan -RenderOffscreen -ResX=640 -ResY=480"
  X_ARGS=()
  echo "Mode: HEADLESS (no window; control via API on 127.0.0.1:41451)"
else
  MODE_ARGS="-vulkan -windowed -ResX=1280 -ResY=720"
  X_ARGS=(-e "DISPLAY=$DISPLAY" -e QT_X11_NO_MITSHM=1 -e SDL_VIDEODRIVER=x11 \
          -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
  echo "Mode: GUI window on DISPLAY=$DISPLAY (XWayland)"
  xhost +local:root >/dev/null 2>&1 || true   # let the container reach your X server
fi

echo "Environment:            $ENV_NAME"
echo "Env dir (host):         $ENV_DIR"
echo "Settings (host):        $SETTINGS"

docker run --rm -it \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  "${X_ARGS[@]}" \
  --net=host \
  -v "$SETTINGS":/home/airsim_user/Documents/AirSim/settings.json:ro \
  -v "$ENV_DIR":"$CONTAINER_ENV":rw \
  "$IMAGE" \
  bash -lc "$CONTAINER_ENV/${ENV_NAME}.sh $MODE_ARGS ${*:-}"
