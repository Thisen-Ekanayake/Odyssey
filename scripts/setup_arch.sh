#!/usr/bin/env bash
# Project-level setup for AirSim swarm on Arch Linux.
# Builds the Vulkan-fixed Docker image, downloads the Blocks env,
# and creates the Python venv with the airsim client.
# Run AFTER install_arch.sh (and after re-login for the docker group).
# Idempotent: skips any step whose output already exists.
set -euo pipefail

log() { printf '\n=== %s ===\n' "$*"; }

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

# ---------------------------------------------------------------------------
log "0/4  Sanity checks"
command -v docker >/dev/null || { echo "docker missing — run install_arch.sh first" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "docker daemon unreachable — re-login or 'sudo systemctl start docker'" >&2; exit 1; }
groups | grep -qw docker || { echo "your user is not in the 'docker' group yet — log out + back in" >&2; exit 1; }
sudo pacman -S --noconfirm --needed python python-pip python-virtualenv

# ---------------------------------------------------------------------------
log "1/4  Base AirSim image (airsim_binary:10.0-devel-ubuntu18.04)"
# This image is built from the AirSim repo's docker/ helper script. ~5 GB, ~10–30 min.
if docker image inspect airsim_binary:10.0-devel-ubuntu18.04 >/dev/null 2>&1; then
  echo "Base image already present — skipping."
else
  if [[ ! -d "$WORKDIR/AirSim" ]]; then
    git clone --depth 1 https://github.com/microsoft/AirSim.git "$WORKDIR/AirSim"
  fi
  # The build script wants the host to be able to docker-pull nvidia/cudagl:10.0-devel-ubuntu18.04
  # and then layer AirSim on top of it.
  python "$WORKDIR/AirSim/docker/build_airsim_image.py" \
    --base_image nvidia/cudagl:10.0-devel-ubuntu18.04 \
    --target_image airsim_binary:10.0-devel-ubuntu18.04
fi

# ---------------------------------------------------------------------------
log "2/4  Vulkan-loader layer (airsim_swarm:vk)"
# The base image lacks libvulkan1; the NVIDIA runtime injects the ICD but not the loader.
# Without -vulkan the UE4 build defaults to OpenGL and instantly exits.
if docker image inspect airsim_swarm:vk >/dev/null 2>&1; then
  echo "airsim_swarm:vk already present — skipping. Force-rebuild: 'docker rmi airsim_swarm:vk' first."
else
  docker build -f "$WORKDIR/Dockerfile.vk" -t airsim_swarm:vk "$WORKDIR"
fi

# ---------------------------------------------------------------------------
log "3/4  Blocks environment (AirSim v1.8.1, Linux)"
if [[ -x "$WORKDIR/Blocks/LinuxNoEditor/Blocks.sh" ]]; then
  echo "Blocks already unpacked — skipping."
else
  if [[ ! -f "$WORKDIR/Blocks.zip" ]]; then
    curl -L --fail -o "$WORKDIR/Blocks.zip" \
      https://github.com/microsoft/AirSim/releases/download/v1.8.1/Blocks.zip
  fi
  mkdir -p "$WORKDIR/Blocks"
  unzip -q "$WORKDIR/Blocks.zip" -d "$WORKDIR/Blocks"
  chmod +x "$WORKDIR/Blocks/LinuxNoEditor/Blocks.sh"
fi

# ---------------------------------------------------------------------------
log "4/4  Python venv with airsim client"
if [[ ! -d "$WORKDIR/airsim_venv" ]]; then
  python -m venv "$WORKDIR/airsim_venv"
fi
# shellcheck disable=SC1091
source "$WORKDIR/airsim_venv/bin/activate"
pip install --upgrade pip setuptools wheel
# The airsim sdist imports numpy at setup time, so PEP 517 build isolation breaks it.
# Install numpy first into the venv, then airsim with --no-build-isolation.
pip install numpy
pip install airsim --no-build-isolation
deactivate

cat <<EOF

Project setup done. Run the sim:

  Terminal A:  ./scripts/run_swarm.sh                 # GUI on XWayland
               HEADLESS=1 ./scripts/run_swarm.sh      # off-screen

  Terminal B:  ./airsim_venv/bin/python swarm_demo.py
EOF
