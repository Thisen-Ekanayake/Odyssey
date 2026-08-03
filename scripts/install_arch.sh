#!/usr/bin/env bash
# System-level install for AirSim swarm on Arch Linux.
# Installs: NVIDIA driver, Docker, nvidia-container-toolkit, CDI spec, XWayland.
# Idempotent: safe to re-run. Requires sudo.
#
# After this completes, log out + log back in (for docker group), then run setup_arch.sh.
set -euo pipefail

log() { printf '\n=== %s ===\n' "$*"; }

if [[ "$EUID" -eq 0 ]]; then
  echo "Run as your normal user (script will sudo as needed)." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
log "1/6  Sync pacman + base tools"
sudo pacman -Sy --noconfirm --needed base-devel git curl unzip

# ---------------------------------------------------------------------------
log "2/6  NVIDIA driver"
# Use nvidia-open for modern GPUs (RTX 4060, etc). Falls back to nvidia for older cards.
if ! pacman -Qq nvidia-open >/dev/null 2>&1 && ! pacman -Qq nvidia >/dev/null 2>&1; then
  sudo pacman -S --noconfirm --needed nvidia-open nvidia-utils
fi
sudo pacman -S --noconfirm --needed nvidia-utils libglvnd
nvidia-smi || { echo "nvidia-smi failed — reboot may be required after fresh driver install." >&2; }

# ---------------------------------------------------------------------------
log "3/6  Docker"
sudo pacman -S --noconfirm --needed docker
sudo systemctl enable --now docker.service

if ! groups "$USER" | grep -qw docker; then
  sudo usermod -aG docker "$USER"
  echo "Added $USER to docker group. You MUST log out and log back in before the rest works."
fi

# ---------------------------------------------------------------------------
log "4/6  nvidia-container-toolkit (from AUR; uses yay if present, else builds via makepkg)"
if ! pacman -Qq nvidia-container-toolkit >/dev/null 2>&1; then
  if command -v yay >/dev/null 2>&1; then
    yay -S --noconfirm nvidia-container-toolkit
  else
    tmp="$(mktemp -d)"
    git clone https://aur.archlinux.org/nvidia-container-toolkit.git "$tmp/nct"
    ( cd "$tmp/nct" && makepkg -si --noconfirm )
    rm -rf "$tmp"
  fi
fi

# Register the nvidia runtime with Docker and set the data-root used on this host.
log "5/6  Docker daemon config (nvidia runtime + data-root)"
sudo install -d -m 0755 /etc/docker
if [[ ! -f /etc/docker/daemon.json ]]; then
  sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "data-root": "/home/docker"
}
JSON
  sudo systemctl restart docker
else
  echo "/etc/docker/daemon.json already exists — leaving untouched. Verify nvidia runtime is registered."
fi

# Generate the CDI spec so `--runtime=nvidia` (or `--device nvidia.com/gpu=all`) works.
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# ---------------------------------------------------------------------------
log "6/6  XWayland + X11 client libs (Wayland sessions need XWayland for the UE4 window)"
sudo pacman -S --noconfirm --needed xorg-xwayland xorg-xhost

# Quick GPU-in-container sanity check.
log "Smoke test: nvidia-smi inside a container"
if groups | grep -qw docker; then
  docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi || \
    echo "Smoke test failed — check daemon config and CDI spec."
else
  echo "Skipped (you need to re-login for the docker group first)."
fi

cat <<EOF

System install done.

Next steps:
  1. Log out + log back in (or reboot) so your shell picks up the 'docker' group.
  2. Run:  ./scripts/setup_arch.sh
EOF
