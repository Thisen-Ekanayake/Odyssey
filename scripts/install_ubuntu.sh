#!/usr/bin/env bash
# System-level install for AirSim swarm on Ubuntu (22.04 / 24.04).
# Installs: NVIDIA driver, Docker CE, nvidia-container-toolkit, CDI spec, XWayland.
# Idempotent: safe to re-run. Requires sudo.
#
# After this completes, log out + log back in (for docker group), then run setup_ubuntu.sh.
set -euo pipefail

log() { printf '\n=== %s ===\n' "$*"; }

if [[ "$EUID" -eq 0 ]]; then
  echo "Run as your normal user (script will sudo as needed)." >&2
  exit 1
fi

. /etc/os-release
case "$VERSION_ID" in
  22.04|24.04) ;;
  *) echo "Untested on Ubuntu $VERSION_ID; proceeding anyway." >&2 ;;
esac

# ---------------------------------------------------------------------------
log "1/6  Base packages"
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release git unzip software-properties-common

# ---------------------------------------------------------------------------
log "2/6  NVIDIA driver (recommended version via ubuntu-drivers)"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  sudo apt-get install -y ubuntu-drivers-common
  sudo ubuntu-drivers install
  echo "Driver installed. A REBOOT is required before nvidia-smi (and Docker GPU) will work."
fi

# ---------------------------------------------------------------------------
log "3/6  Docker CE (official repo, not the snap/distro docker.io)"
if ! command -v docker >/dev/null 2>&1; then
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
sudo systemctl enable --now docker

if ! groups "$USER" | grep -qw docker; then
  sudo usermod -aG docker "$USER"
  echo "Added $USER to docker group. You MUST log out + back in before the rest works."
fi

# ---------------------------------------------------------------------------
log "4/6  nvidia-container-toolkit (official NVIDIA repo)"
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
fi

# ---------------------------------------------------------------------------
log "5/6  Docker daemon config (nvidia runtime + data-root) and CDI spec"
sudo nvidia-ctk runtime configure --runtime=docker
# Optional: pin data-root to /home/docker to match this project's convention.
# Comment out the next block if you want the default /var/lib/docker.
if [[ ! -d /home/docker ]]; then
  sudo install -d -m 0711 /home/docker
fi
if ! grep -q '"data-root"' /etc/docker/daemon.json 2>/dev/null; then
  tmp="$(mktemp)"
  sudo cp /etc/docker/daemon.json "$tmp"
  python3 - <<'PY' "$tmp"
import json, sys
p = sys.argv[1]
with open(p) as f: d = json.load(f)
d["data-root"] = "/home/docker"
with open(p, "w") as f: json.dump(d, f, indent=4)
PY
  sudo install -m 0644 "$tmp" /etc/docker/daemon.json
  rm "$tmp"
fi
sudo systemctl restart docker

sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# ---------------------------------------------------------------------------
log "6/6  XWayland + X11 client libs"
sudo apt-get install -y xwayland x11-xserver-utils

# Sanity check
log "Smoke test: nvidia-smi inside a container"
if groups | grep -qw docker; then
  docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi || \
    echo "Smoke test failed — driver may need a reboot, or daemon.json/CDI is misconfigured."
else
  echo "Skipped (you need to re-login for the docker group first)."
fi

cat <<EOF

System install done.

Next steps:
  1. If the driver was freshly installed: REBOOT.
  2. Otherwise: log out + log back in so the 'docker' group takes effect.
  3. Run:  ./scripts/setup_ubuntu.sh
EOF
