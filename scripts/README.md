# scripts/

Sim launcher + one-time host setup.

| Script | What it does |
|---|---|
| `run_swarm.sh` | launches the sim (GPU + Vulkan + X11 + mounts); pick the env as an arg, `HEADLESS=1` for off-screen |
| `fetch_envs.sh` | downloads/extracts AirSim v1.8.1 environments into `envs/` |
| `install_arch.sh` | system-level install for Arch Linux: NVIDIA driver, Docker, nvidia-container-toolkit, CDI, XWayland |
| `setup_arch.sh` | project-level setup for Arch Linux: builds the Vulkan-fixed image, downloads Blocks, creates the venv |
| `install_ubuntu.sh` | system-level install for Ubuntu 22.04/24.04 (same as `install_arch.sh`) |
| `setup_ubuntu.sh` | project-level setup for Ubuntu (same as `setup_arch.sh`) |

## Run

```bash
# one-time host setup (Arch example; swap _arch for _ubuntu on Ubuntu):
./scripts/install_arch.sh      # sudo, then log out/in for the docker group
./scripts/setup_arch.sh

# every session:
./scripts/fetch_envs.sh AirSimNH         # once per environment
./scripts/run_swarm.sh AirSimNH
HEADLESS=1 ./scripts/run_swarm.sh AirSimNH
```

## Common instructions

- `install_*`/`setup_*` are idempotent — safe to re-run, `install_*` needs sudo.
- Run `install_*` before `setup_*`, and re-login (docker group) in between.
- Never drop `-vulkan` — `run_swarm.sh` already passes it; OpenGL is broken in this image.
