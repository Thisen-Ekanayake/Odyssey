# rl/

Reinforcement learning: single-drone ring-orbit formation-keeping.

| Script | What it does |
|---|---|
| `airsim_gym_env.py` | Gymnasium env (`AirSimFormationEnv`) wrapping AirSim for ring-orbit formation-keeping |
| `train_drl.py` | PPO training entry point |

## Run

```bash
./airsim_venv/bin/python rl/train_drl.py --check      # validate the env (sim must be running)
./airsim_venv/bin/python rl/train_drl.py               # train
./airsim_venv/bin/python rl/train_drl.py --timesteps 500000 --save models/ppo_formation
./airsim_venv/bin/tensorboard --logdir logs/ppo_formation
```

## Common instructions

- Sim must be running before `--check` or training.
- Checkpoints go to `models/`, tensorboard logs go to `logs/` — both gitignored, created at repo root.
- Ring geometry constants in `airsim_gym_env.py` must match `swarm/swarm_circle.py`.
