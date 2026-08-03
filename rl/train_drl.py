#!/usr/bin/env python3
"""
Train a PPO formation-keeping policy in AirSim.

    # validate the environment first (sim must be running):
    ./airsim_venv/bin/python rl/train_drl.py --check

    # train:
    ./airsim_venv/bin/python rl/train_drl.py
    ./airsim_venv/bin/python rl/train_drl.py --timesteps 500000 --save models/ppo_formation

    # monitor with tensorboard:
    ./airsim_venv/bin/tensorboard --logdir logs/ppo_formation
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from airsim_gym_env import AirSimFormationEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--save",      default="models/ppo_formation")
    parser.add_argument("--check",     action="store_true",
                        help="Run the Gymnasium env checker and exit")
    args = parser.parse_args()

    env = AirSimFormationEnv(drone_name="Drone1")

    if args.check:
        print("Checking environment against Gymnasium API...")
        check_env(env, warn=True)
        print("OK.")
        env.close()
        return

    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="logs/ppo_formation",
    )

    print(f"Training PPO for {args.timesteps:,} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=CheckpointCallback(
            save_freq=10_000,
            save_path="models/checkpoints/",
            name_prefix="ppo_formation",
        ),
    )
    model.save(args.save)
    print(f"Saved → {args.save}.zip")
    env.close()


if __name__ == "__main__":
    main()
