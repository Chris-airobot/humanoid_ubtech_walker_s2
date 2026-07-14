#!/usr/bin/env python3
"""Smoke-test Gym registration for the Walker S2 IK pick/place task.

Run from this repository root:

    /home/chris/IsaacLab/isaaclab.sh -p scripts/test_walker_s2_registered_task.py
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Test registered Walker S2 Gym task loading.")
parser.add_argument("--task", default="Isaac-WalkerS2-PickPlace-IK-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--print_every", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import isaaclab_walker_s2.tasks.pick_place  # noqa: F401, E402


def main() -> None:
    spec = gym.spec(args_cli.task)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.events.reset_object_position = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    print(f"[INFO] Loaded registered task: {args_cli.task}")
    print(f"[INFO] observation shape: {obs['policy'].shape}")
    print(f"[INFO] action space: {env.action_space}")
    print(f"[INFO] action terms: {env.unwrapped.action_manager.active_terms}")

    for step in range(args_cli.steps):
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, rew, terminated, truncated, _ = env.step(actions)
        if step % args_cli.print_every == 0:
            print(f"[STEP {step:04d}] reward={rew.detach().cpu().tolist()}")
        if bool(torch.any(terminated | truncated)):
            env.reset()

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Walker S2 registered task test failed:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
