#!/usr/bin/env python3
"""Train the registered Walker S2 IK pick/place task with RSL-RL.

Run from this repository root:

    /home/chris/IsaacLab/isaaclab.sh -p scripts/train_walker_s2_rsl_rl.py --num_envs 16 --max_iterations 100
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train Walker S2 IK pick/place with RSL-RL PPO.")
parser.add_argument("--task", default="Isaac-WalkerS2-PickPlace-IK-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="")
parser.add_argument("--randomize_object", action="store_true", help="Enable reset-time object pose randomization.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import isaaclab_walker_s2.tasks.pick_place  # noqa: F401, E402


def _load_cfg_from_entry_point(entry_point):
    """Instantiate a config from a Gym registry entry point."""
    if isinstance(entry_point, str):
        module_name, attr_name = entry_point.split(":")
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)()
    return entry_point()


def main() -> None:
    spec = gym.spec(args_cli.task)
    env_cfg: ManagerBasedRLEnvCfg = _load_cfg_from_entry_point(spec.kwargs["env_cfg_entry_point"])
    agent_cfg = _load_cfg_from_entry_point(spec.kwargs["rsl_rl_cfg_entry_point"])

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 40.0
    if not args_cli.randomize_object:
        env_cfg.events.reset_object_position = None

    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device
    agent_cfg.run_name = args_cli.run_name
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    installed_version = "0.0.0"
    try:
        import importlib.metadata as metadata

        installed_version = metadata.version("rsl-rl-lib")
    except Exception:
        pass
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    print(f"[INFO] Training task: {args_cli.task}")
    print(f"[INFO] num_envs={env_cfg.scene.num_envs} max_iterations={agent_cfg.max_iterations}")
    print(f"[INFO] object_randomization={args_cli.randomize_object}")
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    start_time = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[INFO] Training finished in {time.time() - start_time:.1f}s")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Walker S2 RSL-RL training failed:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
