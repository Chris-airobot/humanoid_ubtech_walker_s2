#!/usr/bin/env python3
"""Smoke-test the registered Walker S2 IK task with scripted actions.

Run from this repository root:

    /home/chris/IsaacLab/isaaclab.sh -p scripts/test_walker_s2_registered_ik_action_env.py
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Test registered Walker S2 IK task with scripted grasp/lift actions.")
parser.add_argument("--task", default="Isaac-WalkerS2-PickPlace-IK-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--print_every", type=int, default=60)
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


def _scripted_action(env, step: int) -> torch.Tensor:
    unwrapped = env.unwrapped
    action = torch.zeros((unwrapped.num_envs, unwrapped.action_manager.total_action_dim), device=unwrapped.device)

    # Same staged action used by test_walker_s2_ik_action_env.py:
    # hold pregrasp, approach, close, lift, then keep holding.
    if 180 <= step < 360:
        action[:, 0] = -0.045
    elif 360 <= step < 600:
        action[:, 6] = 1.0
    elif 600 <= step < 780:
        action[:, 2] = 0.045
        action[:, 6] = 1.0
    elif step >= 780:
        action[:, 6] = 1.0

    return action


def _print_debug(env, rew: torch.Tensor, step: int) -> None:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    obj = unwrapped.scene["object"]
    palm_ids, _ = robot.find_bodies("hand3_v1_right_R_palm_link", preserve_order=True)
    palm = (robot.data.body_pos_w[:, palm_ids[0]] - unwrapped.scene.env_origins).detach().cpu().tolist()
    obj_pos = (obj.data.root_pos_w - unwrapped.scene.env_origins).detach().cpu().tolist()
    action_term = unwrapped.action_manager.get_term("palm_ik")
    hand_ids = action_term.right_hand_joint_ids
    hand_actual = robot.data.joint_pos[:, hand_ids].detach().cpu()
    hand_target = action_term.joint_target[:, hand_ids].detach().cpu()
    hand_error = hand_actual - hand_target

    print(f"\n[STEP {step:04d}] reward={rew.detach().cpu().tolist()}")
    print(f"  action_raw={action_term.raw_actions.detach().cpu().tolist()}")
    print(
        "  action_processed=[target_nudge_xyz, target_rpy, grip, arm_offsets]="
        f"{action_term.processed_actions.detach().cpu().tolist()}"
    )
    print(f"  right_hand_target={hand_target.tolist()}")
    print(f"  right_hand_actual={hand_actual.tolist()}")
    print(f"  right_hand_error={hand_error.tolist()}")
    print(f"  palm={palm}")
    print(f"  object={obj_pos}")


def main() -> None:
    spec = gym.spec(args_cli.task)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = 40.0
    env_cfg.events.reset_object_position = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    print(f"[INFO] Loaded registered task: {args_cli.task}")
    print(f"[INFO] observation shape: {obs['policy'].shape}")
    print(f"[INFO] action space: {env.action_space}")
    print(f"[INFO] action dim: {env.unwrapped.action_manager.total_action_dim}")
    print(f"[INFO] action terms: {env.unwrapped.action_manager.active_terms}")

    for step in range(args_cli.steps):
        actions = _scripted_action(env, step)
        obs, rew, terminated, truncated, _ = env.step(actions)
        if step % args_cli.print_every == 0:
            _print_debug(env, rew, step)
        if bool(torch.any(terminated | truncated)):
            env.reset()

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Walker S2 registered IK-action task test failed:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
