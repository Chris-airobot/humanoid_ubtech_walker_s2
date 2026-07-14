#!/usr/bin/env python3
"""Smoke-test the Walker S2 palm-IK action interface.

Run from this repository root:

    /home/chris/IsaacLab/isaaclab.sh -p scripts/test_walker_s2_ik_action_env.py
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Test Walker S2 palm-IK pick/place action interface.")
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--print_every", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab_walker_s2.tasks.pick_place.walker_s2_pick_place_env_cfg import WalkerS2IKPickPlaceEnvCfg  # noqa: E402


def _print_debug(env: ManagerBasedRLEnv, obs: dict, rew: torch.Tensor, step: int):
    robot = env.scene["robot"]
    obj = env.scene["object"]
    palm_ids, _ = robot.find_bodies("hand3_v1_right_R_palm_link", preserve_order=True)
    palm = (robot.data.body_pos_w[:, palm_ids[0]] - env.scene.env_origins).detach().cpu().tolist()
    obj_pos = (obj.data.root_pos_w - env.scene.env_origins).detach().cpu().tolist()
    action_term = env.action_manager.get_term("palm_ik")
    hand_ids = action_term.right_hand_joint_ids
    hand_actual = robot.data.joint_pos[:, hand_ids].detach().cpu()
    hand_target = action_term.joint_target[:, hand_ids].detach().cpu()
    hand_error = hand_actual - hand_target
    print(f"\n[STEP {step:04d}] reward={rew.detach().cpu().tolist()}")
    print(f"  action_raw={action_term.raw_actions.detach().cpu().tolist()}")
    print(f"  action_processed=[target_nudge_xyz, target_rpy, grip]={action_term.processed_actions.detach().cpu().tolist()}")
    print(f"  right_hand_target={hand_target.tolist()}")
    print(f"  right_hand_actual={hand_actual.tolist()}")
    print(f"  right_hand_error={hand_error.tolist()}")
    print(f"  palm={palm}")
    print(f"  object={obj_pos}")


def _scripted_action(env: ManagerBasedRLEnv, step: int) -> torch.Tensor:
    action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

    # The IK action starts at a persistent pregrasp nudge of +0.08 m in world X.
    # These small deltas move that nudge toward grasp, close/squeeze, then lift.
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


def main() -> None:
    cfg = WalkerS2IKPickPlaceEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.episode_length_s = 40.0
    cfg.events.reset_object_position = None

    env = ManagerBasedRLEnv(cfg)
    obs, _ = env.reset()

    print("[INFO] Walker S2 IK-action pick/place env created")
    print(f"[INFO] action dim: {env.action_manager.total_action_dim}")
    print(f"[INFO] action terms: {env.action_manager.active_terms}")
    print(f"[INFO] observation shape: {obs['policy'].shape}")

    for step in range(args_cli.steps):
        actions = _scripted_action(env, step)
        obs, rew, terminated, truncated, _ = env.step(actions)
        if step % args_cli.print_every == 0:
            _print_debug(env, obs, rew, step)
        if bool(torch.any(terminated | truncated)):
            env.reset()

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Walker S2 IK-action env test failed:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
