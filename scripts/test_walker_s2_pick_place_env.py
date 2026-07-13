#!/usr/bin/env python3
"""Smoke-test the local Walker S2 pick/place IsaacLab environment.

Run from this repository root:

    /home/chris/IsaacLab/isaaclab.sh -p scripts/test_walker_s2_pick_place_env.py
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Test Walker S2 pick/place env actions and reward terms.")
parser.add_argument("--steps", type=int, default=360, help="Number of env steps to run.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of env instances.")
parser.add_argument("--print_every", type=int, default=30, help="Print debug state every N env steps.")
parser.add_argument(
    "--mode",
    choices=("hold", "wave", "reach"),
    default="hold",
    help="Scripted action pattern. Use 'hold' for neutral action validation, 'wave' for obvious arm motion, 'reach' for rough cube reaching.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab_walker_s2.tasks.pick_place.walker_s2_pick_place_env_cfg import (  # noqa: E402
    RIGHT_ARM_JOINTS,
    RIGHT_HAND_CLOSE_COMMAND,
    RIGHT_HAND_JOINTS,
    RIGHT_HAND_OPEN_COMMAND,
    WalkerS2PickPlaceEnvCfg,
)


def _reward_terms(env: ManagerBasedRLEnv) -> dict[str, list[float]]:
    step_reward = env.reward_manager._step_reward.detach().cpu()
    return {
        name: step_reward[:, term_idx].tolist()
        for term_idx, name in enumerate(env.reward_manager.active_terms)
    }


def _action_debug(env: ManagerBasedRLEnv) -> dict[str, list[list[float]]]:
    debug = {}
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        debug[f"{name}_raw"] = term.raw_actions.detach().cpu().tolist()
        debug[f"{name}_processed"] = term.processed_actions.detach().cpu().tolist()
    return debug


def _action_slices(env: ManagerBasedRLEnv) -> dict[str, slice]:
    slices = {}
    start = 0
    for name, dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim, strict=True):
        slices[name] = slice(start, start + dim)
        start += dim
    return slices


def _right_arm_default(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(RIGHT_ARM_JOINTS, preserve_order=True)
    return robot.data.default_joint_pos[:, joint_ids]


def _right_hand_pose(env: ManagerBasedRLEnv, command: dict[str, float]) -> torch.Tensor:
    values = [[command[joint_name] for joint_name in RIGHT_HAND_JOINTS]]
    return torch.tensor(values, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)


def _print_debug(env: ManagerBasedRLEnv, obs: dict, rew: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, step: int):
    policy_obs = obs["policy"]
    right_arm_q = policy_obs[:, 0:7].detach().cpu().tolist()
    right_hand_q = policy_obs[:, 14:25].detach().cpu().tolist()
    right_palm = policy_obs[:, 25:28].detach().cpu().tolist()
    object_pos = policy_obs[:, 28:31].detach().cpu().tolist()

    print(f"\n[STEP {step:04d}] total_reward={rew.detach().cpu().tolist()}")
    print(f"  terminated={terminated.detach().cpu().tolist()} truncated={truncated.detach().cpu().tolist()}")
    print(f"  action_debug={_action_debug(env)}")
    print(f"  reward_terms={_reward_terms(env)}")
    print(f"  right_arm_q_rel={right_arm_q}")
    print(f"  right_hand_q={right_hand_q}")
    print(f"  right_palm={right_palm}")
    print(f"  object_pos={object_pos}")


def _scripted_action(env: ManagerBasedRLEnv, step: int) -> torch.Tensor:
    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    slices = _action_slices(env)

    # The env action is absolute joint-position targets, matching the old full-q control style.
    actions[:, slices["right_arm"]] = _right_arm_default(env)
    actions[:, slices["right_hand"]] = _right_hand_pose(env, RIGHT_HAND_OPEN_COMMAND)

    if args_cli.mode == "hold":
        return actions

    if args_cli.mode == "wave":
        # These are absolute target offsets around the ready pose. This mode is
        # intentionally exaggerated for visual validation, not task performance.
        phase = (step // 90) % 4
        if phase == 0:
            actions[:, slices["right_arm"].start + 0] += 1.0
            actions[:, slices["right_arm"].start + 1] += -0.75
            actions[:, slices["right_arm"].start + 3] += -0.75
        elif phase == 1:
            actions[:, slices["right_arm"].start + 0] += -1.0
            actions[:, slices["right_arm"].start + 1] += 0.75
            actions[:, slices["right_arm"].start + 3] += 0.75
        elif phase == 2:
            actions[:, slices["right_arm"].start + 2] += 0.75
            actions[:, slices["right_arm"].start + 4] += -0.75
            actions[:, slices["right_arm"].start + 6] += 0.75
        else:
            actions[:, slices["right_arm"].start + 2] += -0.75
            actions[:, slices["right_arm"].start + 4] += 0.75
            actions[:, slices["right_arm"].start + 6] += -0.75
        if 180 <= step < 270:
            actions[:, slices["right_hand"]] = _right_hand_pose(env, RIGHT_HAND_CLOSE_COMMAND)
        return actions

    if 60 <= step < 140:
        actions[:, slices["right_arm"].start + 0] += 0.15
        actions[:, slices["right_arm"].start + 1] += -0.10
        actions[:, slices["right_arm"].start + 3] += -0.125

    if 140 <= step < 220:
        actions[:, slices["right_hand"]] = _right_hand_pose(env, RIGHT_HAND_CLOSE_COMMAND)

    if 220 <= step < 300:
        actions[:, slices["right_arm"].start + 0] += -0.10
        actions[:, slices["right_arm"].start + 3] += 0.10
        actions[:, slices["right_hand"]] = _right_hand_pose(env, RIGHT_HAND_CLOSE_COMMAND)

    return actions


def main() -> None:
    cfg = WalkerS2PickPlaceEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg)
    obs, _ = env.reset()

    print("[INFO] Walker S2 pick/place env created")
    print(f"[INFO] action dim: {env.action_manager.total_action_dim}")
    print(f"[INFO] action terms: {env.action_manager.active_terms}")
    print(f"[INFO] reward terms: {env.reward_manager.active_terms}")

    for step in range(args_cli.steps):
        actions = _scripted_action(env, step)
        obs, rew, terminated, truncated, _ = env.step(actions)

        if step % args_cli.print_every == 0:
            _print_debug(env, obs, rew, terminated, truncated, step)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Walker S2 pick/place env test failed:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
