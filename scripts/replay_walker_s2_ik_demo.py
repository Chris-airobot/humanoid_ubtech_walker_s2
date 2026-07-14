#!/usr/bin/env python3
"""Replay a recorded Walker S2 IK teleop demo in the registered IsaacLab task.

Run from this repository root:

    /home/chris/IsaacLab/isaaclab.sh -p scripts/replay_walker_s2_ik_demo.py \
        --demo demos/walker_s2_pick_place_success/walker_s2_pick_place_ep000_20260714_161434.npz
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO = REPO_ROOT / "demos" / "walker_s2_pick_place_success" / "walker_s2_pick_place_ep000_20260714_161434.npz"

parser = argparse.ArgumentParser(description="Replay a saved Walker S2 IK teleop demo.")
parser.add_argument("--task", default="Isaac-WalkerS2-PickPlace-IK-v0")
parser.add_argument("--demo", type=Path, default=DEFAULT_DEMO)
parser.add_argument("--steps", type=int, default=0, help="Maximum replay steps. 0 means use the whole demo.")
parser.add_argument("--print_every", type=int, default=30)
parser.add_argument(
    "--compare_recorded",
    action="store_true",
    help="Print object-position error against the recorded demo when object_pos exists in the npz.",
)
parser.add_argument("--randomize_object", action="store_true", help="Enable reset-time object randomization.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import isaaclab_walker_s2.tasks.pick_place  # noqa: F401, E402


def _load_actions(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path)
    if "action" not in data.files:
        raise KeyError(f"Demo file {path} does not contain an 'action' array. Keys: {data.files}")

    actions = np.asarray(data["action"], dtype=np.float32)
    if actions.ndim == 3:
        if actions.shape[1] != 1:
            raise ValueError(f"Only one recorded env is supported, got action shape {actions.shape}")
        actions = actions[:, 0, :]
    elif actions.ndim != 2:
        raise ValueError(f"Expected action shape (T, A) or (T, 1, A), got {actions.shape}")

    arrays = {key: np.asarray(data[key]) for key in data.files}
    return actions, arrays


def _env_pos(unwrapped, world_tensor: torch.Tensor) -> torch.Tensor:
    return world_tensor - unwrapped.scene.env_origins


def _print_debug(env, rew: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, step: int, recorded=None):
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    obj = unwrapped.scene["object"]
    palm_ids, _ = robot.find_bodies("hand3_v1_right_R_palm_link", preserve_order=True)
    palm = _env_pos(unwrapped, robot.data.body_pos_w[:, palm_ids[0]]).detach().cpu()
    obj_pos = _env_pos(unwrapped, obj.data.root_pos_w).detach().cpu()
    action_term = unwrapped.action_manager.get_term("palm_ik")
    processed = action_term.processed_actions.detach().cpu()

    print(f"\n[REPLAY {step:04d}] reward={rew.detach().cpu().tolist()}")
    print(f"  terminated={terminated.detach().cpu().tolist()} truncated={truncated.detach().cpu().tolist()}")
    print(f"  action_raw={action_term.raw_actions.detach().cpu().tolist()}")
    print(f"  action_processed=[target_nudge_xyz, target_rpy, grip, arm_offsets]={processed.tolist()}")
    print(f"  palm={palm.tolist()}")
    print(f"  object={obj_pos.tolist()}")
    if recorded is not None:
        recorded_pos = torch.as_tensor(recorded, dtype=obj_pos.dtype)
        error = torch.linalg.norm(obj_pos.cpu() - recorded_pos, dim=1)
        print(f"  recorded_object={recorded_pos.tolist()}")
        print(f"  object_replay_error={error.tolist()}")


def main() -> None:
    actions_np, demo_arrays = _load_actions(args_cli.demo.expanduser())

    spec = gym.spec(args_cli.task)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = 40.0
    if not args_cli.randomize_object:
        env_cfg.events.reset_object_position = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    action_dim = unwrapped.action_manager.total_action_dim

    if actions_np.shape[1] != action_dim:
        raise ValueError(
            f"Demo action dim {actions_np.shape[1]} does not match env action dim {action_dim}. "
            "This usually means the env action config changed after recording."
        )

    max_steps = len(actions_np) if args_cli.steps <= 0 else min(args_cli.steps, len(actions_np))
    recorded_object = demo_arrays.get("object_pos") if args_cli.compare_recorded else None

    print(f"[INFO] Loaded registered task: {args_cli.task}")
    print(f"[INFO] demo: {args_cli.demo}")
    print(f"[INFO] demo action shape: {actions_np.shape}")
    print(f"[INFO] env observation shape: {obs['policy'].shape}")
    print(f"[INFO] env action dim: {action_dim}")
    print(f"[INFO] replay steps: {max_steps}")

    success_step = None
    last_rew = None
    last_terminated = None
    last_truncated = None
    for step in range(max_steps):
        action = torch.tensor(actions_np[step : step + 1], device=unwrapped.device, dtype=torch.float32)
        obs, rew, terminated, truncated, _ = env.step(action)
        last_rew = rew
        last_terminated = terminated
        last_truncated = truncated

        should_print = step % args_cli.print_every == 0 or bool(torch.any(terminated | truncated))
        if should_print:
            recorded = None
            if recorded_object is not None and step < len(recorded_object):
                recorded = recorded_object[step]
            _print_debug(env, rew, terminated, truncated, step, recorded)

        if bool(torch.any(terminated | truncated)):
            if bool(torch.any(terminated)):
                success_step = step
            break

    if success_step is not None:
        print(f"\n[RESULT] Replay terminated successfully at step {success_step}.")
    else:
        print("\n[RESULT] Replay did not hit success before actions ended.")
        if last_rew is not None:
            print(f"[RESULT] final_reward={last_rew.detach().cpu().tolist()}")
            print(
                "[RESULT] final_done="
                f"terminated={last_terminated.detach().cpu().tolist()} truncated={last_truncated.detach().cpu().tolist()}"
            )

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Walker S2 IK demo replay failed:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
