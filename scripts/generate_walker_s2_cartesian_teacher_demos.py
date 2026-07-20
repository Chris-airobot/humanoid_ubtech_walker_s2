#!/usr/bin/env python3
"""Generate state-gated demonstrations in the Cartesian policy action space."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Generate Walker S2 Cartesian teacher demonstrations.")
parser.add_argument("--task", default="Isaac-WalkerS2-PickPlace-ObjectRelative-v0")
parser.add_argument("--num_demos", type=int, default=1)
parser.add_argument("--max_attempts", type=int, default=0, help="Zero uses three attempts per requested demo.")
parser.add_argument("--max_steps", type=int, default=600)
parser.add_argument("--print_every", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--randomize_object", action="store_true")
parser.add_argument("--keep_failures", action="store_true")
parser.add_argument("--output_dir", default="demos/walker_s2_cartesian_teacher_gate")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import isaaclab_walker_s2.tasks.pick_place  # noqa: F401, E402
from isaaclab_walker_s2.tasks.pick_place.walker_s2_pick_place_env_cfg import (  # noqa: E402
    CUBE_CENTER,
    TARGET_POS,
    TARGET_SIZE,
)
from walker_s2_cartesian_teacher import CartesianPickPlaceTeacher  # noqa: E402


def _termination_summary(env, terminated: torch.Tensor, truncated: torch.Tensor) -> dict[str, bool]:
    manager = env.unwrapped.termination_manager
    terms = {name: bool(torch.any(manager.get_term(name)).item()) for name in manager.active_terms}
    return {
        "terminated": bool(torch.any(terminated).item()),
        "truncated": bool(torch.any(truncated).item()),
        "success": bool(terms.get("success", False)),
        "object_dropping": bool(terms.get("object_dropping", False)),
        "time_out": bool(terms.get("time_out", False) or torch.any(truncated).item()),
        **{f"termination_{name}": value for name, value in terms.items()},
    }


def _state(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    obj = unwrapped.scene["object"]
    palm_ids, _ = robot.find_bodies("hand3_v1_right_R_palm_link", preserve_order=True)
    palm_pos = robot.data.body_pos_w[:, palm_ids[0]] - unwrapped.scene.env_origins
    palm_quat = robot.data.body_quat_w[:, palm_ids[0]]
    object_pos = obj.data.root_pos_w - unwrapped.scene.env_origins
    object_quat = obj.data.root_quat_w
    return palm_pos, palm_quat, object_pos, object_quat


def _save(path: Path, frames: list[dict[str, np.ndarray]], summary: dict[str, bool]) -> None:
    arrays = {key: np.stack([frame[key] for frame in frames], axis=0) for key in frames[0]}
    arrays["created_unix_time"] = np.array([time.time()], dtype=np.float64)
    arrays["episode_success"] = np.array([summary["success"]], dtype=np.bool_)
    arrays["episode_object_dropping"] = np.array([summary["object_dropping"]], dtype=np.bool_)
    arrays["episode_time_out"] = np.array([summary["time_out"]], dtype=np.bool_)
    np.savez_compressed(path, **arrays)


def main() -> None:
    output_dir = (REPO_ROOT / args_cli.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = gym.spec(args_cli.task)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    if not args_cli.randomize_object:
        env_cfg.events.reset_object_position = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    arm_action_term = env.unwrapped.action_manager.get_term("arm_ik")
    if env.unwrapped.action_manager.total_action_dim != 7:
        raise RuntimeError(f"Expected Cartesian action dim 7, got {env.unwrapped.action_manager.total_action_dim}.")

    print(f"[INFO] task={args_cli.task} obs_dim={obs['policy'].shape[-1]} action_dim=7")
    print(f"[INFO] randomize_object={args_cli.randomize_object} output_dir={output_dir}")

    saved = 0
    attempted = 0
    max_attempts = args_cli.max_attempts if args_cli.max_attempts > 0 else max(3 * args_cli.num_demos, 1)
    while simulation_app.is_running() and saved < args_cli.num_demos and attempted < max_attempts:
        attempted += 1
        obs, _ = env.reset()
        _, _, object_pos, _ = _state(env)
        teacher = CartesianPickPlaceTeacher(TARGET_POS, TARGET_SIZE, CUBE_CENTER[2])
        teacher.reset(object_pos[0])
        frames: list[dict[str, np.ndarray]] = []
        summary = {"success": False, "object_dropping": False, "time_out": False}
        previous_stage = teacher.stage_name

        for step in range(args_cli.max_steps):
            action = teacher.command(env, arm_action_term)
            palm_pos, palm_quat, object_pos, object_quat = _state(env)
            stage_id = teacher.stage_id
            stage_name = teacher.stage_name
            if stage_name != previous_stage:
                print(f"[TEACHER] step={step:04d} transition={previous_stage}->{stage_name}")
                previous_stage = stage_name

            frame = {
                "obs_policy": obs["policy"].detach().cpu().numpy().copy(),
                "action": action.detach().cpu().numpy().copy(),
                "teacher_stage": np.array([stage_id], dtype=np.int64),
                "palm_pos": palm_pos.detach().cpu().numpy().copy(),
                "palm_quat": palm_quat.detach().cpu().numpy().copy(),
                "object_pos": object_pos.detach().cpu().numpy().copy(),
                "object_quat": object_quat.detach().cpu().numpy().copy(),
            }
            obs_next, reward, terminated, truncated, _ = env.step(action)
            frame["reward"] = reward.detach().cpu().numpy().copy()
            frame["terminated"] = terminated.detach().cpu().numpy().copy()
            frame["truncated"] = truncated.detach().cpu().numpy().copy()
            frames.append(frame)
            obs = obs_next

            if step % args_cli.print_every == 0:
                distance = float(torch.linalg.vector_norm(palm_pos[0] - object_pos[0]).item())
                print(
                    f"[DEMO try={attempted:03d} step={step:04d}] stage={stage_name} "
                    f"reward={float(reward[0].detach().cpu()):.4f} grip={float(action[0, 6]):.3f} "
                    f"palm_object_dist={distance:.4f} target_error={teacher.last_metrics.get('target_error', 0.0):.4f} "
                    f"object_lift={teacher.last_metrics.get('object_lift', 0.0):.4f} "
                    f"in_target={teacher.last_metrics.get('in_target', False)} "
                    f"palm={palm_pos[0].detach().cpu().tolist()} object={object_pos[0].detach().cpu().tolist()}"
                )

            if bool(torch.any(terminated | truncated).item()):
                summary = _termination_summary(env, terminated, truncated)
                break
        else:
            summary["time_out"] = True

        should_save = bool(summary["success"] or args_cli.keep_failures)
        if should_save and frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status = "success" if summary["success"] else "failure"
            path = output_dir / f"walker_s2_cartesian_teacher_{status}_ep{saved:04d}_{timestamp}.npz"
            _save(path, frames, summary)
            saved += 1
            print(
                f"[INFO] Saved {path} frames={len(frames)} success={summary['success']} "
                f"timeout={summary['time_out']} drop={summary['object_dropping']}"
            )
        else:
            print(f"[WARN] Skipped attempt {attempted}: {summary}")

    print(f"[RESULT] saved={saved} attempted={attempted} output_dir={output_dir}")
    env.close()
    if saved < args_cli.num_demos:
        raise RuntimeError(f"Teacher gate failed: saved {saved}/{args_cli.num_demos} successful demonstrations.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
