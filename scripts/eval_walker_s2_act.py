#!/usr/bin/env python3
"""Teacher-free closed-loop evaluation for Walker S2 state-based ACT."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Evaluate Walker S2 ACT.")
parser.add_argument("--task", default="Isaac-WalkerS2-PickPlace-ObjectRelative-v0")
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=REPO_ROOT / "logs" / "walker_s2_act" / "object_relative_teacher64" / "best.pt",
)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--print_every", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--randomize_object", action="store_true")
parser.add_argument("--query_interval", type=int, default=5)
parser.add_argument("--ensemble_decay", type=float, default=0.15)
parser.add_argument("--binary_grip", action="store_true")
parser.add_argument("--grip_threshold", type=float, default=0.5)
parser.add_argument("--result_json", type=Path, default=None)
parser.add_argument("--policy_device", default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch


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
from walker_s2_act_common import (  # noqa: E402
    ACTION_DIM,
    ENV_OBS_DIM,
    FEATURE_CONTRACT,
    load_walker_s2_act,
    normalize_observation,
)


def _policy_device() -> str:
    if args_cli.policy_device:
        return args_cli.policy_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _state(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    obj = unwrapped.scene["object"]
    palm_ids, _ = robot.find_bodies("hand3_v1_right_R_palm_link", preserve_order=True)
    palm = robot.data.body_pos_w[:, palm_ids[0]] - unwrapped.scene.env_origins
    object_pos = obj.data.root_pos_w - unwrapped.scene.env_origins
    object_speed = torch.linalg.vector_norm(obj.data.root_vel_w[:, :3], dim=1)
    distance = torch.linalg.vector_norm(palm - object_pos, dim=1)
    return palm, object_pos, object_speed, distance


def _in_target(object_pos: torch.Tensor) -> torch.Tensor:
    target = torch.as_tensor(TARGET_POS, device=object_pos.device)
    size = torch.as_tensor(TARGET_SIZE, device=object_pos.device)
    return torch.logical_and(
        torch.abs(object_pos[:, 0] - target[0]) <= size[0] * 0.5,
        torch.abs(object_pos[:, 1] - target[1]) <= size[1] * 0.5,
    )


def _termination_summary(env, terminated: torch.Tensor, truncated: torch.Tensor) -> dict[str, bool]:
    manager = env.unwrapped.termination_manager
    terms = {name: bool(torch.any(manager.get_term(name)).item()) for name in manager.active_terms}
    return {
        "success": bool(terms.get("success", False)),
        "object_dropping": bool(terms.get("object_dropping", False)),
        "time_out": bool(terms.get("time_out", False) or torch.any(truncated).item()),
        "terminated": bool(torch.any(terminated).item()),
        "truncated": bool(torch.any(truncated).item()),
    }


def _normalized_observation(
    observation: torch.Tensor,
    device: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    if observation.shape != (1, ENV_OBS_DIM):
        raise RuntimeError(f"Expected one observation with {ENV_OBS_DIM} features, got {tuple(observation.shape)}")
    return normalize_observation(observation.to(device), mean, std, clip)[0]


def _ensemble_action(active_chunks: list[tuple[int, torch.Tensor]], step: int, decay: float) -> torch.Tensor:
    candidates: list[torch.Tensor] = []
    weights: list[float] = []
    for start_step, chunk in active_chunks:
        offset = step - start_step
        if 0 <= offset < len(chunk):
            candidates.append(chunk[offset])
            weights.append(math.exp(-decay * offset))
    if not candidates:
        raise RuntimeError(f"No ACT prediction covers step {step}.")
    stacked = torch.stack(candidates, dim=0)
    weight_tensor = torch.tensor(weights, device=stacked.device, dtype=stacked.dtype)
    return torch.sum(stacked * weight_tensor.unsqueeze(-1), dim=0) / torch.sum(weight_tensor)


def main() -> None:
    if args_cli.query_interval < 1:
        raise ValueError("--query_interval must be positive.")
    policy_device = _policy_device()
    model, obs_mean, obs_std, checkpoint = load_walker_s2_act(args_cli.checkpoint, policy_device)
    obs_clip = float(checkpoint.get("obs_clip", 5.0))
    history_len = model.history_len
    chunk_len = model.chunk_len
    if args_cli.query_interval > chunk_len:
        raise ValueError(f"query_interval={args_cli.query_interval} cannot exceed chunk_len={chunk_len}.")

    spec = gym.spec(args_cli.task)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    if not args_cli.randomize_object:
        env_cfg.events.reset_object_position = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    action_manager = env.unwrapped.action_manager
    if obs["policy"].shape[-1] != ENV_OBS_DIM or action_manager.total_action_dim != ACTION_DIM:
        raise RuntimeError(
            f"Environment contract mismatch: obs={obs['policy'].shape[-1]}, action={action_manager.total_action_dim}; "
            f"expected obs={ENV_OBS_DIM}, action={ACTION_DIM}."
        )
    if action_manager.active_terms != ["arm_ik", "palm_ik"]:
        raise RuntimeError(f"Expected Cartesian IK plus grip terms, got {action_manager.active_terms}")

    print(f"[INFO] checkpoint={args_cli.checkpoint.expanduser().resolve()}")
    print(
        f"[INFO] contract={FEATURE_CONTRACT} checkpoint_epoch={checkpoint.get('epoch')} "
        f"task={args_cli.task} history={history_len} chunk={chunk_len}"
    )
    print(
        f"[INFO] randomize_object={args_cli.randomize_object} query_interval={args_cli.query_interval} "
        f"ensemble_decay={args_cli.ensemble_decay} policy_device={policy_device} sim_device={env.unwrapped.device}"
    )
    print("[INFO] autonomous=True teacher=False stage_input=False phase_clock=False scripted_grip=False")
    results: list[dict] = []

    with torch.inference_mode():
        for episode in range(args_cli.episodes):
            obs, _ = env.reset()
            initial = _normalized_observation(obs["policy"], policy_device, obs_mean, obs_std, obs_clip)
            history: deque[torch.Tensor] = deque([initial.clone() for _ in range(history_len)], maxlen=history_len)
            active_chunks: list[tuple[int, torch.Tensor]] = []
            previous_action = None
            max_action_jump = 0.0
            min_distance = float("inf")
            max_lift = 0.0
            episode_return = 0.0
            counters = {"grasp": 0, "lift": 0, "carry": 0, "release": 0}
            milestones: dict[str, int | None] = {name: None for name in counters}
            summary = {"success": False, "object_dropping": False, "time_out": True}

            for step in range(args_cli.max_steps):
                active_chunks = [(start, chunk) for start, chunk in active_chunks if step - start < chunk_len]
                if step % args_cli.query_interval == 0:
                    history_tensor = torch.stack(list(history), dim=0).unsqueeze(0)
                    predicted_chunk = model(history_tensor)[0]
                    active_chunks.append((step, predicted_chunk))
                action_policy = _ensemble_action(active_chunks, step, args_cli.ensemble_decay)
                action_policy[:6] = torch.clamp(action_policy[:6], -1.0, 1.0)
                action_policy[6] = torch.clamp(action_policy[6], 0.0, 1.0)
                if args_cli.binary_grip:
                    action_policy[6] = float(action_policy[6].item() >= args_cli.grip_threshold)
                action = action_policy.unsqueeze(0).to(env.unwrapped.device)

                action_jump = 0.0 if previous_action is None else float(
                    torch.max(torch.abs(action - previous_action)).item()
                )
                max_action_jump = max(max_action_jump, action_jump)
                previous_action = action.detach().clone()

                palm, object_pos, object_speed, distance = _state(env)
                lift = float(object_pos[0, 2].item() - CUBE_CENTER[2])
                grip = float(action[0, 6].item())
                in_target = bool(_in_target(object_pos)[0].item())
                min_distance = min(min_distance, float(distance[0].item()))
                max_lift = max(max_lift, lift)
                conditions = {
                    "grasp": grip >= 0.7 and distance[0].item() <= 0.075,
                    "lift": milestones["grasp"] is not None
                    and grip >= 0.7
                    and distance[0].item() <= 0.12
                    and lift >= 0.03,
                    "carry": milestones["lift"] is not None
                    and grip >= 0.7
                    and distance[0].item() <= 0.12
                    and lift >= 0.015
                    and in_target,
                    "release": milestones["carry"] is not None
                    and grip <= 0.2
                    and in_target
                    and object_speed[0].item() <= 0.25,
                }
                for name, condition in conditions.items():
                    if milestones[name] is None:
                        counters[name] = counters[name] + 1 if condition else 0
                        if counters[name] >= 5:
                            milestones[name] = step
                            print(f"[EPISODE {episode:03d}] milestone={name} step={step:04d}")

                obs, reward, terminated, truncated, _ = env.step(action)
                normalized = _normalized_observation(obs["policy"], policy_device, obs_mean, obs_std, obs_clip)
                history.append(normalized)
                episode_return += float(reward[0].item())
                if step % args_cli.print_every == 0:
                    print(
                        f"[ACT step={step:04d}] reward={float(reward[0]):.4f} grip={grip:.3f} "
                        f"distance={float(distance[0]):.4f} lift={lift:.4f} in_target={in_target} "
                        f"chunks={len(active_chunks)} action_jump={action_jump:.4f}"
                    )
                    print(f"  action={action[0].detach().cpu().tolist()}")
                if bool(torch.any(terminated | truncated).item()):
                    summary = _termination_summary(env, terminated, truncated)
                    steps = step + 1
                    break
            else:
                steps = args_cli.max_steps

            ordered_complete = all(value is not None for value in milestones.values())
            result = {
                "episode": episode,
                **summary,
                "steps": steps,
                "return": episode_return,
                "min_palm_object_distance": min_distance,
                "max_object_lift": max_lift,
                "max_action_jump": max_action_jump,
                "milestones": milestones,
                "ordered_complete": ordered_complete,
            }
            results.append(result)
            print(
                f"[EPISODE {episode:03d}] success={summary['success']} drop={summary['object_dropping']} "
                f"timeout={summary['time_out']} steps={steps} return={episode_return:.4f} "
                f"min_distance={min_distance:.4f} max_lift={max_lift:.4f} max_action_jump={max_action_jump:.4f}"
            )
            print(f"  milestones={milestones} ordered_complete={ordered_complete}")

    success_count = sum(int(result["success"]) for result in results)
    payload = {
        "checkpoint": str(args_cli.checkpoint.expanduser().resolve()),
        "feature_contract": FEATURE_CONTRACT,
        "episodes": len(results),
        "successes": success_count,
        "success_rate": success_count / max(len(results), 1),
        "results": results,
    }
    print(f"[RESULT] episodes={len(results)} success={success_count} success_rate={payload['success_rate']:.3f}")
    if args_cli.result_json is not None:
        result_path = args_cli.result_json.expanduser()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[INFO] wrote result JSON: {result_path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
