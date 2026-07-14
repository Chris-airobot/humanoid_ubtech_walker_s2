#!/usr/bin/env python3
"""Evaluate a behavior cloning checkpoint in the Walker S2 IK pick/place task.

Run from this repository root:

    /home/chris/IsaacLab/isaaclab.sh -p scripts/eval_walker_s2_bc.py \
        --checkpoint logs/walker_s2_bc/<run>/best.pt
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Evaluate Walker S2 BC policy in IsaacLab.")
parser.add_argument("--task", default="Isaac-WalkerS2-PickPlace-IK-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=1200)
parser.add_argument("--print_every", type=int, default=30)
parser.add_argument(
    "--phase_steps",
    type=int,
    default=0,
    help="Override rollout phase horizon for checkpoints trained with append_phase. 0 uses checkpoint metadata.",
)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=0,
    help="After reset, hold zero action for this many sim steps before running the BC policy.",
)
parser.add_argument("--randomize_object", action="store_true")
parser.add_argument("--action_scale", type=float, default=1.0, help="Optional multiplier applied after policy output.")
parser.add_argument(
    "--binary_grip",
    action="store_true",
    help="Threshold action dim 6 to 0/1 during eval. Useful because teleop grip demos are binary.",
)
parser.add_argument("--grip_threshold", type=float, default=0.7)
parser.add_argument(
    "--binary_grip_close_distance",
    type=float,
    default=0.0,
    help=(
        "If > 0, binary grip can close only when right palm is within this distance of the object. "
        "This avoids closing at the ready pose when the BC policy is uncertain."
    ),
)
parser.add_argument(
    "--force_grip_close_distance",
    type=float,
    default=0.0,
    help=(
        "If > 0, force grip action to 1 when the palm is this close to the object. "
        "Use only as an eval/debug aid to separate grasp-timing errors from pose errors."
    ),
)
parser.add_argument(
    "--force_grip_after_step",
    type=int,
    default=-1,
    help="If >= 0, force grip action to 1 after this many policy steps. Eval/debug aid only.",
)
parser.add_argument(
    "--scripted_grip_close_step",
    type=int,
    default=-1,
    help="If >= 0, override grip to close at this phase step.",
)
parser.add_argument(
    "--scripted_grip_release_step",
    type=int,
    default=-1,
    help="If >= 0, override grip to open again at this phase step.",
)
parser.add_argument(
    "--phase_pause_step",
    type=int,
    default=-1,
    help="If >= 0, hold the phase-conditioned policy at this phase step for phase_pause_duration sim steps.",
)
parser.add_argument(
    "--phase_pause_duration",
    type=int,
    default=0,
    help="Number of sim steps to pause the phase-conditioned policy at phase_pause_step.",
)
parser.add_argument(
    "--latch_grip",
    action="store_true",
    help="Once eval postprocessing closes the grip, keep it closed for the rest of the episode.",
)
parser.add_argument(
    "--action_deadband",
    type=float,
    default=0.0,
    help="Set small absolute policy outputs to zero after scaling. Disabled by default.",
)
parser.add_argument(
    "--policy_device",
    default="",
    help="Torch device for the BC network. Default uses CUDA if available, otherwise CPU.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torch import nn


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import isaaclab_walker_s2.tasks.pick_place  # noqa: F401, E402


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int], output_activation: str) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ELU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, action_dim))
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation != "none":
            raise ValueError(f"Unsupported output activation: {output_activation}")
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _resolve_policy_device() -> str:
    if args_cli.policy_device:
        return args_cli.policy_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_policy(path: Path, device: str) -> tuple[BCPolicy, torch.Tensor, torch.Tensor, dict]:
    checkpoint = torch.load(path.expanduser(), map_location="cpu")
    output_activation = checkpoint.get("output_activation", "tanh")
    model = BCPolicy(
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dims=list(checkpoint["hidden_dims"]),
        output_activation=output_activation,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    obs_mean = checkpoint["obs_mean"].to(device)
    obs_std = checkpoint["obs_std"].to(device)
    return model, obs_mean, obs_std, checkpoint


def _decode_policy_output(policy_out: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    transform = checkpoint.get("target_transform", {"mode": "none"})
    mode = transform.get("mode", "none")
    if mode == "none":
        return policy_out
    if mode != "demo_range":
        raise ValueError(f"Unsupported checkpoint target transform mode: {mode}")
    center = transform["center"].to(policy_out.device)
    half_range = transform["half_range"].to(policy_out.device)
    return policy_out * half_range + center


def _policy_obs_with_phase(obs_policy: torch.Tensor, policy_step: int, checkpoint: dict) -> torch.Tensor:
    if not checkpoint.get("append_phase", False):
        return obs_policy
    phase_steps = args_cli.phase_steps if args_cli.phase_steps > 0 else int(checkpoint.get("phase_steps", 1))
    phase_steps = max(phase_steps - 1, 1)
    phase = min(float(policy_step) / float(phase_steps), 1.0)
    phase_tensor = torch.full((obs_policy.shape[0], 1), phase, device=obs_policy.device, dtype=obs_policy.dtype)
    return torch.cat([obs_policy, phase_tensor], dim=-1)


def _effective_policy_step(step: int) -> int:
    if args_cli.phase_pause_step < 0 or args_cli.phase_pause_duration <= 0:
        return step
    pause_start = args_cli.phase_pause_step
    pause_end = pause_start + args_cli.phase_pause_duration
    if step < pause_start:
        return step
    if step < pause_end:
        return pause_start
    return step - args_cli.phase_pause_duration


def _env_pos(unwrapped, world_tensor: torch.Tensor) -> torch.Tensor:
    return world_tensor - unwrapped.scene.env_origins


def _done_summary(unwrapped, terminated: torch.Tensor, truncated: torch.Tensor) -> dict[str, bool]:
    terms: dict[str, bool] = {}
    log = getattr(unwrapped, "extras", {}).get("log", {})
    prefix = "Episode_Termination/"
    for key, value in log.items():
        if key.startswith(prefix):
            terms[key.removeprefix(prefix)] = bool(value)
    if not terms:
        manager = getattr(unwrapped, "termination_manager", None)
        if manager is not None:
            for name in manager.active_terms:
                terms[name] = bool(torch.any(manager.get_term(name)).item())
    terms["terminated"] = bool(torch.any(terminated).item())
    terms["truncated"] = bool(torch.any(truncated).item())
    return terms


def _print_debug(
    env,
    rew: torch.Tensor,
    step: int,
    action: torch.Tensor,
    policy_out: torch.Tensor,
    palm_object_dist: torch.Tensor,
) -> None:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    obj = unwrapped.scene["object"]
    palm_ids, _ = robot.find_bodies("hand3_v1_right_R_palm_link", preserve_order=True)
    palm = _env_pos(unwrapped, robot.data.body_pos_w[:, palm_ids[0]]).detach().cpu().tolist()
    obj_pos = _env_pos(unwrapped, obj.data.root_pos_w).detach().cpu().tolist()
    action_term = unwrapped.action_manager.get_term("palm_ik")
    print(f"[BC STEP {step:04d}] reward={rew.detach().cpu().tolist()}")
    print(f"  policy_out={policy_out.detach().cpu().tolist()}")
    print(f"  env_action={action.detach().cpu().tolist()}")
    print(
        "  action_processed=[target_nudge_xyz, target_rpy, grip, arm_offsets]="
        f"{action_term.processed_actions.detach().cpu().tolist()}"
    )
    print(f"  palm_object_dist={palm_object_dist.detach().cpu().tolist()}")
    print(f"  palm={palm}")
    print(f"  object={obj_pos}")


def _palm_object_distance(unwrapped, palm_body_id: int) -> torch.Tensor:
    robot = unwrapped.scene["robot"]
    obj = unwrapped.scene["object"]
    palm_pos = _env_pos(unwrapped, robot.data.body_pos_w[:, palm_body_id])
    obj_pos = _env_pos(unwrapped, obj.data.root_pos_w)
    return torch.linalg.norm(palm_pos - obj_pos, dim=-1)


def _apply_grip_postprocess(
    action: torch.Tensor,
    unwrapped,
    palm_body_id: int,
    policy_step: int | None = None,
) -> torch.Tensor:
    if action.shape[-1] <= 6:
        if args_cli.binary_grip or args_cli.force_grip_close_distance > 0.0:
            raise ValueError(f"Grip postprocess requires action dim > 6, got {action.shape[-1]}")
        return action
    action = action.clone()
    palm_object_dist = _palm_object_distance(unwrapped, palm_body_id).to(action.device)
    if args_cli.binary_grip:
        should_close = action[..., 6] >= args_cli.grip_threshold
        if args_cli.binary_grip_close_distance > 0.0:
            should_close = should_close & (palm_object_dist <= args_cli.binary_grip_close_distance)
        action[..., 6] = should_close.to(action.dtype)
    if args_cli.force_grip_close_distance > 0.0:
        action[..., 6] = torch.where(
            palm_object_dist <= args_cli.force_grip_close_distance,
            torch.ones_like(action[..., 6]),
            action[..., 6],
        )
    if args_cli.force_grip_after_step >= 0 and policy_step is not None and policy_step >= args_cli.force_grip_after_step:
        action[..., 6] = 1.0
    if args_cli.scripted_grip_close_step >= 0 and policy_step is not None:
        close_active = policy_step >= args_cli.scripted_grip_close_step
        if args_cli.scripted_grip_release_step >= 0:
            close_active = close_active and policy_step < args_cli.scripted_grip_release_step
        action[..., 6] = 1.0 if close_active else 0.0
    return action


def _apply_grip_latch(action: torch.Tensor, grip_latched: torch.Tensor | None) -> torch.Tensor:
    if grip_latched is None:
        return action
    if action.shape[-1] <= 6:
        return action
    action = action.clone()
    action[..., 6] = torch.where(grip_latched.to(action.device), torch.ones_like(action[..., 6]), action[..., 6])
    return action


def _postprocess_raw_action(
    action: torch.Tensor,
    unwrapped,
    palm_body_id: int,
    policy_step: int | None = None,
) -> torch.Tensor:
    action = torch.clamp(action * args_cli.action_scale, -1.0, 1.0)
    if args_cli.action_deadband > 0.0:
        action = torch.where(torch.abs(action) < args_cli.action_deadband, torch.zeros_like(action), action)
    return _apply_grip_postprocess(action, unwrapped, palm_body_id, policy_step)


def _processed_target_to_raw_action(
    desired: torch.Tensor, unwrapped, palm_body_id: int, action_term, policy_step: int | None = None
) -> torch.Tensor:
    desired = desired.to(unwrapped.device)
    desired = _apply_grip_postprocess(desired, unwrapped, palm_body_id, policy_step)
    current = action_term.processed_actions
    raw = torch.zeros_like(current)

    desired_nudge = torch.clamp(desired[:, :3], min=action_term._nudge_min, max=action_term._nudge_max)
    raw[:, :3] = (desired_nudge - current[:, :3]) / action_term._delta_scale

    desired_rpy = torch.clamp(desired[:, 3:6], min=action_term._rpy_min, max=action_term._rpy_max)
    raw[:, 3:6] = (desired_rpy - current[:, 3:6]) / action_term._rpy_delta_scale

    raw[:, 6] = torch.clamp(desired[:, 6], 0.0, 1.0)

    if raw.shape[-1] > 7:
        desired_offsets = torch.clamp(
            desired[:, 7:],
            min=action_term._arm_offset_min,
            max=action_term._arm_offset_max,
        )
        raw[:, 7:] = (desired_offsets - current[:, 7:]) / action_term._arm_offset_delta_scale

    raw = torch.clamp(raw * args_cli.action_scale, -1.0, 1.0)
    if args_cli.action_deadband > 0.0:
        raw = torch.where(torch.abs(raw) < args_cli.action_deadband, torch.zeros_like(raw), raw)
    return raw


def main() -> None:
    policy_device = _resolve_policy_device()
    model, obs_mean, obs_std, checkpoint = _load_policy(args_cli.checkpoint, policy_device)

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
    palm_ids, _ = unwrapped.scene["robot"].find_bodies("hand3_v1_right_R_palm_link", preserve_order=True)
    palm_body_id = palm_ids[0]
    env_action_dim = unwrapped.action_manager.total_action_dim
    if int(checkpoint["action_dim"]) != env_action_dim:
        raise ValueError(f"Checkpoint action dim {checkpoint['action_dim']} != env action dim {env_action_dim}")
    env_policy_obs_dim = obs["policy"].shape[-1]
    expected_obs_dim = env_policy_obs_dim + (1 if checkpoint.get("append_phase", False) else 0)
    if int(checkpoint["obs_dim"]) != expected_obs_dim:
        raise ValueError(
            f"Checkpoint obs dim {checkpoint['obs_dim']} != expected eval obs dim {expected_obs_dim} "
            f"(env obs dim {env_policy_obs_dim}, append_phase={checkpoint.get('append_phase', False)})"
        )
    target_key = checkpoint.get("target_key", "action")
    if target_key not in {"action", "processed_action"}:
        raise ValueError(f"Unsupported checkpoint target_key: {target_key}")
    action_term = unwrapped.action_manager.get_term("palm_ik")

    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")
    print(f"[INFO] checkpoint epoch={checkpoint.get('epoch')} val_mse={checkpoint.get('val_loss')}")
    print(f"[INFO] task={args_cli.task} randomize_object={args_cli.randomize_object}")
    print(f"[INFO] sim_device={args_cli.device} policy_device={policy_device}")
    print(
        f"[INFO] obs_dim={checkpoint['obs_dim']} action_dim={checkpoint['action_dim']} "
        f"target_key={target_key} output_activation={checkpoint.get('output_activation', 'tanh')} "
        f"append_phase={checkpoint.get('append_phase', False)} phase_steps={checkpoint.get('phase_steps', 0)}"
    )
    print(
        f"[INFO] eval postprocess: action_scale={args_cli.action_scale} "
        f"binary_grip={args_cli.binary_grip} grip_threshold={args_cli.grip_threshold} "
        f"binary_grip_close_distance={args_cli.binary_grip_close_distance} "
        f"force_grip_close_distance={args_cli.force_grip_close_distance} "
        f"force_grip_after_step={args_cli.force_grip_after_step} "
        f"scripted_grip_close_step={args_cli.scripted_grip_close_step} "
        f"scripted_grip_release_step={args_cli.scripted_grip_release_step} "
        f"phase_pause_step={args_cli.phase_pause_step} "
        f"phase_pause_duration={args_cli.phase_pause_duration} "
        f"latch_grip={args_cli.latch_grip} "
        f"action_deadband={args_cli.action_deadband} settle_steps={args_cli.settle_steps}"
    )

    successes = 0
    failures = 0
    timeouts = 0
    with torch.no_grad():
        for episode in range(args_cli.episodes):
            obs, _ = env.reset()
            grip_latched = torch.zeros(1, dtype=torch.bool, device=unwrapped.device)
            min_palm_object_dist = float("inf")
            first_close_step = None
            first_near_step = None
            final_terms = {}
            final_reward = None
            if args_cli.settle_steps > 0:
                zero_action = torch.zeros((unwrapped.num_envs, env_action_dim), device=unwrapped.device)
                for settle_step in range(args_cli.settle_steps):
                    obs, rew, terminated, truncated, _ = env.step(zero_action)
                    final_reward = rew
                    if settle_step % args_cli.print_every == 0:
                        palm_object_dist = _palm_object_distance(unwrapped, palm_body_id)
                        print(f"[SETTLE {settle_step:04d}] reward={rew.detach().cpu().tolist()}")
                        print(f"  palm_object_dist={palm_object_dist.detach().cpu().tolist()}")
                    if bool(torch.any(terminated | truncated)):
                        final_terms = _done_summary(unwrapped, terminated, truncated)
                        print(
                            f"[EPISODE {episode}] ended during settle at step {settle_step}: "
                            f"{final_terms}"
                        )
                        break
            for step in range(args_cli.max_steps if not final_terms else 0):
                policy_step = _effective_policy_step(step)
                policy_obs = _policy_obs_with_phase(obs["policy"].to(policy_device), policy_step, checkpoint)
                norm_obs = (policy_obs - obs_mean) / obs_std
                policy_out = model(norm_obs)
                decoded_policy_out = _decode_policy_output(policy_out, checkpoint)
                if target_key == "processed_action":
                    action = _processed_target_to_raw_action(
                        decoded_policy_out, unwrapped, palm_body_id, action_term, policy_step
                    )
                else:
                    action = _postprocess_raw_action(decoded_policy_out, unwrapped, palm_body_id, policy_step).to(
                        unwrapped.device
                    )
                if args_cli.latch_grip and args_cli.scripted_grip_release_step < 0:
                    grip_latched |= action[..., 6] >= 0.99
                    action = _apply_grip_latch(action, grip_latched)
                obs, rew, terminated, truncated, _ = env.step(action)
                final_reward = rew
                palm_object_dist = _palm_object_distance(unwrapped, palm_body_id)
                dist_value = float(torch.min(palm_object_dist).detach().cpu())
                min_palm_object_dist = min(min_palm_object_dist, dist_value)
                if first_near_step is None and dist_value <= 0.10:
                    first_near_step = step
                if first_close_step is None and bool(torch.any(action[..., 6] >= 0.99)):
                    first_close_step = step
                if step % args_cli.print_every == 0:
                    _print_debug(env, rew, step, action, decoded_policy_out, palm_object_dist)
                    if policy_step != step:
                        print(f"  effective_policy_step={policy_step}")
                if bool(torch.any(terminated | truncated)):
                    final_terms = _done_summary(unwrapped, terminated, truncated)
                    print(f"[EPISODE {episode}] done at step {step}: {final_terms}")
                    break
            else:
                final_terms = {"time_out": True, "terminated": False, "truncated": True}
                print(f"[EPISODE {episode}] max_steps reached")

            if final_terms.get("success", False):
                successes += 1
            elif final_terms.get("time_out", False) or final_terms.get("truncated", False):
                timeouts += 1
            else:
                failures += 1
            if final_reward is not None:
                print(f"[EPISODE {episode}] final_reward={final_reward.detach().cpu().tolist()}")
            print(
                f"[EPISODE {episode}] grasp_debug: min_palm_object_dist={min_palm_object_dist:.4f} "
                f"first_near_10cm_step={first_near_step} first_close_step={first_close_step}"
            )

    print(f"[RESULT] episodes={args_cli.episodes} success={successes} failure={failures} timeout={timeouts}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Walker S2 BC evaluation failed:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
