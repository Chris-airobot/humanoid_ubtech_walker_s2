#!/usr/bin/env python3
"""Train Walker S2 pick/place with PPO and a frozen BC action prior.

This script uses the local registered IsaacLab env directly.  The BC checkpoint
is not treated as the final policy; it is a frozen teacher that adds a decaying
action-prior penalty so PPO starts near the demonstrated arm trajectory while
still optimizing the real task reward.

Run from this repository root:

    /home/chris/IsaacLab/isaaclab.sh -p scripts/train_walker_s2_bc_prior_ppo.py \
        --bc_checkpoint logs/walker_s2_bc/single_demo_phase_processed/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="PPO training for Walker S2 with a frozen BC action prior.")
parser.add_argument("--task", default="Isaac-WalkerS2-PickPlace-IK-v0")
parser.add_argument("--bc_checkpoint", type=Path, default=REPO_ROOT / "logs/walker_s2_bc/single_demo_phase_processed/best.pt")
parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "logs/walker_s2_bc_prior_ppo")
parser.add_argument("--run_name", default="bc_prior_ppo")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--horizon", type=int, default=128)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_ratio", type=float, default=0.2)
parser.add_argument("--entropy_coef", type=float, default=0.005)
parser.add_argument("--value_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--bc_coef_start", type=float, default=0.25)
parser.add_argument("--bc_coef_end", type=float, default=0.02)
parser.add_argument("--bc_coef_decay_iters", type=int, default=500)
parser.add_argument("--action_std_init", type=float, default=0.35)
parser.add_argument("--settle_steps", type=int, default=180)
parser.add_argument("--save_every", type=int, default=25)
parser.add_argument("--print_every", type=int, default=1)
parser.add_argument(
    "--rollout_progress_every",
    type=int,
    default=0,
    help="If > 0, print progress every N rollout steps inside each PPO iteration.",
)
parser.add_argument("--randomize_object", action="store_true")
parser.add_argument("--policy_device", default="", help="Torch device for PPO/BC networks. Default follows env device.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import isaaclab_walker_s2.tasks.pick_place  # noqa: F401, E402


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], output_activation: str = "none") -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ELU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation != "none":
            raise ValueError(f"Unsupported output activation: {output_activation}")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int], action_std_init: float) -> None:
        super().__init__()
        self.actor = MLP(obs_dim, action_dim, hidden_dims, output_activation="tanh")
        self.critic = MLP(obs_dim, 1, hidden_dims, output_activation="none")
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(action_std_init)))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        raw_action = dist.rsample()
        action = torch.clamp(raw_action, -1.0, 1.0)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, value, entropy

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value


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


def _run_dir() -> Path:
    run_name = Path(args_cli.run_name).name if args_cli.run_name else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = args_cli.output_dir.expanduser() / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_bc(path: Path, device: str) -> tuple[BCPolicy, torch.Tensor, torch.Tensor, dict]:
    print(f"[BOOT] _load_bc: torch.load({path.expanduser()}) on CPU...", flush=True)
    checkpoint = torch.load(path.expanduser(), map_location="cpu")
    print("[BOOT] _load_bc: checkpoint loaded. Building BCPolicy on CPU...", flush=True)
    state_dict = checkpoint["model_state_dict"]
    state_keys = list(state_dict.keys())
    print(f"[BOOT] _load_bc: checkpoint has {len(state_keys)} state_dict keys; first keys={state_keys[:3]}", flush=True)
    model = BCPolicy(
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dims=list(checkpoint["hidden_dims"]),
        output_activation=checkpoint.get("output_activation", "tanh"),
    )
    print("[BOOT] _load_bc: loading state_dict...", flush=True)
    model_keys = list(model.state_dict().keys())
    print(f"[BOOT] _load_bc: model expects {len(model_keys)} state_dict keys; first keys={model_keys[:3]}", flush=True)
    model.load_state_dict(state_dict, strict=True)
    print("[BOOT] _load_bc: state_dict loaded.", flush=True)
    print(f"[BOOT] _load_bc: moving BCPolicy to {device}...", flush=True)
    model = model.to(device)
    model.eval()
    print("[BOOT] _load_bc: moving obs stats to device...", flush=True)
    return model, checkpoint["obs_mean"].to(device), checkpoint["obs_std"].to(device), checkpoint


def _decode_bc_output(policy_out: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    transform = checkpoint.get("target_transform", {"mode": "none"})
    mode = transform.get("mode", "none")
    if mode == "none":
        return policy_out
    if mode != "demo_range":
        raise ValueError(f"Unsupported BC target transform mode: {mode}")
    return policy_out * transform["half_range"].to(policy_out.device) + transform["center"].to(policy_out.device)


def _bc_obs_with_phase(obs_policy: torch.Tensor, episode_steps: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    if not checkpoint.get("append_phase", False):
        return obs_policy
    phase_steps = max(int(checkpoint.get("phase_steps", 1)) - 1, 1)
    phase = torch.clamp(episode_steps.float() / float(phase_steps), 0.0, 1.0).unsqueeze(-1)
    return torch.cat([obs_policy, phase.to(obs_policy.device, obs_policy.dtype)], dim=-1)


def _bc_processed_target_to_raw_action(desired: torch.Tensor, action_term) -> torch.Tensor:
    current = action_term.processed_actions
    raw = torch.zeros_like(current)

    desired_nudge = torch.clamp(desired[:, :3], min=action_term._nudge_min, max=action_term._nudge_max)
    raw[:, :3] = (desired_nudge - current[:, :3]) / action_term._delta_scale

    desired_rpy = torch.clamp(desired[:, 3:6], min=action_term._rpy_min, max=action_term._rpy_max)
    raw[:, 3:6] = (desired_rpy - current[:, 3:6]) / action_term._rpy_delta_scale

    raw[:, 6] = torch.clamp(desired[:, 6], 0.0, 1.0)

    if raw.shape[-1] > 7:
        desired_offsets = torch.clamp(desired[:, 7:], min=action_term._arm_offset_min, max=action_term._arm_offset_max)
        raw[:, 7:] = (desired_offsets - current[:, 7:]) / action_term._arm_offset_delta_scale
    return torch.clamp(raw, -1.0, 1.0)


@torch.no_grad()
def _teacher_action(
    bc_model: BCPolicy,
    obs_policy: torch.Tensor,
    episode_steps: torch.Tensor,
    bc_mean: torch.Tensor,
    bc_std: torch.Tensor,
    bc_checkpoint: dict,
    action_term,
) -> torch.Tensor:
    bc_obs = _bc_obs_with_phase(obs_policy, episode_steps, bc_checkpoint)
    norm_obs = (bc_obs - bc_mean) / bc_std
    decoded = _decode_bc_output(bc_model(norm_obs), bc_checkpoint)
    if bc_checkpoint.get("target_key", "action") == "processed_action":
        return _bc_processed_target_to_raw_action(decoded.to(action_term.processed_actions.device), action_term)
    return torch.clamp(decoded, -1.0, 1.0).to(action_term.processed_actions.device)


def _bc_coef(iteration: int) -> float:
    if args_cli.bc_coef_decay_iters <= 0:
        return args_cli.bc_coef_end
    frac = min(float(iteration) / float(args_cli.bc_coef_decay_iters), 1.0)
    return args_cli.bc_coef_start + frac * (args_cli.bc_coef_end - args_cli.bc_coef_start)


def _normalize_obs(obs: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 1:
        return obs
    return (obs - mean) / torch.sqrt(var / max(count - 1, 1) + 1e-6)


def _update_running_stats(obs: torch.Tensor, mean: torch.Tensor, m2: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    batch = obs.detach()
    batch_count = batch.shape[0]
    if batch_count == 0:
        return mean, m2, count
    batch_mean = batch.mean(dim=0)
    batch_m2 = torch.sum((batch - batch_mean) ** 2, dim=0)
    if count == 0:
        return batch_mean, batch_m2, batch_count
    delta = batch_mean - mean
    total = count + batch_count
    new_mean = mean + delta * batch_count / total
    new_m2 = m2 + batch_m2 + delta**2 * count * batch_count / total
    return new_mean, new_m2, total


def main() -> None:
    device = args_cli.policy_device or args_cli.device
    run_dir = _run_dir()
    print(f"[BOOT] Using policy_device={device} sim_device={args_cli.device}", flush=True)
    print(f"[BOOT] Run directory will be {run_dir}", flush=True)
    print(f"[BOOT] Loading BC checkpoint before creating IsaacLab env: {args_cli.bc_checkpoint}", flush=True)
    bc_model, bc_mean, bc_std, bc_checkpoint = _load_bc(args_cli.bc_checkpoint, device)
    print("[BOOT] BC checkpoint loaded before env creation.", flush=True)

    spec = gym.spec(args_cli.task)
    print(f"[BOOT] Loaded gym spec for task={args_cli.task}", flush=True)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = 40.0
    if not args_cli.randomize_object:
        env_cfg.events.reset_object_position = None

    print(f"[BOOT] Creating env with num_envs={args_cli.num_envs}...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[BOOT] gym.make returned. Calling env.reset()...", flush=True)
    obs, _ = env.reset()
    print("[BOOT] env.reset() returned. Reading action manager...", flush=True)
    unwrapped = env.unwrapped
    action_term = unwrapped.action_manager.get_term("palm_ik")
    obs_dim = int(obs["policy"].shape[-1])
    action_dim = int(unwrapped.action_manager.total_action_dim)

    print("[BOOT] Creating PPO networks...", flush=True)
    expected_bc_obs = obs_dim + (1 if bc_checkpoint.get("append_phase", False) else 0)
    if int(bc_checkpoint["obs_dim"]) != expected_bc_obs:
        raise ValueError(
            f"BC checkpoint obs dim {bc_checkpoint['obs_dim']} != expected {expected_bc_obs} "
            f"(env obs {obs_dim}, append_phase={bc_checkpoint.get('append_phase', False)})"
        )
    if int(bc_checkpoint["action_dim"]) != action_dim:
        raise ValueError(f"BC checkpoint action dim {bc_checkpoint['action_dim']} != env action dim {action_dim}")

    model = ActorCritic(obs_dim, action_dim, hidden_dims=[256, 256, 128], action_std_init=args_cli.action_std_init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_cli.lr)
    print("[BOOT] PPO networks ready.", flush=True)

    print("[BOOT] Creating PPO bookkeeping tensors...", flush=True)
    obs_mean = torch.zeros(obs_dim, device=device)
    obs_m2 = torch.zeros(obs_dim, device=device)
    obs_count = 0
    episode_steps = torch.zeros(args_cli.num_envs, dtype=torch.long, device=unwrapped.device)
    episode_returns = torch.zeros(args_cli.num_envs, device=unwrapped.device)
    completed_returns: list[float] = []
    print("[BOOT] PPO bookkeeping tensors ready.", flush=True)

    print("[BOOT] Writing config.json...", flush=True)
    config = {}
    for key, value in vars(args_cli).items():
        config[key] = str(value) if isinstance(value, Path) else value
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("[BOOT] config.json written.", flush=True)

    print(f"[INFO] PPO run_dir={run_dir}", flush=True)
    print(f"[INFO] obs_dim={obs_dim} action_dim={action_dim} num_envs={args_cli.num_envs}", flush=True)
    print(f"[INFO] BC checkpoint={args_cli.bc_checkpoint}", flush=True)
    print(
        f"[INFO] BC target_key={bc_checkpoint.get('target_key')} append_phase={bc_checkpoint.get('append_phase', False)} "
        f"phase_steps={bc_checkpoint.get('phase_steps', 0)}",
        flush=True,
    )

    if args_cli.settle_steps > 0:
        settle_start = time.time()
        print(f"[INFO] Settling env for {args_cli.settle_steps} steps before PPO rollout...", flush=True)
        zero_action = torch.zeros((args_cli.num_envs, action_dim), device=unwrapped.device)
        for settle_step in range(args_cli.settle_steps):
            obs, _, terminated, truncated, _ = env.step(zero_action)
            done = terminated | truncated
            episode_steps[done] = 0
            if (settle_step + 1) % max(args_cli.print_every, 1) == 0 or settle_step + 1 == args_cli.settle_steps:
                elapsed = time.time() - settle_start
                print(
                    f"[SETTLE {settle_step + 1:04d}/{args_cli.settle_steps}] elapsed_s={elapsed:.1f}",
                    flush=True,
                )
        print("[INFO] Settling complete. Starting PPO updates.", flush=True)
    else:
        print("[INFO] No settle steps requested. Starting PPO updates.", flush=True)

    for iteration in range(1, args_cli.iterations + 1):
        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_values = []
        rollout_rewards = []
        rollout_dones = []
        rollout_teacher_mse = []

        bc_coef = _bc_coef(iteration)
        rollout_start = time.time()
        for rollout_step in range(args_cli.horizon):
            policy_obs = obs["policy"].to(device)
            obs_mean, obs_m2, obs_count = _update_running_stats(policy_obs, obs_mean, obs_m2, obs_count)
            norm_obs = _normalize_obs(policy_obs, obs_mean, obs_m2, obs_count)
            with torch.no_grad():
                action, log_prob, value, _ = model.act(norm_obs)
                teacher = _teacher_action(
                    bc_model,
                    policy_obs,
                    episode_steps.to(device),
                    bc_mean,
                    bc_std,
                    bc_checkpoint,
                    action_term,
                )
                teacher_mse = torch.mean((action - teacher.to(device)) ** 2, dim=-1)

            next_obs, reward, terminated, truncated, _ = env.step(action.to(unwrapped.device))
            done = terminated | truncated
            shaped_reward = reward.to(device) - bc_coef * teacher_mse

            rollout_obs.append(norm_obs)
            rollout_actions.append(action.detach())
            rollout_log_probs.append(log_prob.detach())
            rollout_values.append(value.detach())
            rollout_rewards.append(shaped_reward.detach())
            rollout_dones.append(done.to(device).float())
            rollout_teacher_mse.append(teacher_mse.detach())

            episode_returns += reward
            for idx in torch.nonzero(done, as_tuple=False).flatten().tolist():
                completed_returns.append(float(episode_returns[idx].detach().cpu()))
                episode_returns[idx] = 0.0
            episode_steps += 1
            episode_steps[done] = 0
            obs = next_obs
            if args_cli.rollout_progress_every > 0 and (
                (rollout_step + 1) % args_cli.rollout_progress_every == 0 or rollout_step + 1 == args_cli.horizon
            ):
                print(
                    f"[ROLLOUT iter={iteration:04d} step={rollout_step + 1:04d}/{args_cli.horizon}] "
                    f"elapsed_s={time.time() - rollout_start:.1f} reward_mean={float(torch.mean(reward).detach().cpu()):.4f} "
                    f"teacher_mse={float(torch.mean(teacher_mse).detach().cpu()):.4f}",
                    flush=True,
                )

        with torch.no_grad():
            policy_obs = obs["policy"].to(device)
            norm_obs = _normalize_obs(policy_obs, obs_mean, obs_m2, obs_count)
            next_value = model.critic(norm_obs).squeeze(-1)

        obs_b = torch.stack(rollout_obs)
        actions_b = torch.stack(rollout_actions)
        old_log_probs_b = torch.stack(rollout_log_probs)
        values_b = torch.stack(rollout_values)
        rewards_b = torch.stack(rollout_rewards)
        dones_b = torch.stack(rollout_dones)

        advantages = torch.zeros_like(rewards_b)
        last_adv = torch.zeros(args_cli.num_envs, device=device)
        for t in reversed(range(args_cli.horizon)):
            next_nonterminal = 1.0 - dones_b[t]
            next_values = next_value if t == args_cli.horizon - 1 else values_b[t + 1]
            delta = rewards_b[t] + args_cli.gamma * next_values * next_nonterminal - values_b[t]
            last_adv = delta + args_cli.gamma * args_cli.gae_lambda * next_nonterminal * last_adv
            advantages[t] = last_adv
        returns = advantages + values_b

        flat_obs = obs_b.reshape(-1, obs_dim)
        flat_actions = actions_b.reshape(-1, action_dim)
        flat_old_log_probs = old_log_probs_b.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = values_b.reshape(-1)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        batch_size = flat_obs.shape[0]
        minibatch_size = max(batch_size // args_cli.minibatches, 1)
        policy_losses = []
        value_losses = []
        entropies = []
        for _ in range(args_cli.epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb = indices[start : start + minibatch_size]
                new_log_probs, entropy, new_values = model.evaluate(flat_obs[mb], flat_actions[mb])
                ratio = torch.exp(new_log_probs - flat_old_log_probs[mb])
                unclipped = ratio * flat_advantages[mb]
                clipped = torch.clamp(ratio, 1.0 - args_cli.clip_ratio, 1.0 + args_cli.clip_ratio) * flat_advantages[mb]
                policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
                value_loss = torch.mean((new_values - flat_returns[mb]) ** 2)
                entropy_loss = torch.mean(entropy)
                loss = policy_loss + args_cli.value_coef * value_loss - args_cli.entropy_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args_cli.max_grad_norm)
                optimizer.step()

                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                entropies.append(float(entropy_loss.detach().cpu()))

        if iteration % args_cli.save_every == 0 or iteration == args_cli.iterations:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "obs_mean": obs_mean.detach().cpu(),
                    "obs_m2": obs_m2.detach().cpu(),
                    "obs_count": obs_count,
                    "iteration": iteration,
                    "bc_checkpoint": str(args_cli.bc_checkpoint),
                },
                run_dir / "latest.pt",
            )

        if iteration % args_cli.print_every == 0:
            recent_return = np.mean(completed_returns[-20:]) if completed_returns else 0.0
            print(
                f"[ITER {iteration:04d}] reward_mean={float(torch.mean(rewards_b).detach().cpu()):.4f} "
                f"recent_ep_return={recent_return:.4f} bc_coef={bc_coef:.4f} "
                f"teacher_mse={float(torch.mean(torch.stack(rollout_teacher_mse)).detach().cpu()):.4f} "
                f"policy_loss={np.mean(policy_losses):.4f} value_loss={np.mean(value_losses):.4f} "
                f"entropy={np.mean(entropies):.4f}"
            )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
