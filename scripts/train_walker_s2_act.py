#!/usr/bin/env python3
"""Train a stage-free state-based ACT policy for Walker S2 pick/place."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from walker_s2_act_common import (  # noqa: E402
    ACTION_DIM,
    ARM_ACTION_DIM,
    ENV_OBS_DIM,
    FEATURE_CONTRACT,
    GRIP_ACTION_INDEX,
    WalkerS2StateACT,
)


parser = argparse.ArgumentParser(description="Train state-based Walker S2 ACT.")
parser.add_argument(
    "--demo_roots",
    nargs="+",
    type=Path,
    default=[REPO_ROOT / "demos" / "walker_s2_object_relative_act_teacher64"],
)
parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "logs" / "walker_s2_act")
parser.add_argument("--run_name", default="object_relative_teacher64")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--val_fraction", type=float, default=0.2)
parser.add_argument("--history_len", type=int, default=8)
parser.add_argument("--chunk_len", type=int, default=20)
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--nhead", type=int, default=4)
parser.add_argument("--encoder_layers", type=int, default=3)
parser.add_argument("--decoder_layers", type=int, default=3)
parser.add_argument("--dim_feedforward", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--grip_loss_weight", type=float, default=2.0)
parser.add_argument("--delta_loss_weight", type=float, default=0.25)
parser.add_argument("--stage_balance_power", type=float, default=0.75)
parser.add_argument("--obs_std_floor", type=float, default=0.05)
parser.add_argument("--obs_clip", type=float, default=5.0)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_files", type=int, default=0)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()


STAGE_NAMES = (
    "settle",
    "pregrasp",
    "approach",
    "close",
    "lift",
    "carry",
    "lower",
    "release",
    "retreat",
)


@dataclass
class Trajectory:
    path: Path
    observations: np.ndarray
    actions: np.ndarray
    stages: np.ndarray

    @property
    def length(self) -> int:
        return len(self.observations)


def _single_env_frames(array: np.ndarray, key: str, expected_dim: int) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 3:
        if value.shape[1] != 1:
            raise ValueError(f"{key} contains multiple environments: {value.shape}")
        value = value[:, 0]
    if value.ndim != 2 or value.shape[1] != expected_dim:
        raise ValueError(f"{key} must have shape [T, {expected_dim}], got {value.shape}")
    return value.astype(np.float32)


def _bool_scalar(data: np.lib.npyio.NpzFile, key: str) -> bool | None:
    if key not in data.files:
        return None
    value = np.asarray(data[key]).reshape(-1)
    return bool(value[0]) if value.size else None


def _find_files() -> list[Path]:
    files: set[Path] = set()
    for root_arg in args.demo_roots:
        root = root_arg.expanduser()
        if root.is_file() and root.suffix == ".npz":
            files.add(root.resolve())
        elif root.is_dir():
            files.update(path.resolve() for path in root.glob("*.npz"))
    ordered = sorted(files)
    return ordered[: args.max_files] if args.max_files > 0 else ordered


def _load_dataset(files: list[Path]) -> tuple[list[Trajectory], list[dict]]:
    trajectories: list[Trajectory] = []
    report: list[dict] = []
    fingerprints: dict[str, Path] = {}
    required_stages = set(range(len(STAGE_NAMES)))
    for path in files:
        try:
            with np.load(path) as data:
                if _bool_scalar(data, "episode_success") is not True:
                    raise ValueError("not_strict_success")
                if _bool_scalar(data, "episode_object_dropping") is True:
                    raise ValueError("object_dropping")
                missing = {"obs_policy", "action", "teacher_stage"}.difference(data.files)
                if missing:
                    raise ValueError(f"missing_{sorted(missing)}")

                observations = _single_env_frames(data["obs_policy"], "obs_policy", ENV_OBS_DIM)
                actions = _single_env_frames(data["action"], "action", ACTION_DIM)
                stages = np.asarray(data["teacher_stage"], dtype=np.int64).reshape(-1)
                steps = min(len(observations), len(actions), len(stages))
                observations = observations[:steps]
                actions = actions[:steps]
                stages = stages[:steps]
                if steps < args.history_len or not np.isfinite(observations).all() or not np.isfinite(actions).all():
                    raise ValueError("too_short_or_non_finite")
                if set(np.unique(stages).tolist()) != required_stages:
                    raise ValueError("trajectory_missing_one_or_more_teacher_stages")
                if np.max(np.abs(actions[:, :ARM_ACTION_DIM])) > 1.0001:
                    raise ValueError("arm_action_outside_bounds")
                if np.min(actions[:, GRIP_ACTION_INDEX]) < -1e-4 or np.max(actions[:, GRIP_ACTION_INDEX]) > 1.0001:
                    raise ValueError("grip_action_outside_bounds")

                fingerprint_source = np.concatenate((np.round(observations, 5), np.round(actions, 5)), axis=1)
                fingerprint = hashlib.sha256(fingerprint_source.tobytes()).hexdigest()
                if fingerprint in fingerprints:
                    raise ValueError(f"duplicate_content_of_{fingerprints[fingerprint]}")
                fingerprints[fingerprint] = path
                trajectories.append(Trajectory(path, observations, actions, stages))
                report.append(
                    {
                        "path": str(path),
                        "used": True,
                        "frames": steps,
                        "stage_counts": {
                            STAGE_NAMES[stage]: int(np.sum(stages == stage)) for stage in range(len(STAGE_NAMES))
                        },
                    }
                )
        except Exception as exc:
            report.append({"path": str(path), "used": False, "reason": str(exc)})
    if not trajectories:
        raise RuntimeError("No strict-success trajectories passed the ACT dataset contract.")
    return trajectories, report


def _split(trajectories: list[Trajectory]) -> tuple[list[Trajectory], list[Trajectory]]:
    shuffled = trajectories.copy()
    random.Random(args.seed).shuffle(shuffled)
    if len(shuffled) < 2:
        raise RuntimeError("ACT requires at least two trajectories for a trajectory-level validation split.")
    val_count = min(max(1, round(len(shuffled) * args.val_fraction)), len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def _statistics(trajectories: list[Trajectory]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = np.concatenate([item.observations for item in trajectories], axis=0)
    mean = observations.mean(axis=0).astype(np.float32)
    raw_std = observations.std(axis=0).astype(np.float32)
    std = np.maximum(raw_std, args.obs_std_floor).astype(np.float32)
    return mean, std, raw_std


class ACTWindowDataset(Dataset):
    def __init__(
        self,
        trajectories: list[Trajectory],
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        obs_clip: float,
        history_len: int,
        chunk_len: int,
    ) -> None:
        self.trajectories = trajectories
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.obs_clip = obs_clip
        self.history_len = history_len
        self.chunk_len = chunk_len
        self.indices = [(trajectory_id, step) for trajectory_id, item in enumerate(trajectories) for step in range(item.length)]
        self.sample_stages = np.asarray(
            [trajectories[trajectory_id].stages[step] for trajectory_id, step in self.indices],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectory_id, step = self.indices[index]
        trajectory = self.trajectories[trajectory_id]

        history_start = max(0, step - self.history_len + 1)
        history = trajectory.observations[history_start : step + 1]
        if len(history) < self.history_len:
            padding = np.repeat(history[:1], self.history_len - len(history), axis=0)
            history = np.concatenate((padding, history), axis=0)
        history = (history - self.obs_mean) / self.obs_std
        if self.obs_clip > 0.0:
            history = np.clip(history, -self.obs_clip, self.obs_clip)

        chunk_end = min(trajectory.length, step + self.chunk_len)
        valid_actions = trajectory.actions[step:chunk_end]
        action_chunk = np.zeros((self.chunk_len, ACTION_DIM), dtype=np.float32)
        action_mask = np.zeros(self.chunk_len, dtype=np.bool_)
        action_chunk[: len(valid_actions)] = valid_actions
        action_mask[: len(valid_actions)] = True
        return (
            torch.from_numpy(history.astype(np.float32)),
            torch.from_numpy(action_chunk),
            torch.from_numpy(action_mask),
            torch.tensor(int(trajectory.stages[step]), dtype=torch.long),
        )


def _stage_sampler(dataset: ACTWindowDataset) -> tuple[WeightedRandomSampler, np.ndarray, np.ndarray]:
    counts = np.bincount(dataset.sample_stages, minlength=len(STAGE_NAMES))
    if np.any(counts == 0):
        missing = [STAGE_NAMES[index] for index in np.flatnonzero(counts == 0)]
        raise RuntimeError(f"Training dataset is missing stages: {missing}")
    weights = (counts.sum() / (len(STAGE_NAMES) * counts.astype(np.float64))) ** args.stage_balance_power
    weights = np.minimum(weights, 5.0)
    sample_weights = torch.as_tensor(weights[dataset.sample_stages], dtype=torch.double)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    return sampler, counts, weights.astype(np.float32)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * mask.float()) / torch.clamp(torch.sum(mask), min=1)


def _losses(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    arm_per_step = F.smooth_l1_loss(
        prediction[..., :ARM_ACTION_DIM],
        target[..., :ARM_ACTION_DIM],
        reduction="none",
        beta=0.05,
    ).mean(dim=-1)
    grip_per_step = F.binary_cross_entropy(
        torch.clamp(prediction[..., GRIP_ACTION_INDEX], 1e-6, 1.0 - 1e-6),
        target[..., GRIP_ACTION_INDEX],
        reduction="none",
    )
    arm_loss = _masked_mean(arm_per_step, mask)
    grip_loss = _masked_mean(grip_per_step, mask)

    pair_mask = torch.logical_and(mask[:, 1:], mask[:, :-1])
    predicted_delta = prediction[:, 1:, :ARM_ACTION_DIM] - prediction[:, :-1, :ARM_ACTION_DIM]
    target_delta = target[:, 1:, :ARM_ACTION_DIM] - target[:, :-1, :ARM_ACTION_DIM]
    delta_per_step = torch.abs(predicted_delta - target_delta).mean(dim=-1)
    delta_loss = _masked_mean(delta_per_step, pair_mask)
    total = arm_loss + args.grip_loss_weight * grip_loss + args.delta_loss_weight * delta_loss
    return {"total": total, "arm": arm_loss, "grip": grip_loss, "delta": delta_loss}


def _run_epoch(
    model: WalkerS2StateACT,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float | list[float]]:
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("total", "arm", "grip", "delta")}
    valid_steps = 0
    arm_error_sum = 0.0
    grip_error_sum = 0.0
    first_arm_error_sum = 0.0
    first_grip_error_sum = 0.0
    samples = 0
    stage_frames = np.zeros(len(STAGE_NAMES), dtype=np.int64)
    stage_first_arm = np.zeros(len(STAGE_NAMES), dtype=np.float64)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for history, action_chunk, action_mask, stages in loader:
            history = history.to(device)
            action_chunk = action_chunk.to(device)
            action_mask = action_mask.to(device)
            stages = stages.to(device)
            prediction = model(history)
            losses = _losses(prediction, action_chunk, action_mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            count = int(action_mask.sum().item())
            valid_steps += count
            for name, loss in losses.items():
                totals[name] += float(loss.detach().cpu()) * count
            arm_error = torch.abs(prediction[..., :ARM_ACTION_DIM] - action_chunk[..., :ARM_ACTION_DIM]).mean(dim=-1)
            grip_error = torch.abs(prediction[..., GRIP_ACTION_INDEX] - action_chunk[..., GRIP_ACTION_INDEX])
            arm_error_sum += float(arm_error[action_mask].sum().detach().cpu())
            grip_error_sum += float(grip_error[action_mask].sum().detach().cpu())
            first_arm_error = arm_error[:, 0]
            first_grip_error = grip_error[:, 0]
            first_arm_error_sum += float(first_arm_error.sum().detach().cpu())
            first_grip_error_sum += float(first_grip_error.sum().detach().cpu())
            samples += len(history)
            for stage_id in range(len(STAGE_NAMES)):
                stage_mask = stages == stage_id
                stage_count = int(stage_mask.sum().item())
                if stage_count:
                    stage_frames[stage_id] += stage_count
                    stage_first_arm[stage_id] += float(first_arm_error[stage_mask].sum().detach().cpu())

    step_denominator = max(valid_steps, 1)
    sample_denominator = max(samples, 1)
    metrics: dict[str, float | list[float]] = {
        name: total / step_denominator for name, total in totals.items()
    }
    metrics.update(
        {
            "chunk_arm_mae": arm_error_sum / step_denominator,
            "chunk_grip_mae": grip_error_sum / step_denominator,
            "first_arm_mae": first_arm_error_sum / sample_denominator,
            "first_grip_mae": first_grip_error_sum / sample_denominator,
            "stage_first_arm_mae": (stage_first_arm / np.maximum(stage_frames, 1)).tolist(),
        }
    )
    metrics["deployment_score"] = float(metrics["first_arm_mae"]) + float(metrics["first_grip_mae"])
    return metrics


def _serializable_args() -> dict:
    payload = vars(args).copy()
    payload["demo_roots"] = [str(path) for path in args.demo_roots]
    payload["output_dir"] = str(args.output_dir)
    return payload


def main() -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    files = _find_files()
    if not files:
        raise FileNotFoundError(f"No .npz files found under {[str(path) for path in args.demo_roots]}")
    trajectories, report = _load_dataset(files)
    train_trajectories, val_trajectories = _split(trajectories)
    obs_mean, obs_std, obs_std_raw = _statistics(train_trajectories)
    train_dataset = ACTWindowDataset(
        train_trajectories,
        obs_mean,
        obs_std,
        args.obs_clip,
        args.history_len,
        args.chunk_len,
    )
    val_dataset = ACTWindowDataset(
        val_trajectories,
        obs_mean,
        obs_std,
        args.obs_clip,
        args.history_len,
        args.chunk_len,
    )
    sampler, stage_counts, stage_weights = _stage_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = WalkerS2StateACT(
        history_len=args.history_len,
        chunk_len=args.chunk_len,
        d_model=args.d_model,
        nhead=args.nhead,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    run_name = Path(args.run_name).name
    if not run_name:
        raise ValueError("--run_name must not be empty.")
    run_dir = args.output_dir.expanduser() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "accepted": [item for item in report if item.get("used")],
        "rejected": [item for item in report if not item.get("used")],
        "train_files": [str(item.path) for item in train_trajectories],
        "val_files": [str(item.path) for item in val_trajectories],
    }
    (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"[INFO] contract={FEATURE_CONTRACT} stage_input=False history={args.history_len} "
        f"chunk={args.chunk_len} temporal_model=transformer"
    )
    print(
        f"[INFO] trajectories={len(trajectories)} train={len(train_trajectories)} "
        f"val={len(val_trajectories)} windows={len(train_dataset)}/{len(val_dataset)} device={device}"
    )
    print(f"[INFO] stage_counts={dict(zip(STAGE_NAMES, stage_counts.tolist(), strict=True))}")
    print(f"[INFO] sampler_weights={dict(zip(STAGE_NAMES, np.round(stage_weights, 3).tolist(), strict=True))}")
    for item in report:
        print(f"[{'USE' if item.get('used') else 'SKIP'}] {item['path']} {item.get('frames', item.get('reason'))}")

    best_score = float("inf")
    best_epoch = 0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step()
        checkpoint = {
            "checkpoint_type": "walker_s2_state_act",
            "checkpoint_version": 1,
            "feature_contract": FEATURE_CONTRACT,
            "environment_obs_dim": ENV_OBS_DIM,
            "action_dim": ACTION_DIM,
            "model_config": model.export_config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_mean": torch.from_numpy(obs_mean),
            "obs_std": torch.from_numpy(obs_std),
            "obs_std_raw": torch.from_numpy(obs_std_raw),
            "obs_clip": args.obs_clip,
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "task": "Isaac-WalkerS2-PickPlace-ObjectRelative-v0",
            "stage_input": False,
            "action_layout": [
                "object_relative_x",
                "object_relative_y",
                "object_relative_z",
                "roll",
                "pitch",
                "yaw",
                "grip",
            ],
            "args": _serializable_args(),
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if float(val_metrics["deployment_score"]) < best_score:
            best_score = float(val_metrics["deployment_score"])
            best_epoch = epoch
            torch.save(checkpoint, run_dir / "best.pt")

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"[EPOCH {epoch:04d}] train_total={train_metrics['total']:.6f} "
                f"val_total={val_metrics['total']:.6f} "
                f"first_arm_mae={val_metrics['first_arm_mae']:.5f} "
                f"first_grip_mae={val_metrics['first_grip_mae']:.5f} "
                f"chunk_arm_mae={val_metrics['chunk_arm_mae']:.5f} "
                f"score={val_metrics['deployment_score']:.5f}"
            )

    metadata = {
        "feature_contract": FEATURE_CONTRACT,
        "run_name": run_name,
        "accepted_trajectories": len(trajectories),
        "rejected_trajectories": len(manifest["rejected"]),
        "train_frames": sum(item.length for item in train_trajectories),
        "val_frames": sum(item.length for item in val_trajectories),
        "best_epoch": best_epoch,
        "best_deployment_score": best_score,
        "elapsed_seconds": time.time() - start_time,
        "args": _serializable_args(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[INFO] saved best checkpoint: {run_dir / 'best.pt'}")
    print(f"[INFO] saved dataset manifest: {run_dir / 'dataset_manifest.json'}")
    print(f"[INFO] saved metadata: {run_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
