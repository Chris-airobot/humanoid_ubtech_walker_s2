#!/usr/bin/env python3
"""Train a behavior cloning policy from Walker S2 IK teleop demos.

Run from this repository root:

    python3 scripts/train_walker_s2_bc.py \
        --demo_roots demos/walker_s2_pick_place_success_random demos/walker_s2_pick_place_success
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]


parser = argparse.ArgumentParser(description="Train Walker S2 pick/place BC from .npz demos.")
parser.add_argument(
    "--demo_roots",
    nargs="+",
    type=Path,
    default=[
        REPO_ROOT / "demos" / "walker_s2_pick_place_success_random",
        REPO_ROOT / "demos" / "walker_s2_pick_place_success",
    ],
)
parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "logs" / "walker_s2_bc")
parser.add_argument(
    "--run_name",
    default="",
    help="Optional fixed run directory name under output_dir. If omitted, a timestamped directory is used.",
)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--val_fraction", type=float, default=0.15)
parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256, 128])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument(
    "--append_phase",
    action="store_true",
    help="Append normalized episode phase [0, 1] to observations. Useful for single-demo BC replay.",
)
parser.add_argument(
    "--target_key",
    choices=["action", "processed_action"],
    default="action",
    help="Demo array to imitate. processed_action is usually better for compact IK teleop demos.",
)
parser.add_argument(
    "--output_activation",
    choices=["auto", "tanh", "none"],
    default="auto",
    help="Policy output activation. auto uses tanh for bounded/normalized targets.",
)
parser.add_argument(
    "--target_normalization",
    choices=["auto", "none", "demo_range"],
    default="auto",
    help="Normalize BC targets before training. auto uses demo_range for processed_action and none for raw action.",
)
parser.add_argument(
    "--target_range_margin",
    type=float,
    default=0.10,
    help="Extra fractional margin added to demo min/max ranges when target_normalization=demo_range.",
)
parser.add_argument(
    "--target_range_floor",
    type=float,
    default=0.02,
    help="Minimum half-range per action dimension when target_normalization=demo_range.",
)
parser.add_argument(
    "--action_nonzero_threshold",
    type=float,
    default=0.05,
    help="Absolute action value treated as an intentional non-zero demo command.",
)
parser.add_argument(
    "--nonzero_action_weight",
    type=float,
    default=8.0,
    help="Extra loss weight for sparse non-zero action dimensions.",
)
parser.add_argument(
    "--grip_action_weight",
    type=float,
    default=3.0,
    help="Loss multiplier for the grip action dimension.",
)
parser.add_argument(
    "--arm_offset_action_weight",
    type=float,
    default=6.0,
    help="Loss multiplier for the arm-offset action dimensions.",
)
parser.add_argument(
    "--require_success",
    action="store_true",
    help="Use only demos explicitly labeled episode_success=True. Unlabeled files are skipped.",
)
parser.add_argument(
    "--include_failed",
    action="store_true",
    help="Also include demos explicitly labeled as failed. Default skips explicit failures.",
)
parser.add_argument("--max_files", type=int, default=0)
args = parser.parse_args()


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


def _as_2d(array: np.ndarray, key: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[1] != 1:
            raise ValueError(f"Only single-env demos are supported for {key}, got shape {array.shape}")
        array = array[:, 0, :]
    if array.ndim != 2:
        raise ValueError(f"Expected {key} to be 2D or 3D single-env, got shape {array.shape}")
    return array


def _bool_scalar(data: np.lib.npyio.NpzFile, key: str) -> bool | None:
    if key not in data.files:
        return None
    value = np.asarray(data[key]).reshape(-1)
    if value.size == 0:
        return None
    return bool(value[0])


def _should_use_demo(data: np.lib.npyio.NpzFile) -> tuple[bool, str]:
    success = _bool_scalar(data, "episode_success")
    failed = _bool_scalar(data, "episode_object_dropping")
    timeout = _bool_scalar(data, "episode_time_out")

    if args.require_success and success is not True:
        return False, "missing_or_false_success_label"
    if not args.include_failed and (success is False or failed is True or timeout is True):
        return False, "explicit_failure_label"
    if success is None:
        return True, "unlabeled_included"
    return True, "success_labeled" if success else "included_by_flag"


def _find_demo_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        root = root.expanduser()
        if root.is_file() and root.suffix == ".npz":
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.glob("*.npz")))
    unique = []
    seen = set()
    for path in sorted(files):
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    if args.max_files > 0:
        unique = unique[: args.max_files]
    return unique


def _load_dataset(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    obs_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    report: list[dict] = []

    for path in paths:
        data = np.load(path)
        if "obs_policy" not in data.files or args.target_key not in data.files:
            report.append({"path": str(path), "used": False, "reason": f"missing_obs_or_{args.target_key}"})
            continue
        use, reason = _should_use_demo(data)
        if not use:
            report.append({"path": str(path), "used": False, "reason": reason})
            continue

        obs = _as_2d(data["obs_policy"], "obs_policy")
        actions = _as_2d(data[args.target_key], args.target_key)
        steps = min(len(obs), len(actions))
        if steps <= 0:
            report.append({"path": str(path), "used": False, "reason": "empty"})
            continue
        obs = obs[:steps]
        actions = actions[:steps]
        if args.append_phase:
            denom = max(steps - 1, 1)
            phase = (np.arange(steps, dtype=np.float32) / float(denom))[:, None]
            obs = np.concatenate([obs, phase], axis=1)
        if args.target_key == "action":
            actions = np.clip(actions, -1.0, 1.0)
        obs_chunks.append(obs)
        action_chunks.append(actions)
        report.append(
            {
                "path": str(path),
                "used": True,
                "reason": reason,
                "frames": int(steps),
                "obs_dim": int(obs.shape[1]),
                "action_dim": int(actions.shape[1]),
            }
        )

    if not obs_chunks:
        raise RuntimeError("No demos were selected for training.")

    obs_all = np.concatenate(obs_chunks, axis=0).astype(np.float32)
    action_all = np.concatenate(action_chunks, axis=0).astype(np.float32)
    if obs_all.shape[0] != action_all.shape[0]:
        raise RuntimeError("Observation/action frame count mismatch after loading.")
    return obs_all, action_all, report


def _make_loss_weights(actions: np.ndarray) -> np.ndarray:
    weights = np.ones_like(actions, dtype=np.float32)
    active = np.abs(actions) >= args.action_nonzero_threshold
    weights += active.astype(np.float32) * args.nonzero_action_weight

    # Current IK action layout: [xyz, rpy, grip, arm_offsets...].
    if actions.shape[1] > 6:
        weights[:, 6] *= args.grip_action_weight
    if actions.shape[1] > 7:
        weights[:, 7:] *= args.arm_offset_action_weight
    return weights.astype(np.float32)


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.mean(weights * torch.square(pred - target))


def _prepare_bc_targets(actions: np.ndarray, normalization: str) -> tuple[np.ndarray, dict]:
    if normalization == "none":
        return actions.astype(np.float32), {"mode": "none"}
    if normalization != "demo_range":
        raise ValueError(f"Unsupported target normalization: {normalization}")

    target_min = actions.min(axis=0).astype(np.float32)
    target_max = actions.max(axis=0).astype(np.float32)
    center = ((target_min + target_max) * 0.5).astype(np.float32)
    half_range = ((target_max - target_min) * 0.5).astype(np.float32)
    half_range = half_range * (1.0 + args.target_range_margin)
    half_range = np.maximum(half_range, args.target_range_floor).astype(np.float32)
    normalized = np.clip((actions - center) / half_range, -1.0, 1.0).astype(np.float32)
    return normalized, {
        "mode": "demo_range",
        "center": center,
        "half_range": half_range,
        "target_min": target_min,
        "target_max": target_max,
        "margin": args.target_range_margin,
        "floor": args.target_range_floor,
    }


def _make_run_dir() -> Path:
    run_name = Path(args.run_name.strip()).name if args.run_name else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not run_name:
        raise ValueError("--run_name must not be empty after path normalization.")
    run_dir = args.output_dir.expanduser() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    demo_files = _find_demo_files(args.demo_roots)
    if not demo_files:
        raise FileNotFoundError(f"No .npz demos found under: {[str(p) for p in args.demo_roots]}")

    obs_np, action_np, report = _load_dataset(demo_files)
    obs_dim = obs_np.shape[1]
    action_dim = action_np.shape[1]
    phase_steps = 0
    if args.append_phase:
        used_frame_counts = [int(item["frames"]) for item in report if item.get("used")]
        phase_steps = int(np.median(used_frame_counts)) if used_frame_counts else 0
    if args.target_normalization == "auto":
        target_normalization = "demo_range" if args.target_key == "processed_action" else "none"
    else:
        target_normalization = args.target_normalization
    if args.output_activation == "auto":
        output_activation = "tanh" if target_normalization == "demo_range" or args.target_key == "action" else "none"
    else:
        output_activation = args.output_activation
    train_target_np, target_transform = _prepare_bc_targets(action_np, target_normalization)

    used = [item for item in report if item.get("used")]
    skipped = [item for item in report if not item.get("used")]
    print(f"[INFO] selected demos: {len(used)} used, {len(skipped)} skipped")
    print(
        f"[INFO] frames={len(obs_np)} obs_dim={obs_dim} action_dim={action_dim} "
        f"target_key={args.target_key} output_activation={output_activation} "
        f"target_normalization={target_normalization} append_phase={args.append_phase}"
    )
    for item in report:
        status = "USE" if item.get("used") else "SKIP"
        detail = f"frames={item.get('frames', '-')}" if item.get("used") else item["reason"]
        print(f"[{status}] {item['path']}  {detail}")

    obs_mean = obs_np.mean(axis=0)
    obs_std = obs_np.std(axis=0)
    obs_std = np.maximum(obs_std, 1e-6)
    obs_norm = (obs_np - obs_mean) / obs_std
    loss_weights_np = _make_loss_weights(action_np)

    indices = np.arange(len(obs_norm))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    val_count = max(1, int(len(indices) * args.val_fraction))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    if len(train_indices) == 0:
        raise RuntimeError("Not enough frames for a train/val split.")

    train_ds = TensorDataset(
        torch.from_numpy(obs_norm[train_indices]).float(),
        torch.from_numpy(train_target_np[train_indices]).float(),
        torch.from_numpy(loss_weights_np[train_indices]).float(),
    )
    val_obs = torch.from_numpy(obs_norm[val_indices]).float().to(args.device)
    val_actions = torch.from_numpy(train_target_np[val_indices]).float().to(args.device)
    val_weights = torch.from_numpy(loss_weights_np[val_indices]).float().to(args.device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = BCPolicy(obs_dim, action_dim, args.hidden_dims, output_activation).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    run_dir = _make_run_dir()
    best_val = float("inf")
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for obs_batch, action_batch, weight_batch in train_loader:
            obs_batch = obs_batch.to(args.device)
            action_batch = action_batch.to(args.device)
            weight_batch = weight_batch.to(args.device)
            pred = model(obs_batch)
            loss = _weighted_mse(pred, action_batch, weight_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_obs)
            val_loss = float(_weighted_mse(val_pred, val_actions, val_weights).detach().cpu())
            val_mse = float(loss_fn(val_pred, val_actions).detach().cpu())
            val_mae = float(torch.mean(torch.abs(val_pred - val_actions)).detach().cpu())
            val_dim_mae = torch.mean(torch.abs(val_pred - val_actions), dim=0).detach().cpu().tolist()
        train_loss = float(np.mean(train_losses))

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "obs_mean": torch.from_numpy(obs_mean).float(),
            "obs_std": torch.from_numpy(obs_std).float(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dims": args.hidden_dims,
            "target_key": args.target_key,
            "output_activation": output_activation,
            "append_phase": args.append_phase,
            "phase_steps": phase_steps,
            "target_transform": {
                key: torch.from_numpy(value).float() if isinstance(value, np.ndarray) else value
                for key, value in target_transform.items()
            },
            "epoch": epoch,
            "val_loss": val_loss,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_dim_mae": val_dim_mae,
            "action_nonzero_threshold": args.action_nonzero_threshold,
            "nonzero_action_weight": args.nonzero_action_weight,
            "grip_action_weight": args.grip_action_weight,
            "arm_offset_action_weight": args.arm_offset_action_weight,
        }
        torch.save(checkpoint, last_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, best_path)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"[EPOCH {epoch:04d}] train_weighted_mse={train_loss:.6f} "
                f"val_weighted_mse={val_loss:.6f} val_mse={val_mse:.6f} val_mae={val_mae:.6f}"
            )
            print(f"  val_dim_mae={np.round(np.asarray(val_dim_mae), 5).tolist()}")

    metadata = {
        "demo_roots": [str(path) for path in args.demo_roots],
        "run_name": args.run_name,
        "used_files": used,
        "skipped_files": skipped,
        "frames": int(len(obs_np)),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "hidden_dims": args.hidden_dims,
        "target_key": args.target_key,
        "output_activation": output_activation,
        "append_phase": args.append_phase,
        "phase_steps": phase_steps,
        "target_normalization": target_normalization,
        "target_transform": {
            key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in target_transform.items()
        },
        "epochs": args.epochs,
        "best_val_weighted_mse": best_val,
        "action_nonzero_threshold": args.action_nonzero_threshold,
        "nonzero_action_weight": args.nonzero_action_weight,
        "grip_action_weight": args.grip_action_weight,
        "arm_offset_action_weight": args.arm_offset_action_weight,
        "elapsed_s": time.time() - start,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[INFO] saved best checkpoint: {best_path}")
    print(f"[INFO] saved metadata: {run_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
