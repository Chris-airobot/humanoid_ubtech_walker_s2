#!/usr/bin/env python3
"""Keyboard teleoperation for the registered Walker S2 IK pick/place task.

Run from this repository root:

    /home/chris/IsaacLab/isaaclab.sh -p scripts/teleop_walker_s2_ik_env.py
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Keyboard teleop for Walker S2 palm-IK pick/place.")
parser.add_argument("--task", default="Isaac-WalkerS2-PickPlace-IK-v0")
parser.add_argument("--steps", type=int, default=0, help="Maximum sim steps. 0 means run until quit.")
parser.add_argument("--print_every", type=int, default=30)
parser.add_argument("--record_dir", default="demos/walker_s2_pick_place")
parser.add_argument("--record", action="store_true", help="Start recording immediately.")
parser.add_argument("--randomize_object", action="store_true", help="Enable reset-time object pose randomization.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.input
import gymnasium as gym
import numpy as np
import omni.appwindow
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import isaaclab_walker_s2.tasks.pick_place  # noqa: F401, E402


class WalkerS2IKKeyboardTeleop:
    """Keyboard state machine for the compact Walker S2 IK action."""

    ARM_OFFSET_KEYS = {
        "U": (7, 1.0),
        "J": (7, -1.0),
        "I": (8, 1.0),
        "K": (8, -1.0),
        "O": (9, 1.0),
        "L": (9, -1.0),
        "N": (10, 1.0),
        "M": (10, -1.0),
    }

    def __init__(self, env, record_dir: Path, record_on_start: bool = False) -> None:
        self.env = env
        self.unwrapped = env.unwrapped
        self.device = self.unwrapped.device
        self.record_dir = record_dir
        self.record_dir.mkdir(parents=True, exist_ok=True)

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

        self._pressed: set[str] = set()
        self._arm_offset_pulses: dict[int, float] = {}
        self._grip = 0.0
        self._recording = record_on_start
        self._quit = False
        self._reset_requested = False
        self._episode_index = 0
        self._frames: list[dict[str, np.ndarray | float | bool]] = []
        self._last_done_info: dict[str, bool | dict[str, bool]] = {}
        self._last_obs = None
        self._last_info = None

        print("\n[INFO] Walker S2 IK keyboard teleop ready")
        print("  W/S: palm target x -/+")
        print("  A/D: palm target y +/-")
        print("  E/Q: palm target z +/-")
        print("  Z/X: palm roll +/-")
        print("  T/Y: palm pitch +/-")
        print("  C/V: palm yaw +/-")
        print("  U/J: shoulder yaw offset +/-")
        print("  I/K: elbow yaw offset +/-")
        print("  O/L: wrist pitch offset +/-")
        print("  N/M: wrist roll offset +/-")
        print("  G: toggle grip close/open")
        print("  R: reset environment")
        print("  P: start/stop recording")
        print("  ESC: save current recording and quit\n")

    @property
    def should_quit(self) -> bool:
        return self._quit

    def close(self) -> None:
        if self._recording and self._frames:
            self._save_recording()
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)

    def reset(self) -> None:
        if self._recording and self._frames:
            self._save_recording()
        self._frames = []
        self._last_done_info = {}
        self._pressed.clear()
        self._arm_offset_pulses.clear()
        self._grip = 0.0
        self._episode_index += 1
        self._last_obs, self._last_info = self.env.reset()
        self._reset_requested = False
        print("[INFO] Environment reset")

    def action(self) -> torch.Tensor:
        action = torch.zeros(
            (self.unwrapped.num_envs, self.unwrapped.action_manager.total_action_dim), device=self.device
        )

        if "W" in self._pressed:
            action[:, 0] += -1.0
        if "S" in self._pressed:
            action[:, 0] += 1.0
        if "A" in self._pressed:
            action[:, 1] += 1.0
        if "D" in self._pressed:
            action[:, 1] += -1.0
        if "E" in self._pressed:
            action[:, 2] += 1.0
        if "Q" in self._pressed:
            action[:, 2] += -1.0

        if "Z" in self._pressed:
            action[:, 3] += 1.0
        if "X" in self._pressed:
            action[:, 3] += -1.0
        if "T" in self._pressed:
            action[:, 4] += 1.0
        if "Y" in self._pressed:
            action[:, 4] += -1.0
        if "C" in self._pressed:
            action[:, 5] += 1.0
        if "V" in self._pressed:
            action[:, 5] += -1.0

        action[:, 6] = self._grip
        for index, value in self._arm_offset_pulses.items():
            if index < action.shape[1]:
                action[:, index] += value
        self._arm_offset_pulses.clear()
        return torch.clamp(action, -1.0, 1.0)

    def step(self, step_count: int):
        if self._reset_requested:
            self.reset()

        action = self.action()
        obs, rew, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        self._last_info = info

        if self._recording:
            self._append_frame(obs, action, rew, terminated, truncated)

        if bool(torch.any(terminated | truncated)):
            self._last_done_info = self._termination_summary(terminated, truncated)
            print(f"[INFO] Episode ended at step {step_count}: terminated={terminated.tolist()} truncated={truncated.tolist()}")
            if self._last_done_info.get("terms"):
                print(f"[INFO] termination_terms={self._last_done_info['terms']}")
            self.reset()

        return obs, rew, terminated, truncated, info

    def print_debug(self, rew: torch.Tensor, step_count: int) -> None:
        robot = self.unwrapped.scene["robot"]
        obj = self.unwrapped.scene["object"]
        palm_ids, _ = robot.find_bodies("hand3_v1_right_R_palm_link", preserve_order=True)
        palm = (robot.data.body_pos_w[:, palm_ids[0]] - self.unwrapped.scene.env_origins).detach().cpu().tolist()
        obj_pos = (obj.data.root_pos_w - self.unwrapped.scene.env_origins).detach().cpu().tolist()
        action_term = self.unwrapped.action_manager.get_term("palm_ik")
        processed = action_term.processed_actions.detach()
        target_pos = None
        reference_object_pos = getattr(action_term, "_reference_object_pos_w", None)
        if reference_object_pos is not None:
            target_pos = (reference_object_pos + processed[:, :3] - self.unwrapped.scene.env_origins).cpu().tolist()

        print(f"[STEP {step_count:05d}] reward={rew.detach().cpu().tolist()} recording={self._recording}")
        print(f"  action_raw={action_term.raw_actions.detach().cpu().tolist()}")
        print(f"  action_processed=[target_nudge_xyz, target_rpy, grip, arm_offsets]={processed.cpu().tolist()}")
        arm_offset_names = getattr(action_term, "arm_offset_joint_names", [])
        if arm_offset_names:
            print(f"  arm_offset_joints={arm_offset_names}")
        if target_pos is not None:
            print(f"  ik_target_pos={target_pos}")
        print(f"  palm={palm}")
        print(f"  object={obj_pos}")

    def _append_frame(
        self,
        obs: dict,
        action: torch.Tensor,
        rew: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        robot = self.unwrapped.scene["robot"]
        obj = self.unwrapped.scene["object"]
        palm_ids, _ = robot.find_bodies("hand3_v1_right_R_palm_link", preserve_order=True)
        action_term = self.unwrapped.action_manager.get_term("palm_ik")

        frame = {
            "obs_policy": obs["policy"].detach().cpu().numpy().copy(),
            "action": action.detach().cpu().numpy().copy(),
            "processed_action": action_term.processed_actions.detach().cpu().numpy().copy(),
            "reward": rew.detach().cpu().numpy().copy(),
            "terminated": terminated.detach().cpu().numpy().copy(),
            "truncated": truncated.detach().cpu().numpy().copy(),
            "palm_pos": (robot.data.body_pos_w[:, palm_ids[0]] - self.unwrapped.scene.env_origins)
            .detach()
            .cpu()
            .numpy()
            .copy(),
            "object_pos": (obj.data.root_pos_w - self.unwrapped.scene.env_origins).detach().cpu().numpy().copy(),
            "object_quat": obj.data.root_quat_w.detach().cpu().numpy().copy(),
        }
        self._frames.append(frame)

    def _termination_summary(self, terminated: torch.Tensor, truncated: torch.Tensor) -> dict[str, bool | dict[str, bool]]:
        terms: dict[str, bool] = {}
        log = getattr(self.unwrapped, "extras", {}).get("log", {})
        prefix = "Episode_Termination/"
        for key, value in log.items():
            if key.startswith(prefix):
                terms[key.removeprefix(prefix)] = bool(value)
        if not terms:
            manager = getattr(self.unwrapped, "termination_manager", None)
            if manager is not None:
                for name in manager.active_terms:
                    with torch.no_grad():
                        terms[name] = bool(torch.any(manager.get_term(name)).item())
        return {
            "terminated": bool(torch.any(terminated).item()),
            "truncated": bool(torch.any(truncated).item()),
            "success": bool(terms.get("success", False)),
            "object_dropping": bool(terms.get("object_dropping", False)),
            "time_out": bool(terms.get("time_out", False) or torch.any(truncated).item()),
            "terms": terms,
        }

    def _save_recording(self) -> None:
        if not self._frames:
            print("[INFO] No recorded frames to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.record_dir / f"walker_s2_pick_place_ep{self._episode_index:03d}_{timestamp}.npz"
        keys = self._frames[0].keys()
        arrays = {key: np.stack([frame[key] for frame in self._frames], axis=0) for key in keys}
        arrays["created_unix_time"] = np.array([time.time()], dtype=np.float64)
        arrays["episode_success"] = np.array([bool(self._last_done_info.get("success", False))], dtype=np.bool_)
        arrays["episode_object_dropping"] = np.array(
            [bool(self._last_done_info.get("object_dropping", False))], dtype=np.bool_
        )
        arrays["episode_time_out"] = np.array([bool(self._last_done_info.get("time_out", False))], dtype=np.bool_)
        terms = self._last_done_info.get("terms", {})
        if isinstance(terms, dict):
            for name, value in terms.items():
                arrays[f"termination_{name}"] = np.array([bool(value)], dtype=np.bool_)
        np.savez_compressed(path, **arrays)
        print(f"[INFO] Saved {len(self._frames)} frames to {path}")
        self._frames = []
        self._last_done_info = {}

    def _on_keyboard_event(self, event):
        key = event.input if isinstance(event.input, str) else event.input.name
        key = key.upper()
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key in {"W", "A", "S", "D", "Q", "E", "Z", "X", "T", "Y", "C", "V"}:
                self._pressed.add(key)
            elif key in self.ARM_OFFSET_KEYS:
                if key not in self._pressed:
                    index, value = self.ARM_OFFSET_KEYS[key]
                    self._arm_offset_pulses[index] = self._arm_offset_pulses.get(index, 0.0) + value
                    self._pressed.add(key)
            elif key == "G":
                self._grip = 0.0 if self._grip > 0.5 else 1.0
                print(f"[INFO] grip={self._grip}")
            elif key == "R":
                self._reset_requested = True
            elif key == "P":
                self._recording = not self._recording
                print(f"[INFO] recording={self._recording}")
                if not self._recording:
                    self._save_recording()
            elif key == "ESCAPE":
                self._quit = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key in {"W", "A", "S", "D", "Q", "E", "Z", "X", "T", "Y", "C", "V"} | set(self.ARM_OFFSET_KEYS):
                self._pressed.discard(key)


def main() -> None:
    spec = gym.spec(args_cli.task)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = 40.0
    if not args_cli.randomize_object:
        env_cfg.events.reset_object_position = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    print(f"[INFO] Loaded registered task: {args_cli.task}")
    print(f"[INFO] observation shape: {obs['policy'].shape}")
    print(f"[INFO] action terms: {env.unwrapped.action_manager.active_terms}")

    teleop = WalkerS2IKKeyboardTeleop(env, REPO_ROOT / args_cli.record_dir, args_cli.record)

    try:
        step_count = 0
        while simulation_app.is_running() and not teleop.should_quit:
            _, rew, _, _, _ = teleop.step(step_count)
            if step_count % args_cli.print_every == 0:
                teleop.print_debug(rew, step_count)
            step_count += 1
            if args_cli.steps > 0 and step_count >= args_cli.steps:
                break
    finally:
        teleop.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Walker S2 IK teleop failed:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
