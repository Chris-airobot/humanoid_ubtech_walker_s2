"""Custom action terms for Walker S2 pick/place tasks."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from dataclasses import MISSING
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URDF_PATH = (
    REPO_ROOT
    / "assets"
    / "resources"
    / "walker_s2_description_hand3_v1_left_hand3_v1_right"
    / "walker_s2_description_hand3_v1_left_hand3_v1_right_isaac_simple_hand_collision.urdf"
)


def _load_fixed_grasp_demo_module():
    module_path = REPO_ROOT / "scripts" / "isaac_walker_fixed_grasp_demo.py"
    spec = importlib.util.spec_from_file_location("walker_fixed_grasp_demo_for_action", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WalkerS2PalmIKAction(ActionTerm):
    """Convert compact palm/grip commands into full-body joint position targets."""

    cfg: "WalkerS2PalmIKActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "WalkerS2PalmIKActionCfg", env) -> None:
        super().__init__(cfg, env)

        self._object: RigidObject = env.scene[self.cfg.object_asset_name]
        self._dof_names = list(self._asset.joint_names)
        self._right_hand_ids = [self._dof_names.index(name) for name in self.cfg.right_hand_joint_names]

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._target_nudge = torch.tensor(self.cfg.default_nudge, device=self.device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self._target_rpy = torch.tensor(self.cfg.default_rpy, device=self.device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self._nudge_min = torch.tensor(self.cfg.nudge_min, device=self.device, dtype=torch.float32)
        self._nudge_max = torch.tensor(self.cfg.nudge_max, device=self.device, dtype=torch.float32)
        self._delta_scale = torch.tensor(self.cfg.delta_scale, device=self.device, dtype=torch.float32)
        self._rpy_min = torch.tensor(self.cfg.rpy_min, device=self.device, dtype=torch.float32)
        self._rpy_max = torch.tensor(self.cfg.rpy_max, device=self.device, dtype=torch.float32)
        self._rpy_delta_scale = torch.tensor(self.cfg.rpy_delta_scale, device=self.device, dtype=torch.float32)
        self._joint_target = self._asset.data.default_joint_pos.clone()
        self._reference_object_pos_w = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)

        hand_open = np.array(
            [self.cfg.right_hand_open_command[name] for name in self.cfg.right_hand_joint_names], dtype=float
        )
        hand_close = np.array(
            [self.cfg.right_hand_close_command[name] for name in self.cfg.right_hand_joint_names], dtype=float
        )
        thumb_mask = np.array(["thumb" in name for name in self.cfg.right_hand_joint_names], dtype=bool)
        self._hand_open = hand_open
        self._hand_close = hand_open.copy()
        self._hand_close[thumb_mask] = hand_open[thumb_mask] + self.cfg.thumb_close_scale * (
            hand_close[thumb_mask] - hand_open[thumb_mask]
        )
        self._hand_close[~thumb_mask] = hand_open[~thumb_mask] + self.cfg.finger_close_scale * (
            hand_close[~thumb_mask] - hand_open[~thumb_mask]
        )

        fixed_demo = _load_fixed_grasp_demo_module()
        self._solve_right_arm_to_cube = fixed_demo.solve_right_arm_to_cube

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def joint_target(self) -> torch.Tensor:
        return self._joint_target

    @property
    def right_hand_joint_ids(self) -> list[int]:
        return self._right_hand_ids

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
        self._target_nudge[:] = torch.clamp(
            self._target_nudge + self._raw_actions[:, :3] * self._delta_scale,
            min=self._nudge_min,
            max=self._nudge_max,
        )
        self._target_rpy[:] = torch.clamp(
            self._target_rpy + self._raw_actions[:, 3:6] * self._rpy_delta_scale,
            min=self._rpy_min,
            max=self._rpy_max,
        )
        grip = torch.clamp(self._raw_actions[:, 6], 0.0, 1.0)
        self._processed_actions[:, :3] = self._target_nudge
        self._processed_actions[:, 3:6] = self._target_rpy
        self._processed_actions[:, 6] = grip

        joint_pos = self._asset.data.joint_pos.detach().cpu().numpy()
        if self.cfg.track_current_object:
            object_pos_w = self._object.data.root_pos_w.detach().cpu().numpy()
        else:
            object_pos_w = self._reference_object_pos_w.detach().cpu().numpy()
        robot_pos_w = self._asset.data.root_pos_w.detach().cpu().numpy()
        target_nudge = self._target_nudge.detach().cpu().numpy()
        target_rpy = self._target_rpy.detach().cpu().numpy()
        grip_np = grip.detach().cpu().numpy()

        joint_targets = []
        for env_id in range(self.num_envs):
            q_seed = joint_pos[env_id].astype(float).copy()
            q_target = self._solve_ik_quiet(
                q_seed=q_seed,
                cube_world=object_pos_w[env_id],
                robot_xyz=robot_pos_w[env_id],
                palm_world_nudge=target_nudge[env_id],
                palm_rpy_nudge=target_rpy[env_id],
            )
            hand_target = self._hand_open + float(grip_np[env_id]) * (self._hand_close - self._hand_open)
            q_target[self._right_hand_ids] = hand_target
            joint_targets.append(q_target)

        self._joint_target[:] = torch.tensor(np.asarray(joint_targets), device=self.device, dtype=torch.float32)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._joint_target)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._target_nudge[env_ids] = torch.tensor(self.cfg.default_nudge, device=self.device, dtype=torch.float32)
        self._target_rpy[env_ids] = torch.tensor(self.cfg.default_rpy, device=self.device, dtype=torch.float32)
        self._joint_target[env_ids] = self._asset.data.default_joint_pos[env_ids]
        self._reference_object_pos_w[env_ids] = self._object.data.root_pos_w[env_ids]

    def _solve_ik_quiet(
        self,
        q_seed: np.ndarray,
        cube_world: np.ndarray,
        robot_xyz: np.ndarray,
        palm_world_nudge: np.ndarray,
        palm_rpy_nudge: np.ndarray,
    ) -> np.ndarray:
        if self.cfg.quiet_ik:
            with contextlib.redirect_stdout(io.StringIO()):
                q_target, _ = self._solve_right_arm_to_cube(
                    self.cfg.urdf_path,
                    self._dof_names,
                    q_seed,
                    cube_world,
                    robot_xyz,
                    self.cfg.robot_yaw_deg,
                    self.cfg.palm_tcp_offset,
                    palm_world_nudge,
                    palm_rpy_nudge,
                )
        else:
            q_target, _ = self._solve_right_arm_to_cube(
                self.cfg.urdf_path,
                self._dof_names,
                q_seed,
                cube_world,
                robot_xyz,
                self.cfg.robot_yaw_deg,
                self.cfg.palm_tcp_offset,
                palm_world_nudge,
                palm_rpy_nudge,
            )
        return np.asarray(q_target, dtype=float)


@configclass
class WalkerS2PalmIKActionCfg(ActionTermCfg):
    """Palm delta + grip action that solves Walker S2 right-arm IK.

    The action is seven dimensional:

    - ``a[0:3]``: incremental world-frame palm target nudge, scaled by ``delta_scale``.
    - ``a[3:6]``: incremental palm roll/pitch/yaw nudge, scaled by ``rpy_delta_scale``.
    - ``a[6]``: continuous grip command. Values <= 0 keep the hand open, values >= 1 close fully.

    The IK target is built from the current object center plus the persistent nudge.
    """

    class_type: type[ActionTerm] = WalkerS2PalmIKAction
    asset_name: str = "robot"
    object_asset_name: str = "object"
    urdf_path: str = str(DEFAULT_URDF_PATH)
    right_arm_joint_names: list[str] = MISSING
    right_hand_joint_names: list[str] = MISSING
    right_hand_open_command: dict[str, float] = MISSING
    right_hand_close_command: dict[str, float] = MISSING
    palm_tcp_offset: tuple[float, float, float] = (0.005, -0.018, 0.025)
    default_nudge: tuple[float, float, float] = (0.08, 0.0, 0.0)
    nudge_min: tuple[float, float, float] = (-0.03, -0.08, -0.04)
    nudge_max: tuple[float, float, float] = (0.12, 0.08, 0.14)
    delta_scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
    default_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_min: tuple[float, float, float] = (-0.6, -0.6, -0.6)
    rpy_max: tuple[float, float, float] = (0.6, 0.6, 0.6)
    rpy_delta_scale: tuple[float, float, float] = (0.03, 0.03, 0.03)
    robot_yaw_deg: float = 90.0
    thumb_close_scale: float = 0.8
    finger_close_scale: float = 1.25
    track_current_object: bool = False
    quiet_ik: bool = True
