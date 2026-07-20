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
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils


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


class WalkerS2DirectArmGripAction(ActionTerm):
    """Map compact direct policy actions to Walker S2 right-arm and right-hand joint targets.

    Raw action layout:

    - ``a[0:7]``: normalized right-arm joint target offsets around the robot default pose.
    - ``a[7]``: grip command. Values <= 0 keep the hand open, values >= 1 close fully.

    This action term intentionally has no IK, no object-relative target, and no staged phase.
    """

    cfg: "WalkerS2DirectArmGripActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "WalkerS2DirectArmGripActionCfg", env) -> None:
        super().__init__(cfg, env)

        self._dof_names = list(self._asset.joint_names)
        self._right_arm_ids = [self._dof_names.index(name) for name in self.cfg.right_arm_joint_names]
        self._right_hand_ids = [self._dof_names.index(name) for name in self.cfg.right_hand_joint_names]

        if len(self.cfg.arm_joint_scale) != len(self._right_arm_ids):
            raise ValueError(
                "arm_joint_scale must match right_arm_joint_names length "
                f"{len(self._right_arm_ids)}; got {len(self.cfg.arm_joint_scale)}."
            )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._arm_joint_scale = torch.tensor(self.cfg.arm_joint_scale, device=self.device, dtype=torch.float32)
        self._default_arm_pos = self._asset.data.default_joint_pos[:, self._right_arm_ids].clone()
        self._joint_target_arm = self._default_arm_pos.clone()
        self._joint_target_hand = self._asset.data.default_joint_pos[:, self._right_hand_ids].clone()

        soft_limits = self._asset.data.soft_joint_pos_limits[:, self._right_arm_ids]
        self._arm_lower = soft_limits[..., 0]
        self._arm_upper = soft_limits[..., 1]

        hand_open = torch.tensor(
            [self.cfg.right_hand_open_command[name] for name in self.cfg.right_hand_joint_names],
            device=self.device,
            dtype=torch.float32,
        )
        hand_close = torch.tensor(
            [self.cfg.right_hand_close_command[name] for name in self.cfg.right_hand_joint_names],
            device=self.device,
            dtype=torch.float32,
        )
        thumb_mask = torch.tensor(
            ["thumb" in name for name in self.cfg.right_hand_joint_names],
            device=self.device,
            dtype=torch.bool,
        )
        self._hand_open = hand_open
        self._hand_close = hand_open.clone()
        self._hand_close[thumb_mask] = hand_open[thumb_mask] + self.cfg.thumb_close_scale * (
            hand_close[thumb_mask] - hand_open[thumb_mask]
        )
        self._hand_close[~thumb_mask] = hand_open[~thumb_mask] + self.cfg.finger_close_scale * (
            hand_close[~thumb_mask] - hand_open[~thumb_mask]
        )

    @property
    def action_dim(self) -> int:
        return 8

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def joint_target(self) -> torch.Tensor:
        return self._asset.data.joint_pos_target

    @property
    def right_hand_joint_ids(self) -> list[int]:
        return self._right_hand_ids

    @property
    def grip_index(self) -> int:
        return 7

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
        arm_target = self._default_arm_pos + self._raw_actions[:, :7] * self._arm_joint_scale
        self._joint_target_arm[:] = torch.clamp(arm_target, min=self._arm_lower, max=self._arm_upper)

        grip = torch.clamp(self._raw_actions[:, 7], 0.0, 1.0)
        self._joint_target_hand[:] = self._hand_open + grip.unsqueeze(-1) * (self._hand_close - self._hand_open)

        self._processed_actions[:, :7] = self._joint_target_arm
        self._processed_actions[:, 7] = grip

    def apply_actions(self):
        self._asset.set_joint_position_target(self._joint_target_arm, joint_ids=self._right_arm_ids)
        self._asset.set_joint_position_target(self._joint_target_hand, joint_ids=self._right_hand_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._joint_target_arm[env_ids] = self._default_arm_pos[env_ids]
        self._joint_target_hand[env_ids] = self._hand_open


class WalkerS2HandGripAction(ActionTerm):
    """Map one continuous grip command to all right-hand joint targets.

    This term deliberately controls only the hand.  Keeping it separate from
    the Cartesian arm term gives the policy the conventional seven-dimensional
    action contract ``[delta_pose_6, grip]`` without routing arm commands
    through the legacy CPU IK solver.
    """

    cfg: "WalkerS2HandGripActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "WalkerS2HandGripActionCfg", env) -> None:
        super().__init__(cfg, env)

        dof_names = list(self._asset.joint_names)
        self._right_hand_ids = [dof_names.index(name) for name in self.cfg.right_hand_joint_names]
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        hand_open = torch.tensor(
            [self.cfg.right_hand_open_command[name] for name in self.cfg.right_hand_joint_names],
            device=self.device,
            dtype=torch.float32,
        )
        hand_close = torch.tensor(
            [self.cfg.right_hand_close_command[name] for name in self.cfg.right_hand_joint_names],
            device=self.device,
            dtype=torch.float32,
        )
        thumb_mask = torch.tensor(
            ["thumb" in name for name in self.cfg.right_hand_joint_names],
            device=self.device,
            dtype=torch.bool,
        )
        self._hand_open = hand_open
        self._hand_close = hand_open.clone()
        self._hand_close[thumb_mask] = hand_open[thumb_mask] + self.cfg.thumb_close_scale * (
            hand_close[thumb_mask] - hand_open[thumb_mask]
        )
        self._hand_close[~thumb_mask] = hand_open[~thumb_mask] + self.cfg.finger_close_scale * (
            hand_close[~thumb_mask] - hand_open[~thumb_mask]
        )
        self._joint_target_hand = self._hand_open.repeat(self.num_envs, 1)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def grip_index(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def right_hand_joint_ids(self) -> list[int]:
        return self._right_hand_ids

    def process_actions(self, actions: torch.Tensor) -> None:
        grip = torch.clamp(actions[:, :1], 0.0, 1.0)
        self._raw_actions[:] = actions[:, :1]
        self._processed_actions[:] = grip
        self._joint_target_hand[:] = self._hand_open + grip * (self._hand_close - self._hand_open)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._joint_target_hand, joint_ids=self._right_hand_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._joint_target_hand[env_ids] = self._hand_open


class WalkerS2AbsoluteCartesianAction(DifferentialInverseKinematicsAction):
    """Servo a normalized absolute palm pose with native differential IK.

    The policy controls a stable goal rather than imitating the teacher's
    state-dependent feedback correction. Raw actions are
    ``[x, y, z, roll, pitch, yaw]`` in ``[-1, 1]``. Position is mapped into a
    world-aligned workspace; orientation is an offset around the validated
    grasp quaternion.
    """

    cfg: "WalkerS2AbsoluteCartesianActionCfg"

    def __init__(self, cfg: "WalkerS2AbsoluteCartesianActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._workspace_min = torch.tensor(cfg.workspace_min, device=self.device, dtype=torch.float32)
        self._workspace_max = torch.tensor(cfg.workspace_max, device=self.device, dtype=torch.float32)
        self._workspace_center = 0.5 * (self._workspace_min + self._workspace_max)
        self._workspace_half_range = 0.5 * (self._workspace_max - self._workspace_min)
        self._orientation_range = torch.tensor(cfg.orientation_range, device=self.device, dtype=torch.float32)
        self._nominal_quat_w = torch.tensor(
            cfg.nominal_quat_w, device=self.device, dtype=torch.float32
        ).repeat(self.num_envs, 1)

    @property
    def action_dim(self) -> int:
        return 6

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = torch.clamp(actions[:, :6], -1.0, 1.0)
        self._processed_actions[:] = self._raw_actions

        desired_pos_env = self._workspace_center + self._raw_actions[:, :3] * self._workspace_half_range
        rpy = self._raw_actions[:, 3:6] * self._orientation_range
        delta_quat = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        desired_quat_w = math_utils.quat_mul(self._nominal_quat_w, delta_quat)
        desired_pos_w = desired_pos_env + self._env.scene.env_origins
        desired_pos_b, desired_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w,
            self._asset.data.root_quat_w,
            desired_pos_w,
            desired_quat_w,
        )
        command = torch.cat((desired_pos_b, desired_quat_b), dim=-1)
        self._ik_controller.set_command(command)

    def encode_env_pose(self, desired_pos_env: torch.Tensor) -> torch.Tensor:
        """Encode an environment-frame position at the nominal orientation."""
        desired_pos_env = desired_pos_env.reshape(-1, 3).to(device=self.device, dtype=torch.float32)
        normalized_pos = (desired_pos_env - self._workspace_center) / self._workspace_half_range
        encoded = torch.zeros((desired_pos_env.shape[0], 6), device=self.device, dtype=torch.float32)
        encoded[:, :3] = torch.clamp(normalized_pos, -1.0, 1.0)
        return encoded

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._processed_actions[env_ids] = 0.0


class WalkerS2ObjectRelativeCartesianAction(WalkerS2AbsoluteCartesianAction):
    """Servo an absolute palm goal expressed relative to the current object.

    The coordinate convention is generic: raw XYZ controls a bounded offset
    from the object, while orientation remains relative to the validated palm
    quaternion. No task phase, waypoint, grasp offset, or target location is
    encoded in this action term.
    """

    cfg: "WalkerS2ObjectRelativeCartesianActionCfg"

    def __init__(self, cfg: "WalkerS2ObjectRelativeCartesianActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._object: RigidObject = env.scene[cfg.object_asset_name]
        self._offset_min = torch.tensor(cfg.offset_min, device=self.device, dtype=torch.float32)
        self._offset_max = torch.tensor(cfg.offset_max, device=self.device, dtype=torch.float32)
        self._offset_center = 0.5 * (self._offset_min + self._offset_max)
        self._offset_half_range = 0.5 * (self._offset_max - self._offset_min)

    def _object_pos_env(self) -> torch.Tensor:
        return self._object.data.root_pos_w - self._env.scene.env_origins

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = torch.clamp(actions[:, :6], -1.0, 1.0)
        self._processed_actions[:] = self._raw_actions

        desired_offset = self._offset_center + self._raw_actions[:, :3] * self._offset_half_range
        desired_pos_env = self._object_pos_env() + desired_offset
        desired_pos_env = torch.clamp(desired_pos_env, min=self._workspace_min, max=self._workspace_max)
        rpy = self._raw_actions[:, 3:6] * self._orientation_range
        delta_quat = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        desired_quat_w = math_utils.quat_mul(self._nominal_quat_w, delta_quat)
        desired_pos_w = desired_pos_env + self._env.scene.env_origins
        desired_pos_b, desired_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w,
            self._asset.data.root_quat_w,
            desired_pos_w,
            desired_quat_w,
        )
        self._ik_controller.set_command(torch.cat((desired_pos_b, desired_quat_b), dim=-1))

    def encode_env_pose(self, desired_pos_env: torch.Tensor) -> torch.Tensor:
        """Encode an environment-frame palm goal in the object-relative convention."""

        desired_pos_env = desired_pos_env.reshape(-1, 3).to(device=self.device, dtype=torch.float32)
        object_pos_env = self._object_pos_env()
        if object_pos_env.shape[0] != desired_pos_env.shape[0]:
            if desired_pos_env.shape[0] != 1:
                raise ValueError(
                    f"Cannot broadcast {desired_pos_env.shape[0]} goals to {object_pos_env.shape[0]} objects."
                )
            object_pos_env = object_pos_env[:1]
        normalized_offset = (desired_pos_env - object_pos_env - self._offset_center) / self._offset_half_range
        encoded = torch.zeros((desired_pos_env.shape[0], 6), device=self.device, dtype=torch.float32)
        encoded[:, :3] = torch.clamp(normalized_offset, -1.0, 1.0)
        return encoded


class WalkerS2PalmIKAction(ActionTerm):
    """Convert compact palm/grip commands into full-body joint position targets."""

    cfg: "WalkerS2PalmIKActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "WalkerS2PalmIKActionCfg", env) -> None:
        super().__init__(cfg, env)

        self._object: RigidObject = env.scene[self.cfg.object_asset_name]
        self._dof_names = list(self._asset.joint_names)
        self._right_hand_ids = [self._dof_names.index(name) for name in self.cfg.right_hand_joint_names]
        self._arm_offset_joint_names = list(self.cfg.arm_offset_joint_names)
        self._arm_offset_ids = [self._dof_names.index(name) for name in self._arm_offset_joint_names]
        self._arm_offset_dim = len(self._arm_offset_ids)
        if self._arm_offset_dim > 0:
            expected = self._arm_offset_dim
            actual = (
                len(self.cfg.arm_offset_min),
                len(self.cfg.arm_offset_max),
                len(self.cfg.arm_offset_delta_scale),
            )
            if actual != (expected, expected, expected):
                raise ValueError(
                    "arm_offset_min, arm_offset_max, and arm_offset_delta_scale must match "
                    f"arm_offset_joint_names length {expected}; got {actual}."
                )

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
        self._target_arm_offsets = torch.zeros(self.num_envs, self._arm_offset_dim, device=self.device)
        self._arm_offset_min = torch.tensor(self.cfg.arm_offset_min, device=self.device, dtype=torch.float32)
        self._arm_offset_max = torch.tensor(self.cfg.arm_offset_max, device=self.device, dtype=torch.float32)
        self._arm_offset_delta_scale = torch.tensor(
            self.cfg.arm_offset_delta_scale, device=self.device, dtype=torch.float32
        )
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
        return 7 + len(self.cfg.arm_offset_joint_names)

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

    def capture_current_object_reference(self, env_ids: Sequence[int] | None = None) -> None:
        """Use the object's current pose as the fixed IK reference for selected environments."""
        if env_ids is None:
            env_ids = slice(None)
        self._reference_object_pos_w[env_ids] = self._object.data.root_pos_w[env_ids]

    @property
    def arm_offset_joint_names(self) -> list[str]:
        return self._arm_offset_joint_names

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
        if self._arm_offset_dim > 0:
            self._target_arm_offsets[:] = torch.clamp(
                self._target_arm_offsets + self._raw_actions[:, 7:] * self._arm_offset_delta_scale,
                min=self._arm_offset_min,
                max=self._arm_offset_max,
            )
        self._processed_actions[:, :3] = self._target_nudge
        self._processed_actions[:, 3:6] = self._target_rpy
        self._processed_actions[:, 6] = grip
        if self._arm_offset_dim > 0:
            self._processed_actions[:, 7:] = self._target_arm_offsets

        joint_pos = self._asset.data.joint_pos.detach().cpu().numpy()
        if self.cfg.track_current_object:
            object_pos_w = self._object.data.root_pos_w.detach().cpu().numpy()
        else:
            object_pos_w = self._reference_object_pos_w.detach().cpu().numpy()
        robot_pos_w = self._asset.data.root_pos_w.detach().cpu().numpy()
        target_nudge = self._target_nudge.detach().cpu().numpy()
        target_rpy = self._target_rpy.detach().cpu().numpy()
        grip_np = grip.detach().cpu().numpy()
        arm_offsets = self._target_arm_offsets.detach().cpu().numpy()

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
            if self._arm_offset_dim > 0:
                q_target[self._arm_offset_ids] += arm_offsets[env_id]
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
        if self._arm_offset_dim > 0:
            self._target_arm_offsets[env_ids] = 0.0
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


class WalkerS2CartesianDeltaAction(WalkerS2PalmIKAction):
    """Control the right palm with object-agnostic Cartesian deltas and a grip command.

    Raw action layout:

    - ``a[0:3]``: world-frame palm position delta.
    - ``a[3:6]``: palm roll/pitch/yaw delta around the validated grasp orientation.
    - ``a[6]``: absolute grip command in ``[0, 1]``.

    The persistent palm target is initialized from the measured palm pose after
    each reset. The object pose is never used to transform or offset policy
    actions; IK is only the low-level mapping from a palm target to arm joints.
    """

    cfg: "WalkerS2CartesianDeltaActionCfg"

    def __init__(self, cfg: "WalkerS2CartesianDeltaActionCfg", env) -> None:
        super().__init__(cfg, env)

        palm_ids, _ = self._asset.find_bodies(self.cfg.palm_body_name, preserve_order=True)
        if len(palm_ids) != 1:
            raise ValueError(f"Expected one palm body named {self.cfg.palm_body_name!r}, got {palm_ids}.")
        self._palm_body_id = palm_ids[0]

        self._position_delta_scale = torch.tensor(
            self.cfg.position_delta_scale, device=self.device, dtype=torch.float32
        )
        self._orientation_delta_scale = torch.tensor(
            self.cfg.orientation_delta_scale, device=self.device, dtype=torch.float32
        )
        self._workspace_min = torch.tensor(self.cfg.workspace_min, device=self.device, dtype=torch.float32)
        self._workspace_max = torch.tensor(self.cfg.workspace_max, device=self.device, dtype=torch.float32)
        self._cartesian_target_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._cartesian_target_rpy = torch.tensor(
            self.cfg.default_rpy, device=self.device, dtype=torch.float32
        ).repeat(self.num_envs, 1)
        self._target_initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def grip_index(self) -> int:
        return 6

    @property
    def target_position(self) -> torch.Tensor:
        """Persistent palm position target in each environment's world-aligned frame."""

        return self._cartesian_target_pos

    @property
    def target_rpy(self) -> torch.Tensor:
        return self._cartesian_target_rpy

    def _initialize_target_from_palm(self) -> None:
        uninitialized = torch.logical_not(self._target_initialized)
        if not torch.any(uninitialized):
            return
        palm_pos = self._asset.data.body_pos_w[:, self._palm_body_id] - self._env.scene.env_origins
        self._cartesian_target_pos[uninitialized] = palm_pos[uninitialized]
        self._cartesian_target_rpy[uninitialized] = torch.tensor(
            self.cfg.default_rpy, device=self.device, dtype=torch.float32
        )
        self._target_initialized[uninitialized] = True

    @staticmethod
    def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = (float(value) for value in rpy)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
        rotation_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
        rotation_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        return rotation_z @ rotation_y @ rotation_x

    def _tcp_nudge_world(self, palm_rpy: np.ndarray) -> np.ndarray:
        robot_yaw = np.deg2rad(self.cfg.robot_yaw_deg)
        world_from_base = np.array(
            [
                [np.cos(robot_yaw), -np.sin(robot_yaw), 0.0],
                [np.sin(robot_yaw), np.cos(robot_yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        validated_palm_rotation = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=float
        )
        palm_rotation_world = world_from_base @ validated_palm_rotation @ self._rpy_matrix(palm_rpy)
        return palm_rotation_world @ np.asarray(self.cfg.palm_tcp_offset, dtype=float)

    def process_actions(self, actions: torch.Tensor):
        self._initialize_target_from_palm()
        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
        self._cartesian_target_pos[:] = torch.clamp(
            self._cartesian_target_pos + self._raw_actions[:, :3] * self._position_delta_scale,
            min=self._workspace_min,
            max=self._workspace_max,
        )
        self._cartesian_target_rpy[:] = torch.clamp(
            self._cartesian_target_rpy + self._raw_actions[:, 3:6] * self._orientation_delta_scale,
            min=self._rpy_min,
            max=self._rpy_max,
        )
        grip = torch.clamp(self._raw_actions[:, 6], 0.0, 1.0)

        self._processed_actions[:, :3] = self._cartesian_target_pos
        self._processed_actions[:, 3:6] = self._cartesian_target_rpy
        self._processed_actions[:, 6] = grip

        joint_pos = self._asset.data.joint_pos.detach().cpu().numpy()
        robot_pos_w = self._asset.data.root_pos_w.detach().cpu().numpy()
        env_origins = self._env.scene.env_origins.detach().cpu().numpy()
        target_pos = self._cartesian_target_pos.detach().cpu().numpy()
        target_rpy = self._cartesian_target_rpy.detach().cpu().numpy()
        grip_np = grip.detach().cpu().numpy()
        previous_target = self._joint_target.detach().cpu().numpy()
        pose_action_active = torch.any(torch.abs(self._raw_actions[:, :6]) > 1e-6, dim=1).detach().cpu().numpy()

        joint_targets = []
        for env_id in range(self.num_envs):
            if not bool(pose_action_active[env_id]):
                q_target = previous_target[env_id].copy()
                hand_target = self._hand_open + float(grip_np[env_id]) * (self._hand_close - self._hand_open)
                q_target[self._right_hand_ids] = hand_target
                joint_targets.append(q_target)
                continue

            q_seed = joint_pos[env_id].astype(float).copy()
            target_pos_w = target_pos[env_id] + env_origins[env_id]
            tcp_nudge_world = self._tcp_nudge_world(target_rpy[env_id])
            q_target = self._solve_ik_quiet(
                q_seed=q_seed,
                cube_world=target_pos_w,
                robot_xyz=robot_pos_w[env_id],
                palm_world_nudge=tcp_nudge_world,
                palm_rpy_nudge=target_rpy[env_id],
            )

            previous_arm = previous_target[env_id, self._right_arm_ids]
            candidate_arm = q_target[self._right_arm_ids]
            arm_delta = np.clip(
                candidate_arm - previous_arm,
                -self.cfg.max_arm_joint_target_step,
                self.cfg.max_arm_joint_target_step,
            )
            q_target[self._right_arm_ids] = previous_arm + arm_delta
            hand_target = self._hand_open + float(grip_np[env_id]) * (self._hand_close - self._hand_open)
            q_target[self._right_hand_ids] = hand_target
            joint_targets.append(q_target)

        self._joint_target[:] = torch.tensor(np.asarray(joint_targets), device=self.device, dtype=torch.float32)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._target_initialized[env_ids] = False
        self._cartesian_target_pos[env_ids] = 0.0
        self._cartesian_target_rpy[env_ids] = torch.tensor(
            self.cfg.default_rpy, device=self.device, dtype=torch.float32
        )
        self._joint_target[env_ids][:, self._right_hand_ids] = torch.tensor(
            self._hand_open, device=self.device, dtype=torch.float32
        )


@configclass
class WalkerS2PalmIKActionCfg(ActionTermCfg):
    """Palm delta + grip action that solves Walker S2 right-arm IK.

    The action is compact but includes an optional arm-posture residual:

    - ``a[0:3]``: incremental world-frame palm target nudge, scaled by ``delta_scale``.
    - ``a[3:6]``: incremental palm roll/pitch/yaw nudge, scaled by ``rpy_delta_scale``.
    - ``a[6]``: continuous grip command. Values <= 0 keep the hand open, values >= 1 close fully.
    - ``a[7:]``: incremental offsets for ``arm_offset_joint_names``.

    The IK target is built from the current object center plus the persistent nudge.
    """

    class_type: type[ActionTerm] = WalkerS2PalmIKAction
    asset_name: str = "robot"
    object_asset_name: str = "object"
    urdf_path: str = str(DEFAULT_URDF_PATH)
    right_arm_joint_names: list[str] = MISSING
    right_hand_joint_names: list[str] = MISSING
    arm_offset_joint_names: tuple[str, ...] = ()
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
    arm_offset_min: tuple[float, ...] = ()
    arm_offset_max: tuple[float, ...] = ()
    arm_offset_delta_scale: tuple[float, ...] = ()
    robot_yaw_deg: float = 90.0
    thumb_close_scale: float = 0.8
    finger_close_scale: float = 1.25
    track_current_object: bool = False
    quiet_ik: bool = True


@configclass
class WalkerS2CartesianDeltaActionCfg(WalkerS2PalmIKActionCfg):
    """Object-agnostic Cartesian palm delta and absolute grip action."""

    class_type: type[ActionTerm] = WalkerS2CartesianDeltaAction
    palm_body_name: str = "hand3_v1_right_R_palm_link"
    position_delta_scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
    orientation_delta_scale: tuple[float, float, float] = (0.05, 0.05, 0.05)
    workspace_min: tuple[float, float, float] = (0.54, 0.04, 1.06)
    workspace_max: tuple[float, float, float] = (1.10, 0.40, 1.36)
    max_arm_joint_target_step: float = 0.15


@configclass
class WalkerS2DirectArmGripActionCfg(ActionTermCfg):
    """Direct 8D right-arm + grip action for student policies."""

    class_type: type[ActionTerm] = WalkerS2DirectArmGripAction
    asset_name: str = "robot"
    right_arm_joint_names: list[str] = MISSING
    right_hand_joint_names: list[str] = MISSING
    right_hand_open_command: dict[str, float] = MISSING
    right_hand_close_command: dict[str, float] = MISSING
    arm_joint_scale: tuple[float, float, float, float, float, float, float] = (
        1.2,
        1.2,
        1.5,
        1.5,
        1.5,
        1.2,
        1.2,
    )
    thumb_close_scale: float = 0.95
    finger_close_scale: float = 2.1


@configclass
class WalkerS2HandGripActionCfg(ActionTermCfg):
    """Configuration for the independent right-hand grip action term."""

    class_type: type[ActionTerm] = WalkerS2HandGripAction
    asset_name: str = "robot"
    right_hand_joint_names: list[str] = MISSING
    right_hand_open_command: dict[str, float] = MISSING
    right_hand_close_command: dict[str, float] = MISSING
    thumb_close_scale: float = 0.95
    finger_close_scale: float = 2.1


@configclass
class WalkerS2AbsoluteCartesianActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for normalized absolute palm-pose actions."""

    class_type: type[ActionTerm] = WalkerS2AbsoluteCartesianAction
    workspace_min: tuple[float, float, float] = (0.54, 0.04, 1.06)
    workspace_max: tuple[float, float, float] = (1.10, 0.40, 1.36)
    nominal_quat_w: tuple[float, float, float, float] = (0.5, -0.5, -0.5, 0.5)
    orientation_range: tuple[float, float, float] = (0.4, 0.4, 0.4)


@configclass
class WalkerS2ObjectRelativeCartesianActionCfg(WalkerS2AbsoluteCartesianActionCfg):
    """Configuration for object-relative absolute palm goals."""

    class_type: type[ActionTerm] = WalkerS2ObjectRelativeCartesianAction
    object_asset_name: str = "object"
    offset_min: tuple[float, float, float] = (-0.35, -0.25, -0.10)
    offset_max: tuple[float, float, float] = (0.25, 0.25, 0.35)


class WalkerS2StagedPickPlaceAction(WalkerS2PalmIKAction):
    """Stage-based pick/place primitive with residual policy controls.

    Raw action layout:

    - ``a[0:3]``: bounded residual added to the staged palm nudge.
    - ``a[3:6]``: bounded residual added to the staged palm orientation.
    - ``a[6]``: grip residual added to the staged grip command.
    - ``a[7]``: stage-speed control. ``-1`` holds, ``0`` uses nominal speed, ``1`` doubles speed.
    - ``a[8:]``: optional arm-posture residuals for configured arm-offset joints.

    Processed action layout keeps grip at index 6 for reward helpers:
    ``[target_nudge_xyz, target_rpy, grip, phase, arm_offsets...]``.
    """

    cfg: "WalkerS2StagedPickPlaceActionCfg"

    def __init__(self, cfg: "WalkerS2StagedPickPlaceActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._residual_xyz_scale = torch.tensor(self.cfg.residual_xyz_scale, device=self.device, dtype=torch.float32)
        self._residual_rpy_scale = torch.tensor(self.cfg.residual_rpy_scale, device=self.device, dtype=torch.float32)
        self._target_pos = torch.tensor(self.cfg.target_pos, device=self.device, dtype=torch.float32)
        self._base_waypoint_phase = torch.tensor(
            [item[0] for item in self.cfg.waypoints], device=self.device, dtype=torch.float32
        )
        self._base_waypoint_nudge = torch.tensor(
            [item[1] for item in self.cfg.waypoints], device=self.device, dtype=torch.float32
        )
        self._base_waypoint_rpy = torch.tensor(
            [item[2] for item in self.cfg.waypoints], device=self.device, dtype=torch.float32
        )
        self._base_waypoint_grip = torch.tensor(
            [item[3] for item in self.cfg.waypoints], device=self.device, dtype=torch.float32
        )
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        return 8 + len(self.cfg.arm_offset_joint_names)

    @property
    def phase(self) -> torch.Tensor:
        """Current primitive phase for each environment."""
        return self._phase

    def set_phase(self, phase: float | torch.Tensor) -> None:
        """Override primitive phase without changing the normal staged action API."""
        phase_tensor = torch.as_tensor(phase, device=self.device, dtype=torch.float32)
        if phase_tensor.ndim == 0:
            self._phase.fill_(float(phase_tensor.item()))
        else:
            self._phase[:] = phase_tensor.reshape(self._phase.shape)
        self._phase.clamp_(0.0, 1.0)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)

        phase_speed = torch.clamp(1.0 + self.cfg.phase_action_scale * self._raw_actions[:, 7], 0.0, 2.0)
        self._phase[:] = torch.clamp(self._phase + self.cfg.phase_rate * phase_speed, 0.0, 1.0)

        base_nudge, base_rpy, base_grip = self._interpolate_stage(self._phase)
        place_nudge = self._target_place_nudge()
        base_nudge = self._apply_place_nudge(base_nudge, place_nudge)

        target_nudge = torch.clamp(
            base_nudge + self._raw_actions[:, :3] * self._residual_xyz_scale,
            min=self._nudge_min,
            max=self._nudge_max,
        )
        target_rpy = torch.clamp(
            base_rpy + self._raw_actions[:, 3:6] * self._residual_rpy_scale,
            min=self._rpy_min,
            max=self._rpy_max,
        )
        grip = torch.clamp(base_grip + self.cfg.grip_residual_scale * self._raw_actions[:, 6], 0.0, 1.0)

        if self._arm_offset_dim > 0:
            arm_raw = self._raw_actions[:, 8:]
            self._target_arm_offsets[:] = torch.clamp(
                arm_raw * self._arm_offset_delta_scale,
                min=self._arm_offset_min,
                max=self._arm_offset_max,
            )

        self._target_nudge[:] = target_nudge
        self._target_rpy[:] = target_rpy
        self._processed_actions[:, :3] = target_nudge
        self._processed_actions[:, 3:6] = target_rpy
        self._processed_actions[:, 6] = grip
        self._processed_actions[:, 7] = self._phase
        if self._arm_offset_dim > 0:
            self._processed_actions[:, 8:] = self._target_arm_offsets

        joint_pos = self._asset.data.joint_pos.detach().cpu().numpy()
        if self.cfg.track_current_object:
            object_pos_w = self._object.data.root_pos_w.detach().cpu().numpy()
        else:
            object_pos_w = self._reference_object_pos_w.detach().cpu().numpy()
        robot_pos_w = self._asset.data.root_pos_w.detach().cpu().numpy()
        target_nudge_np = target_nudge.detach().cpu().numpy()
        target_rpy_np = target_rpy.detach().cpu().numpy()
        grip_np = grip.detach().cpu().numpy()
        arm_offsets = self._target_arm_offsets.detach().cpu().numpy()

        joint_targets = []
        for env_id in range(self.num_envs):
            q_seed = joint_pos[env_id].astype(float).copy()
            q_target = self._solve_ik_quiet(
                q_seed=q_seed,
                cube_world=object_pos_w[env_id],
                robot_xyz=robot_pos_w[env_id],
                palm_world_nudge=target_nudge_np[env_id],
                palm_rpy_nudge=target_rpy_np[env_id],
            )
            if self._arm_offset_dim > 0:
                q_target[self._arm_offset_ids] += arm_offsets[env_id]
            hand_target = self._hand_open + float(grip_np[env_id]) * (self._hand_close - self._hand_open)
            q_target[self._right_hand_ids] = hand_target
            joint_targets.append(q_target)

        self._joint_target[:] = torch.tensor(np.asarray(joint_targets), device=self.device, dtype=torch.float32)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._phase[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

    def _interpolate_stage(self, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase_count = self._base_waypoint_phase.shape[0]
        nudge = torch.zeros(self.num_envs, 3, device=self.device)
        rpy = torch.zeros(self.num_envs, 3, device=self.device)
        grip = torch.zeros(self.num_envs, device=self.device)

        for waypoint_id in range(phase_count - 1):
            phase0 = self._base_waypoint_phase[waypoint_id]
            phase1 = self._base_waypoint_phase[waypoint_id + 1]
            if waypoint_id == phase_count - 2:
                mask = torch.logical_and(phase >= phase0, phase <= phase1)
            else:
                mask = torch.logical_and(phase >= phase0, phase < phase1)
            alpha = torch.clamp((phase - phase0) / torch.clamp(phase1 - phase0, min=1e-6), 0.0, 1.0)
            alpha_n = alpha.unsqueeze(-1)
            nudge_i = (1.0 - alpha_n) * self._base_waypoint_nudge[waypoint_id] + alpha_n * self._base_waypoint_nudge[
                waypoint_id + 1
            ]
            rpy_i = (1.0 - alpha_n) * self._base_waypoint_rpy[waypoint_id] + alpha_n * self._base_waypoint_rpy[
                waypoint_id + 1
            ]
            grip_i = (1.0 - alpha) * self._base_waypoint_grip[waypoint_id] + alpha * self._base_waypoint_grip[
                waypoint_id + 1
            ]
            nudge = torch.where(mask.unsqueeze(-1), nudge_i, nudge)
            rpy = torch.where(mask.unsqueeze(-1), rpy_i, rpy)
            grip = torch.where(mask, grip_i, grip)

        return nudge, rpy, grip

    def _target_place_nudge(self) -> torch.Tensor:
        if self.cfg.track_current_object:
            reference_object_pos = self._object.data.root_pos_w - self._env.scene.env_origins
        else:
            reference_object_pos = self._reference_object_pos_w - self._env.scene.env_origins
        target = self._target_pos.repeat(self.num_envs, 1)
        place_xy = target[:, :2] - reference_object_pos[:, :2]
        lower_alpha = torch.clamp(
            (self._phase - self.cfg.release_blend_start)
            / max(self.cfg.release_blend_end - self.cfg.release_blend_start, 1e-6),
            0.0,
            1.0,
        )
        place_z = (1.0 - lower_alpha) * self.cfg.place_lift_nudge_z + lower_alpha * self.cfg.place_release_nudge_z
        retreat_alpha = torch.clamp(
            (self._phase - self.cfg.release_retreat_start) / max(1.0 - self.cfg.release_retreat_start, 1e-6),
            0.0,
            1.0,
        )
        retreat_offset = torch.tensor(self.cfg.release_retreat_nudge, device=self.device, dtype=torch.float32)
        place_xyz = torch.cat(
            [
                place_xy + torch.tensor(self.cfg.place_palm_xy_offset, device=self.device, dtype=torch.float32),
                place_z.unsqueeze(-1),
            ],
            dim=1,
        )
        place_xyz = place_xyz + retreat_alpha.unsqueeze(-1) * retreat_offset
        place_xyz[:, 2] = torch.clamp(place_xyz[:, 2], self._nudge_min[2], self._nudge_max[2])
        place_xyz[:, :2] = torch.clamp(place_xyz[:, :2], self._nudge_min[:2], self._nudge_max[:2])
        return place_xyz

    def _apply_place_nudge(self, base_nudge: torch.Tensor, place_nudge: torch.Tensor) -> torch.Tensor:
        place_alpha = torch.clamp(
            (self._phase - self.cfg.place_blend_start) / max(self.cfg.place_blend_end - self.cfg.place_blend_start, 1e-6),
            0.0,
            1.0,
        ).unsqueeze(-1)
        return (1.0 - place_alpha) * base_nudge + place_alpha * place_nudge


@configclass
class WalkerS2StagedPickPlaceActionCfg(WalkerS2PalmIKActionCfg):
    """Stage-based pick/place primitive with residual RL controls."""

    class_type: type[ActionTerm] = WalkerS2StagedPickPlaceAction
    phase_rate: float = 1.0 / 900.0
    phase_action_scale: float = 1.0
    residual_xyz_scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
    residual_rpy_scale: tuple[float, float, float] = (0.04, 0.04, 0.04)
    grip_residual_scale: float = 0.15
    target_pos: tuple[float, float, float] = (0.62, 0.24, 1.045)
    place_palm_xy_offset: tuple[float, float] = (-0.001, 0.0)
    place_lift_nudge_z: float = 0.08
    place_release_nudge_z: float = 0.02
    place_blend_start: float = 0.64
    place_blend_end: float = 0.88
    release_blend_start: float = 0.90
    release_blend_end: float = 0.94
    release_retreat_start: float = 0.95
    release_retreat_nudge: tuple[float, float, float] = (0.06, 0.0, 0.08)
    waypoints: tuple[
        tuple[float, tuple[float, float, float], tuple[float, float, float], float],
        ...,
    ] = (
        (0.00, (0.08, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
        (0.20, (0.08, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
        (0.42, (-0.001, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
        (0.56, (-0.001, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0),
        (0.70, (-0.001, 0.0, 0.081), (0.0, 0.0, 0.0), 1.0),
        (0.88, (-0.001, 0.0, 0.081), (0.0, 0.0, 0.0), 1.0),
        (0.91, (-0.001, 0.0, 0.02), (0.0, 0.0, 0.0), 1.0),
        (0.94, (-0.001, 0.0, 0.02), (0.0, 0.0, 0.0), 0.0),
        (1.00, (0.06, 0.0, 0.10), (0.0, 0.0, 0.0), 0.0),
    )
