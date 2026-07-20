"""State-gated Cartesian teacher for the Walker S2 pick/place proof pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import isaaclab.utils.math as math_utils


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
class CartesianTeacherCfg:
    settle_steps: int = 10
    pregrasp_offset: tuple[float, float, float] = (0.106, -0.003, -0.001)
    grasp_offset: tuple[float, float, float] = (0.033, -0.005, -0.001)
    lift_height: float = 0.080
    carry_palm_offset: tuple[float, float, float] = (0.039, 0.0, 0.115)
    lower_palm_offset: tuple[float, float, float] = (0.039, 0.0, 0.068)
    retreat_offset: tuple[float, float, float] = (0.080, 0.0, 0.080)
    pregrasp_tolerance: float = 0.015
    approach_tolerance: float = 0.012
    motion_tolerance: float = 0.030
    stable_steps: int = 4
    close_steps: int = 12
    release_steps: int = 8
    target_margin_xy: float = 0.015
    orientation_tolerance: float = 0.15
    carry_min_lift: float = 0.015


class CartesianPickPlaceTeacher:
    """A deterministic oracle whose transitions are gated by measured state."""

    def __init__(self, target_pos, target_size, initial_object_height: float, cfg=None) -> None:
        self.cfg = cfg or CartesianTeacherCfg()
        self.target_pos = torch.as_tensor(target_pos, dtype=torch.float32)
        self.target_size = torch.as_tensor(target_size, dtype=torch.float32)
        self.initial_object_height = float(initial_object_height)
        self.stage_id = 0
        self.stage_step = 0
        self.stable_count = 0
        self.object_reference = None
        self.grasp_target = None
        self.lift_target = None
        self.retreat_target = None
        self.initial_palm_reference = None
        self.desired_palm_quat_w = torch.tensor((0.5, -0.5, -0.5, 0.5), dtype=torch.float32)
        self.last_metrics: dict[str, float | bool] = {}

    @property
    def stage_name(self) -> str:
        return STAGE_NAMES[self.stage_id]

    def reset(self, object_position: torch.Tensor) -> None:
        self.stage_id = 0
        self.stage_step = 0
        self.stable_count = 0
        self.object_reference = object_position.detach().clone()
        self.grasp_target = None
        self.lift_target = None
        self.retreat_target = None
        self.initial_palm_reference = None
        self.last_metrics = {}

    def _transition(self, stage_id: int, palm_pos: torch.Tensor, object_pos: torch.Tensor) -> None:
        self.stage_id = stage_id
        self.stage_step = 0
        self.stable_count = 0
        if STAGE_NAMES[stage_id] == "close":
            offset = torch.tensor(self.cfg.grasp_offset, device=object_pos.device, dtype=torch.float32)
            self.grasp_target = object_pos.detach().clone() + offset
        elif STAGE_NAMES[stage_id] == "lift":
            self.lift_target = palm_pos.detach().clone()
            self.lift_target[2] += self.cfg.lift_height
        elif STAGE_NAMES[stage_id] == "retreat":
            offset = torch.tensor(
                self.cfg.retreat_offset,
                device=palm_pos.device,
                dtype=torch.float32,
            )
            self.retreat_target = palm_pos.detach().clone() + offset

    def _stable(self, condition: bool) -> bool:
        self.stable_count = self.stable_count + 1 if condition else 0
        return self.stable_count >= self.cfg.stable_steps

    def _target_for_stage(
        self, palm_pos: torch.Tensor, object_pos: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        device = palm_pos.device
        target_pos = self.target_pos.to(device=device)
        if self.stage_id == 0:
            desired = self.initial_palm_reference.to(device=device)
            grip = 0.0
        elif self.stage_id == 1:
            desired = object_pos + torch.tensor(self.cfg.pregrasp_offset, device=device)
            grip = 0.0
        elif self.stage_id == 2:
            desired = object_pos + torch.tensor(self.cfg.grasp_offset, device=device)
            grip = 0.0
        elif self.stage_id == 3:
            desired = self.grasp_target.to(device=device)
            grip = 1.0
        elif self.stage_id == 4:
            desired = self.lift_target.to(device=device)
            grip = 1.0
        elif self.stage_id == 5:
            desired = target_pos + torch.tensor(self.cfg.carry_palm_offset, device=device)
            grip = 1.0
        elif self.stage_id == 6:
            desired = target_pos + torch.tensor(self.cfg.lower_palm_offset, device=device)
            grip = 1.0
        elif self.stage_id == 7:
            desired = target_pos + torch.tensor(self.cfg.lower_palm_offset, device=device)
            grip = 0.0
        else:
            desired = self.retreat_target.to(device=device)
            grip = 0.0
        return desired, grip

    def _object_in_target(self, object_pos: torch.Tensor) -> bool:
        target = self.target_pos.to(device=object_pos.device)
        size = self.target_size.to(device=object_pos.device)
        half_extent = torch.clamp(size[:2] * 0.5 - self.cfg.target_margin_xy, min=0.01)
        return bool(torch.all(torch.abs(object_pos[:2] - target[:2]) <= half_extent).item())

    def _advance(
        self,
        palm_pos: torch.Tensor,
        palm_quat: torch.Tensor,
        palm_velocity: torch.Tensor,
        object_pos: torch.Tensor,
        object_velocity: torch.Tensor,
    ) -> None:
        desired, _ = self._target_for_stage(palm_pos, object_pos)
        target_error = float(torch.linalg.vector_norm(palm_pos - desired).item())
        desired_quat = self.desired_palm_quat_w.to(device=palm_pos.device)
        quat_error = math_utils.quat_mul(desired_quat.unsqueeze(0), math_utils.quat_inv(palm_quat.unsqueeze(0)))
        orientation_error = float(torch.linalg.vector_norm(math_utils.axis_angle_from_quat(quat_error)[0]).item())
        palm_speed = float(torch.linalg.vector_norm(palm_velocity[:3]).item())
        object_speed = float(torch.linalg.vector_norm(object_velocity[:3]).item())
        palm_object_distance = float(torch.linalg.vector_norm(palm_pos - object_pos).item())
        object_lift = float(object_pos[2].item() - self.initial_object_height)
        in_target = self._object_in_target(object_pos)
        self.last_metrics = {
            "target_error": target_error,
            "orientation_error": orientation_error,
            "palm_speed": palm_speed,
            "object_speed": object_speed,
            "palm_object_distance": palm_object_distance,
            "object_lift": object_lift,
            "in_target": in_target,
        }

        ready = False
        if self.stage_id == 0:
            ready = self.stage_step >= self.cfg.settle_steps
        elif self.stage_id == 1:
            ready = self._stable(
                target_error <= self.cfg.pregrasp_tolerance
                and orientation_error <= self.cfg.orientation_tolerance
                and palm_speed <= 0.08
            )
        elif self.stage_id == 2:
            ready = self._stable(
                target_error <= self.cfg.approach_tolerance
                and orientation_error <= self.cfg.orientation_tolerance
                and palm_speed <= 0.06
            )
        elif self.stage_id == 3:
            ready = self._stable(
                self.stage_step >= self.cfg.close_steps and palm_object_distance <= 0.075 and palm_speed <= 0.08
            )
        elif self.stage_id == 4:
            ready = self._stable(object_lift >= 0.035 and target_error <= self.cfg.motion_tolerance)
        elif self.stage_id == 5:
            # The strict lift milestone was already proven in stage 4.  During
            # transport the compliant fingers sag slightly, so this gate only
            # verifies that the object remains held above the table.
            ready = self._stable(
                in_target and object_lift >= self.cfg.carry_min_lift and target_error <= self.cfg.motion_tolerance
            )
        elif self.stage_id == 6:
            ready = self._stable(
                in_target
                and object_pos[2].item() <= self.initial_object_height + 0.020
                and target_error <= self.cfg.motion_tolerance
                and object_speed <= 0.15
            )
        elif self.stage_id == 7:
            ready = self._stable(
                self.stage_step >= self.cfg.release_steps and in_target and object_speed <= 0.10
            )

        if ready and self.stage_id < len(STAGE_NAMES) - 1:
            self._transition(self.stage_id + 1, palm_pos, object_pos)

    def command(self, env, action_term) -> torch.Tensor:
        """Return one action after updating state-gated stage transitions."""

        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        obj = unwrapped.scene["object"]
        palm_ids, _ = robot.find_bodies(action_term.cfg.body_name, preserve_order=True)
        palm_pos_w = robot.data.body_pos_w[0, palm_ids[0]]
        palm_quat_w = robot.data.body_quat_w[0, palm_ids[0]]
        palm_pos = palm_pos_w - unwrapped.scene.env_origins[0]
        palm_velocity = robot.data.body_vel_w[0, palm_ids[0]]
        object_pos = obj.data.root_pos_w[0] - unwrapped.scene.env_origins[0]
        object_velocity = obj.data.root_vel_w[0]

        if self.initial_palm_reference is None:
            self.initial_palm_reference = palm_pos.detach().clone()

        self._advance(palm_pos, palm_quat_w, palm_velocity, object_pos, object_velocity)
        desired_pos, grip = self._target_for_stage(palm_pos, object_pos)
        action = torch.zeros((1, 7), device=unwrapped.device, dtype=torch.float32)

        if hasattr(action_term, "encode_env_pose"):
            action[0, :6] = action_term.encode_env_pose(desired_pos.unsqueeze(0))[0]
        elif self.stage_id != 0:
            desired_pos_w = desired_pos + unwrapped.scene.env_origins[0]
            desired_quat_w = self.desired_palm_quat_w.to(device=unwrapped.device)
            palm_pos_b, palm_quat_b = math_utils.subtract_frame_transforms(
                robot.data.root_pos_w[0:1],
                robot.data.root_quat_w[0:1],
                palm_pos_w.unsqueeze(0),
                palm_quat_w.unsqueeze(0),
            )
            desired_pos_b, desired_quat_b = math_utils.subtract_frame_transforms(
                robot.data.root_pos_w[0:1],
                robot.data.root_quat_w[0:1],
                desired_pos_w.unsqueeze(0),
                desired_quat_w.unsqueeze(0),
            )
            position_error = desired_pos_b[0] - palm_pos_b[0]
            quat_error = math_utils.quat_mul(desired_quat_b, math_utils.quat_inv(palm_quat_b))
            orientation_error = math_utils.axis_angle_from_quat(quat_error)[0]
            scale = action_term._scale[0]
            action[0, :3] = torch.clamp(position_error / scale[:3], -1.0, 1.0)
            action[0, 3:6] = torch.clamp(orientation_error / scale[3:6], -1.0, 1.0)
        action[0, 6] = grip
        self.stage_step += 1
        return action
