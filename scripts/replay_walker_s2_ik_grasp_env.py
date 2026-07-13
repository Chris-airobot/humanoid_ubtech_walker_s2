#!/usr/bin/env python3
"""Replay the old Walker S2 IK grasp sequence in the IsaacLab scene.

Run from this repository root:

    /home/chris/IsaacLab/isaaclab.sh -p scripts/replay_walker_s2_ik_grasp_env.py
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import traceback
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = (
    REPO_ROOT
    / "assets"
    / "resources"
    / "walker_s2_description_hand3_v1_left_hand3_v1_right"
    / "walker_s2_description_hand3_v1_left_hand3_v1_right_isaac_simple_hand_collision.urdf"
)


parser = argparse.ArgumentParser(description="Replay old Walker S2 IK grasp targets in the IsaacLab env.")
parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Walker S2 URDF used by Pinocchio IK.")
parser.add_argument("--num_envs", type=int, default=1, help="Only 1 is supported for IK replay.")
parser.add_argument("--open_steps", type=int, default=180, help="Initial steps to force and hold the right hand open.")
parser.add_argument(
    "--ready_settle_steps",
    "--ready-settle-steps",
    type=int,
    default=180,
    help="Steps to hold the pre-trigger pose as a target before waiting for G.",
)
parser.add_argument("--pregrasp_steps", type=int, default=240)
parser.add_argument("--approach_steps", type=int, default=180)
parser.add_argument("--close_steps", type=int, default=240)
parser.add_argument(
    "--post_close_hold_steps",
    "--post-close-hold-steps",
    type=int,
    default=120,
    help="Steps to hold the squeezed grasp before lifting so finger/object contacts settle.",
)
parser.add_argument("--lift_steps", type=int, default=180)
parser.add_argument("--hold_steps", type=int, default=240)
parser.add_argument("--print_every", type=int, default=60)
parser.add_argument(
    "--grip_strategy",
    "--grip-strategy",
    choices=("all_together", "fingers_then_thumb"),
    default="all_together",
    help="Close all fingers together, or close non-thumb fingers first and bring the thumb in later.",
)
parser.add_argument(
    "--thumb_start_fraction",
    "--thumb-start-fraction",
    type=float,
    default=0.45,
    help="For --grip_strategy fingers_then_thumb, fraction of close phase before the thumb starts closing.",
)
parser.add_argument(
    "--thumb_close_scale",
    "--thumb-close-scale",
    type=float,
    default=1.0,
    help="Scale thumb closing distance from open to closed target. Use <1.0 if the thumb tips the object.",
)
parser.add_argument(
    "--finger_close_scale",
    "--finger-close-scale",
    type=float,
    default=1.15,
    help="Scale non-thumb finger closing distance from open to closed target. Use >1.0 for extra squeeze.",
)
parser.add_argument(
    "--debug_ready_pose",
    "--debug-ready-pose",
    action="store_true",
    help="Print full ready-pose diagnostics before waiting for G.",
)
parser.add_argument(
    "--debug_settle_trace",
    "--debug-settle-trace",
    action="store_true",
    help="Print target-vs-actual joint/body diagnostics during the pre-G settle hold.",
)
parser.add_argument(
    "--settle_trace_steps",
    "--settle-trace-steps",
    type=int,
    nargs="*",
    default=[0, 1, 5, 10, 30, 60, 120, 180],
    help="Ready-settle step indices to print when --debug-settle-trace is enabled.",
)
parser.add_argument(
    "--auto_start",
    "--auto-start",
    action="store_true",
    help="Start the grasp immediately. By default, GUI runs wait for G like the old fixed demo.",
)
parser.add_argument(
    "--hold_pose",
    "--hold-pose",
    choices=("pregrasp", "ready"),
    default="ready",
    help="Pose to hold before G. The old fixed demo holds 'ready' before the G-triggered trajectory.",
)
parser.add_argument(
    "--control_profile",
    "--control-profile",
    choices=("demo_sanity", "env"),
    default="demo_sanity",
    help=(
        "demo_sanity uses strong fixed-base manipulation settings so the old demo trajectory can be "
        "validated inside IsaacLab. env keeps the normal Walker S2 env actuator/gravity settings."
    ),
)
parser.add_argument(
    "--hand_physics",
    "--hand-physics",
    choices=("sanity", "dynamic", "max_grip"),
    default="sanity",
    help=(
        "sanity keeps strong demo hand drives. dynamic keeps the working demo arm profile, "
        "but re-enables hand rigid-body gravity and uses normal hand actuator limits. "
        "max_grip also re-enables hand gravity but uses stronger hand drives for grasp-hold debugging."
    ),
)
parser.add_argument(
    "--show_hand_colliders",
    "--show-hand-colliders",
    action="store_true",
    help="Draw red debug boxes for the simplified hand colliders used by the fixed grasp demo.",
)
parser.add_argument(
    "--debug_collider_visual_scale",
    "--debug-collider-visual-scale",
    type=float,
    default=1.0,
    help="Visual-only scale for the red hand collider debug boxes.",
)
parser.add_argument(
    "--robot_z",
    "--robot-z",
    type=float,
    default=0.91,
    help="IsaacLab USD root z for Walker S2. Use 0.86 only when comparing against the old fixed demo importer.",
)
parser.add_argument("--pregrasp_distance", "--pregrasp-distance", type=float, default=0.08)
parser.add_argument(
    "--grasp_clearance",
    "--grasp-clearance",
    type=float,
    default=0.0,
    help="Palm-normal retreat before grasp. Negative values move the palm deeper into the object.",
)
parser.add_argument("--lift_height", "--lift-height", type=float, default=0.08)
parser.add_argument("--palm_tcp_offset", "--palm-tcp-offset", type=float, nargs=3, default=(0.005, -0.018, 0.025))
parser.add_argument("--palm_world_nudge", "--palm-world-nudge", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument(
    "--object_world_offset",
    "--object-world-offset",
    type=float,
    nargs=3,
    default=(-0.03, 0.0, 0.0),
    help="World-frame offset applied only to the placed object. Default moves it left by 3 cm.",
)
parser.add_argument(
    "--object_palm_offset",
    "--object-palm-offset",
    type=float,
    nargs=3,
    default=None,
    help="Optional object center in the solved actual palm frame, matching the fixed demo option.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab_walker_s2.tasks.pick_place.walker_s2_pick_place_env_cfg import (  # noqa: E402
    RIGHT_ARM_JOINTS,
    RIGHT_HAND_JOINTS,
    RIGHT_HAND_CLOSE_COMMAND,
    RIGHT_HAND_OPEN_COMMAND,
    WalkerS2PickPlaceEnvCfg,
)


def _load_fixed_grasp_demo_module():
    module_path = REPO_ROOT / "scripts" / "isaac_walker_fixed_grasp_demo.py"
    spec = importlib.util.spec_from_file_location("walker_fixed_grasp_demo", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fixed_grasp_demo = _load_fixed_grasp_demo_module()
rotation_z = _fixed_grasp_demo.rotation_z
solve_right_arm_to_cube = _fixed_grasp_demo.solve_right_arm_to_cube
is_hand_link_name = _fixed_grasp_demo.is_hand_link_name
hand_collider_box_size = _fixed_grasp_demo.hand_collider_box_size
hand_collider_origin = _fixed_grasp_demo.hand_collider_origin


FIXED_DEMO_READY_ARM_POSE = {
    "L_shoulder_pitch_joint": 0.09322471888572098,
    "L_shoulder_roll_joint": -0.5933223843430208,
    "L_shoulder_yaw_joint": -1.595878574835185,
    "L_elbow_roll_joint": -1.8963565338596158,
    "L_elbow_yaw_joint": 1.4000461262831179,
    "L_wrist_pitch_joint": -0.00048740902645395785,
    "L_wrist_roll_joint": 0.0998718010009366,
    "R_shoulder_pitch_joint": -0.09321727661087699,
    "R_shoulder_roll_joint": -0.5933455607833843,
    "R_shoulder_yaw_joint": 1.595869459316937,
    "R_elbow_roll_joint": -1.8963607249359917,
    "R_elbow_yaw_joint": -1.4000874256427638,
    "R_wrist_pitch_joint": 0.00048144049606466176,
    "R_wrist_roll_joint": 0.09985407619802703,
    "head_pitch_joint": -0.785398163,
    "head_yaw_joint": 1.9677590016147396e-07,
}

READY_DEBUG_JOINTS = [
    "waist_yaw_joint",
    "waist_pitch_joint",
    "L_shoulder_pitch_joint",
    "L_shoulder_roll_joint",
    "L_shoulder_yaw_joint",
    "L_elbow_roll_joint",
    "L_elbow_yaw_joint",
    "L_wrist_pitch_joint",
    "L_wrist_roll_joint",
    "R_shoulder_pitch_joint",
    "R_shoulder_roll_joint",
    "R_shoulder_yaw_joint",
    "R_elbow_roll_joint",
    "R_elbow_yaw_joint",
    "R_wrist_pitch_joint",
    "R_wrist_roll_joint",
]

READY_DEBUG_BODIES = [
    "L_shoulder_pitch_link",
    "L_elbow_yaw_link",
    "L_wrist_roll_link",
    "hand3_v1_left_L_palm_link",
    "R_shoulder_pitch_link",
    "R_elbow_yaw_link",
    "R_wrist_roll_link",
    "hand3_v1_right_R_palm_link",
]


def _smooth_step(alpha: float) -> float:
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _as_env_pos(env: ManagerBasedRLEnv, world_pos: torch.Tensor) -> np.ndarray:
    return (world_pos[0] - env.scene.env_origins[0]).detach().cpu().numpy()


def _object_pos(env: ManagerBasedRLEnv) -> np.ndarray:
    return _as_env_pos(env, env.scene["object"].data.root_pos_w)


def _set_object_world_pos(env: ManagerBasedRLEnv, world_pos: np.ndarray) -> None:
    obj = env.scene["object"]
    root_state = obj.data.root_state_w.clone()
    root_state[0, :3] = torch.tensor(world_pos, device=env.device, dtype=torch.float32)
    root_state[0, 7:] = 0.0
    obj.write_root_state_to_sim(root_state)


def _palm_pose_to_object_world(palm_pos_world: np.ndarray, palm_R_world: np.ndarray, object_palm_offset: np.ndarray):
    return np.asarray(palm_pos_world, dtype=float) + np.asarray(palm_R_world, dtype=float) @ np.asarray(
        object_palm_offset, dtype=float
    )


def _palm_pos(env: ManagerBasedRLEnv) -> np.ndarray:
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies("hand3_v1_right_R_palm_link", preserve_order=True)
    return _as_env_pos(env, robot.data.body_pos_w[:, body_ids[0]])


def _right_arm_actual(env: ManagerBasedRLEnv) -> np.ndarray:
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(RIGHT_ARM_JOINTS, preserve_order=True)
    return robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().astype(float)


def _right_hand_actual(env: ManagerBasedRLEnv) -> np.ndarray:
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(RIGHT_HAND_JOINTS, preserve_order=True)
    return robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().astype(float)


def _joint_values(env: ManagerBasedRLEnv, joint_names: list[str]) -> np.ndarray:
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
    return robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().astype(float)


def _joint_targets(q_target: np.ndarray, dof_names: list[str], joint_names: list[str]) -> np.ndarray:
    return np.asarray([q_target[dof_names.index(joint_name)] for joint_name in joint_names], dtype=float)


def _apply_named_pose(q: np.ndarray, dof_names: list[str], values: dict[str, float]) -> np.ndarray:
    q_named = q.copy()
    missing = []
    for joint_name, value in values.items():
        if joint_name in dof_names:
            q_named[dof_names.index(joint_name)] = float(value)
        else:
            missing.append(joint_name)
    if missing:
        print(f"[WARN] Missing fixed-demo joints, skipped: {missing}", flush=True)
    return q_named


def _left_hand_open_command() -> dict[str, float]:
    return {
        name.replace("hand3_v1_right_R_", "hand3_v1_left_L_"): (
            -value if "thumb_cmp_joint" in name else value
        )
        for name, value in RIGHT_HAND_OPEN_COMMAND.items()
    }


def _fixed_demo_ready_open_q(q_home: np.ndarray, dof_names: list[str]) -> np.ndarray:
    q_ready = _apply_named_pose(q_home, dof_names, FIXED_DEMO_READY_ARM_POSE)
    q_ready = _apply_named_pose(q_ready, dof_names, _left_hand_open_command())
    q_ready = _apply_named_pose(q_ready, dof_names, RIGHT_HAND_OPEN_COMMAND)
    return q_ready


def _robot_prim_roots(env: ManagerBasedRLEnv) -> list[str]:
    robot = env.scene["robot"]
    roots = []
    root_view = getattr(robot, "root_physx_view", None)
    prim_paths = getattr(root_view, "prim_paths", None)
    if prim_paths:
        roots.append(str(prim_paths[0]))
    cfg_prim_path = str(getattr(robot.cfg, "prim_path", ""))
    if cfg_prim_path:
        roots.append(cfg_prim_path.replace("env_.*", "env_0").replace("{ENV_REGEX_NS}", "/World/envs/env_0"))
    roots.append("/World/envs/env_0/Robot")

    unique_roots = []
    for root in roots:
        if root and root not in unique_roots:
            unique_roots.append(root)
    return unique_roots


def _set_actuator_cfg_value(actuator_cfg, attr_name: str, value: float) -> None:
    if hasattr(actuator_cfg, attr_name):
        setattr(actuator_cfg, attr_name, value)


def _apply_demo_sanity_control_profile(cfg: WalkerS2PickPlaceEnvCfg, hand_physics: str) -> None:
    """Make the IsaacLab asset suitable for validating the old fixed-demo trajectory.

    The old standalone demo imports the URDF and drives full joint-position
    targets without the same IsaacLab actuator effort clipping. This profile is
    intentionally for sanity replay, not the final dynamics profile.
    """

    if hasattr(cfg.scene.robot, "copy"):
        cfg.scene.robot = cfg.scene.robot.copy()

    cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
    cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4

    for name, actuator_cfg in cfg.scene.robot.actuators.items():
        if name in ("hands",):
            if hand_physics == "max_grip":
                _set_actuator_cfg_value(actuator_cfg, "effort_limit", 2000.0)
                _set_actuator_cfg_value(actuator_cfg, "effort_limit_sim", 2000.0)
                actuator_cfg.stiffness = 500.0
                actuator_cfg.damping = 30.0
            elif hand_physics == "dynamic":
                _set_actuator_cfg_value(actuator_cfg, "effort_limit", 80.0)
                _set_actuator_cfg_value(actuator_cfg, "effort_limit_sim", 80.0)
                actuator_cfg.stiffness = 120.0
                actuator_cfg.damping = 8.0
            else:
                _set_actuator_cfg_value(actuator_cfg, "effort_limit", 500.0)
                _set_actuator_cfg_value(actuator_cfg, "effort_limit_sim", 500.0)
                actuator_cfg.stiffness = 160.0
                actuator_cfg.damping = 10.0
        elif name in ("head",):
            _set_actuator_cfg_value(actuator_cfg, "effort_limit", 100.0)
            _set_actuator_cfg_value(actuator_cfg, "effort_limit_sim", 100.0)
            actuator_cfg.stiffness = 80.0
            actuator_cfg.damping = 8.0
        else:
            _set_actuator_cfg_value(actuator_cfg, "effort_limit", 5000.0)
            _set_actuator_cfg_value(actuator_cfg, "effort_limit_sim", 5000.0)
            if "shoulder" in name or "elbow" in name or "wrist" in name:
                actuator_cfg.stiffness = 3000.0
                actuator_cfg.damping = 180.0
            elif name == "waist":
                actuator_cfg.stiffness = 3000.0
                actuator_cfg.damping = 180.0


def _enable_hand_rigid_body_gravity(env: ManagerBasedRLEnv) -> int:
    """Re-enable gravity on hand rigid bodies after demo_sanity disabled robot gravity."""

    from pxr import PhysxSchema, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    robot_roots = _robot_prim_roots(env)

    count = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_name = prim.GetName()
        if not any(prim_path.startswith(root) for root in robot_roots):
            continue
        if not (prim_name.startswith("hand3_v1_left") or prim_name.startswith("hand3_v1_right")):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        physx_rb = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not physx_rb:
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rb.CreateDisableGravityAttr(False)
        physx_rb.GetDisableGravityAttr().Set(False)
        count += 1
    return count


def _draw_hand_collider_debug_boxes(env: ManagerBasedRLEnv, visual_scale: float) -> int:
    """Draw visual-only red boxes matching the fixed demo's simplified hand colliders."""

    from pxr import Gf, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    robot_roots = _robot_prim_roots(env)
    count = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        name = prim.GetName()
        if not any(prim_path.startswith(root) for root in robot_roots):
            continue
        if not is_hand_link_name(name):
            continue

        size = np.fromstring(hand_collider_box_size(name), sep=" ", dtype=float)
        if size.size != 3:
            continue
        origin = hand_collider_origin(name)
        box_path = f"{prim_path}/debug_collider_box"
        box = UsdGeom.Cube.Define(stage, box_path)
        box.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(box.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(origin[0]), float(origin[1]), float(origin[2])))
        xform.AddScaleOp().Set(
            Gf.Vec3f(
                float(size[0] * visual_scale),
                float(size[1] * visual_scale),
                float(size[2] * visual_scale),
            )
        )
        box.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.0, 0.0)])
        box.CreateDisplayOpacityAttr([0.85])
        count += 1
    return count


def _right_arm_target(q_target: np.ndarray, dof_names: list[str]) -> np.ndarray:
    return np.asarray([q_target[dof_names.index(joint_name)] for joint_name in RIGHT_ARM_JOINTS], dtype=float)


def _q_tensor(env: ManagerBasedRLEnv, q: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(q, dtype=np.float32), device=env.device).unsqueeze(0)


def _write_full_body_state(env: ManagerBasedRLEnv, q: np.ndarray) -> None:
    """Reset actual joint state and target to a full-body pose."""
    robot = env.scene["robot"]
    q_tensor = _q_tensor(env, q)
    robot.write_joint_state_to_sim(q_tensor, torch.zeros_like(q_tensor))
    robot.set_joint_position_target(q_tensor)
    env.scene.write_data_to_sim()
    env.scene.update(dt=env.physics_dt)


def _apply_full_body_target(env: ManagerBasedRLEnv, q: np.ndarray) -> None:
    """Match the old demo's robot.apply_action(joint_positions=q) control style."""
    env.scene["robot"].set_joint_position_target(_q_tensor(env, q))


def _step_full_body_target(env: ManagerBasedRLEnv, q: np.ndarray) -> None:
    _apply_full_body_target(env, q)
    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    for _ in range(env.cfg.decimation):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    if is_rendering:
        env.sim.render()


def _print_settle_trace(env: ManagerBasedRLEnv, label: str, step: int, q_target: np.ndarray, dof_names: list[str]) -> None:
    robot = env.scene["robot"]
    actual_q = robot.data.joint_pos[0].detach().cpu().numpy().astype(float)
    actual_v = robot.data.joint_vel[0].detach().cpu().numpy().astype(float)
    root_pos = (robot.data.root_pos_w[0] - env.scene.env_origins[0]).detach().cpu().numpy().astype(float)

    print(
        f"[SETTLE_TRACE {label} {step:04d}] "
        f"root={root_pos.tolist()} "
        f"max_abs_joint_error={float(np.max(np.abs(actual_q - q_target))):.6f} "
        f"max_abs_joint_vel={float(np.max(np.abs(actual_v))):.6f}",
        flush=True,
    )
    for body_name in (
        "L_wrist_roll_link",
        "hand3_v1_left_L_palm_link",
        "R_wrist_roll_link",
        "hand3_v1_right_R_palm_link",
    ):
        body_ids, _ = robot.find_bodies(body_name, preserve_order=True)
        if body_ids:
            pos = _as_env_pos(env, robot.data.body_pos_w[:, body_ids[0]])
            print(f"  body {body_name} pos={pos.tolist()}", flush=True)

    left_hand_joints = list(_left_hand_open_command().keys())
    for group_name, joint_names in (
        ("left_arm", [name.replace("R_", "L_", 1) for name in RIGHT_ARM_JOINTS]),
        ("right_arm", RIGHT_ARM_JOINTS),
        ("left_hand", left_hand_joints),
        ("right_hand", RIGHT_HAND_JOINTS),
    ):
        target = _joint_targets(q_target, dof_names, joint_names)
        actual = _joint_values(env, joint_names)
        error = actual - target
        print(
            f"  {group_name}_target={target.tolist()}\n"
            f"  {group_name}_actual={actual.tolist()}\n"
            f"  {group_name}_error={error.tolist()}",
            flush=True,
        )


def _settle_full_body_target(env: ManagerBasedRLEnv, q: np.ndarray, steps: int, dof_names: list[str]) -> None:
    trace_steps = set(args_cli.settle_trace_steps)
    if args_cli.debug_settle_trace and 0 in trace_steps:
        _print_settle_trace(env, "READY_HOLD", 0, q, dof_names)
    for step in range(1, steps + 1):
        _step_full_body_target(env, q)
        if args_cli.debug_settle_trace and step in trace_steps:
            _print_settle_trace(env, "READY_HOLD", step, q, dof_names)


def _print_state(
    env: ManagerBasedRLEnv,
    label: str,
    step: int,
    reward: torch.Tensor | None = None,
    q_target: np.ndarray | None = None,
    dof_names: list[str] | None = None,
    arm_raw: np.ndarray | None = None,
    hand_target: torch.Tensor | None = None,
) -> None:
    obj = _object_pos(env)
    palm = _palm_pos(env)
    msg = (
        f"[{label} {step:04d}] "
        f"object={obj.tolist()} palm={palm.tolist()} object_height={obj[2]:.4f}"
    )
    if reward is not None:
        msg += f" reward={reward.detach().cpu().tolist()}"
    print(msg, flush=True)
    if q_target is not None and dof_names is not None:
        target = _right_arm_target(q_target, dof_names)
        actual = _right_arm_actual(env)
        error = actual - target
        print(f"  right_arm_target={target.tolist()}", flush=True)
        print(f"  right_arm_actual={actual.tolist()}", flush=True)
        print(f"  right_arm_error={error.tolist()}", flush=True)
    if arm_raw is not None:
        print(f"  right_arm_raw_action={arm_raw.tolist()}", flush=True)
    right_hand_actual = _right_hand_actual(env)
    print(f"  right_hand_actual={right_hand_actual.tolist()}", flush=True)
    if hand_target is not None:
        target = hand_target.detach().cpu().numpy()[0].astype(float)
        print(f"  right_hand_target={target.tolist()}", flush=True)
        print(f"  right_hand_error={(right_hand_actual - target).tolist()}", flush=True)


def _print_pose_consistency(
    env: ManagerBasedRLEnv,
    label: str,
    q_target: np.ndarray,
    dof_names: list[str],
    expected_palm_world: np.ndarray,
) -> None:
    actual_palm = _palm_pos(env)
    print(
        f"[CHECK {label}] ik_predicted_palm={np.asarray(expected_palm_world, dtype=float).tolist()} "
        f"isaaclab_actual_palm={actual_palm.tolist()} "
        f"palm_error={(actual_palm - np.asarray(expected_palm_world, dtype=float)).tolist()}",
        flush=True,
    )
    _print_state(env, label, 0, q_target=q_target, dof_names=dof_names)


def _print_ready_pose_debug(env: ManagerBasedRLEnv, label: str, q_target: np.ndarray, dof_names: list[str]) -> None:
    robot = env.scene["robot"]
    actual_q = robot.data.joint_pos[0].detach().cpu().numpy().astype(float)
    print(f"[READY_DEBUG {label}] num_joints={len(dof_names)}", flush=True)
    for joint_name in READY_DEBUG_JOINTS:
        if joint_name not in dof_names:
            print(f"[READY_DEBUG {label}] joint {joint_name}: missing", flush=True)
            continue
        index = dof_names.index(joint_name)
        print(
            f"[READY_DEBUG {label}] joint {joint_name}: "
            f"target={float(q_target[index]): .9f} actual={float(actual_q[index]): .9f} "
            f"error={float(actual_q[index] - q_target[index]): .9f}",
            flush=True,
        )
    for body_name in READY_DEBUG_BODIES:
        body_ids, _ = robot.find_bodies(body_name, preserve_order=True)
        if not body_ids:
            print(f"[READY_DEBUG {label}] body {body_name}: missing", flush=True)
            continue
        pos = _as_env_pos(env, robot.data.body_pos_w[:, body_ids[0]])
        print(f"[READY_DEBUG {label}] body {body_name}: pos={pos.tolist()}", flush=True)
    print(f"[READY_DEBUG {label}] full_q_target={q_target.tolist()}", flush=True)
    print(f"[READY_DEBUG {label}] full_q_actual={actual_q.tolist()}", flush=True)


def _run_full_q_phase(
    env: ManagerBasedRLEnv,
    label: str,
    q_start: np.ndarray,
    q_end: np.ndarray,
    dof_names: list[str],
    steps: int,
    hand_target_indices: np.ndarray | None = None,
) -> np.ndarray:
    for i in range(steps):
        alpha = _smooth_step((i + 1) / float(steps))
        q_target = (1.0 - alpha) * q_start + alpha * q_end
        _step_full_body_target(env, q_target)
        if i % args_cli.print_every == 0 or i == steps - 1:
            hand_target = None
            if hand_target_indices is not None:
                hand_target = torch.tensor(
                    q_target[hand_target_indices], device=env.device, dtype=torch.float32
                ).unsqueeze(0)
            _print_state(env, label, i + 1, q_target=q_target, dof_names=dof_names, hand_target=hand_target)
    return q_end.copy()


def _scaled_hand_close_pos(
    right_hand_joint_names: list[str],
    right_hand_open_pos: np.ndarray,
    right_hand_close_pos: np.ndarray,
) -> np.ndarray:
    thumb_mask = np.asarray(["thumb" in joint_name for joint_name in right_hand_joint_names], dtype=bool)
    thumb_scale = min(max(float(args_cli.thumb_close_scale), 0.0), 1.5)
    finger_scale = min(max(float(args_cli.finger_close_scale), 0.0), 1.8)
    close_pos = right_hand_open_pos.copy()
    close_pos[thumb_mask] = right_hand_open_pos[thumb_mask] + thumb_scale * (
        right_hand_close_pos[thumb_mask] - right_hand_open_pos[thumb_mask]
    )
    close_pos[~thumb_mask] = right_hand_open_pos[~thumb_mask] + finger_scale * (
        right_hand_close_pos[~thumb_mask] - right_hand_open_pos[~thumb_mask]
    )
    return close_pos


def _run_full_q_hand_close_phase(
    env: ManagerBasedRLEnv,
    q_grasp_open: np.ndarray,
    right_hand_indices: np.ndarray,
    right_hand_joint_names: list[str],
    right_hand_open_pos: np.ndarray,
    right_hand_close_pos: np.ndarray,
    dof_names: list[str],
    steps: int,
) -> np.ndarray:
    q_target = q_grasp_open.copy()
    thumb_mask = np.asarray(["thumb" in joint_name for joint_name in right_hand_joint_names], dtype=bool)
    thumb_start = min(max(float(args_cli.thumb_start_fraction), 0.0), 0.95)
    scaled_close_pos = _scaled_hand_close_pos(right_hand_joint_names, right_hand_open_pos, right_hand_close_pos)
    for i in range(steps):
        alpha = _smooth_step((i + 1) / float(steps))
        hand_alpha = np.full_like(right_hand_open_pos, alpha, dtype=float)
        if args_cli.grip_strategy == "fingers_then_thumb":
            thumb_alpha = 0.0
            if alpha > thumb_start:
                thumb_alpha = _smooth_step((alpha - thumb_start) / (1.0 - thumb_start))
            hand_alpha[thumb_mask] = thumb_alpha
        hand_pos = right_hand_open_pos + hand_alpha * (scaled_close_pos - right_hand_open_pos)
        q_target = q_grasp_open.copy()
        q_target[right_hand_indices] = hand_pos
        _step_full_body_target(env, q_target)
        if i % args_cli.print_every == 0 or i == steps - 1:
            hand_target = torch.tensor(hand_pos, device=env.device, dtype=torch.float32).unsqueeze(0)
            _print_state(env, "CLOSE", i + 1, q_target=q_target, dof_names=dof_names, hand_target=hand_target)
    return q_target.copy()


def _run_full_q_hold_phase(
    env: ManagerBasedRLEnv,
    label: str,
    q_target: np.ndarray,
    dof_names: list[str],
    steps: int,
    hand_target_indices: np.ndarray | None = None,
) -> np.ndarray:
    for i in range(steps):
        _step_full_body_target(env, q_target)
        if i % args_cli.print_every == 0 or i == steps - 1:
            hand_target = None
            if hand_target_indices is not None:
                hand_target = torch.tensor(
                    q_target[hand_target_indices], device=env.device, dtype=torch.float32
                ).unsqueeze(0)
            _print_state(env, label, i + 1, q_target=q_target, dof_names=dof_names, hand_target=hand_target)
    return q_target.copy()


def _wait_for_grasp_key(env: ManagerBasedRLEnv, q_ready: np.ndarray, dof_names: list[str]) -> None:
    if args_cli.headless or args_cli.auto_start:
        print("[INFO] Auto-starting grasp replay.", flush=True)
        return

    from walker_s2_grasp_sim import WalkerS2GraspKeyboard

    keyboard = WalkerS2GraspKeyboard()
    keyboard.connect()
    print("[TELEOP] Holding ready pose. Press G to start grasp replay, or Q to quit.", flush=True)
    try:
        while simulation_app.is_running():
            command = keyboard.sample()
            if command.quit:
                raise KeyboardInterrupt("Replay cancelled by Q")
            if command.assisted_grasp:
                print("[TELEOP] G pressed. Starting grasp replay.", flush=True)
                return
            _step_full_body_target(env, q_ready)
    finally:
        keyboard.close()


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("This IK replay script currently supports --num_envs 1 only.")
    if not args_cli.urdf.is_file():
        raise FileNotFoundError(args_cli.urdf)

    cfg = WalkerS2PickPlaceEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device
    cfg.episode_length_s = 40.0
    cfg.events.reset_object_position = None
    cfg.scene.robot.init_state.pos = (0.75, -0.2, args_cli.robot_z)
    if args_cli.control_profile == "demo_sanity":
        _apply_demo_sanity_control_profile(cfg, args_cli.hand_physics)
    env = ManagerBasedRLEnv(cfg)
    env.reset()
    hand_gravity_count = 0
    if args_cli.control_profile == "demo_sanity" and args_cli.hand_physics in ("dynamic", "max_grip"):
        hand_gravity_count = _enable_hand_rigid_body_gravity(env)
    hand_collider_debug_count = 0
    if args_cli.show_hand_colliders:
        hand_collider_debug_count = _draw_hand_collider_debug_boxes(env, args_cli.debug_collider_visual_scale)

    robot = env.scene["robot"]
    dof_names = list(robot.joint_names)
    q_home = robot.data.default_joint_pos[0].detach().cpu().numpy().astype(float)
    q_ready = _fixed_demo_ready_open_q(q_home, dof_names)

    original_object_center = env.scene["object"].data.root_pos_w[0].detach().cpu().numpy().astype(float)
    robot_xyz = np.asarray(cfg.scene.robot.init_state.pos, dtype=float)
    robot_yaw_deg = 90.0
    object_palm_offset = (
        None if args_cli.object_palm_offset is None else np.asarray(args_cli.object_palm_offset, dtype=float)
    )
    object_world_offset = np.asarray(args_cli.object_world_offset, dtype=float)
    pregrasp_reference_center = original_object_center.copy()
    grasp_reference_center = original_object_center + object_world_offset

    palm_normal_world = rotation_z(math.radians(robot_yaw_deg)) @ np.array([0.0, 1.0, 0.0])
    grasp_world_nudge = np.asarray(args_cli.palm_world_nudge, dtype=float) - palm_normal_world * args_cli.grasp_clearance

    print(f"[INFO] Using IK URDF: {args_cli.urdf}", flush=True)
    print(f"[INFO] original object center: {original_object_center.tolist()}", flush=True)
    print(f"[INFO] pregrasp reference center: {pregrasp_reference_center.tolist()}", flush=True)
    print(f"[INFO] grasp/lift reference center: {grasp_reference_center.tolist()}", flush=True)
    print(f"[INFO] grasp clearance: {args_cli.grasp_clearance:.4f}", flush=True)
    print(f"[INFO] grasp world nudge: {grasp_world_nudge.tolist()}", flush=True)
    print(
        "[INFO] object palm offset: "
        f"{None if object_palm_offset is None else object_palm_offset.tolist()}",
        flush=True,
    )
    print(f"[INFO] object world offset: {object_world_offset.tolist()}", flush=True)
    print(f"[INFO] robot_xyz: {robot_xyz.tolist()}, robot_yaw_deg={robot_yaw_deg}", flush=True)
    print("[INFO] replay control path: direct full-body joint position targets", flush=True)
    print(f"[INFO] replay control profile: {args_cli.control_profile}", flush=True)
    print(f"[INFO] hand physics profile: {args_cli.hand_physics}", flush=True)
    print(
        "[INFO] grip strategy: "
        f"{args_cli.grip_strategy}, thumb_start_fraction={args_cli.thumb_start_fraction:.3f}, "
        f"thumb_close_scale={args_cli.thumb_close_scale:.3f}, "
        f"finger_close_scale={args_cli.finger_close_scale:.3f}, "
        f"post_close_hold_steps={args_cli.post_close_hold_steps}",
        flush=True,
    )
    print(
        "[INFO] robot gravity disabled: "
        f"{cfg.scene.robot.spawn.rigid_props.disable_gravity}",
        flush=True,
    )
    if args_cli.control_profile == "demo_sanity" and args_cli.hand_physics in ("dynamic", "max_grip"):
        print(f"[INFO] hand rigid bodies with gravity re-enabled: {hand_gravity_count}", flush=True)
    if args_cli.show_hand_colliders:
        print(
            "[INFO] drew hand collider debug boxes: "
            f"{hand_collider_debug_count}, visual scale={args_cli.debug_collider_visual_scale}",
            flush=True,
        )
    print(
        "[INFO] replay timing: "
        f"physics_dt={env.physics_dt:.4f}, decimation={env.cfg.decimation}, command_dt={env.step_dt:.4f}",
        flush=True,
    )
    print(f"[INFO] RL action dim in env: {env.action_manager.total_action_dim}", flush=True)
    print(f"[INFO] RL action terms in env: {env.action_manager.active_terms}", flush=True)
    print(f"[INFO] RL action dims in env: {env.action_manager.action_term_dim}", flush=True)
    print(
        "[INFO] forcing fixed-demo right hand open pose: "
        f"{[RIGHT_HAND_OPEN_COMMAND[joint_name] for joint_name in RIGHT_HAND_JOINTS]}",
        flush=True,
    )
    print("[INFO] using fixed-demo ready arm/head pose as q_ready_open", flush=True)

    q_grasp, palm_debug = solve_right_arm_to_cube(
        args_cli.urdf,
        dof_names,
        q_ready,
        grasp_reference_center,
        robot_xyz,
        robot_yaw_deg,
        args_cli.palm_tcp_offset,
        grasp_world_nudge,
    )
    object_world = grasp_reference_center.copy()
    if object_palm_offset is not None:
        object_world = _palm_pose_to_object_world(
            palm_debug["actual_pos_world"],
            palm_debug["actual_R_world"],
            object_palm_offset,
        )
    _set_object_world_pos(env, object_world)
    print(f"[INFO] IK grasp target palm world: {palm_debug['target_pos_world'].tolist()}", flush=True)
    print(f"[INFO] IK grasp predicted palm world: {palm_debug['actual_pos_world'].tolist()}", flush=True)
    print(f"[INFO] placed object world: {object_world.tolist()}", flush=True)
    print(
        "[INFO] predicted palm to placed object: "
        f"{(object_world - palm_debug['actual_pos_world']).tolist()}",
        flush=True,
    )
    pregrasp_world_nudge = grasp_world_nudge - palm_debug["target_R_world"][:, 2] * args_cli.pregrasp_distance
    q_pregrasp, pregrasp_debug = solve_right_arm_to_cube(
        args_cli.urdf,
        dof_names,
        q_ready,
        pregrasp_reference_center,
        robot_xyz,
        robot_yaw_deg,
        args_cli.palm_tcp_offset,
        pregrasp_world_nudge,
    )
    q_lift, _ = solve_right_arm_to_cube(
        args_cli.urdf,
        dof_names,
        q_grasp,
        grasp_reference_center,
        robot_xyz,
        robot_yaw_deg,
        args_cli.palm_tcp_offset,
        grasp_world_nudge + np.array([0.0, 0.0, args_cli.lift_height], dtype=float),
    )
    right_hand_indices = np.array([dof_names.index(joint_name) for joint_name in RIGHT_HAND_JOINTS], dtype=int)
    right_hand_open_pos = np.array([RIGHT_HAND_OPEN_COMMAND[joint_name] for joint_name in RIGHT_HAND_JOINTS], dtype=float)
    right_hand_close_pos = np.array(
        [RIGHT_HAND_CLOSE_COMMAND[joint_name] for joint_name in RIGHT_HAND_JOINTS], dtype=float
    )
    right_hand_hold_close_pos = _scaled_hand_close_pos(RIGHT_HAND_JOINTS, right_hand_open_pos, right_hand_close_pos)
    q_grasp_closed = q_grasp.copy()
    q_grasp_closed[right_hand_indices] = right_hand_hold_close_pos
    q_lift_closed = q_lift.copy()
    q_lift_closed[right_hand_indices] = right_hand_hold_close_pos

    q_hold = q_pregrasp.copy() if args_cli.hold_pose == "pregrasp" else q_ready.copy()
    _write_full_body_state(env, q_hold)
    print(
        f"[INFO] settling pre-trigger pose as target for {args_cli.ready_settle_steps} steps",
        flush=True,
    )
    _settle_full_body_target(env, q_hold, args_cli.ready_settle_steps, dof_names)
    if args_cli.hold_pose == "pregrasp":
        _print_pose_consistency(env, "START_PREGRASP", q_hold, dof_names, pregrasp_debug["actual_pos_world"])
    else:
        _print_state(env, "START_READY", args_cli.ready_settle_steps, q_target=q_hold, dof_names=dof_names)
    if args_cli.debug_ready_pose:
        _print_ready_pose_debug(env, "ISAACLAB_REPLAY_BEFORE_G", q_hold, dof_names)
    print(f"[INFO] holding pre-trigger pose: {args_cli.hold_pose}", flush=True)
    _wait_for_grasp_key(env, q_hold, dof_names)
    if args_cli.hold_pose == "ready":
        q_current = _run_full_q_phase(env, "PREGRASP", q_hold, q_pregrasp, dof_names, args_cli.pregrasp_steps)
    else:
        q_current = q_pregrasp.copy()
    q_current = _run_full_q_phase(env, "APPROACH", q_current, q_grasp, dof_names, args_cli.approach_steps)
    _run_full_q_hand_close_phase(
        env,
        q_grasp,
        right_hand_indices,
        RIGHT_HAND_JOINTS,
        right_hand_open_pos,
        right_hand_close_pos,
        dof_names,
        args_cli.close_steps,
    )
    _run_full_q_hold_phase(
        env,
        "SQUEEZE",
        q_grasp_closed,
        dof_names,
        args_cli.post_close_hold_steps,
        hand_target_indices=right_hand_indices,
    )
    q_current = _run_full_q_phase(
        env,
        "LIFT",
        q_grasp_closed,
        q_lift_closed,
        dof_names,
        args_cli.lift_steps,
        hand_target_indices=right_hand_indices,
    )
    _run_full_q_hold_phase(env, "HOLD", q_current, dof_names, args_cli.hold_steps)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Walker S2 IK grasp env replay failed:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
