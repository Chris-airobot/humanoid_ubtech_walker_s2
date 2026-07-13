#!/usr/bin/env python3
"""Spawn Walker S2 in a simple table-top pick/place layout.

Run from this repository root, for example:

    /home/chris/IsaacLab/isaaclab.sh -p scripts/spawn_walker_s2_table_scene.py --steps 2400
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Spawn Walker S2 with a table, cube, and target marker.")
parser.add_argument("--steps", type=int, default=2400, help="Number of simulation steps to run after reset.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Walker S2 scenes to spawn.")
parser.add_argument("--env_spacing", type=float, default=3.0, help="Spacing between spawned environments.")
parser.add_argument("--cube_center", type=float, nargs=3, default=(0.92, 0.20, 1.105))
parser.add_argument("--target_center", type=float, nargs=2, default=(0.62, 0.24))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab_walker_s2 import WALKER_S2_CFG, WALKER_S2_USD_PATH  # noqa: E402


TABLE_CENTER = (0.75, 0.30, 1.02)
TABLE_SIZE = (1.20, 0.65, 0.04)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] * 0.5
TARGET_SIZE = (0.28, 0.20, 0.01)


@configclass
class WalkerS2TableSceneCfg(InteractiveSceneCfg):
    """Minimal manipulation layout for Walker S2."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_CENTER),
    )

    target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            size=TARGET_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.7, 0.15), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(args_cli.target_center[0], args_cli.target_center[1], TABLE_TOP_Z + TARGET_SIZE[2] * 0.5)
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.07, 0.07, 0.13),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.05, 0.05), roughness=0.7),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(args_cli.cube_center)),
    )

    robot = WALKER_S2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _body_position(robot, body_name: str) -> list[list[float]]:
    body_ids, body_names = robot.find_bodies(body_name, preserve_order=True)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one body for {body_name!r}, found {body_names}")
    return robot.data.body_pos_w[:, body_ids[0]].detach().cpu().tolist()


def _reset_scene(scene: InteractiveScene) -> None:
    robot = scene["robot"]
    cube = scene["cube"]

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())

    cube_state = cube.data.default_root_state.clone()
    cube_state[:, :3] += scene.env_origins
    cube.write_root_pose_to_sim(cube_state[:, :7])
    cube.write_root_velocity_to_sim(cube_state[:, 7:])

    scene.reset()


def _print_scene_summary(scene: InteractiveScene) -> None:
    robot = scene["robot"]
    cube = scene["cube"]

    left_palm = _body_position(robot, "hand3_v1_left_L_palm_link")
    right_palm = _body_position(robot, "hand3_v1_right_R_palm_link")
    cube_pos = cube.data.root_pos_w.detach().cpu().tolist()
    target_pos = [
        [
            args_cli.target_center[0] + float(origin[0]),
            args_cli.target_center[1] + float(origin[1]),
            TABLE_TOP_Z + TARGET_SIZE[2] * 0.5,
        ]
        for origin in scene.env_origins.detach().cpu()
    ]

    print(f"[INFO] Walker S2 USD: {WALKER_S2_USD_PATH}")
    print(f"[INFO] table center={TABLE_CENTER}, size={TABLE_SIZE}, top_z={TABLE_TOP_Z:.3f}")
    print(f"[INFO] cube position(s): {cube_pos}")
    print(f"[INFO] target position(s): {target_pos}")
    print(f"[INFO] left palm position(s): {left_palm}")
    print(f"[INFO] right palm position(s): {right_palm}")


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    sim_dt = sim.get_physics_dt()
    _reset_scene(scene)
    _print_scene_summary(scene)

    robot = scene["robot"]
    for step in range(args_cli.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if step in (0, args_cli.steps - 1) or (step + 1) % 240 == 0:
            cube_pos = scene["cube"].data.root_pos_w.detach().cpu().tolist()
            right_palm = _body_position(robot, "hand3_v1_right_R_palm_link")
            print(f"[STEP {step + 1:04d}] cube_pos={cube_pos} right_palm={right_palm}")

    print("[INFO] Walker S2 table scene test complete.")


def main() -> None:
    if not WALKER_S2_USD_PATH.is_file():
        raise FileNotFoundError(f"Walker S2 USD not found: {WALKER_S2_USD_PATH}")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, -1.25, 1.55], [0.75, 0.15, 1.0])

    scene_cfg = WalkerS2TableSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] IsaacLab Walker S2 table scene setup complete.", flush=True)
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Walker S2 table scene test failed before normal completion:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
