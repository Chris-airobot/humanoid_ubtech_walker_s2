#!/usr/bin/env python3
"""Spawn Walker S2 from its IsaacLab ArticulationCfg and run a short sim test.

Run from the IsaacLab checkout, for example:

    /home/chris/IsaacLab/isaaclab.sh -p \
      /home/chris/Projects/internship/zollent_technology/GlobalHumanoidRobotChallenge_2026_Baseline/scripts/spawn_walker_s2_isaaclab.py \
      --headless --steps 240
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Spawn-test the Walker S2 IsaacLab articulation config.")
parser.add_argument("--steps", type=int, default=240, help="Number of simulation steps to run after reset.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Walker S2 instances to spawn.")
parser.add_argument("--env_spacing", type=float, default=3.0, help="Spacing between spawned environments.")
parser.add_argument("--spawn_height", type=float, default=None, help="Override the configured root spawn height.")
parser.add_argument("--free-base", action="store_true", help="Do not fix the robot root link for this spawn test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab_walker_s2 import WALKER_S2_CFG, WALKER_S2_USD_PATH  # noqa: E402


def _walker_cfg():
    cfg = WALKER_S2_CFG.copy()
    if args_cli.free_base:
        cfg.spawn.articulation_props.fix_root_link = False
    if args_cli.spawn_height is not None:
        cfg.init_state.pos = (cfg.init_state.pos[0], cfg.init_state.pos[1], float(args_cli.spawn_height))
    return cfg


@configclass
class WalkerS2SpawnSceneCfg(InteractiveSceneCfg):
    """Minimal scene for validating the Walker S2 articulation."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = _walker_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")


def _print_names(label: str, names: list[str], max_items: int = 200) -> None:
    print(f"[INFO] {label}: count={len(names)}")
    for idx, name in enumerate(names[:max_items]):
        print(f"  [{idx:03d}] {name}")
    if len(names) > max_items:
        print(f"  ... {len(names) - max_items} more")


def _print_robot_summary(robot) -> None:
    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)

    print(f"[INFO] Walker S2 USD: {WALKER_S2_USD_PATH}")
    print(f"[INFO] robot prim path expression: {robot.cfg.prim_path}")
    _print_names("joint names", joint_names)
    _print_names("body names", body_names)

    root_pos = robot.data.root_pos_w.detach().cpu()
    print(f"[INFO] initial root position(s): {root_pos.tolist()}")

    for expected in (
        "base_link",
        "torso_link",
        "L_ankle_roll_link",
        "R_ankle_roll_link",
        "hand3_v1_left_L_palm_link",
        "hand3_v1_right_R_palm_link",
    ):
        found = expected in body_names
        print(f"[CHECK] body {expected}: {'FOUND' if found else 'MISSING'}")


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    print("[INFO] Resetting Walker S2 to configured default state...", flush=True)

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    scene.reset()

    _print_robot_summary(robot)
    print("[INFO] Starting simulation steps with default joint-position targets...", flush=True)

    for step in range(args_cli.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if step in (0, args_cli.steps - 1) or (step + 1) % 60 == 0:
            root_pos = robot.data.root_pos_w.detach().cpu()
            root_quat = robot.data.root_quat_w.detach().cpu()
            print(
                f"[STEP {step + 1:04d}] "
                f"root_pos={root_pos.tolist()} root_quat={root_quat.tolist()}"
            )

    print("[INFO] Spawn test complete.")


def main() -> None:
    if not WALKER_S2_USD_PATH.is_file():
        raise FileNotFoundError(f"Walker S2 USD not found: {WALKER_S2_USD_PATH}")

    print(f"[INFO] Using Walker S2 USD: {WALKER_S2_USD_PATH}", flush=True)
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.6, -2.2, 1.8], [0.75, -0.2, 0.8])

    scene_cfg = WalkerS2SpawnSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    print("[INFO] Creating IsaacLab scene...", flush=True)
    scene = InteractiveScene(scene_cfg)

    print("[INFO] Calling sim.reset()...", flush=True)
    sim.reset()
    print("[INFO] IsaacLab simulation setup complete.", flush=True)
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Walker S2 spawn test failed before normal completion:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
