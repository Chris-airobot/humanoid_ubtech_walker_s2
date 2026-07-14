#!/usr/bin/env python3
"""Load the Walker S2 USD into a minimal Isaac Sim scene."""

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = (
    REPO_ROOT
    / "WalkerS2-Model"
    / "Collected_WalkerS2"
    / "SubUSDs"
    / "s2_hand4_v1.usd"
)
DEFAULT_SPAWN_Z = 0.0
DEFAULT_GROUND_CLEARANCE = 0.02

# Conservative free-base pose: slight knee bend, flat-ish feet, arms tucked.
STANDING_JOINT_POSE = {
    "L_hip_roll_joint": 0.04,
    "L_hip_yaw_joint": 0.0,
    "L_hip_pitch_joint": 0.22,
    "L_knee_pitch_joint": -0.44,
    "L_ankle_pitch_joint": 0.22,
    "L_ankle_roll_joint": -0.04,
    "R_hip_roll_joint": -0.04,
    "R_hip_yaw_joint": 0.0,
    "R_hip_pitch_joint": 0.22,
    "R_knee_pitch_joint": -0.44,
    "R_ankle_pitch_joint": 0.22,
    "R_ankle_roll_joint": 0.04,
    "waist_yaw_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "L_elbow_roll_joint": -1.8963565338596158,
    "L_elbow_yaw_joint": 1.4000461262831179,
    "L_shoulder_pitch_joint": 0.09322471888572098,
    "L_shoulder_roll_joint": -0.5933223843430208,
    "L_shoulder_yaw_joint": -1.595878574835185,
    "L_wrist_pitch_joint": -0.00048740902645395785,
    "L_wrist_roll_joint": 0.0998718010009366,
    "R_elbow_roll_joint": -1.8963607249359917,
    "R_elbow_yaw_joint": -1.4000874256427638,
    "R_shoulder_pitch_joint": -0.09321727661087699,
    "R_shoulder_roll_joint": -0.5933455607833843,
    "R_shoulder_yaw_joint": 1.595869459316937,
    "R_wrist_pitch_joint": 0.00048144049606466176,
    "R_wrist_roll_joint": 0.09985407619802703,
    "head_pitch_joint": -0.785398163,
    "head_yaw_joint": 1.9677590016147396e-07,
}

HIGH_GAIN_JOINTS = {
    "hip",
    "knee",
    "ankle",
    "waist",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Import Walker S2 into Isaac Sim.")
    parser.add_argument("--usd-path", type=Path, default=DEFAULT_USD, help="Path to the Walker S2 USD file.")
    parser.add_argument("--prim-path", default="/World/WalkerS2", help="Stage prim path for the robot reference.")
    parser.add_argument("--headless", action="store_true", help="Run without the Isaac Sim GUI.")
    parser.add_argument("--play", action="store_true", help="Start stepping physics after loading.")
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Load and pose the robot without stepping physics.",
    )
    parser.add_argument(
        "--show-collisions",
        action="store_true",
        help="Reveal collision prims so they can be inspected in the viewport.",
    )
    parser.add_argument(
        "--spawn-z",
        type=float,
        default=None,
        help="Robot root Z height in metres. Defaults to aligning the loaded robot bounds above the ground.",
    )
    parser.add_argument(
        "--no-stand",
        action="store_true",
        help="Skip standing-pose setup (robot is loaded at the origin with default joint angles).",
    )
    parser.add_argument(
        "--free-base",
        action="store_true",
        help="Deprecated alias; the robot is free-base by default unless --fix-base is set.",
    )
    parser.add_argument(
        "--fix-base",
        action="store_true",
        help="Add a world fixed joint to the root body for debugging.",
    )
    parser.add_argument("--steps", type=int, default=0, help="Number of frames to run; 0 keeps the app open.")
    return parser.parse_args()


args = parse_args()

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "width": 1280,
        "height": 720,
        "renderer": "RaytracedLighting",
    }
)

from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, is_stage_loading
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt


def compute_spawn_height(usd_path: Path, ground_clearance: float = 0.01) -> float:
    """Return the Z translation needed so the robot feet sit on z=0."""
    stage = Usd.Stage.Open(str(usd_path))
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        return DEFAULT_SPAWN_Z

    bound = UsdGeom.Boundable(default_prim).ComputeWorldBound(
        Usd.TimeCode.Default(),
        UsdGeom.Tokens.default_,
    )
    bbox = bound.ComputeAlignedBox()
    if bbox.IsEmpty():
        return DEFAULT_SPAWN_Z

    lowest_z = bbox.GetMin()[2]
    return max(0.0, -lowest_z + ground_clearance)


def set_prim_translation(stage, prim_path: str, translation) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2])))


def get_prim_translation(stage, prim_path: str) -> Gf.Vec3d:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            value = op.Get()
            return Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]))
    return Gf.Vec3d(0.0, 0.0, 0.0)


def align_prim_bottom_to_ground(prim_path: str, ground_clearance: float = DEFAULT_GROUND_CLEARANCE):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if bbox.IsEmpty():
        return None

    min_z = float(bbox.GetMin()[2])
    translation = get_prim_translation(stage, prim_path)
    z_delta = ground_clearance - min_z
    aligned_translation = Gf.Vec3d(translation[0], translation[1], translation[2] + z_delta)
    set_prim_translation(stage, prim_path, aligned_translation)
    return {
        "old_z": float(translation[2]),
        "new_z": float(aligned_translation[2]),
        "min_z_before": min_z,
        "z_delta": float(z_delta),
        "ground_clearance": ground_clearance,
    }


def add_lights():
    stage = get_current_stage()
    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome_light.CreateIntensityAttr(450.0)

    key_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
    key_light.CreateIntensityAttr(3000.0)
    key_light.CreateAngleAttr(0.35)

    fill_light = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLight"))
    fill_light.CreateIntensityAttr(3500.0)
    fill_light.CreateRadiusAttr(2.0)
    UsdGeom.Xformable(fill_light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(2.5, -3.5, 4.0))


def add_visual_ground():
    stage = get_current_stage()
    ground = UsdGeom.Cube.Define(stage, Sdf.Path("/World/VisualGround"))
    ground.CreateSizeAttr(1.0)
    ground.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.35, 0.35, 0.35)]))

    xformable = UsdGeom.Xformable(ground.GetPrim())
    xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.005))
    xformable.AddScaleOp().Set(Gf.Vec3f(10.0, 10.0, 0.01))


def add_visual_scene():
    add_visual_ground()
    add_lights()


def add_simple_scene():
    from pxr import PhysicsSchemaTools

    stage = get_current_stage()
    PhysicsSchemaTools.addGroundPlane(
        stage,
        "/World/GroundPlane",
        "Z",
        20.0,
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(0.35, 0.35, 0.35),
    )
    add_lights()


def reveal_collision_prims(prim_path: str) -> tuple[list[str], list[str]]:
    stage = get_current_stage()
    physics_collision_paths = []
    named_collision_paths = []

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(prim_path):
            continue

        applied_schemas = {str(schema) for schema in prim.GetAppliedSchemas()}
        has_collision_api = "PhysicsCollisionAPI" in applied_schemas
        has_collision_name = "collision" in prim.GetName().lower()
        if not has_collision_api and not has_collision_name:
            continue

        if has_collision_api:
            physics_collision_paths.append(path)
        else:
            named_collision_paths.append(path)

        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeVisible()
            imageable.CreatePurposeAttr(UsdGeom.Tokens.default_)

        gprim = UsdGeom.Gprim(prim)
        if gprim:
            gprim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.85, 1.0)]))
            gprim.CreateDisplayOpacityAttr(Vt.FloatArray([0.35]))

    return physics_collision_paths, named_collision_paths


def disable_robot_physics(prim_path: str) -> int:
    disabled_count = 0
    for prim in get_current_stage().Traverse():
        path = str(prim.GetPath())
        if not path.startswith(prim_path):
            continue

        applied_schemas = {str(schema) for schema in prim.GetAppliedSchemas()}
        if "PhysicsRigidBodyAPI" not in applied_schemas:
            continue

        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        disabled_count += 1

    return disabled_count


def add_root_fixed_joint(prim_path: str) -> str:
    root_body_path = f"{prim_path}/base_link"
    root_body = get_current_stage().GetPrimAtPath(root_body_path)
    if not root_body.IsValid():
        raise RuntimeError(f"Walker S2 root body not found: {root_body_path}")

    fixed_joint_path = f"{prim_path}/world_fixed_joint"
    fixed_joint = UsdPhysics.FixedJoint.Define(get_current_stage(), Sdf.Path(fixed_joint_path))
    fixed_joint.GetBody1Rel().SetTargets([Sdf.Path(root_body_path)])

    root_transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(root_body)
    root_translation = root_transform.ExtractTranslation()
    fixed_joint.CreateLocalPos0Attr(
        Gf.Vec3f(float(root_translation[0]), float(root_translation[1]), float(root_translation[2]))
    )
    fixed_joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return fixed_joint_path


def setup_standing_robot(world, prim_path: str):
    """Place the robot on the ground and hold the default standing pose."""
    import torch

    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.types import ArticulationActions

    robot = Articulation(prim_paths_expr=prim_path, name="walker_s2")
    robot.initialize()

    dof_names = robot.dof_names
    target_positions = [STANDING_JOINT_POSE.get(name, 0.0) for name in dof_names]
    joint_indices = list(range(len(dof_names)))
    kps = [900.0 if any(token in name for token in HIGH_GAIN_JOINTS) else 180.0 for name in dof_names]
    kds = [70.0 if any(token in name for token in HIGH_GAIN_JOINTS) else 18.0 for name in dof_names]
    max_efforts = [350.0 if any(token in name for token in HIGH_GAIN_JOINTS) else 120.0 for name in dof_names]

    robot.set_joint_positions(
        torch.tensor(target_positions, dtype=torch.float32),
        joint_indices=torch.tensor(joint_indices, dtype=torch.int32),
    )
    robot.set_joint_velocities(
        torch.zeros(len(dof_names), dtype=torch.float32),
        joint_indices=torch.tensor(joint_indices, dtype=torch.int32),
    )
    robot.set_gains(
        kps=torch.tensor([kps], dtype=torch.float32),
        kds=torch.tensor([kds], dtype=torch.float32),
    )
    robot.set_max_efforts(torch.tensor([max_efforts], dtype=torch.float32))

    def hold_standing_pose(_step_size):
        robot.apply_action(
            ArticulationActions(
                joint_positions=torch.tensor([target_positions], dtype=torch.float32),
                joint_indices=torch.tensor(joint_indices, dtype=torch.int32),
            )
        )

    world.add_physics_callback("walker_s2_hold_standing_pose", hold_standing_pose)
    return robot, target_positions


def print_collision_report(physics_collision_paths: list[str], named_collision_paths: list[str]) -> None:
    print(f"Robot PhysicsCollisionAPI prim(s): {len(physics_collision_paths)}", flush=True)
    for path in physics_collision_paths[:20]:
        print(f"  {path}", flush=True)
    if len(physics_collision_paths) > 20:
        print(f"  ... {len(physics_collision_paths) - 20} more", flush=True)

    print(f"Robot collision-named container prim(s): {len(named_collision_paths)}", flush=True)
    for path in named_collision_paths[:20]:
        print(f"  {path}", flush=True)
    if len(named_collision_paths) > 20:
        print(f"  ... {len(named_collision_paths) - 20} more", flush=True)


def run_view_only(usd_path: Path, spawn_z: float) -> int:
    import omni.timeline

    omni.timeline.get_timeline_interface().stop()
    add_visual_scene()
    add_reference_to_stage(usd_path=str(usd_path), prim_path=args.prim_path)
    set_prim_translation(get_current_stage(), args.prim_path, (0.0, 0.0, spawn_z))

    simulation_app.update()
    while is_stage_loading():
        simulation_app.update()

    alignment = None if args.spawn_z is not None else align_prim_bottom_to_ground(args.prim_path)
    final_spawn_z = get_prim_translation(get_current_stage(), args.prim_path)[2]
    set_camera_view(
        eye=[3.5, -5.0, final_spawn_z + 2.2],
        target=[0.0, 0.0, final_spawn_z + 1.0],
        camera_prim_path="/OmniverseKit_Persp",
    )

    disabled_body_count = disable_robot_physics(args.prim_path)
    physics_collision_paths, named_collision_paths = (
        reveal_collision_prims(args.prim_path) if args.show_collisions else ([], [])
    )

    print(f"Loaded Walker S2 from {usd_path}", flush=True)
    print(f"Robot prim: {args.prim_path}", flush=True)
    print(f"View-only mode at z={final_spawn_z:.3f} m", flush=True)
    if alignment:
        print(
            "Auto-aligned robot bottom: "
            f"min_z_before={alignment['min_z_before']:.3f} m, "
            f"z_delta={alignment['z_delta']:.3f} m, "
            f"clearance={alignment['ground_clearance']:.3f} m",
            flush=True,
        )
    print("No World, no world.reset(), no Articulation, no physics stepping", flush=True)
    print(f"Disabled robot rigid bodies in-memory: {disabled_body_count}", flush=True)
    if not args.no_stand:
        print("Standing pose is skipped in view-only mode; showing the USD authored pose", flush=True)
    if args.show_collisions:
        print_collision_report(physics_collision_paths, named_collision_paths)

    if args.steps > 0:
        for _ in range(args.steps):
            simulation_app.update()
    else:
        while simulation_app.is_running():
            simulation_app.update()

    simulation_app.close()
    return 0


def main():
    usd_path = args.usd_path.expanduser().resolve()
    if not usd_path.exists():
        print(f"Walker S2 USD not found: {usd_path}", file=sys.stderr, flush=True)
        simulation_app.close()
        return 1

    stand = not args.no_stand
    spawn_z = args.spawn_z if args.spawn_z is not None else DEFAULT_SPAWN_Z
    if args.view_only:
        return run_view_only(usd_path, spawn_z)

    run_physics = False if args.view_only else args.play or stand

    from isaacsim.core.api import World

    world = World(stage_units_in_meters=1.0)
    add_simple_scene()

    add_reference_to_stage(usd_path=str(usd_path), prim_path=args.prim_path)
    set_prim_translation(get_current_stage(), args.prim_path, (0.0, 0.0, spawn_z))
    locked_root_body_path = None

    simulation_app.update()
    while is_stage_loading():
        simulation_app.update()

    alignment = None if args.spawn_z is not None else align_prim_bottom_to_ground(args.prim_path)
    final_spawn_z = get_prim_translation(get_current_stage(), args.prim_path)[2]
    if stand and args.fix_base:
        locked_root_body_path = add_root_fixed_joint(args.prim_path)
    set_camera_view(
        eye=[3.5, -5.0, final_spawn_z + 2.2],
        target=[0.0, 0.0, final_spawn_z + 1.0],
        camera_prim_path="/OmniverseKit_Persp",
    )

    physics_collision_paths, named_collision_paths = (
        reveal_collision_prims(args.prim_path) if args.show_collisions else ([], [])
    )

    world.reset()

    if stand:
        setup_standing_robot(world, args.prim_path)
        if not args.view_only:
            for _ in range(10):
                world.step(render=False)

    print(f"Loaded Walker S2 from {usd_path}", flush=True)
    print(f"Robot prim: {args.prim_path}", flush=True)
    if stand:
        print(f"Standing pose enabled at z={final_spawn_z:.3f} m (feet on ground plane)", flush=True)
        if alignment:
            print(
                "Auto-aligned robot bottom: "
                f"min_z_before={alignment['min_z_before']:.3f} m, "
                f"z_delta={alignment['z_delta']:.3f} m, "
                f"clearance={alignment['ground_clearance']:.3f} m",
                flush=True,
            )
        if locked_root_body_path:
            print(f"Root body locked for stable standing by: {locked_root_body_path}", flush=True)
        else:
            print("Root body is free (use --fix-base only for debug pinning)", flush=True)
    else:
        print("Standing pose disabled (--no-stand)", flush=True)
    if args.view_only:
        print("View-only mode: physics is not being stepped", flush=True)
    if args.show_collisions:
        print_collision_report(physics_collision_paths, named_collision_paths)

    if args.steps > 0:
        for _ in range(args.steps):
            if run_physics:
                world.step(render=not args.headless)
            else:
                simulation_app.update()
    else:
        while simulation_app.is_running():
            if run_physics:
                world.step(render=not args.headless)
            else:
                simulation_app.update()

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
