#!/usr/bin/env python3
"""Import the official merged Walker S2 + hand v3 URDF into Isaac Sim.

Unlike import_walker_s2.py (USD), this loads the merged URDF so the dexterous
hand meshes and finger joints are present and actuated.

Build the URDF first:
  python3 scripts/setup_official_walker_s2.py

Then:
  /home/chris/isaacsim/python.sh scripts/import_walker_s2_urdf.py --view-only
  /home/chris/isaacsim/python.sh scripts/import_walker_s2_urdf.py --play
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = (
    REPO_ROOT
    / "WalkerS2-Model"
    / "walker_s2_description_hand3_v1_left_hand3_v1_right"
    / "walker_s2_description_hand3_v1_left_hand3_v1_right.urdf"
)
DEFAULT_SPAWN_Z = 0.0
DEFAULT_GROUND_CLEARANCE = 0.02

STANDING_JOINT_POSE = {
    "L_hip_roll_joint": 0.0,
    "L_hip_yaw_joint": 0.0,
    "L_hip_pitch_joint": 0.0,
    "L_knee_pitch_joint": 0.0,
    "L_ankle_pitch_joint": 0.0,
    "L_ankle_roll_joint": 0.0,
    "R_hip_roll_joint": 0.0,
    "R_hip_yaw_joint": 0.0,
    "R_hip_pitch_joint": 0.0,
    "R_knee_pitch_joint": 0.0,
    "R_ankle_pitch_joint": 0.0,
    "R_ankle_roll_joint": 0.0,
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

HIGH_GAIN_JOINTS = {"hip", "knee", "ankle", "waist"}


def parse_args():
    parser = argparse.ArgumentParser(description="Import official Walker S2 URDF into Isaac Sim.")
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--play",
        action="store_true",
        help="Step physics and hold standing pose (implies physics, not view-only).",
    )
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Load the URDF for inspection without stepping physics (recommended first).",
    )
    parser.add_argument("--fix-base", action="store_true", help="Fix base_link to world (debug).")
    parser.add_argument(
        "--spawn-z",
        type=float,
        default=None,
        help="Robot root Z height in metres. Defaults to auto-align feet above ground.",
    )
    parser.add_argument(
        "--ground-clearance",
        type=float,
        default=DEFAULT_GROUND_CLEARANCE,
        help="Gap between lowest robot point and the ground after auto-align.",
    )
    parser.add_argument("--steps", type=int, default=0)
    return parser.parse_args()


def resolve_robot_xform_path(stage, articulation_root: str) -> str:
    """URDF import often returns the root link; translate the parent xform when present."""
    prim = stage.GetPrimAtPath(articulation_root)
    if not prim.IsValid():
        return articulation_root
    if articulation_root.endswith("/base_link"):
        parent = prim.GetParent()
        if parent.IsValid() and parent.GetName() not in ("World", ""):
            return str(parent.GetPath())
    return articulation_root


def set_prim_translation(stage, prim_path: str, translation) -> None:
    from pxr import Gf, UsdGeom

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


def get_prim_translation(stage, prim_path: str):
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            value = op.Get()
            return Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]))
    return Gf.Vec3d(0.0, 0.0, 0.0)


def align_prim_bottom_to_ground(stage, prim_path: str, ground_clearance: float = DEFAULT_GROUND_CLEARANCE):
    from pxr import Gf, Usd, UsdGeom

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


def add_lights(stage) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome_light.CreateIntensityAttr(450.0)

    key_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
    key_light.CreateIntensityAttr(3000.0)
    key_light.CreateAngleAttr(0.35)

    fill_light = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLight"))
    fill_light.CreateIntensityAttr(3500.0)
    fill_light.CreateRadiusAttr(2.0)
    UsdGeom.Xformable(fill_light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(2.5, -3.5, 4.0))


def add_visual_ground(stage) -> None:
    from pxr import Gf, Sdf, UsdGeom, Vt

    ground = UsdGeom.Cube.Define(stage, Sdf.Path("/World/VisualGround"))
    ground.CreateSizeAttr(1.0)
    ground.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.35, 0.35, 0.35)]))

    xformable = UsdGeom.Xformable(ground.GetPrim())
    xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.005))
    xformable.AddScaleOp().Set(Gf.Vec3f(10.0, 10.0, 0.01))


def add_visual_scene(stage) -> None:
    add_visual_ground(stage)
    add_lights(stage)


def add_physics_scene(stage) -> None:
    from pxr import Gf, PhysicsSchemaTools

    PhysicsSchemaTools.addGroundPlane(
        stage,
        "/World/GroundPlane",
        "Z",
        20.0,
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(0.35, 0.35, 0.35),
    )
    add_lights(stage)


def main() -> int:
    args = parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "width": 1280,
            "height": 720,
        }
    )

    import omni.kit.commands
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.viewports import set_camera_view
    from pxr import Gf, Sdf, UsdPhysics

    urdf_path = args.urdf_path.expanduser().resolve()
    if not urdf_path.is_file():
        print(f"URDF not found: {urdf_path}", file=sys.stderr)
        print("Run: python3 scripts/setup_official_walker_s2.py", file=sys.stderr)
        simulation_app.close()
        return 1

    meshes_dir = urdf_path.parent / "meshes"
    if not meshes_dir.is_dir():
        print(f"Meshes directory not found: {meshes_dir}", file=sys.stderr)
        simulation_app.close()
        return 1

    # Let Kit finish booting before touching USD / URDF importer.
    for _ in range(5):
        simulation_app.update()

    enable_extension("isaacsim.asset.importer.urdf")

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    if args.play and not args.view_only:
        add_physics_scene(stage)
    else:
        add_visual_scene(stage)

    try:
        from isaacsim.asset.importer.urdf import _urdf
    except ImportError:
        import isaacsim.asset.importer.urdf as _urdf_mod

        _urdf = _urdf_mod._urdf

    print(f"Importing URDF: {urdf_path}", flush=True)
    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.fix_base = False
    import_config.import_inertia_tensor = True
    import_config.self_collision = False
    import_config.make_default_prim = True
    import_config.create_physics_scene = False
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.default_drive_strength = 1e4
    import_config.default_position_drive_damping = 1e3

    # dest_path must be a USD file path or empty string for in-memory import.
    # Do NOT pass a prim path like /World/WalkerS2 — that crashes CreateNew().

    result = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
        dest_path="",
        get_articulation_root=True,
    )
    if isinstance(result, tuple):
        _, prim_path = result
    else:
        prim_path = result

    if not prim_path:
        print("URDF import failed (empty prim path).", file=sys.stderr)
        simulation_app.close()
        return 1

    print(f"Robot articulation root: {prim_path}", flush=True)

    robot_xform_path = resolve_robot_xform_path(stage, prim_path)
    spawn_z = args.spawn_z if args.spawn_z is not None else DEFAULT_SPAWN_Z
    set_prim_translation(stage, robot_xform_path, (0.0, 0.0, spawn_z))
    for _ in range(3):
        simulation_app.update()

    alignment = None
    if args.spawn_z is None:
        alignment = align_prim_bottom_to_ground(stage, robot_xform_path, args.ground_clearance)
    final_spawn_z = float(get_prim_translation(stage, robot_xform_path)[2])
    if alignment:
        print(
            "Auto-aligned robot bottom: "
            f"min_z_before={alignment['min_z_before']:.3f} m, "
            f"z_delta={alignment['z_delta']:.3f} m, "
            f"spawn_z={final_spawn_z:.3f} m",
            flush=True,
        )
    else:
        print(f"Robot spawn z={final_spawn_z:.3f} m", flush=True)

    if args.fix_base:
        base_path = f"{prim_path}/base_link"
        if stage.GetPrimAtPath(base_path).IsValid():
            fixed = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(f"{prim_path}/world_fixed_joint"))
            fixed.GetBody1Rel().SetTargets([Sdf.Path(base_path)])

    set_camera_view(
        eye=[3.5, -5.0, final_spawn_z + 2.2],
        target=[0.0, 0.0, final_spawn_z + 1.0],
        camera_prim_path="/OmniverseKit_Persp",
    )

    run_physics = args.play and not args.view_only

    if run_physics:
        import torch
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.types import ArticulationActions

        world.reset()
        robot = Articulation(prim_paths_expr=prim_path, name="walker_s2_official")
        robot.initialize()

        dof_names = robot.dof_names
        print(f"Loaded {len(dof_names)} DOFs", flush=True)
        hand_dofs = [n for n in dof_names if any(k in n for k in ("thumb", "index", "middle", "ring", "little"))]
        print(f"Hand DOFs ({len(hand_dofs)}): {hand_dofs[:8]}{'...' if len(hand_dofs) > 8 else ''}", flush=True)

        target_positions = [STANDING_JOINT_POSE.get(name, 0.0) for name in dof_names]
        joint_indices = list(range(len(dof_names)))
        kps = [900.0 if any(t in name for t in HIGH_GAIN_JOINTS) else 180.0 for name in dof_names]
        kds = [70.0 if any(t in name for t in HIGH_GAIN_JOINTS) else 18.0 for name in dof_names]

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

        for _ in range(5):
            simulation_app.update()
        if args.spawn_z is None:
            play_alignment = align_prim_bottom_to_ground(stage, robot_xform_path, args.ground_clearance)
            if play_alignment:
                final_spawn_z = float(get_prim_translation(stage, robot_xform_path)[2])
                print(
                    "Re-aligned for standing pose: "
                    f"min_z_before={play_alignment['min_z_before']:.3f} m, "
                    f"spawn_z={final_spawn_z:.3f} m",
                    flush=True,
                )
                set_camera_view(
                    eye=[3.5, -5.0, final_spawn_z + 2.2],
                    target=[0.0, 0.0, final_spawn_z + 1.0],
                    camera_prim_path="/OmniverseKit_Persp",
                )

        def hold_pose(_step_size):
            robot.apply_action(
                ArticulationActions(
                    joint_positions=torch.tensor([target_positions], dtype=torch.float32),
                    joint_indices=torch.tensor(joint_indices, dtype=torch.int32),
                )
            )

        world.add_physics_callback("walker_s2_official_hold_pose", hold_pose)
        for _ in range(10):
            world.step(render=not args.headless)

    if args.steps > 0 and run_physics:
        for _ in range(args.steps):
            world.step(render=not args.headless)
    elif run_physics:
        while simulation_app.is_running():
            world.step(render=not args.headless)
    elif args.headless or args.steps > 0:
        updates = max(args.steps, 1)
        print(f"View-only mode: holding for {updates} update(s), then exiting.", flush=True)
        for _ in range(updates):
            simulation_app.update()
    else:
        print("View-only falling back to interactive loop (Ctrl+C to quit).", flush=True)
        while simulation_app.is_running():
            simulation_app.update()

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
