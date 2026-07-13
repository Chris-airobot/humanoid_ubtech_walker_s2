#!/usr/bin/env python3
"""Export the Walker S2 + Hand3 URDF to a robot-only USD and validate it.

Run with Isaac Sim Python:
  ~/isaacsim/python.sh scripts/export_walker_s2_usd.py

The default input is the tuned simple-hand-collision URDF used by the grasp demo.
The output USD is intended to be referenced later from an IsaacLab ArticulationCfg.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_RESOURCE_DIR = (
    REPO_ROOT
    / "assets"
    / "resources"
    / "walker_s2_description_hand3_v1_left_hand3_v1_right"
)
DEFAULT_URDF = ROBOT_RESOURCE_DIR / "walker_s2_description_hand3_v1_left_hand3_v1_right_isaac_simple_hand_collision.urdf"
DEFAULT_USD = ROBOT_RESOURCE_DIR / "walker_s2_with_hands_isaaclab.usd"

REQUIRED_LINKS = (
    "base_link",
    "R_wrist_roll_link",
    "L_wrist_roll_link",
    "hand3_v1_right_R_palm_link",
    "hand3_v1_left_L_palm_link",
)
REQUIRED_JOINTS = (
    "R_shoulder_pitch_joint",
    "R_shoulder_roll_joint",
    "R_shoulder_yaw_joint",
    "R_elbow_roll_joint",
    "R_elbow_yaw_joint",
    "R_wrist_pitch_joint",
    "R_wrist_roll_joint",
    "L_shoulder_pitch_joint",
    "L_shoulder_roll_joint",
    "L_shoulder_yaw_joint",
    "L_elbow_roll_joint",
    "L_elbow_yaw_joint",
    "L_wrist_pitch_joint",
    "L_wrist_roll_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Walker S2 + Hand3 URDF to USD and validate the result.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Input Walker S2 URDF path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_USD, help="Output robot-only USD path.")
    parser.add_argument("--headless", action="store_true", default=True, help="Run Isaac Sim headless.")
    parser.add_argument("--show-window", dest="headless", action="store_false", help="Open the Isaac Sim window.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing USD.")
    parser.add_argument(
        "--fix-base",
        action="store_true",
        help="Bake a fixed base joint into the USD. Leave off for IsaacLab, then use fix_root_link=True there.",
    )
    parser.add_argument("--stiffness", type=float, default=10000.0, help="Default position drive stiffness.")
    parser.add_argument("--damping", type=float, default=1000.0, help="Default position drive damping.")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Only validate --out. This does not launch Isaac Sim.",
    )
    return parser.parse_args()


def _set_import_config_value(config, name: str, value) -> None:
    setter = getattr(config, f"set_{name}", None)
    if callable(setter):
        setter(value)
        return
    if hasattr(config, name):
        setattr(config, name, value)


def _resolve_robot_default_prim(stage, articulation_root_path: str):
    prim = stage.GetPrimAtPath(articulation_root_path)
    if not prim.IsValid():
        return None

    # The URDF importer often returns the root link path. If it is nested under a
    # robot xform, make that xform the default prim instead of only base_link.
    if prim.GetName() == "base_link":
        parent = prim.GetParent()
        if parent.IsValid() and parent.GetName() not in ("", "World"):
            return parent
    return prim


def _make_import_config(omni, _urdf, *, fix_base: bool, stiffness: float, damping: float):
    _, config = omni.kit.commands.execute("URDFCreateImportConfig")

    values = {
        "merge_fixed_joints": False,
        "fix_base": fix_base,
        "import_inertia_tensor": True,
        "self_collision": False,
        "make_default_prim": True,
        "create_physics_scene": False,
        "distance_scale": 1.0,
        "density": 0.0,
        "convex_decomp": False,
        "parse_mimic": False,
        "replace_cylinders_with_capsules": True,
        "override_joint_dynamics": False,
    }
    for name, value in values.items():
        _set_import_config_value(config, name, value)

    target_position = getattr(_urdf.UrdfJointTargetType, "JOINT_DRIVE_POSITION", None)
    if target_position is not None:
        _set_import_config_value(config, "default_drive_type", target_position)
    _set_import_config_value(config, "default_drive_strength", stiffness)
    _set_import_config_value(config, "default_position_drive_damping", damping)
    return config


def export_usd(args: argparse.Namespace) -> Path:
    urdf_path = args.urdf.expanduser().resolve()
    usd_path = args.out.expanduser().resolve()

    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if not (urdf_path.parent / "meshes").is_dir():
        raise FileNotFoundError(f"Expected meshes directory next to URDF: {urdf_path.parent / 'meshes'}")
    if usd_path.exists() and not args.force:
        raise FileExistsError(f"Output USD already exists: {usd_path}. Use --force to overwrite it.")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})
    try:
        import omni.kit.commands
        import omni.usd
        from isaacsim.core.utils.extensions import enable_extension

        try:
            from isaacsim.asset.importer.urdf import _urdf
        except ImportError:
            import isaacsim.asset.importer.urdf as _urdf_mod

            _urdf = _urdf_mod._urdf

        for _ in range(5):
            simulation_app.update()
        enable_extension("isaacsim.asset.importer.urdf")
        for _ in range(5):
            simulation_app.update()

        omni.usd.get_context().new_stage()
        for _ in range(5):
            simulation_app.update()

        import_config = _make_import_config(
            omni,
            _urdf,
            fix_base=args.fix_base,
            stiffness=args.stiffness,
            damping=args.damping,
        )

        print(f"[INFO] Importing URDF: {urdf_path}", flush=True)
        result = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(urdf_path),
            import_config=import_config,
            dest_path="",
            get_articulation_root=True,
        )
        prim_path = result[1] if isinstance(result, tuple) else result
        if not prim_path:
            raise RuntimeError("URDF import failed: empty articulation root path")

        stage = omni.usd.get_context().get_stage()
        default_prim = _resolve_robot_default_prim(stage, prim_path)
        if default_prim is not None and default_prim.IsValid():
            stage.SetDefaultPrim(default_prim)

        usd_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving USD: {usd_path}", flush=True)
        if not omni.usd.get_context().save_as_stage(str(usd_path)):
            raise RuntimeError(f"Failed to save USD: {usd_path}")

        for _ in range(3):
            simulation_app.update()
        print(f"[INFO] Imported articulation root: {prim_path}", flush=True)
        if default_prim is not None and default_prim.IsValid():
            print(f"[INFO] USD default prim: {default_prim.GetPath()}", flush=True)
        return usd_path
    finally:
        simulation_app.close()


def _prim_names(stage, predicate):
    return sorted(str(prim.GetName()) for prim in stage.Traverse() if predicate(prim))


def validate_usd(usd_path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {usd_path}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise RuntimeError("USD has no default prim")

    xformable_links = set(_prim_names(stage, lambda prim: prim.IsA(UsdGeom.Xformable)))
    missing_links = [name for name in REQUIRED_LINKS if name not in xformable_links]

    joint_type_names = {
        "PhysicsRevoluteJoint",
        "PhysicsPrismaticJoint",
        "PhysicsFixedJoint",
        "PhysicsSphericalJoint",
        "PhysicsJoint",
    }
    joint_names = set(_prim_names(stage, lambda prim: prim.GetTypeName() in joint_type_names))
    missing_joints = [name for name in REQUIRED_JOINTS if name not in joint_names]

    rigid_body_count = sum(1 for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI))
    collision_count = sum(1 for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI))
    articulation_roots = [str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]

    problems = []
    if missing_links:
        problems.append(f"missing required links: {missing_links}")
    if missing_joints:
        problems.append(f"missing required arm joints: {missing_joints}")
    if not articulation_roots:
        problems.append("no UsdPhysics.ArticulationRootAPI found")
    if rigid_body_count == 0:
        problems.append("no rigid bodies found")
    if collision_count == 0:
        problems.append("no collision shapes found")

    print(f"[VALIDATE] USD: {usd_path}", flush=True)
    print(f"[VALIDATE] default prim: {default_prim.GetPath()}", flush=True)
    print(f"[VALIDATE] articulation roots: {articulation_roots}", flush=True)
    print(f"[VALIDATE] joints: {len(joint_names)}", flush=True)
    print(f"[VALIDATE] rigid bodies: {rigid_body_count}", flush=True)
    print(f"[VALIDATE] collision shapes: {collision_count}", flush=True)

    if problems:
        raise RuntimeError("USD validation failed: " + "; ".join(problems))
    print(
        "[VALIDATE] OK: required Walker links, arm joints, articulation, rigid bodies, and collisions are present.",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    try:
        usd_path = args.out.expanduser().resolve()
        if not args.skip_export:
            usd_path = export_usd(args)
            validate_usd(usd_path)
        else:
            simulation_app = None
            try:
                from pxr import Usd  # noqa: F401
            except ModuleNotFoundError:
                from isaacsim import SimulationApp

                simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})
            try:
                validate_usd(usd_path)
            finally:
                if simulation_app is not None:
                    simulation_app.close()
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
