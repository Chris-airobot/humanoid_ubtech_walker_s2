#!/usr/bin/env python3
"""Merge Walker S2 body URDF with official UBT hand v3 (三代手) URDF packages.

The zip provides standalone left/right hand models:
  ubt_left_hand_v3_description/urdf/hand3_v1/hand3_v1.urdf
  ubt_right_hand_v3_description/urdf/hand3_v1/hand3_v1.urdf

This script:
  1. Takes the Walker S2 body (default: walker_s2_urdf_with_hand3 without hand links)
  2. Attaches the official hand URDFs at the wrist
  3. Copies hand meshes and rewrites mesh paths for Pinocchio / RViz

Output (default):
  WalkerS2-Model/walker_s2_official/walker_s2.urdf
  WalkerS2-Model/walker_s2_official/meshes/
"""

from __future__ import annotations

import argparse
import math
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HAND_ROOT = REPO_ROOT / "WalkerS2-Model" / "ubt_hand_v3"
DEFAULT_BODY_URDF = (
    REPO_ROOT
    / "WalkerS2-Model"
    / "walker_s2_description"
    / "walker_s2_description"
    / "urdf"
    / "s2"
    / "s2.urdf"
)
DEFAULT_REFERENCE_URDF = (
    REPO_ROOT / "WalkerS2-Model" / "walker_s2_urdf_with_hand3" / "walker_s2.urdf"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "WalkerS2-Model" / "walker_s2_official"
DEFAULT_MOUNT_LINK = "L_sixforce_link"
DEFAULT_RIGHT_MOUNT_RPY = "0 0 3.1415927"

# Links/joints to strip from the bundled body URDF before attaching official hands.
HAND_LINK_PREFIXES = (
    "L_hand_",
    "R_hand_",
    "L_palm",
    "R_palm",
    "L_thumb_",
    "R_thumb_",
    "L_index_",
    "R_index_",
    "L_middle_",
    "R_middle_",
    "L_ring_",
    "R_ring_",
    "L_little_",
    "R_little_",
)
HAND_LINK_EXACT = {
    "L_wrist_roll_link_geom_2",
    "R_wrist_roll_link_geom_2",
}
HAND_JOINT_KEYWORDS = (
    "hand_base",
    "palm",
    "thumb",
    "index",
    "middle",
    "ring",
    "little",
    "wrist_roll_link_geom_2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hand-root",
        type=Path,
        default=DEFAULT_HAND_ROOT,
        help="Directory containing ubt_*_hand_v3_description/ from the zip.",
    )
    parser.add_argument(
        "--body-urdf",
        type=Path,
        default=DEFAULT_BODY_URDF,
        help="Walker S2 body URDF (hand links will be removed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for merged URDF and meshes.",
    )
    parser.add_argument(
        "--mount-link",
        default=DEFAULT_MOUNT_LINK,
        help="Body link used as hand mount parent (L_/R_ prefix applied per side).",
    )
    parser.add_argument(
        "--mount-xyz",
        default="0 0 0",
        help="Fixed joint origin xyz from mount link to hand_base (calibrate if misaligned).",
    )
    parser.add_argument(
        "--mount-rpy",
        default="0 0 0",
        help="Fixed joint origin rpy from mount link to hand_base (calibrate if misaligned).",
    )
    parser.add_argument(
        "--reference-urdf",
        type=Path,
        default=DEFAULT_REFERENCE_URDF,
        help="Bundled Walker S2+hand URDF used when --use-reference-right-mount is set.",
    )
    parser.add_argument(
        "--use-reference-right-mount",
        action="store_true",
        help="Derive the right hand mount rotation from the bundled Walker S2+hand URDF.",
    )
    parser.add_argument(
        "--right-mount-rpy",
        default=None,
        help=f"Override right sixforce->hand rpy (xyz stays 0). Default: {DEFAULT_RIGHT_MOUNT_RPY}.",
    )
    parser.add_argument(
        "--detailed-hand-collisions",
        action="store_true",
        help="Keep official detailed STL hand colliders instead of stable primitives.",
    )
    return parser.parse_args()


def _is_hand_link(name: str) -> bool:
    if name in HAND_LINK_EXACT:
        return True
    return any(name.startswith(prefix) for prefix in HAND_LINK_PREFIXES)


def _is_hand_joint(name: str) -> bool:
    lower = name.lower()
    return any(keyword in lower for keyword in HAND_JOINT_KEYWORDS)


def _rewrite_mesh_paths(root: ET.Element, mesh_prefix: str) -> None:
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        basename = Path(filename.replace("package://", "")).name
        mesh.set("filename", f"{mesh_prefix}/{basename}")


def _load_urdf(path: Path) -> ET.Element:
    tree = ET.parse(path)
    return tree.getroot()


def _body_elements(body_root: ET.Element) -> tuple[list[ET.Element], list[ET.Element]]:
    links: list[ET.Element] = []
    joints: list[ET.Element] = []
    for child in body_root:
        tag = child.tag
        if tag == "link":
            name = child.get("name", "")
            if _is_hand_link(name):
                continue
            links.append(child)
        elif tag == "joint":
            name = child.get("name", "")
            parent = child.find("parent")
            child_link = child.find("child")
            parent_name = parent.get("link") if parent is not None else ""
            child_name = child_link.get("link") if child_link is not None else ""
            if _is_hand_joint(name) or _is_hand_link(parent_name) or _is_hand_link(child_name):
                continue
            joints.append(child)
        else:
            links.append(child)
    return links, joints


def _hand_elements(hand_root: ET.Element, side_prefix: str) -> tuple[list[ET.Element], list[ET.Element]]:
    links: list[ET.Element] = []
    joints: list[ET.Element] = []
    for child in hand_root:
        tag = child.tag
        if tag == "link":
            name = child.get("name", "")
            if not name.startswith(side_prefix):
                continue
            links.append(child)
        elif tag == "joint":
            name = child.get("name", "")
            if not name.startswith(side_prefix):
                continue
            joints.append(child)
        elif tag == "mujoco":
            continue
    return links, joints


def _side_mount_link(mount_link: str, side: str) -> str:
    if mount_link.startswith(("L_", "R_")):
        return f"{side}_" + mount_link[2:]
    return f"{side}_{mount_link}"


def _parse_vec3(text: str) -> tuple[float, float, float]:
    parts = text.split()
    if len(parts) != 3:
        raise ValueError(f"Expected 3 values, got {text!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _format_vec3(values: tuple[float, float, float]) -> str:
    return " ".join(f"{v:.7f}" for v in values)


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _matrix_to_rpy(matrix: list[list[float]]) -> tuple[float, float, float]:
    sy = math.hypot(matrix[0][0], matrix[1][0])
    if sy > 1e-6:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        pitch = math.atan2(-matrix[2][0], sy)
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[1][2], matrix[1][1])
        pitch = math.atan2(-matrix[2][0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def _transform_from_xyz_rpy(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> tuple[tuple[float, float, float], list[list[float]]]:
    return xyz, _rpy_to_matrix(*rpy)


def _invert_transform(xyz: tuple[float, float, float], rot: list[list[float]]) -> tuple[tuple[float, float, float], list[list[float]]]:
    inv_rot = [
        [rot[0][0], rot[1][0], rot[2][0]],
        [rot[0][1], rot[1][1], rot[2][1]],
        [rot[0][2], rot[1][2], rot[2][2]],
    ]
    inv_xyz = (
        -(inv_rot[0][0] * xyz[0] + inv_rot[0][1] * xyz[1] + inv_rot[0][2] * xyz[2]),
        -(inv_rot[1][0] * xyz[0] + inv_rot[1][1] * xyz[1] + inv_rot[1][2] * xyz[2]),
        -(inv_rot[2][0] * xyz[0] + inv_rot[2][1] * xyz[1] + inv_rot[2][2] * xyz[2]),
    )
    return inv_xyz, inv_rot


def _compose_transform(
    xyz_a: tuple[float, float, float],
    rot_a: list[list[float]],
    xyz_b: tuple[float, float, float],
    rot_b: list[list[float]],
) -> tuple[tuple[float, float, float], list[list[float]]]:
    rot = [
        [
            rot_a[i][0] * rot_b[0][0] + rot_a[i][1] * rot_b[1][0] + rot_a[i][2] * rot_b[2][0],
            rot_a[i][0] * rot_b[0][1] + rot_a[i][1] * rot_b[1][1] + rot_a[i][2] * rot_b[2][1],
            rot_a[i][0] * rot_b[0][2] + rot_a[i][1] * rot_b[1][2] + rot_a[i][2] * rot_b[2][2],
        ]
        for i in range(3)
    ]
    xyz = (
        rot_a[0][0] * xyz_b[0] + rot_a[0][1] * xyz_b[1] + rot_a[0][2] * xyz_b[2] + xyz_a[0],
        rot_a[1][0] * xyz_b[0] + rot_a[1][1] * xyz_b[1] + rot_a[1][2] * xyz_b[2] + xyz_a[1],
        rot_a[2][0] * xyz_b[0] + rot_a[2][1] * xyz_b[1] + rot_a[2][2] * xyz_b[2] + xyz_a[2],
    )
    return xyz, rot


def _joint_origin(root: ET.Element, joint_name: str) -> tuple[str, str, str] | None:
    for joint in root.findall("joint"):
        if joint.get("name") != joint_name:
            continue
        parent = joint.find("parent")
        origin = joint.find("origin")
        if parent is None:
            return None
        xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        return parent.get("link"), xyz, rpy
    return None


def _compose_sixforce_to_hand_mount(
    body_root: ET.Element,
    reference_root: ET.Element,
    side: str,
    *,
    rotation_only: bool = False,
) -> tuple[str, str] | None:
    """Derive sixforce_link -> hand_base from bundled wrist mount + body sixforce joint."""
    wrist_to_hand = _joint_origin(reference_root, f"{side}_wrist_roll_link_to_{side}_hand_base_link")
    wrist_to_sixforce = _joint_origin(body_root, f"{side}_sixforce_joint")
    if wrist_to_hand is None or wrist_to_sixforce is None:
        return None

    _, wh_xyz, wh_rpy = wrist_to_hand
    _, ws_xyz, ws_rpy = wrist_to_sixforce
    t_wh = _transform_from_xyz_rpy(_parse_vec3(wh_xyz), _parse_vec3(wh_rpy))
    t_ws = _transform_from_xyz_rpy(_parse_vec3(ws_xyz), _parse_vec3(ws_rpy))
    inv_ws = _invert_transform(*t_ws)
    t_sh = _compose_transform(*inv_ws, *t_wh)
    xyz = "0 0 0" if rotation_only else _format_vec3(t_sh[0])
    return xyz, _format_vec3(_matrix_to_rpy(t_sh[1]))


def _find_mount_origin(
    body_root: ET.Element,
    reference_root: ET.Element | None,
    side: str,
    mount_link: str,
    mount_xyz: str,
    mount_rpy: str,
    use_reference_right_mount: bool = False,
    right_mount_rpy: str | None = None,
) -> tuple[str, str, str]:
    """Return xyz/rpy/parent for mounting hand_base on the body.

    Both hands attach to {side}_sixforce_link so the flange and hand share one chain:
      wrist_roll -> sixforce -> hand_base
    """
    parent = _side_mount_link(mount_link, side)
    if side == "L":
        return mount_xyz, mount_rpy, parent
    if right_mount_rpy is not None:
        return "0 0 0", right_mount_rpy, parent
    if use_reference_right_mount and reference_root is not None:
        calibrated = _compose_sixforce_to_hand_mount(
            body_root, reference_root, side, rotation_only=True
        )
        if calibrated is not None:
            return calibrated[0], calibrated[1], parent
    return "0 0 0", DEFAULT_RIGHT_MOUNT_RPY, parent


def _make_mount_joint(side: str, parent_link: str, xyz: str, rpy: str) -> ET.Element:
    joint = ET.Element("joint", name=f"{side}_wrist_to_hand_base_joint", type="fixed")
    origin = ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)
    _ = origin
    ET.SubElement(joint, "parent", link=parent_link)
    ET.SubElement(joint, "child", link=f"{side}_hand_base_link")
    return joint


def _indent(elem: ET.Element, level: int = 0) -> None:
    indent_str = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent_str + "  "
        for child in elem:
            _indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent_str
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent_str


def _copy_hand_meshes(hand_root: Path, output_meshes: Path) -> None:
    for side in ("left", "right"):
        src = hand_root / f"ubt_{side}_hand_v3_description" / "meshes" / "hand3_v1"
        if not src.is_dir():
            raise FileNotFoundError(f"Hand meshes not found: {src}")
        for stl in src.glob("*.STL"):
            shutil.copy2(stl, output_meshes / stl.name)


def _load_stl_vertices(path: Path) -> np.ndarray:
    """Load binary or ASCII STL vertices without depending on mesh libraries."""
    data = path.read_bytes()
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if 84 + triangle_count * 50 == len(data):
            vertices = np.empty((triangle_count * 3, 3), dtype=np.float64)
            for triangle_i in range(triangle_count):
                offset = 84 + triangle_i * 50 + 12
                vertices[triangle_i * 3 : triangle_i * 3 + 3] = np.asarray(
                    struct.unpack_from("<9f", data, offset), dtype=np.float64
                ).reshape(3, 3)
            return vertices

    vertices = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append(tuple(float(value) for value in fields[1:]))
    if not vertices:
        raise ValueError(f"No STL vertices found: {path}")
    return np.asarray(vertices, dtype=np.float64)


def _pca_bounds(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center, rotation and extents for a PCA-aligned bounding box."""
    mean = vertices.mean(axis=0)
    covariance = np.cov(vertices - mean, rowvar=False)
    _, eigenvectors = np.linalg.eigh(covariance)
    rotation = eigenvectors[:, ::-1]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 1] *= -1.0
    local = (vertices - mean) @ rotation
    lower = local.min(axis=0)
    upper = local.max(axis=0)
    local_center = 0.5 * (lower + upper)
    center = mean + rotation @ local_center
    return center, rotation, upper - lower


def _collision_origin(parent: ET.Element, xyz: np.ndarray, rotation: np.ndarray) -> None:
    ET.SubElement(
        parent,
        "origin",
        xyz=_format_vec3(tuple(float(value) for value in xyz)),
        rpy=_format_vec3(_matrix_to_rpy(rotation.tolist())),
    )


def _append_box_collision(
    link: ET.Element,
    vertices: np.ndarray,
    shrink: float | tuple[float, float, float] = 0.9,
) -> None:
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = 0.5 * (lower + upper)
    shrink_vec = np.asarray(shrink, dtype=np.float64)
    if shrink_vec.size == 1:
        shrink_vec = np.repeat(float(shrink_vec), 3)
    size = np.maximum((upper - lower) * shrink_vec, 0.004)
    collision = ET.SubElement(link, "collision", name="simplified_box")
    _collision_origin(collision, center, np.eye(3))
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", size=_format_vec3(tuple(float(v) for v in size)))


def _append_capsule_collisions(
    link: ET.Element,
    vertices: np.ndarray,
    radius_scale: float = 0.32,
    length_scale: float = 0.86,
) -> None:
    center, rotation, extents = _pca_bounds(vertices)
    # PCA's first axis is longest. URDF cylinders use local Z, so reorder axes.
    capsule_rotation = np.column_stack((rotation[:, 1], rotation[:, 2], rotation[:, 0]))
    if np.linalg.det(capsule_rotation) < 0.0:
        capsule_rotation[:, 1] *= -1.0
    radius = max(0.002, float(radius_scale) * float(max(extents[1], extents[2])))
    cylinder_length = max(0.004, float(extents[0]) * float(length_scale) - 2.0 * radius)

    cylinder = ET.SubElement(link, "collision", name="simplified_capsule_body")
    _collision_origin(cylinder, center, capsule_rotation)
    geometry = ET.SubElement(cylinder, "geometry")
    ET.SubElement(
        geometry,
        "cylinder",
        radius=f"{radius:.7f}",
        length=f"{cylinder_length:.7f}",
    )

    long_axis = capsule_rotation[:, 2]
    for suffix, sign in (("a", -1.0), ("b", 1.0)):
        sphere = ET.SubElement(link, "collision", name=f"simplified_capsule_{suffix}")
        sphere_center = center + sign * 0.5 * cylinder_length * long_axis
        _collision_origin(sphere, sphere_center, np.eye(3))
        sphere_geometry = ET.SubElement(sphere, "geometry")
        ET.SubElement(sphere_geometry, "sphere", radius=f"{radius:.7f}")


def _simplify_hand_collisions(links: list[ET.Element], output_meshes: Path) -> None:
    simplified = 0
    for link in links:
        link_name = link.get("name", "")
        mesh_path = output_meshes / f"{link_name}.STL"
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing official hand mesh for collider: {mesh_path}")
        vertices = _load_stl_vertices(mesh_path)
        for collision in list(link.findall("collision")):
            link.remove(collision)
        if link_name.endswith(("_palm_link", "_hand_base_link")):
            # The arm is currently moved with kinematic joint-state following in
            # Isaac. Palm/base contacts therefore behave like an infinite-force
            # paddle against dynamic parts. Keep these links visual/inertial
            # only and let grasp contact come from the fingers/thumb.
            pass
        elif link_name.endswith("_thumb_cmp_link"):
            _append_box_collision(link, vertices, shrink=(0.70, 0.70, 0.65))
        else:
            _append_capsule_collisions(link, vertices)
        simplified += 1
    print(f"Simplified hand collisions: {simplified} links (box/capsule primitives)")


def _resolve_body_mesh_dir(body_urdf: Path) -> Path | None:
    package_root = body_urdf.parent.parent.parent
    for candidate in (
        body_urdf.parent / "meshes",
        package_root / "meshes" / body_urdf.parent.name,
        package_root / "meshes" / "s2",
        body_urdf.parent.parent / "meshes",
    ):
        if candidate.is_dir():
            return candidate
    return None


def _copy_body_meshes(body_urdf: Path, output_meshes: Path) -> None:
    src = _resolve_body_mesh_dir(body_urdf)
    if src is None:
        return
    for stl in src.glob("*.STL"):
        if _is_hand_link(stl.stem) or any(
            part in stl.name for part in ("thumb", "index", "middle", "ring", "little", "palm", "hand_base")
        ):
            continue
        shutil.copy2(stl, output_meshes / stl.name)


def merge(
    hand_root: Path,
    body_urdf: Path,
    output_dir: Path,
    mount_link: str,
    mount_xyz: str = "0 0 0",
    mount_rpy: str = "0 0 0",
    reference_urdf: Path | None = DEFAULT_REFERENCE_URDF,
    use_reference_right_mount: bool = False,
    right_mount_rpy: str | None = None,
    simplify_hand_collisions: bool = True,
) -> Path:
    left_hand_urdf = hand_root / "ubt_left_hand_v3_description" / "urdf" / "hand3_v1" / "hand3_v1.urdf"
    right_hand_urdf = hand_root / "ubt_right_hand_v3_description" / "urdf" / "hand3_v1" / "hand3_v1.urdf"
    for path in (left_hand_urdf, right_hand_urdf, body_urdf):
        if not path.is_file():
            raise FileNotFoundError(path)

    body_root = _load_urdf(body_urdf)
    reference_root = _load_urdf(reference_urdf) if reference_urdf and reference_urdf.is_file() else None
    left_root = _load_urdf(left_hand_urdf)
    right_root = _load_urdf(right_hand_urdf)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_meshes = output_dir / "meshes"
    output_meshes.mkdir(exist_ok=True)
    _copy_body_meshes(body_urdf, output_meshes)
    _copy_hand_meshes(hand_root, output_meshes)

    body_links, body_joints = _body_elements(body_root)
    left_links, left_joints = _hand_elements(left_root, "L_")
    right_links, right_joints = _hand_elements(right_root, "R_")

    for elem in body_links:
        if elem.tag == "link":
            _rewrite_mesh_paths(elem, "meshes")
    for elem in left_links + right_links:
        _rewrite_mesh_paths(elem, "meshes")
    if simplify_hand_collisions:
        _simplify_hand_collisions(left_links + right_links, output_meshes)

    robot = ET.Element("robot", name="walker_s2_official")
    for elem in body_links:
        robot.append(elem)
    for elem in body_joints:
        robot.append(elem)

    l_xyz, l_rpy, l_parent = _find_mount_origin(
        body_root, reference_root, "L", mount_link, mount_xyz, mount_rpy
    )
    r_xyz, r_rpy, r_parent = _find_mount_origin(
        body_root,
        reference_root,
        "R",
        mount_link,
        mount_xyz,
        mount_rpy,
        use_reference_right_mount=use_reference_right_mount,
        right_mount_rpy=right_mount_rpy or None,
    )
    print(
        f"Hand mounts: L parent={l_parent} xyz={l_xyz} rpy={l_rpy} | "
        f"R parent={r_parent} xyz={r_xyz} rpy={r_rpy}"
    )
    robot.append(_make_mount_joint("L", l_parent, l_xyz, l_rpy))
    robot.append(_make_mount_joint("R", r_parent, r_xyz, r_rpy))

    for elem in left_links + left_joints + right_links + right_joints:
        robot.append(elem)

    _indent(robot)
    out_urdf = output_dir / "walker_s2.urdf"
    ET.ElementTree(robot).write(out_urdf, encoding="unicode", xml_declaration=True)
    return out_urdf


def main() -> int:
    args = parse_args()
    out = merge(
        hand_root=args.hand_root.resolve(),
        body_urdf=args.body_urdf.resolve(),
        output_dir=args.output_dir.resolve(),
        mount_link=args.mount_link,
        mount_xyz=args.mount_xyz,
        mount_rpy=args.mount_rpy,
        reference_urdf=args.reference_urdf.resolve() if args.reference_urdf else None,
        use_reference_right_mount=args.use_reference_right_mount,
        right_mount_rpy=args.right_mount_rpy or None,
        simplify_hand_collisions=not args.detailed_hand_collisions,
    )
    print(f"Wrote merged URDF: {out}")
    print(f"Meshes: {out.parent / 'meshes'}")
    print()
    print("Next steps:")
    print("  1. Visualize in RViz / your URDF viewer to verify wrist-to-hand alignment.")
    print("  2. Point IK config at this URDF, e.g.:")
    print(f"     --robot.urdf_path={out.relative_to(REPO_ROOT)}")
    print("  3. Isaac Sim URDF import (hand + finger DOFs):")
    print("     /home/chris/isaacsim/python.sh scripts/import_walker_s2_urdf.py --play")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
