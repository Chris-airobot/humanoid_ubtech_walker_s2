"""World <-> Pinocchio base coordinate transformation utilities.

This module provides rigid body transformation tools between the world coordinate system
and the Pinocchio URDF root (base) coordinate system, primarily used for aligning simulation
world coordinates with Pinocchio robot model coordinates in Isaac Sim environment.
"""

import numpy as np
import pinocchio as pin


class CoordinateTransform:
    """Coordinate transformation manager for rigid body transforms between world and Pinocchio base frames.

    The transformation relationship is computed via the torso_link anchor point, with core formula:
        base_world = torso_world * inv(pin_torso_in_base)
    Where:
    - torso_world: Torso link pose in world coordinate system
    - pin_torso_in_base: Torso link pose in Pinocchio base coordinate system
    - base_world: Pinocchio base pose in world coordinate system (the final transform being solved)

    Attributes:
        robot_world_pos (np.ndarray): Pinocchio base position in world coordinates, shape=(3,), dtype=float
        robot_world_R (np.ndarray): Pinocchio base rotation matrix in world coordinates, shape=(3,3), dtype=float
        robot_world_R_inv (np.ndarray): Inverse (transpose) of rotation matrix for inverse transformation, shape=(3,3), dtype=float
    """

    def __init__(self, robot_world_pos: np.ndarray, robot_world_R: np.ndarray):
        """Initialize coordinate transformation manager

        Args:
            robot_world_pos (np.ndarray): Pinocchio base position in world coordinates, shape=(3,)
            robot_world_R (np.ndarray): Pinocchio base rotation matrix in world coordinates, shape=(3,3)

        Internal processing:
            1. Convert input to float-type numpy arrays for numerical precision
            2. Pre-compute inverse of rotation matrix (orthogonal matrix inverse = transpose) to avoid redundant calculations
        """
        self.robot_world_pos = np.asarray(robot_world_pos, dtype=float)
        self.robot_world_R = np.asarray(robot_world_R, dtype=float)
        self.robot_world_R_inv = self.robot_world_R.T  # Orthogonal matrix inverse = transpose

    @classmethod
    def from_torso_link(
        cls,
        ik_solver,
        torso_prim_path: str = "/Root/Ref_Xform/Ref/torso_link",
    ):
        """Compute transformation via torso_link (legacy USD / challenge URDF layout)."""
        return cls.from_anchor_frame(
            ik_solver=ik_solver,
            frame_name="torso_link",
            frame_prim_path=torso_prim_path,
        )

    @classmethod
    def from_anchor_frame(
        cls,
        ik_solver,
        frame_name: str,
        frame_prim_path: str,
    ):
        """Compute world/base transform using any URDF link present in sim + Pinocchio."""
        from pxr import UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        xc = UsdGeom.XformCache()

        frame_prim = stage.GetPrimAtPath(frame_prim_path)
        if not frame_prim.IsValid():
            raise AssertionError(
                f"{frame_name} prim not found at {frame_prim_path}"
            )

        frame_tf = xc.GetLocalToWorldTransform(frame_prim)
        frame_t = np.array(frame_tf.ExtractTranslation(), dtype=float)
        frame_R_gf = frame_tf.ExtractRotationMatrix()
        frame_R = np.array(
            [[frame_R_gf[i][j] for j in range(3)] for i in range(3)], dtype=float
        ).T
        frame_world = pin.SE3(frame_R, frame_t)

        if not ik_solver.model.existFrame(frame_name):
            raise AssertionError(f"{frame_name} not found in Pinocchio model")

        frame_fid = ik_solver.model.getFrameId(frame_name)
        pin.forwardKinematics(ik_solver.model, ik_solver.data, ik_solver.q)
        pin.updateFramePlacements(ik_solver.model, ik_solver.data)
        pin_frame = ik_solver.data.oMf[frame_fid].copy()

        base_world = frame_world * pin_frame.inverse()
        robot_world_R = np.array(base_world.rotation)
        robot_world_pos = np.array(base_world.translation)

        print(f"[Coordinate] {frame_name} world position: {frame_t}")
        print(f"[Coordinate] URDF root world position:  {robot_world_pos}")
        print(f"[Coordinate] base rotation matrix:\n{robot_world_R}")

        return cls(robot_world_pos, robot_world_R)

    def world_to_robot(self, world_xyz: np.ndarray) -> np.ndarray:
        """World coordinate system -> Pinocchio base coordinate system conversion

        Transformation formula:
            robot_xyz = R_inv @ (world_xyz - base_pos)
        Where:
        - R_inv: Inverse (transpose) of base rotation matrix
        - base_pos: Base position in world coordinate system

        Args:
            world_xyz (np.ndarray): Point coordinates in world coordinate system, shape=(3,) or (N,3) (supports batch conversion)

        Returns:
            np.ndarray: Point coordinates in Pinocchio base frame, same shape as input, dtype=float

        Example:
            world_xyz = np.array([1.0, 2.0, 3.0])
            robot_xyz = transform.world_to_robot(world_xyz)
        """
        # Convert to float array for type consistency
        world_xyz_arr = np.asarray(world_xyz, dtype=float)
        # Calculate offset of world coordinates relative to base origin
        d = world_xyz_arr - self.robot_world_pos
        # Rotate offset vector to base coordinate system (supports batch conversion)
        return self.robot_world_R_inv @ d

    def robot_to_world(self, robot_xyz: np.ndarray) -> np.ndarray:
        """Pinocchio base coordinate system -> World coordinate system conversion

        Transformation formula:
            world_xyz = base_pos + R @ robot_xyz
        Where:
        - R: Base rotation matrix
        - base_pos: Base position in world coordinate system

        Args:
            robot_xyz (np.ndarray): Point coordinates in Pinocchio base frame, shape=(3,) or (N,3) (supports batch conversion)

        Returns:
            np.ndarray: Point coordinates in world coordinate system, same shape as input, dtype=float

        Example:
            robot_xyz = np.array([0.1, 0.2, 0.3])
            world_xyz = transform.robot_to_world(robot_xyz)
        """
        # Convert to float array for type consistency
        robot_xyz_arr = np.asarray(robot_xyz, dtype=float)
        # Rotate base coordinates to world coordinate system, then add base world position
        return self.robot_world_pos + self.robot_world_R @ robot_xyz_arr

    def verify_ee_alignment(self, ik_solver) -> None:
        """Verify coordinate transformation accuracy by comparing Pinocchio FK computed end-effector pose with actual Isaac Sim pose

        Verification logic:
            1. Verify left and right sixforce_link separately
            2. Compute end-effector pose via Pinocchio FK and transform to world coordinates
            3. Read end-effector world pose from Isaac Sim
            4. Calculate Euclidean distance between them and output verification result (smaller distance = more accurate transformation)

        Args:
            ik_solver: Initialized DualArmIK instance for retrieving end-effector FK poses

        Returns:
            None: Directly prints verification results

        Output example:
            [Verify] left sixforce frame diff: 0.0002m (FK->world=[1.0,2.0,3.0], Isaac=[1.0001,2.0001,3.0001])
        """
        from pxr import UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        xc = UsdGeom.XformCache()

        log = []
        for side, link_name in [("left", "L_sixforce_link"), ("right", "R_sixforce_link")]:
            # 1. Get end-effector pose in base frame via Pinocchio FK
            se3 = ik_solver.get_ee_pose(side)
            # 2. Transform to world coordinate system
            pin_world = self.robot_to_world(se3.translation)
            # 3. Read end-effector world position from Isaac Sim
            prim = stage.GetPrimAtPath(f"/Root/Ref_Xform/Ref/{link_name}")
            isaac_pos = np.array(xc.GetLocalToWorldTransform(prim).ExtractTranslation())
            # 4. Calculate Euclidean norm of position difference (verify accuracy)
            diff_norm = np.linalg.norm(isaac_pos - pin_world)
            error_msg = {
                "arm": side,
                "base_pos": se3.translation,
                "world_pos": isaac_pos,
                "base_world_pos": self.robot_world_pos,
                "base_world_R": self.robot_world_R,
            }
            log.append(error_msg)

            # Print verification results
            print(
                f"[Verify] {side} sixforce frame diff: {diff_norm:.4f}m  "
                f"(FK->world={pin_world}, Isaac={isaac_pos})"
            )
        return log
