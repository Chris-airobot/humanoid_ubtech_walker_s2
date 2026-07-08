from __future__ import annotations

import numpy as np


class WalkerS2CartesianController:
    """Persistent dual-arm IK controller for manual Cartesian teleoperation."""

    SMOOTH_ALPHA = 0.3

    def __init__(self, urdf_path, dof_names, ready_positions):
        from Ubtech_sim.source.DualArmIK import DualArmIK

        self.ik = DualArmIK(str(urdf_path))
        self.dof_names = list(dof_names)
        self.name_to_index = {name: index for index, name in enumerate(self.dof_names)}
        self.ik.sync_joint_positions(self.dof_names, ready_positions)
        self.ik.save_initial_q()
        self.ik.set_neutral_config(
            [ready_positions[self.name_to_index[name]] for name in DualArmIK.LEFT_ARM_JOINTS],
            [ready_positions[self.name_to_index[name]] for name in DualArmIK.RIGHT_ARM_JOINTS],
        )
        self.reset(ready_positions)

    def reset(self, joint_positions) -> None:
        self.ik.sync_joint_positions(self.dof_names, joint_positions)
        self.last_arm_positions = {
            "left": np.array(
                [joint_positions[self.name_to_index[name]] for name in self.ik.LEFT_ARM_JOINTS],
                dtype=float,
            ),
            "right": np.array(
                [joint_positions[self.name_to_index[name]] for name in self.ik.RIGHT_ARM_JOINTS],
                dtype=float,
            ),
        }

    def step(self, joint_positions, arm_deltas):
        active = {
            side: np.asarray(delta, dtype=float)
            for side, delta in arm_deltas.items()
            if np.linalg.norm(delta) > 1e-10
        }
        if not active:
            return np.asarray(joint_positions, dtype=float), {}

        # The official controller forms each target from current FK feedback.
        # This prevents unresolved orientation errors from accumulating into a jump.
        self.ik.sync_joint_positions(self.dof_names, joint_positions)
        target_xyzrpy = {}
        for side, delta in active.items():
            current = self.ik.se3_to_xyzrpy(self.ik.get_ee_pose(side))
            target_xyzrpy[side] = current + delta

        result = self.ik.solve_dual_arm(
            left_target_xyzrpy=target_xyzrpy.get("left"),
            right_target_xyzrpy=target_xyzrpy.get("right"),
            isaac_joint_names=self.dof_names,
            isaac_joint_positions=joint_positions,
        )
        q = np.asarray(joint_positions, dtype=float).copy()
        status = {}
        for side in active:
            names = result.get(f"{side}_joint_names", ())
            solved = np.asarray(result.get(f"{side}_joint_positions", ()), dtype=float)
            previous = self.last_arm_positions[side]
            smoothed = previous + self.SMOOTH_ALPHA * (solved - previous)
            self.last_arm_positions[side] = smoothed.copy()
            for name, value in zip(names, smoothed):
                q[self.name_to_index[name]] = float(value)
            status[side] = bool(result.get(f"{side}_success", False))
        return q, status
