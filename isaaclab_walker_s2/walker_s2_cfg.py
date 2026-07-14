"""Walker S2 robot asset configuration for IsaacLab.

This file only defines the robot ``ArticulationCfg``. It does not define an
environment, rewards, observations, or training configuration.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


REPO_ROOT = Path(__file__).resolve().parents[1]
WALKER_S2_USD_PATH = (
    REPO_ROOT
    / "assets"
    / "resources"
    / "walker_s2_description_hand3_v1_left_hand3_v1_right"
    / "walker_s2_with_hands_isaaclab.usd"
)


WALKER_S2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(WALKER_S2_USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.75, -0.2, 0.91),
        rot=(0.70710678, 0.0, 0.0, 0.70710678),
        joint_pos={
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
            "hand3_v1_left_L_thumb_cmp_joint": -0.03,
            "hand3_v1_left_L_thumb_mpp_joint": 0.03,
            "hand3_v1_left_L_thumb_ip_joint": 0.03,
            "hand3_v1_left_L_index_mpp_joint": 0.03,
            "hand3_v1_left_L_index_ip_joint": 0.03,
            "hand3_v1_left_L_middle_mpp_joint": 0.03,
            "hand3_v1_left_L_middle_ip_joint": 0.03,
            "hand3_v1_left_L_ring_mpp_joint": 0.03,
            "hand3_v1_left_L_ring_ip_joint": 0.03,
            "hand3_v1_left_L_little_mpp_joint": 0.03,
            "hand3_v1_left_L_little_ip_joint": 0.03,
            "hand3_v1_right_R_thumb_cmp_joint": 0.03,
            "hand3_v1_right_R_thumb_mpp_joint": 0.03,
            "hand3_v1_right_R_thumb_ip_joint": 0.03,
            "hand3_v1_right_R_index_mpp_joint": 0.03,
            "hand3_v1_right_R_index_ip_joint": 0.03,
            "hand3_v1_right_R_middle_mpp_joint": 0.03,
            "hand3_v1_right_R_middle_ip_joint": 0.03,
            "hand3_v1_right_R_ring_mpp_joint": 0.03,
            "hand3_v1_right_R_ring_ip_joint": 0.03,
            "hand3_v1_right_R_little_mpp_joint": 0.03,
            "hand3_v1_right_R_little_ip_joint": 0.03,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_hip_roll_joint"],
            effort_limit_sim=220.0,
            stiffness=900.0,
            damping=90.0,
        ),
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_hip_yaw_joint"],
            effort_limit_sim=60.0,
            stiffness=900.0,
            damping=90.0,
        ),
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_hip_pitch_joint"],
            effort_limit_sim=200.0,
            stiffness=900.0,
            damping=90.0,
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_knee_pitch_joint"],
            effort_limit_sim=250.0,
            stiffness=900.0,
            damping=90.0,
        ),
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_ankle_pitch_joint", "[LR]_ankle_roll_joint"],
            effort_limit_sim=60.0,
            stiffness=900.0,
            damping=90.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_pitch_joint"],
            effort_limit_sim=250.0,
            stiffness=1500.0,
            damping=150.0,
        ),
        "shoulder_pitch_roll": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_shoulder_pitch_joint", "[LR]_shoulder_roll_joint"],
            effort_limit_sim=300.0,
            stiffness=900.0,
            damping=90.0,
        ),
        "shoulder_yaw_elbow_roll": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_shoulder_yaw_joint", "[LR]_elbow_roll_joint"],
            effort_limit_sim=300.0,
            stiffness=900.0,
            damping=90.0,
        ),
        "elbow_yaw_wrist": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_elbow_yaw_joint", "[LR]_wrist_pitch_joint", "[LR]_wrist_roll_joint"],
            effort_limit_sim=300.0,
            stiffness=900.0,
            damping=90.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim=20.0,
            stiffness=20.0,
            damping=4.0,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                "hand3_v1_left_L_thumb_cmp_joint",
                "hand3_v1_left_L_thumb_mpp_joint",
                "hand3_v1_left_L_thumb_ip_joint",
                "hand3_v1_left_L_index_mpp_joint",
                "hand3_v1_left_L_index_ip_joint",
                "hand3_v1_left_L_middle_mpp_joint",
                "hand3_v1_left_L_middle_ip_joint",
                "hand3_v1_left_L_ring_mpp_joint",
                "hand3_v1_left_L_ring_ip_joint",
                "hand3_v1_left_L_little_mpp_joint",
                "hand3_v1_left_L_little_ip_joint",
                "hand3_v1_right_R_thumb_cmp_joint",
                "hand3_v1_right_R_thumb_mpp_joint",
                "hand3_v1_right_R_thumb_ip_joint",
                "hand3_v1_right_R_index_mpp_joint",
                "hand3_v1_right_R_index_ip_joint",
                "hand3_v1_right_R_middle_mpp_joint",
                "hand3_v1_right_R_middle_ip_joint",
                "hand3_v1_right_R_ring_mpp_joint",
                "hand3_v1_right_R_ring_ip_joint",
                "hand3_v1_right_R_little_mpp_joint",
                "hand3_v1_right_R_little_ip_joint",
            ],
            effort_limit_sim=80.0,
            stiffness=120.0,
            damping=8.0,
        ),
    },
)
