"""Fixed-base Walker S2 pick/place environment configuration.

This follows IsaacLab's manager-based task style while using the local Walker S2
asset and the table layout validated in ``scripts/spawn_walker_s2_table_scene.py``.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_walker_s2 import WALKER_S2_CFG

from . import mdp
from .actions import WalkerS2PalmIKActionCfg


TABLE_CENTER = (0.75, 0.30, 1.02)
TABLE_SIZE = (1.20, 0.65, 0.04)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] * 0.5
CUBE_CENTER = (0.92, 0.20, 1.105)
CUBE_SIZE = (0.035, 0.035, 0.13)
TARGET_CENTER = (0.62, 0.24)
TARGET_SIZE = (0.28, 0.20, 0.01)
TARGET_POS = (TARGET_CENTER[0], TARGET_CENTER[1], TABLE_TOP_Z + TARGET_SIZE[2] * 0.5)

RIGHT_ARM_JOINTS = [
    "R_shoulder_pitch_joint",
    "R_shoulder_roll_joint",
    "R_shoulder_yaw_joint",
    "R_elbow_roll_joint",
    "R_elbow_yaw_joint",
    "R_wrist_pitch_joint",
    "R_wrist_roll_joint",
]

RIGHT_HAND_JOINTS = [
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
]

RIGHT_HAND_OPEN_COMMAND = {joint_name: 0.03 for joint_name in RIGHT_HAND_JOINTS}
RIGHT_HAND_CLOSE_COMMAND = {
    "hand3_v1_right_R_thumb_cmp_joint": 0.95,
    "hand3_v1_right_R_thumb_mpp_joint": 1.03,
    "hand3_v1_right_R_thumb_ip_joint": 1.02,
    "hand3_v1_right_R_index_mpp_joint": 1.20,
    "hand3_v1_right_R_index_ip_joint": 1.30,
    "hand3_v1_right_R_middle_mpp_joint": 1.20,
    "hand3_v1_right_R_middle_ip_joint": 1.30,
    "hand3_v1_right_R_ring_mpp_joint": 1.20,
    "hand3_v1_right_R_ring_ip_joint": 1.30,
    "hand3_v1_right_R_little_mpp_joint": 1.15,
    "hand3_v1_right_R_little_ip_joint": 1.25,
}


@configclass
class WalkerS2PickPlaceSceneCfg(InteractiveSceneCfg):
    """Scene configuration for fixed-base Walker S2 pick/place."""

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())

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
        init_state=AssetBaseCfg.InitialStateCfg(pos=TARGET_POS),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=CUBE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.05, 0.05), roughness=0.7),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_CENTER),
    )

    robot = WALKER_S2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    right_arm = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINTS,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )

    right_hand = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_HAND_JOINTS,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class IKActionsCfg:
    """Compact palm/grip action specifications for RL."""

    palm_ik = WalkerS2PalmIKActionCfg(
        asset_name="robot",
        object_asset_name="object",
        right_arm_joint_names=RIGHT_ARM_JOINTS,
        right_hand_joint_names=RIGHT_HAND_JOINTS,
        right_hand_open_command=RIGHT_HAND_OPEN_COMMAND,
        right_hand_close_command=RIGHT_HAND_CLOSE_COMMAND,
        default_nudge=(0.08, 0.0, 0.0),
        nudge_min=(-0.03, -0.08, -0.04),
        nudge_max=(0.12, 0.08, 0.14),
        delta_scale=(0.01, 0.01, 0.01),
        thumb_close_scale=0.8,
        finger_close_scale=1.8,
        quiet_ik=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        right_arm_joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True)},
        )
        right_arm_joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True)},
        )
        right_hand_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS, preserve_order=True)},
        )
        right_palm_pos = ObsTerm(
            func=mdp.body_position,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="hand3_v1_right_R_palm_link")},
        )
        object_pos = ObsTerm(func=mdp.object_position, params={"asset_cfg": SceneEntityCfg("object")})
        target_pos = ObsTerm(func=mdp.target_position, params={"target_pos": TARGET_POS})
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    reset_object_position = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "z": (0.0, 0.0), "yaw": (-0.15, 0.15)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reach_object = RewTerm(
        func=mdp.reach_object_reward,
        weight=2.0,
        params={
            "std": 0.25,
            "palm_cfg": SceneEntityCfg("robot", body_names="hand3_v1_right_R_palm_link"),
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    lift_object = RewTerm(
        func=mdp.lift_object_reward,
        weight=8.0,
        params={"table_top_z": TABLE_TOP_Z, "lift_height": 0.08, "object_cfg": SceneEntityCfg("object")},
    )

    lift_progress = RewTerm(
        func=mdp.object_lift_progress_reward,
        weight=4.0,
        params={"initial_height": CUBE_CENTER[2], "lift_height": 0.08, "object_cfg": SceneEntityCfg("object")},
    )

    object_to_target = RewTerm(
        func=mdp.lifted_object_target_reward,
        weight=10.0,
        params={
            "target_pos": TARGET_POS,
            "std": 0.25,
            "initial_height": CUBE_CENTER[2],
            "min_lift": 0.03,
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    place_on_target = RewTerm(
        func=mdp.placed_on_target_reward,
        weight=30.0,
        params={
            "target_pos": TARGET_POS,
            "target_size": TARGET_SIZE,
            "initial_height": CUBE_CENTER[2],
            "height_tolerance": 0.04,
            "max_speed": 0.25,
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    success = RewTerm(
        func=mdp.placed_on_target_reward,
        weight=20.0,
        params={
            "target_pos": TARGET_POS,
            "target_size": TARGET_SIZE,
            "initial_height": CUBE_CENTER[2],
            "height_tolerance": 0.04,
            "max_speed": 0.25,
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True)},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.75, "asset_cfg": SceneEntityCfg("object")},
    )

    success = DoneTerm(
        func=mdp.object_placed_on_target,
        params={
            "target_pos": TARGET_POS,
            "target_size": TARGET_SIZE,
            "initial_height": CUBE_CENTER[2],
            "height_tolerance": 0.04,
            "max_speed": 0.25,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class WalkerS2PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Fixed-base Walker S2 pick/place RL environment."""

    scene: WalkerS2PickPlaceSceneCfg = WalkerS2PickPlaceSceneCfg(num_envs=1, env_spacing=3.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0

        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation

        self.viewer.eye = (2.0, -1.25, 1.55)
        self.viewer.lookat = (0.75, 0.15, 1.0)


@configclass
class WalkerS2IKPickPlaceEnvCfg(WalkerS2PickPlaceEnvCfg):
    """Walker S2 pick/place env with a compact palm-IK action interface."""

    actions: IKActionsCfg = IKActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = self.scene.robot.copy()
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
        self.scene.object.init_state.pos = (CUBE_CENTER[0] - 0.03, CUBE_CENTER[1], CUBE_CENTER[2])

        for name, actuator_cfg in self.scene.robot.actuators.items():
            if name == "hands":
                actuator_cfg.effort_limit = 2000.0
                actuator_cfg.effort_limit_sim = 2000.0
                actuator_cfg.stiffness = 500.0
                actuator_cfg.damping = 30.0
            elif name == "head":
                actuator_cfg.effort_limit = 100.0
                actuator_cfg.effort_limit_sim = 100.0
                actuator_cfg.stiffness = 80.0
                actuator_cfg.damping = 8.0
            else:
                actuator_cfg.effort_limit = 5000.0
                actuator_cfg.effort_limit_sim = 5000.0
                if "shoulder" in name or "elbow" in name or "wrist" in name or name == "waist":
                    actuator_cfg.stiffness = 3000.0
                    actuator_cfg.damping = 180.0
