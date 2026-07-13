"""MDP helpers for the fixed-base Walker S2 pick/place task."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def target_position(env, target_pos: tuple[float, float, float]) -> torch.Tensor:
    """Target position in each environment frame."""
    return torch.tensor(target_pos, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)


def body_position(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body position in each environment frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if body_pos.ndim == 3:
        body_pos = body_pos[:, 0, :]
    return body_pos - env.scene.env_origins


def object_position(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Object root position in each environment frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def palm_object_distance(
    env,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Distance from palm body to object center."""
    palm_pos = body_position(env, palm_cfg)
    obj_pos = object_position(env, object_cfg)
    return torch.norm(palm_pos - obj_pos, dim=1)


def reach_object_reward(
    env,
    std: float,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward for moving the right palm near the object."""
    distance = palm_object_distance(env, palm_cfg, object_cfg)
    return torch.exp(-distance / std)


def object_lifted(
    env,
    table_top_z: float,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Whether the object is lifted above the table by a margin."""
    obj_pos = object_position(env, object_cfg)
    return obj_pos[:, 2] > table_top_z + lift_height


def lift_object_reward(
    env,
    table_top_z: float,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sparse lift reward."""
    return object_lifted(env, table_top_z, lift_height, object_cfg).float()


def object_target_distance(
    env,
    target_pos: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Distance from object center to target center."""
    obj_pos = object_position(env, object_cfg)
    target = target_position(env, target_pos)
    return torch.norm(obj_pos - target, dim=1)


def object_target_reward(
    env,
    target_pos: tuple[float, float, float],
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward for moving the object to the target."""
    distance = object_target_distance(env, target_pos, object_cfg)
    return torch.exp(-distance / std)


def object_near_target(
    env,
    target_pos: tuple[float, float, float],
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Success condition for placing the object near the target."""
    return object_target_distance(env, target_pos, object_cfg) < threshold


def success_reward(
    env,
    target_pos: tuple[float, float, float],
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sparse success reward."""
    return object_near_target(env, target_pos, threshold, object_cfg).float()
