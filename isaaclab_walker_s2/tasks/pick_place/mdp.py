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


def body_orientation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body quaternion in world axes using IsaacLab's ``wxyz`` convention."""

    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    if body_quat.ndim == 3:
        body_quat = body_quat[:, 0, :]
    return body_quat


def body_velocity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body linear and angular velocity in world axes."""

    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_vel_w[:, asset_cfg.body_ids]
    if body_vel.ndim == 3:
        body_vel = body_vel[:, 0, :]
    return body_vel


def object_position(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Object root position in each environment frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def object_orientation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Object root quaternion in world axes using IsaacLab's ``wxyz`` convention."""

    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


def object_velocity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Object root linear and angular velocity in world axes."""

    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_vel_w


def object_relative_to_palm(
    env,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object center minus palm center in world-aligned environment coordinates."""

    return object_position(env, object_cfg) - body_position(env, palm_cfg)


def target_relative_to_object(
    env,
    target_pos: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Target center minus object center in world-aligned environment coordinates."""

    return target_position(env, target_pos) - object_position(env, object_cfg)


def grasp_observation(
    env,
    table_top_z: float,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    term_name: str = "palm_ik",
    grip_index: int = 6,
) -> torch.Tensor:
    """Deployable grasp proxies: grip, distance, object height, and object speed."""

    grip = action_grip_command(env, term_name, grip_index)
    distance = palm_object_distance(env, palm_cfg, object_cfg)
    height = object_position(env, object_cfg)[:, 2] - table_top_z
    speed = object_speed(env, object_cfg)
    return torch.stack([grip, distance, height, speed], dim=-1)


def palm_object_distance(
    env,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Distance from palm body to object center."""
    palm_pos = body_position(env, palm_cfg)
    obj_pos = object_position(env, object_cfg)
    return torch.norm(palm_pos - obj_pos, dim=1)


def action_grip_command(env, term_name: str = "palm_ik", grip_index: int = 6) -> torch.Tensor:
    """Current compact-action grip command, clamped to [0, 1]."""
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return torch.zeros(env.num_envs, device=env.device)

    try:
        term = action_manager.get_term(term_name)
        grip_index = getattr(term, "grip_index", grip_index)
        processed_actions = term.processed_actions
        if processed_actions.shape[1] > grip_index:
            return torch.clamp(processed_actions[:, grip_index], 0.0, 1.0)
    except Exception:
        pass

    action = action_manager.action
    term_names = list(action_manager.active_terms)
    term_dims = list(action_manager.action_term_dim)
    if term_name not in term_names:
        return torch.zeros(env.num_envs, device=action.device)

    term_id = term_names.index(term_name)
    try:
        grip_index = getattr(action_manager.get_term(term_name), "grip_index", grip_index)
    except Exception:
        pass
    action_id = sum(term_dims[:term_id]) + grip_index
    if action_id >= action.shape[1]:
        return torch.zeros(env.num_envs, device=action.device)
    return torch.clamp(action[:, action_id], 0.0, 1.0)


def action_grip_available(env, term_name: str = "palm_ik", grip_index: int = 6) -> torch.Tensor:
    """Whether the current action space has a compact grip command."""
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return torch.zeros(env.num_envs, device=env.device)

    action = action_manager.action
    term_names = list(action_manager.active_terms)
    term_dims = list(action_manager.action_term_dim)
    if term_name not in term_names:
        return torch.zeros(env.num_envs, device=action.device)

    term_id = term_names.index(term_name)
    try:
        grip_index = getattr(action_manager.get_term(term_name), "grip_index", grip_index)
    except Exception:
        pass
    action_id = sum(term_dims[:term_id]) + grip_index
    if action_id >= action.shape[1]:
        return torch.zeros(env.num_envs, device=action.device)
    return torch.ones(env.num_envs, device=action.device)


def object_speed(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Object linear speed."""
    asset: RigidObject = env.scene[object_cfg.name]
    return torch.norm(asset.data.root_vel_w[:, :3], dim=1)


def reach_object_reward(
    env,
    std: float,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward for moving the right palm near the object."""
    distance = palm_object_distance(env, palm_cfg, object_cfg)
    return torch.exp(-distance / std)


def grip_near_object_reward(
    env,
    std: float,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    term_name: str = "palm_ik",
    grip_index: int = 6,
) -> torch.Tensor:
    """Reward closing the hand only when the palm is close to the object."""
    grip = action_grip_command(env, term_name, grip_index)
    distance = palm_object_distance(env, palm_cfg, object_cfg)
    return grip * torch.exp(-distance / std)


def early_grip_penalty(
    env,
    close_distance: float,
    far_distance: float,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    term_name: str = "palm_ik",
    grip_index: int = 6,
) -> torch.Tensor:
    """Penalize closing the hand while the palm is still far from the object."""
    grip = action_grip_command(env, term_name, grip_index)
    distance = palm_object_distance(env, palm_cfg, object_cfg)
    far_scale = torch.clamp((distance - close_distance) / max(far_distance - close_distance, 1e-6), 0.0, 1.0)
    return grip * far_scale


def object_lifted(
    env,
    table_top_z: float,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Whether the object is lifted above the table by a margin."""
    obj_pos = object_position(env, object_cfg)
    return obj_pos[:, 2] > table_top_z + lift_height


def _ordered_pick_place_history(
    env,
    target_pos: tuple[float, float, float],
    target_size: tuple[float, float, float],
    initial_height: float,
    min_lift: float,
    carry_min_lift: float | None,
    close_distance: float,
    grip_close_threshold: float,
    held_distance: float,
    grasp_dwell_steps: int,
    lift_dwell_steps: int,
    carry_dwell_steps: int,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    term_name: str = "palm_ik",
    grip_index: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Track a sustained grasp -> lift -> carry sequence for each environment."""

    def state_tensor(name: str, dtype: torch.dtype, fill_value: int | bool = 0) -> torch.Tensor:
        value = getattr(env, name, None)
        if value is None or value.shape[0] != env.num_envs or value.device != torch.device(env.device):
            value = torch.full((env.num_envs,), fill_value, dtype=dtype, device=env.device)
            setattr(env, name, value)
        return value

    grasp_count = state_tensor("_walker_s2_pick_place_grasp_count", torch.long)
    lift_count = state_tensor("_walker_s2_pick_place_lift_count", torch.long)
    carry_count = state_tensor("_walker_s2_pick_place_carry_count", torch.long)
    grasp_confirmed = state_tensor("_walker_s2_pick_place_grasp_confirmed", torch.bool)
    lift_confirmed = state_tensor("_walker_s2_pick_place_lift_confirmed", torch.bool)
    carry_confirmed = state_tensor("_walker_s2_pick_place_carry_confirmed", torch.bool)
    last_step = state_tensor("_walker_s2_pick_place_history_last_step", torch.long, -1)

    episode_step = getattr(env, "episode_length_buf", None)
    if episode_step is None:
        update_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        reset_mask = torch.zeros_like(update_mask)
    else:
        episode_step = episode_step.to(device=env.device, dtype=torch.long)
        reset_mask = torch.logical_and(
            episode_step <= 1,
            torch.logical_or(last_step < 0, episode_step < last_step),
        )
        update_mask = episode_step != last_step

    for value in (grasp_count, lift_count, carry_count):
        value[reset_mask] = 0
    for value in (grasp_confirmed, lift_confirmed, carry_confirmed):
        value[reset_mask] = False

    obj_pos = object_position(env, object_cfg)
    grip = action_grip_command(env, term_name, grip_index)
    distance = palm_object_distance(env, palm_cfg, object_cfg)
    in_target = object_in_target_area(env, target_pos, target_size, object_cfg)
    lifted_now = obj_pos[:, 2] > initial_height + min_lift
    carry_lift_threshold = min_lift if carry_min_lift is None else carry_min_lift
    carry_lifted_now = obj_pos[:, 2] > initial_height + carry_lift_threshold
    closed_near_now = torch.logical_and(grip >= grip_close_threshold, distance <= close_distance)
    held_now = torch.logical_and(grip >= grip_close_threshold, distance <= held_distance)

    def update_counter(counter: torch.Tensor, condition: torch.Tensor, dwell_steps: int) -> torch.Tensor:
        next_count = torch.where(condition, counter + 1, torch.zeros_like(counter))
        counter[update_mask] = next_count[update_mask]
        return counter >= max(int(dwell_steps), 1)

    grasp_confirmed[:] = torch.logical_or(
        grasp_confirmed,
        update_counter(grasp_count, closed_near_now, grasp_dwell_steps),
    )
    lifted_while_held = torch.logical_and(
        torch.logical_and(grasp_confirmed, lifted_now),
        torch.logical_and(held_now, torch.logical_not(in_target)),
    )
    lift_confirmed[:] = torch.logical_or(
        lift_confirmed,
        update_counter(lift_count, lifted_while_held, lift_dwell_steps),
    )
    carried_over_target = torch.logical_and(
        torch.logical_and(lift_confirmed, carry_lifted_now),
        torch.logical_and(held_now, in_target),
    )
    carry_confirmed[:] = torch.logical_or(
        carry_confirmed,
        update_counter(carry_count, carried_over_target, carry_dwell_steps),
    )

    if episode_step is not None:
        last_step[update_mask] = episode_step[update_mask]

    return grasp_confirmed.clone(), lift_confirmed.clone(), carry_confirmed.clone()


def lift_object_reward(
    env,
    table_top_z: float,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sparse lift reward."""
    return object_lifted(env, table_top_z, lift_height, object_cfg).float()


def object_lift_progress_reward(
    env,
    initial_height: float,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward for raising the object above its reset height."""
    obj_pos = object_position(env, object_cfg)
    return torch.clamp((obj_pos[:, 2] - initial_height) / lift_height, min=0.0, max=1.0)


def hold_lifted_object_reward(
    env,
    initial_height: float,
    lift_height: float,
    palm_std: float,
    speed_std: float,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    term_name: str = "palm_ik",
    grip_index: int = 6,
) -> torch.Tensor:
    """Reward keeping the object close to the palm after it has been lifted."""
    grip = action_grip_command(env, term_name, grip_index)
    height_progress = object_lift_progress_reward(env, initial_height, lift_height, object_cfg)
    palm_score = torch.exp(-palm_object_distance(env, palm_cfg, object_cfg) / palm_std)
    speed_score = torch.exp(-object_speed(env, object_cfg) / speed_std)
    return grip * height_progress * palm_score * speed_score


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


def lifted_object_target_reward(
    env,
    target_pos: tuple[float, float, float],
    std: float,
    initial_height: float,
    min_lift: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward moving the object toward the target only after it is lifted."""
    obj_pos = object_position(env, object_cfg)
    target = target_position(env, target_pos)
    xy_distance = torch.norm(obj_pos[:, :2] - target[:, :2], dim=1)
    lifted = obj_pos[:, 2] > initial_height + min_lift
    return lifted.float() * torch.exp(-xy_distance / std)


def carried_object_target_reward(
    env,
    target_pos: tuple[float, float, float],
    target_std: float,
    palm_std: float,
    initial_height: float,
    min_lift: float,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    term_name: str = "palm_ik",
    grip_index: int = 6,
) -> torch.Tensor:
    """Reward carrying a held, lifted object toward the target area."""
    grip = action_grip_command(env, term_name, grip_index)
    obj_pos = object_position(env, object_cfg)
    target = target_position(env, target_pos)
    lifted = (obj_pos[:, 2] > initial_height + min_lift).float()
    target_score = torch.exp(-torch.norm(obj_pos[:, :2] - target[:, :2], dim=1) / target_std)
    palm_score = torch.exp(-palm_object_distance(env, palm_cfg, object_cfg) / palm_std)
    return grip * lifted * palm_score * target_score


def object_in_target_area(
    env,
    target_pos: tuple[float, float, float],
    target_size: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Whether the object root is inside the green target rectangle in x/y."""
    obj_pos = object_position(env, object_cfg)
    target = target_position(env, target_pos)
    half_x = target_size[0] * 0.5
    half_y = target_size[1] * 0.5
    in_x = torch.abs(obj_pos[:, 0] - target[:, 0]) <= half_x
    in_y = torch.abs(obj_pos[:, 1] - target[:, 1]) <= half_y
    return torch.logical_and(in_x, in_y)


def object_placed_on_target(
    env,
    target_pos: tuple[float, float, float],
    target_size: tuple[float, float, float],
    initial_height: float,
    height_tolerance: float,
    max_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    minimum_height: float | None = None,
) -> torch.Tensor:
    """Whether the object has been placed in the green area and is settled."""
    asset: RigidObject = env.scene[object_cfg.name]
    obj_pos = object_position(env, object_cfg)
    speed = torch.norm(asset.data.root_vel_w[:, :3], dim=1)
    in_area = object_in_target_area(env, target_pos, target_size, object_cfg)
    if minimum_height is None:
        near_table_height = torch.abs(obj_pos[:, 2] - initial_height) <= height_tolerance
    else:
        near_table_height = torch.logical_and(
            obj_pos[:, 2] >= minimum_height,
            obj_pos[:, 2] <= initial_height + height_tolerance,
        )
    settled = speed <= max_speed
    return torch.logical_and(torch.logical_and(in_area, near_table_height), settled)


def release_on_target_reward(
    env,
    target_pos: tuple[float, float, float],
    target_size: tuple[float, float, float],
    initial_height: float,
    height_tolerance: float,
    speed_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    term_name: str = "palm_ik",
    grip_index: int = 6,
    minimum_height: float | None = None,
    require_lift_and_grasp: bool = False,
    min_lift: float = 0.03,
    carry_min_lift: float | None = None,
    close_distance: float = 0.10,
    grip_close_threshold: float = 0.7,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    held_distance: float = 0.12,
    grasp_dwell_steps: int = 5,
    lift_dwell_steps: int = 5,
    carry_dwell_steps: int = 5,
) -> torch.Tensor:
    """Reward opening the hand after the object is in the green target area."""
    grip = action_grip_command(env, term_name, grip_index)
    grip_available = action_grip_available(env, term_name, grip_index)
    obj_pos = object_position(env, object_cfg)
    in_area = object_in_target_area(env, target_pos, target_size, object_cfg)
    if minimum_height is None:
        near_table_height = torch.abs(obj_pos[:, 2] - initial_height) <= height_tolerance
    else:
        near_table_height = torch.logical_and(
            obj_pos[:, 2] >= minimum_height,
            obj_pos[:, 2] <= initial_height + height_tolerance,
        )
    speed_score = torch.exp(-object_speed(env, object_cfg) / speed_std)
    ready_to_release = torch.logical_and(in_area, near_table_height).float()
    if require_lift_and_grasp:
        grasped, lifted_while_held, carried_to_target = _ordered_pick_place_history(
            env=env,
            target_pos=target_pos,
            target_size=target_size,
            initial_height=initial_height,
            min_lift=min_lift,
            carry_min_lift=carry_min_lift,
            close_distance=close_distance,
            grip_close_threshold=grip_close_threshold,
            held_distance=held_distance,
            grasp_dwell_steps=grasp_dwell_steps,
            lift_dwell_steps=lift_dwell_steps,
            carry_dwell_steps=carry_dwell_steps,
            palm_cfg=palm_cfg,
            object_cfg=object_cfg,
            term_name=term_name,
            grip_index=grip_index,
        )
        ordered_carry = torch.logical_and(torch.logical_and(grasped, lifted_while_held), carried_to_target)
        ready_to_release = ready_to_release * ordered_carry.float()
    return grip_available * (1.0 - grip) * ready_to_release * speed_score


def release_away_from_target_penalty(
    env,
    target_pos: tuple[float, float, float],
    target_size: tuple[float, float, float],
    initial_height: float,
    min_lift: float,
    height_tolerance: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    term_name: str = "palm_ik",
    grip_index: int = 6,
    minimum_height: float | None = None,
) -> torch.Tensor:
    """Penalize opening while carrying the object before it reaches the target."""
    grip = action_grip_command(env, term_name, grip_index)
    grip_available = action_grip_available(env, term_name, grip_index)
    obj_pos = object_position(env, object_cfg)
    lifted = obj_pos[:, 2] > initial_height + min_lift
    in_area = object_in_target_area(env, target_pos, target_size, object_cfg)
    if minimum_height is None:
        near_table_height = torch.abs(obj_pos[:, 2] - initial_height) <= height_tolerance
    else:
        near_table_height = torch.logical_and(
            obj_pos[:, 2] >= minimum_height,
            obj_pos[:, 2] <= initial_height + height_tolerance,
        )
    release_is_valid = torch.logical_and(in_area, near_table_height)
    return grip_available * (1.0 - grip) * torch.logical_and(lifted, torch.logical_not(release_is_valid)).float()


def placed_on_target_reward(
    env,
    target_pos: tuple[float, float, float],
    target_size: tuple[float, float, float],
    initial_height: float,
    height_tolerance: float,
    max_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    minimum_height: float | None = None,
    require_lift_and_grasp: bool = False,
    min_lift: float = 0.03,
    carry_min_lift: float | None = None,
    close_distance: float = 0.10,
    grip_close_threshold: float = 0.7,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    term_name: str = "palm_ik",
    grip_index: int = 6,
    held_distance: float = 0.12,
    grasp_dwell_steps: int = 5,
    lift_dwell_steps: int = 5,
    carry_dwell_steps: int = 5,
) -> torch.Tensor:
    """Sparse reward for placing the object inside the green target rectangle."""
    placed = object_placed_on_target(
        env, target_pos, target_size, initial_height, height_tolerance, max_speed, object_cfg, minimum_height
    )
    if require_lift_and_grasp:
        grasped, lifted_while_held, carried_to_target = _ordered_pick_place_history(
            env=env,
            target_pos=target_pos,
            target_size=target_size,
            initial_height=initial_height,
            min_lift=min_lift,
            carry_min_lift=carry_min_lift,
            close_distance=close_distance,
            grip_close_threshold=grip_close_threshold,
            held_distance=held_distance,
            grasp_dwell_steps=grasp_dwell_steps,
            lift_dwell_steps=lift_dwell_steps,
            carry_dwell_steps=carry_dwell_steps,
            palm_cfg=palm_cfg,
            object_cfg=object_cfg,
            term_name=term_name,
            grip_index=grip_index,
        )
        ordered_carry = torch.logical_and(torch.logical_and(grasped, lifted_while_held), carried_to_target)
        placed = torch.logical_and(placed, ordered_carry)
    return placed.float()


def object_released_on_target(
    env,
    target_pos: tuple[float, float, float],
    target_size: tuple[float, float, float],
    initial_height: float,
    height_tolerance: float,
    max_speed: float,
    grip_threshold: float,
    min_palm_object_distance: float,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    term_name: str = "palm_ik",
    grip_index: int = 6,
    minimum_height: float | None = None,
    require_lift_and_grasp: bool = True,
    min_lift: float = 0.03,
    carry_min_lift: float | None = None,
    close_distance: float = 0.10,
    grip_close_threshold: float = 0.7,
    held_distance: float = 0.12,
    grasp_dwell_steps: int = 5,
    lift_dwell_steps: int = 5,
    carry_dwell_steps: int = 5,
) -> torch.Tensor:
    """Whether the object has been placed and the hand has actually released it."""
    placed = object_placed_on_target(
        env,
        target_pos,
        target_size,
        initial_height,
        height_tolerance,
        max_speed,
        object_cfg,
        minimum_height,
    )
    grip = action_grip_command(env, term_name, grip_index)
    grip_available = action_grip_available(env, term_name, grip_index).bool()
    opened = torch.logical_or(torch.logical_not(grip_available), grip <= grip_threshold)
    separated = palm_object_distance(env, palm_cfg, object_cfg) >= min_palm_object_distance
    success = torch.logical_and(torch.logical_and(placed, opened), separated)
    if require_lift_and_grasp:
        grasped, lifted_while_held, carried_to_target = _ordered_pick_place_history(
            env=env,
            target_pos=target_pos,
            target_size=target_size,
            initial_height=initial_height,
            min_lift=min_lift,
            carry_min_lift=carry_min_lift,
            close_distance=close_distance,
            grip_close_threshold=grip_close_threshold,
            held_distance=held_distance,
            grasp_dwell_steps=grasp_dwell_steps,
            lift_dwell_steps=lift_dwell_steps,
            carry_dwell_steps=carry_dwell_steps,
            palm_cfg=palm_cfg,
            object_cfg=object_cfg,
            term_name=term_name,
            grip_index=grip_index,
        )
        ordered_carry = torch.logical_and(torch.logical_and(grasped, lifted_while_held), carried_to_target)
        success = torch.logical_and(success, ordered_carry)
    return success


def released_on_target_reward(
    env,
    target_pos: tuple[float, float, float],
    target_size: tuple[float, float, float],
    initial_height: float,
    height_tolerance: float,
    max_speed: float,
    grip_threshold: float,
    min_palm_object_distance: float,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    term_name: str = "palm_ik",
    grip_index: int = 6,
    minimum_height: float | None = None,
    require_lift_and_grasp: bool = True,
    min_lift: float = 0.03,
    carry_min_lift: float | None = None,
    close_distance: float = 0.10,
    grip_close_threshold: float = 0.7,
    held_distance: float = 0.12,
    grasp_dwell_steps: int = 5,
    lift_dwell_steps: int = 5,
    carry_dwell_steps: int = 5,
) -> torch.Tensor:
    """Sparse reward for a completed place-and-release."""
    return object_released_on_target(
        env=env,
        target_pos=target_pos,
        target_size=target_size,
        initial_height=initial_height,
        height_tolerance=height_tolerance,
        max_speed=max_speed,
        grip_threshold=grip_threshold,
        min_palm_object_distance=min_palm_object_distance,
        palm_cfg=palm_cfg,
        object_cfg=object_cfg,
        term_name=term_name,
        grip_index=grip_index,
        minimum_height=minimum_height,
        require_lift_and_grasp=require_lift_and_grasp,
        min_lift=min_lift,
        carry_min_lift=carry_min_lift,
        close_distance=close_distance,
        grip_close_threshold=grip_close_threshold,
        held_distance=held_distance,
        grasp_dwell_steps=grasp_dwell_steps,
        lift_dwell_steps=lift_dwell_steps,
        carry_dwell_steps=carry_dwell_steps,
    ).float()


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
