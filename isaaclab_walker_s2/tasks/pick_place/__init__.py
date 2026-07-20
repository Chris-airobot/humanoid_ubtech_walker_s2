"""Fixed-base Walker S2 pick/place task registrations."""

import gymnasium as gym

from . import agents
from . import walker_s2_pick_place_env_cfg


gym.register(
    id="Isaac-WalkerS2-PickPlace-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": walker_s2_pick_place_env_cfg.WalkerS2IKPickPlaceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkerS2IKPickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-WalkerS2-PickPlace-Staged-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": walker_s2_pick_place_env_cfg.WalkerS2StagedPickPlaceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkerS2StagedPickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-WalkerS2-PickPlace-DirectArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": walker_s2_pick_place_env_cfg.WalkerS2DirectArmPickPlaceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkerS2DirectArmPickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-WalkerS2-PickPlace-Cartesian-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": walker_s2_pick_place_env_cfg.WalkerS2CartesianPickPlaceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkerS2CartesianPickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-WalkerS2-PickPlace-ObjectRelative-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": walker_s2_pick_place_env_cfg.WalkerS2ObjectRelativePickPlaceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkerS2CartesianPickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)
