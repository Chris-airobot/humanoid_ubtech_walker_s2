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
