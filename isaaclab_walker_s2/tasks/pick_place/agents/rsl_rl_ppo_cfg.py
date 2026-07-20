"""RSL-RL PPO configuration for the Walker S2 IK pick/place task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class WalkerS2IKPickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Small PPO setup for the compact 11D palm-IK action space."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "walker_s2_pick_place_ik"
    run_name = ""
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class WalkerS2StagedPickPlacePPORunnerCfg(WalkerS2IKPickPlacePPORunnerCfg):
    """PPO setup for the staged primitive action space."""

    experiment_name = "walker_s2_pick_place_staged"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.05,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )


@configclass
class WalkerS2DirectArmPickPlacePPORunnerCfg(WalkerS2IKPickPlacePPORunnerCfg):
    """PPO setup for the direct right-arm + grip student action space."""

    experiment_name = "walker_s2_pick_place_direct_arm"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )


@configclass
class WalkerS2CartesianPickPlacePPORunnerCfg(WalkerS2IKPickPlacePPORunnerCfg):
    """PPO setup for the object-agnostic Cartesian palm action space."""

    experiment_name = "walker_s2_pick_place_cartesian"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.15,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
