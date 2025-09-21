from typing import Callable, List, Tuple
import gymnasium
import hydra
from omegaconf import DictConfig
from stable_baselines3.sac import SAC
import wandb
from project.environment.ik_rl.ik_rl.environment import InvKinEnvContinuous
from project.environment.ik_rl.ik_rl.wrapper import (
    NormalizeRewardWrapper,
    ActionScaleWrapper,
)
from config2class.api.base import StructuredConfig
from project.models.policy import get_policy
from project.utils.configs.train_sac import Config as SACConfig
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback


def setup_env(config: SACConfig) -> Tuple[SubprocVecEnv, str]:
    _, name = hydra.utils.instantiate(config.env)
    env_fns = [
        lambda: Monitor(hydra.utils.instantiate(config.env)[0])
        for _ in range(config.n_envs)
    ]
    env = SubprocVecEnv(env_fns)
    return env, name


def train_sac(config: DictConfig, force: bool = False, device: str = "cpu"):
    # config: SACConfig = SACConfig.from_dict_config(config)
    env, env_name = setup_env(config)

    wandb_config = (
        config.to_container()
        if isinstance(config, StructuredConfig)
        else config.__dict__
    )
    wandb_config["env"] = env_name
    run = wandb.init(
        project="LatentSAC",
        config=wandb_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    policy, policy_kwargs = get_policy(config.SAC)
    sac = SAC(
        policy=policy,
        env=env,
        learning_rate=config.SAC.learning_rate,
        buffer_size=config.SAC.buffer_size,
        batch_size=config.SAC.batch_size,
        tau=config.SAC.tau,
        tensorboard_log=log_dir,
        gamma=config.SAC.gamma,
        train_freq=config.SAC.train_freq,
        gradient_steps=config.SAC.gradient_steps,
        action_noise=config.SAC.action_noise,
        optimize_memory_usage=config.SAC.optimize_memory_usage,
        ent_coef=config.SAC.ent_coef,
        target_update_interval=config.SAC.target_update_interval,
        target_entropy=config.SAC.target_entropy,
        use_sde=config.SAC.use_sde,
        sde_sample_freq=config.SAC.sde_sample_freq,
        use_sde_at_warmup=config.SAC.use_sde_at_warmup,
        stats_window_size=config.SAC.stats_window_size,
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    
    # setup logger and callbacks
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    sac.set_logger(new_logger)
    # eval_call_back = EvalCallback(
    #     Monitor(
    #         NormalizeRewardWrapper(
    #             InvKinEnvContinuous(
    #                 n_joints=config.n_joints, n_steps=config.episode_steps, seed=42
    #             )
    #         )
    #     ),
    #     verbose=0,
    #     n_eval_episodes=10,
    #     eval_freq=int(10_000 / config.n_envs),
    # )
    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=log_dir,
        verbose=2,
        log="all",
    )
    checkpoint_callback = CheckpointCallback(config.save_interval, log_dir, "sac_model")
    callbacks = CallbackList([wandb_callback, checkpoint_callback])

    print("======== ACTOR =========")
    print(sac.policy)
    if not force:
        question = input("Would you like to start to train? [Y, n]")
        if not (question is None or question.lower().strip() in ["", "y", " yes"]):
            print("Abort training")
            return

    sac.learn(config.step_budget, callback=callbacks)
    run.finish()
