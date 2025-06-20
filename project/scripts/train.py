import hydra
from omegaconf import DictConfig
from stable_baselines3.sac import SAC
from project.environment.ik_rl.ik_rl.environment import InvKinEnvContinuous
from project.environment.ik_rl.ik_rl.wrapper import NormalizeRewardWrapper
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


def train_sac(config: DictConfig, force: bool = False, device: str = "cpu"):
    config: SACConfig = SACConfig.from_dict_config(config)

    env_fns = [
        lambda: Monitor(
            NormalizeRewardWrapper(
                InvKinEnvContinuous(
                    n_joints=config.n_joints, n_steps=config.episode_steps
                )
            )
        )
        for _ in range(config.n_envs)
    ]
    env = SubprocVecEnv(env_fns)

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
    eval_call_back = EvalCallback(
        Monitor(
            NormalizeRewardWrapper(
                InvKinEnvContinuous(
                    n_joints=config.n_joints, n_steps=config.episode_steps, seed=42
                )
            )
        ),
        verbose=0,
        n_eval_episodes=10,
        eval_freq=int(10_000 / config.n_envs),
    )
    checkpoint_callback = CheckpointCallback(config.save_interval, log_dir, "sac_model")
    callbacks = CallbackList([eval_call_back, checkpoint_callback])
    
    print("======== ACTOR =========")
    print(sac.policy)
    if not force:
        question = input("Would you like to start to train? [Y, n]")
        if not (question is None or question.lower().strip() in ["", "y", " yes"]):
            print("Abort training")
            return

    sac.learn(config.step_budget, callback=callbacks)



def train_vae(config: DictConfig, force: bool = False, device: str = "cpu"):
    pass