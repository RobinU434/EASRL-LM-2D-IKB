from pathlib import Path
import hydra
from omegaconf import DictConfig
from stable_baselines3.sac import SAC
from project.environment.ik_rl.ik_rl.environment import InvKinEnvContinuous
from project.utils.configs.train_sac import Config as SACConfig
from stable_baselines3.common.logger import configure


def train_sac(config: DictConfig, force: bool = False, device: str = "cpu"):
    config: SACConfig = SACConfig.from_dict_config(config)
    env = InvKinEnvContinuous(n_joints=config.n_joints, n_steps=config.episode_steps)

    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    sac = SAC(
        "MlpPolicy",
        env,
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
    )
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    sac.set_logger(new_logger)

    if not force:
        question = input("Would you like to start to train? [Y, n]")
        if not (question is None or question.lower().strip() in ["", "y", " yes"]):
            print("Abort training")
            return

    sac.learn(config.step_budget)
