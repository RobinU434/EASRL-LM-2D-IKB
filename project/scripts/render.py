from pathlib import Path
from gymnasium import Env
from stable_baselines3 import SAC
from project.environment.ik_rl.ik_rl.environment import InvKinEnvContinuous
from project.environment.ik_rl.ik_rl.wrapper import NormalizeRewardWrapper
from project.utils.configs.train_sac import Config as SACConfig


def render_sac(checkpoint: str, device: str = "cpu", stochastic: bool = False):
    config_path = Path(checkpoint).parent / ".hydra" / "config.yaml"
    config: SACConfig = SACConfig.from_file(config_path)
    env = InvKinEnvContinuous(
        n_joints=config.n_joints,
        n_steps=config.episode_steps,
        render_mode="human",
        epsilon=0.05,
    )
    sac = SAC.load(checkpoint, env=env, device=device)

    obs, info = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action, _states = sac.predict(obs, deterministic=not stochastic)
        env.render()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.render()
    env.close()
