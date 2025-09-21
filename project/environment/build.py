from typing import Any, Dict, Tuple

import gymnasium


def build_IK_continuous(
    env_name: str,
    env_kwargs: Dict[str, Any],
) -> Tuple[gymnasium.Env, str]:
    from project.environment.ik_rl.ik_rl.environment import InvKinEnvContinuous
    from project.environment.ik_rl.ik_rl.wrapper import (
        ActionScaleWrapper,
        NormalizeRewardWrapper,
    )

    n_joints = env_kwargs["n_joints"]
    episode_steps = env_kwargs["episode_steps"]
    action_space_scale = env_kwargs["action_space_scale"]
    normalize_reward = env_kwargs["normalize_reward"]
    relative_actions = env_kwargs["relative_actions"]

    env = InvKinEnvContinuous(
        n_joints=n_joints,
        n_steps=episode_steps,
        relative_actions=relative_actions,
    )
    robot_arm_length = env.robot_arm.arm_length
    env = ActionScaleWrapper(env, action_space_scale)

    if normalize_reward:
        env = NormalizeRewardWrapper(env, robot_arm_length)

    return env, env_name


def build_gym_env(
    env_name: str, env_kwargs: Dict[str, Any]
) -> Tuple[gymnasium.Env, str]:
    env = gymnasium.make(id=env_name, **env_kwargs)
    return env, env_name
