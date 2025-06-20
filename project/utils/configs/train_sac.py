from dataclasses import dataclass
from types import NoneType
from config2class.api.base import StructuredConfig


@dataclass
class _Actor_kwargs(StructuredConfig):
    latent_dim: int
    latent_arch: list
    conditional_decoder: bool


@dataclass
class _LatentSACPolicy(StructuredConfig):
    actor_kwargs: _Actor_kwargs

    def __post_init__(self):
        self.actor_kwargs = _Actor_kwargs(**self.actor_kwargs)  #pylint: disable=E1134


@dataclass
class _SAC(StructuredConfig):
    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    tau: float
    gamma: float
    train_freq: int
    gradient_steps: int
    action_noise: NoneType
    optimize_memory_usage: bool
    ent_coef: str
    target_update_interval: int
    target_entropy: str
    use_sde: bool
    sde_sample_freq: int
    use_sde_at_warmup: bool
    stats_window_size: int
    policy: str
    LatentSACPolicy: _LatentSACPolicy

    def __post_init__(self):
        self.LatentSACPolicy = _LatentSACPolicy(**self.LatentSACPolicy)  #pylint: disable=E1134


@dataclass
class Config(StructuredConfig):
    step_budget: int
    n_envs: int
    save_interval: int
    n_joints: int
    episode_steps: int
    SAC: _SAC

    def __post_init__(self):
        self.SAC = _SAC(**self.SAC)  #pylint: disable=E1134
