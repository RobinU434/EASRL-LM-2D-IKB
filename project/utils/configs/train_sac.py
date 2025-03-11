from dataclasses import dataclass
from types import NoneType
from config2class.api.base import StructuredConfig


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


@dataclass
class Config(StructuredConfig):
    step_budget: int
    save_interval: int
    n_joints: int
    episode_steps: int
    SAC: _SAC

    def __post_init__(self):
        self.SAC = _SAC(**self.SAC)  # pylint: disable=E1134
