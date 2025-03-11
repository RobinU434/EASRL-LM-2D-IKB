from omegaconf import DictConfig
from pyargwriter.decorator import add_hydra


class Entrypoint:
    def __init__(self) -> None:
        pass

    @add_hydra("config", None, config_path="configs", config_name="train_sac.yaml")
    def train_sac(self, config: DictConfig, force: bool = False, device: str = "cpu"):
        from project.scripts.train import train_sac
        
        train_sac(config, force, device)
