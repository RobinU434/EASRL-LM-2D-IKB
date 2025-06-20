from omegaconf import DictConfig
from pyargwriter.decorator import add_hydra
from pyargwriter import api


class Entrypoint:
    def __init__(self) -> None:
        pass

    @add_hydra("config", None, config_path="configs", config_name="train_sac.yaml")
    def train_sac(
        self,
        config: DictConfig,
        force: bool = False,
        device: str = "cpu",    
    ):
        from project.scripts.train import train_sac
        train_sac(config, force, device)
            
    def render_sac(
        self, checkpoint: str, device: str = "cpu", stochastic: bool = False
    ):
        from project.scripts.render import render_sac

        render_sac(checkpoint, device, stochastic)
