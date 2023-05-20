import glob
import yaml
import torch
import logging
import torch.nn as nn

from typing import List
from torch.utils.data import DataLoader

from algorithms.sac.actor.base_actor import Actor
from vae.utils.loss import VAELoss
from vae.model.vae import VariationalAutoencoder
from vae.utils.post_processing import PostProcessor


def load_checkpoint(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint


def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def store_vae_config(config, path: str):
    with open(path + "/vae_config.yaml", "w") as config_file:
        yaml.dump(config, config_file)


def extract_loss(path: str):
    checkpoint_name = path.split("/")[-1]
    loss = float(checkpoint_name.split("_")[-1][: -3])
    return loss


def load_best_checkpoint(vae_results_dir: str, output_dim: int, latent_dim: int):
    paths = glob.glob(vae_results_dir + f"/{output_dim}_{latent_dim}*/*.pt")
    if len(paths) == 0:
        logging.error(f"there is not VAE trained input dim: {output_dim} and latent dim: {latent_dim}")
    losses = map(extract_loss, paths)
    d = dict(zip(losses, paths))
    path = d[min(d)]
    print(f"use checkpoint for VAE at {path}")
    file_name = path.split("/")[-1]
    config = load_config("/".join(path.split("/")[: -1] + ["config.yaml"]))
    return file_name, load_checkpoint(path), config


class LatentActor(nn.Module):
    def __init__(
        self, 
        device,
        input_dim: int, 
        latent_dim: int, 
        output_dim: int, 
        learning_rate: int, 
        conditional_info_dim: int = 0, 
        architecture: List[int] = [128, 128], 
        kl_loss_weight: float = 1,
        reconstruction_loss_weight: float = 1,
        vae_learning: bool = False,
        checkpoint_dir: str = "results/vae",
        log_dir: str = "",

        ) -> None:
        super().__init__()

        self.device = device

        self.actor = Actor(
            input_dim=input_dim,
            output_dim=latent_dim,
            learning_rate=learning_rate,
            architecture=architecture
            )
        
        self.vae_learning = vae_learning
        self.auto_encoder = VariationalAutoencoder(
            input_dim=output_dim + conditional_info_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            learning_rate=learning_rate,
            conditional_info_dim=conditional_info_dim,
            logger=None,
            post_processor=PostProcessor(enabled=False),  # needs no post processing because it is used in policy net directly
            store_history=False,
            device=device,
            verbose=True,
        ).to(device)

        # checkpoint chosen because of the overall performance reconstruction loss + kl loss
        self.vae_config = None
        if not vae_learning:
            file_name, checkpoint, vae_config = load_best_checkpoint(checkpoint_dir, output_dim, latent_dim)
            self.auto_encoder.load_state_dict(checkpoint["model_state_dict"])
            self.vae_config = vae_config
            self.auto_encoder.store(
                log_dir + "/" + file_name,
                checkpoint["epoch"],
                {"reconstruction_loss": checkpoint["reconstruction_loss"],
                 "kl_loss": checkpoint["kl_loss"],
                 "distance_loss": checkpoint["distance_loss"],
                 "imitation_loss": checkpoint["imitation_loss"]}
                 )
            store_vae_config(self.vae_config, log_dir)
        else:
            self.auto_encoder_loss_func = VAELoss(
                kl_loss_weight=kl_loss_weight,
                reconstruction_loss_weight=reconstruction_loss_weight
            )


    def forward(self, x):
        latent_mu, latent_std = self.actor.forward(x)
        """ideas
        - take just the action from the actor mapped from state space into latent space
        - sample latent x with output from actor = mu, std
        - evaluate all k nearest neighbors from a pre defined test set and take the one with the highest q value 
        (https://arxiv.org/pdf/1512.07679.pdf)
        - evaluate all actions from a pre defined test set by q which are in the ellipse around mu, axis of the ellipse are defined by std
        """

        return latent_mu, latent_std

    def train(self, loss):
        self.actor.train(loss)

    def train_vae(self, data: DataLoader, train_iterations: int):
        r_loss_tensor = torch.tensor([])
        kl_loss_tensor = torch.tensor([])
        for i in range(train_iterations):
            for x, y in data:
                x = x.to(self.device)
                y = y.to(self.device)

                # concat x and y for conditional input
                encoder_input = torch.cat([x, y], dim=1)
                x_hat, mu, log_std = self.auto_encoder(encoder_input, y)  # out shape: (batch_size, number of joints) 
                
                loss = self.auto_encoder_loss_func(x, x_hat, mu, log_std)
                self.auto_encoder.train(loss)
                
                r_loss_tensor = torch.cat([r_loss_tensor, torch.tensor([self.auto_encoder_loss_func.r_loss])])
                kl_loss_tensor = torch.cat([kl_loss_tensor, torch.tensor([self.auto_encoder_loss_func.kl_div])])

        
        return r_loss_tensor.mean().item(), kl_loss_tensor.mean().item()

    @property
    def optimizer(self):
        return self.actor.optimizer