import torch
import torch.nn as nn

from typing import List

from algorithms.sac.actor.base_actor import Actor
from vae.model.vae import VariationalAutoencoder


class LatentActor(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int, 
        output_dim: int, 
        learning_rate: int, 
        enhanced_latent_dim: int = 0, 
        architecture: List[int] = [128, 128], 
        kl_weight: float = 0.01
        ) -> None:
        
        super().__init__()

        self.actor = Actor(
            input_size=input_dim,
            output_size=latent_dim,
            learning_rate=learning_rate,
            architecture=architecture
            )

        self.auto_encoder = VariationalAutoencoder(
            input_dim=output_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            learning_rate=learning_rate,
            enhanced_latent_dim=enhanced_latent_dim,
            logger=None
        )

        # checkpoint chosen because of the overall performance reconstruction loss + kl loss
        if enhanced_latent_dim == 0:
            if latent_dim == 5:
                if kl_weight == 0.1:
                    checkpoint = torch.load("results/vae/unenhanced/5_10/1675462838/model_135_val_r_loss_8.8094.pt")
                elif kl_weight == 0.01:
                    checkpoint = torch.load("results/vae/unenhanced/5_10/1675462847/model_190_val_r_loss_1.1852.pt")
                elif kl_weight == 0.0001:
                    checkpoint = torch.load("results/vae/unenhanced/5_10/1675464735/model_165_val_r_loss_0.0715.pt")
                elif kl_weight == 1.0e-05:
                    checkpoint = torch.load("results/vae/unenhanced/5_10/1675504441/model_40_val_r_loss_0.0740.pt")
                elif kl_weight == 1.0e-06:
                    checkpoint = torch.load("results/vae/unenhanced/5_10/1675504449/model_185_val_r_loss_0.0749.pt")
            elif latent_dim == 3:
                if kl_weight == 0.01:
                    checkpoint = torch.load("results/vae/unenhanced/3_10/1675861960/model_155_val_r_loss_1.1588.pt")
                elif kl_weight == 0.001:
                    checkpoint = torch.load("results/vae/unenhanced/3_10/1675862007/model_110_val_r_loss_0.2422.pt")
                elif kl_weight == 0.0001:
                    checkpoint = torch.load("results/vae/unenhanced/3_10/1675863182/model_150_val_r_loss_0.0951.pt")
                elif kl_weight == 1e-5:
                    checkpoint = torch.load("results/vae/unenhanced/3_10/1675863248/model_105_val_r_loss_0.0849.pt")
            elif latent_dim == 2:
                if kl_weight == 0.01:
                    checkpoint = torch.load("results/vae/unenhanced/2_10/1675866791/model_175_val_r_loss_1.2797.pt")
                elif kl_weight == 0.001:
                    checkpoint = torch.load("results/vae/unenhanced/2_10/1675866798/model_200_val_r_loss_0.4617.pt")
                elif kl_weight == 0.0001:
                    checkpoint = torch.load("results/vae/unenhanced/2_10/1675867964/model_150_val_r_loss_0.3011.pt")
                elif kl_weight == 1e-5:
                    checkpoint = torch.load("results/vae/unenhanced/2_10/1675867972/model_200_val_r_loss_3.8610.pt")
            elif latent_dim == 1:
                if kl_weight == 0.01:
                    checkpoint = torch.load("results/vae/unenhanced/1_10/1675864359/model_190_val_r_loss_2.7298.pt")
                elif kl_weight == 0.001:
                    checkpoint = torch.load("results/vae/unenhanced/1_10/1675864383/model_115_val_r_loss_1.7364.pt")
                elif kl_weight == 0.0001:
                    checkpoint = torch.load("results/vae/unenhanced/1_10/1675865585/model_195_val_r_loss_1.9992.pt")
                elif kl_weight == 1e-5:
                    checkpoint = torch.load("results/vae/unenhanced/1_10/1675865592/model_175_val_r_loss_1.5570.pt")
        else:
            if latent_dim == 8:
                checkpoint = torch.load("results/vae/enhanced/8_10")
            elif latent_dim == 5:
                checkpoint = torch.load("results/vae/enhanced/5_10/1676323228/model_355_val_r_loss_1.4800.pt")

        self.auto_encoder.load_state_dict(checkpoint["model_state_dict"])


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
