from typing import List, Tuple

import torch
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.sac.policies import Actor


class LatentActor(Actor):
    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        features_extractor,
        features_dim,
        latent_dim: int,
        latent_arch: List[int] = [256, 256],
        conditional_decoder: bool = False,
        activation_fn=nn.ReLU,
        use_sde=False,
        log_std_init=-3,
        full_std=True,
        use_expln=False,
        clip_mean=2,
        normalize_images=True,
    ):
        # adapt action space to reduce output dim of Gaussian distribution and input dim of decoder

        assert isinstance(action_space, Box), "The action space has to be Box."
        assert len(action_space.shape) == 1, "Assert vectorized action space."

        latent_action_space = Box(
            low=-1,  # regularize with tanh
            high=1,
            shape=(latent_dim,),
            dtype=action_space.dtype,
            seed=action_space._np_random,
        )
        super().__init__(
            observation_space,
            latent_action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images,
        )

        self.conditional_decoder = conditional_decoder

        input_dim = latent_dim
        if self.conditional_decoder:
            input_dim += self.features_dim

        self.decoder = nn.Sequential(
            *create_mlp(
                input_dim=input_dim,
                output_dim=get_action_dim(action_space),
                net_arch=latent_arch,
                activation_fn=activation_fn,
            )
        )
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, get_action_dim(action_space)),
        # )
        
    def get_latent_action(self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        return super().forward(observation, deterministic)

    def get_decoded_action(self, z: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        # box in latent space -> output space is also boxed
        # z = torch.tanh(z)
        
        # Apply additional network to the sampled action
        if self.conditional_decoder:
            z = torch.cat([z, observation], dim=-1)
        
        action = self.decoder.forward(z)
        return torch.tanh(action)  # Tanh to squash action to [-1, 1]

    def forward(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        z = self.get_latent_action(observation, deterministic)
        action = self.get_decoded_action(z, observation)
        return action
    
    def action_log_prob(self, obs: PyTorchObs) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_action, log_prob = super().action_log_prob(obs)
        action = self.get_decoded_action(latent_action, obs)
        return action, log_prob
    
    def _predict(self, observation, deterministic=False):
        return self.forward(observation, deterministic)
