from typing import List, Tuple

import torch
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.sac.policies import Actor

from project.models.flow import ConditionalRealNVP, RealNVP


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
        constrain_latent_space: bool = False,
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
        self.constrain_latent_space = constrain_latent_space
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

    def get_action_dist_params(self, obs):
        mean_actions, log_std, kwargs = super().get_action_dist_params(obs)
        if self.constrain_latent_space:
            # squeeze mean into a constrained space [-1, 1]
            mean_actions = torch.tanh(mean_actions)
        return mean_actions, log_std, kwargs

    def get_latent_action(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        return super().forward(observation, deterministic)

    def get_decoded_action(
        self, z: torch.Tensor, observation: torch.Tensor
    ) -> torch.Tensor:
        """Get the decoded action from the latent space by passing through the decoder network.

        Args:
            z (torch.Tensor): The latent space representation.
            observation (torch.Tensor): The original observation.

        Returns:
            torch.Tensor: The decoded action.
        """
        # Apply additional network to the sampled action
        if self.conditional_decoder:
            z = torch.cat([z, observation], dim=-1)

        action = self.decoder.forward(z)
        return torch.tanh(action)  # Tanh to squash action to [-1, 1]

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get the action corresponding to the observation by sampling from the latent space and decoding it.

        Args:
            obs (torch.Tensor): The original obs.
            deterministic (bool, optional): Whether to use deterministic action selection. Defaults to False.

        Returns:
            torch.Tensor: The latent action.
        """
        z = self.get_latent_action(obs, deterministic)
        action = self.get_decoded_action(z, obs)
        return action

    def action_log_prob(self, obs: PyTorchObs) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_action, log_prob = super().action_log_prob(obs)
        action = self.get_decoded_action(latent_action, obs)
        return action, log_prob

    def _predict(self, observation, deterministic=False):
        return self.forward(observation, deterministic)


class FlowActor(Actor):
    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        features_extractor,
        features_dim,
        latent_dim: int,
        flow_arch: List[int] = [256, 256],
        conditional_flow: bool = False,
        constrain_latent_space: bool = False,
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

        self.conditional_flow = conditional_flow
        self.constrain_latent_space = constrain_latent_space
        input_dim = latent_dim

        if self.conditional_flow:
            input_dim += self.features_dim
            self.flow_model = ConditionalRealNVP(
                dim=get_action_dim(action_space),
                cond_dim=features_dim,
                n_couplings=len(flow_arch),
                hidden_dim=flow_arch[0],
                device=self.device,
            )
        else:
            self.flow_model = RealNVP(
                dim=get_action_dim(action_space),
                n_couplings=len(flow_arch),
                hidden_dim=flow_arch[0],
                device=self.device,
            )
        self.linear_map = nn.Linear(latent_dim, get_action_dim(action_space))

    def get_action_dist_params(self, obs):
        mean_actions, log_std, kwargs = super().get_action_dist_params(obs)
        if self.constrain_latent_space:
            # squeeze mean into a constrained space [-1, 1]
            mean_actions = torch.tanh(mean_actions)
        return mean_actions, log_std, kwargs

    def get_latent_action(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        return super().forward(observation, deterministic)

    def get_flow_action(self, z: torch.Tensor, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.linear_map.forward(z)
        if self.conditional_flow:
            action, log_det = self.flow_model.forward(z, obs)
        else:
            action, log_det = self.flow_model.forward(z)
        action = torch.tanh(action)
        return action, log_det

    def forward(self, obs, deterministic=False):
        z = self.get_latent_action(obs, deterministic)
        action, _ = self.get_flow_action(z, obs)
        return action

    def action_log_prob(self, obs: PyTorchObs) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_action, log_prob = super().action_log_prob(obs)
        action, logdet = self.get_flow_action(latent_action, obs)
        # p(x): distribution after flow
        # p(z): distribution before flow
        # 1 / |det dx/dz|: change of variables
        # p(x) = p(z) * 1 / |det dx/dz|
        log_prob = log_prob + logdet
        return action, log_prob

    def _predict(self, observation, deterministic=False):
        return self.forward(observation, deterministic)
