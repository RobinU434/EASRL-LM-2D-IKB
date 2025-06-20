from typing import Any, Dict, Tuple

import torch as th
from config2class.api.base import StructuredConfig
from stable_baselines3 import sac
from stable_baselines3.sac.policies import SACPolicy
from torch import nn

from project.models.actor import LatentActor
from project.utils.configs.train_sac import _SAC as SACConfig
from stable_baselines3.common.torch_layers import FlattenExtractor


class LatentSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,
        use_sde=False,
        log_std_init=-3,
        use_expln=False,
        clip_mean=2,
        features_extractor_class=FlattenExtractor,
        features_extractor_kwargs=None,
        normalize_images=True,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=None,
        n_critics=2,
        share_features_extractor=False,
        actor_kwargs = None, 
    ):
        # dont use net_kwargs -> will be overwritten
        self.a_kwargs = actor_kwargs
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )
        
    def make_actor(self, features_extractor = None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs.update(self.a_kwargs)
        return LatentActor(**actor_kwargs).to(self.device)
    
    
def get_policy(config: SACConfig) -> Tuple[str | SACPolicy, Dict[Any, Any]]:    
    if config.policy in globals().keys():
        policy = globals()[config.policy]
    elif config.policy in sac.__all__:
        policy = config.policy
    else:
        raise ValueError(f"No policy called: {config.policy} is locally or in stable baselines defined.")

    policy_config = getattr(config, config.policy, None)
    if isinstance(policy_config, StructuredConfig):
        policy_kwargs = policy_config.to_container()
    else:
        policy_kwargs = {}
    
    return policy, policy_kwargs