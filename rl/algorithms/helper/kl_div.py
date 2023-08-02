from typing import Tuple
import torch


def kl_divergence_from_weights(p, q) -> Tuple[torch.tensor, torch.tensor, float]:
    """function returns kl divergence from two lists of weights fom a model

    Args:
        p (torch.tensor): first list of weights 
        q (torch.tensor): second list of weights 

    Returns:
        Tuple[torch.tensor, torch.tensor, float]: p_density, q_density, kl divergence
    """
    p_dist, p_bins = torch.histogram(p,)
    q_dist, q_bins = torch.histogram(q, p_bins)
    assert torch.equal(p_bins, q_bins)

    # compute density functions
    p_dist /= len(p)
    q_dist /= len(q)
    
    # calculate kl_divergence
    return p_dist, q_dist, torch.sum(p * torch.log(p / q))
