import math
import torch

from typing import Tuple

from torch.distributions import MultivariateNormal

from algorithms.helper.helper import get_dim
from algorithms.common.helper import scale_matrix


def get_distribution(loc: torch.tensor, std: torch.tensor,  mode: str = "independent", decay: float = 0) -> MultivariateNormal:
    """main function. returns a Multivariate Normal distribution with covariance matrix either be: \n
    - independent (identity matrix) \n
    - sparse (identity matrix plus neighbor diagonals have value == decay) \n
    - exp_decay 

    Args:
        loc (torch.tensor): loc for distribution
        mode (str, optional): mode for covariance matrix. Defaults to "independent".
        decay (float, optional): decay for covariance matrix. Defaults to 0.

    Returns:
        MultivariateNormal: configured distribution
    """
    func_dict = {
        independent.__name__: independent,
        sparse.__name__: sparse,
        exp_decay.__name__: exp_decay,
    }       
    batch_size, n = get_dim(loc.size())
    covariance_matrix = func_dict[mode](n, decay)
    covariance_matrix = scale_matrix(covariance_matrix, batch_size)

    # get unit matrix
    unit_matrix = independent(n, 0)
    unit_matrix = scale_matrix(unit_matrix, batch_size)
    unit_matrix *= std.unsqueeze(1)
    covariance_matrix *=  unit_matrix

    distribution = MultivariateNormal(loc, covariance_matrix)

    return distribution


def independent(n: torch.Size, decay: float) -> torch.tensor:
    """returns unit matrix with shape (batch_size, m, m)

    Args:
        size (int): the dimension of the result matrix
        decay (float): place holder... not relevant in this function

    Returns:
        torch.tensor: independent covariance matrix
    """
    # decay has to be there to fit the calling pattern
    covariance_matrix = torch.eye(n)
    return covariance_matrix

def sparse(n: int, decay: float) -> torch.tensor:

    # TODO: do same thing as in function independent
    covariance_matrix = torch.eye(n)

    # create matrix with decay one diagonal above the normal diagonal
    zeros = torch.zeros(1, n)
    above = torch.cat((covariance_matrix[1:], zeros)) * decay
    # create matrix with decay one diagonal below the normal diagonal
    below = torch.cat((zeros, covariance_matrix[:-1])) * decay

    # merge everything into the covariance matrix
    covariance_matrix = covariance_matrix + above + below

    return covariance_matrix

def exp_decay(n: int, decay: float) -> torch.tensor:
    # TODO: do same thing as in function independent
    covariance_matrix = torch.eye(n)
    sum_up_list = [covariance_matrix]
    for i in range(1, n):
        value = math.pow(decay, i)
        
        zeros = torch.zeros(i, n)
        above = torch.cat((covariance_matrix[i:], zeros)) * value
        below = torch.cat((zeros, covariance_matrix[:-i])) * value

        sum_up_list.append(above)
        sum_up_list.append(below)

    covariance_matrix = torch.stack(sum_up_list, dim=0).sum(dim=0)

    return covariance_matrix

if __name__ == "__main__":
    get_distribution(torch.tensor([1, 2, 3, 4]), mode="decay", decay=0.5)
