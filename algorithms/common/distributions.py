import math
import torch
from torch.distributions import MultivariateNormal


def get_distribution(loc: torch.tensor, mode: str = "independent", decay: float = 0) -> MultivariateNormal:
    """main function. returns a Multivariate Normal distribution with covariance matrix either be: \n
    - independent (identity matrix) \n
    - sparse (identity matrix plus neighbour diagonals have value == decay) \n
    - exp_decay 

    Args:
        loc (torch.tensor): loc for distribution
        mode (str, optional): mode for cavariance matrix. Defaults to "independent".
        decay (float, optional): decay for covariance matrix. Defaults to 0.

    Returns:
        MultivariateNormal: configured distribution
    """
    func_dict = {
        independent.__name__: independent,
        sparse.__name__: sparse,
        exp_decay.__name__: exp_decay,
    }       

    convariance_matrix = func_dict[mode](len(loc), decay)

    distribution = MultivariateNormal(loc, convariance_matrix)

    return distribution


def independent(n, decay):
    # decay has to be there to fit the calling pattern
    convariance_matrix = torch.eye(n)
    return convariance_matrix

def sparse(n, decay):
    convariance_matrix = torch.eye(n)

    # create matrix with decay one diagonal above the normal diagonal
    zeros = torch.zeros(1, n)
    above = torch.cat((convariance_matrix[1:], zeros)) * decay
    # create matrix with decay one diagonal below the normal diagonal
    below = torch.cat((zeros, convariance_matrix[:-1])) * decay

    # merge everything into the cavariance matrix
    convariance_matrix = convariance_matrix + above + below

    return convariance_matrix

def exp_decay(n, decay):
    convariance_matrix = torch.eye(n)
    sum_up_list = [convariance_matrix]
    for i in range(1, n):
        value = math.pow(decay, i)
        
        zeros = torch.zeros(i, n)
        above = torch.cat((convariance_matrix[i:], zeros)) * value
        below = torch.cat((zeros, convariance_matrix[:-i])) * value

        sum_up_list.append(above)
        sum_up_list.append(below)

    convariance_matrix = torch.stack(sum_up_list, dim=0).sum(dim=0)

    return convariance_matrix

if __name__ == "__main__":
    get_distribution(torch.tensor([1, 2, 3, 4]), mode="decay", decay=0.5)
