import time
import numpy as np
import torch

from progress.bar import Bar
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vae.data.data_set import ActionDataset
from vae.model.vae import VariationalAutoencoder


def train(
    autoencoder: VariationalAutoencoder, 
    data: DataLoader,
    logger: SummaryWriter = None,
    epochs: int = 20,
    device: str = "cpu",
    learning_rate: float = 1e-3,
    print_interval: int = 5
    ):

    opt = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    
    if logger is not None:
        logger.add_hparams(
            hparam_dict={"learning_rate": learning_rate},
            metric_dict={})

    for epoch_idx in range(epochs):
        mses = []
        kl_div = []
        for x, y in data:
            opt.zero_grad()
            x = x.to(device) # GPU
            
            # normalize input values (mean and std extracted from ./data/distribution.ipynb)
            x = (x - 3.094711356534092) / 1.8342796626648048
            x_hat, mu, log_std = autoencoder(x)  # out shape: (batch_size, number of joints)
        
            # mean squared error
            # sum over the number of joints and mean for the batch
            mse = (torch.float_power(x - x_hat, 2)).sum(dim=1).mean()
            mses.append(mse.item())
            
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            # D_KL(N(mu, sigma)|N(0, 1))
            kl = (torch.exp(2 * log_std) / 2 + torch.float_power(mu, 2) / 2 - log_std - 1/2).sum()
            kl_div.append(kl.item())
        
            loss = mse + kl

            loss.backward()
            opt.step()
        
        mses = np.array(mses)
        kl_div = np.array(kl_div)

        if logger is not None and epoch_idx % print_interval == 0:
            print(f"epoch {epoch_idx}: loss {mses.mean()} kl_div: {kl_div.mean()}")
            logger.add_scalar("vae/loss", mses.mean(), epoch_idx)
            logger.add_scalar("vae/kl", kl_div.mean(), epoch_idx)
            
            # parameter histogram
            param_tensor = torch.tensor([])
            for param in autoencoder.parameters():
                param_tensor = torch.cat([param_tensor, param.flatten()])
            logger.add_histogram("vae/param", param_tensor, epoch_idx)

    return autoencoder

if __name__ == "__main__":
    
    device = "cpu" # "cuda:0" if torch.cuda.is_available() else "cpu"
    
    latent_dim = 5
    num_joints = 10 

    autoencoder = VariationalAutoencoder(input_dim=num_joints, latent_dim=latent_dim, output_dim=num_joints)

    data = ActionDataset("./datasets/10/actions_1674648707.csv")
    dataloader = DataLoader(data, batch_size=64, shuffle=False)
    
    logger = SummaryWriter(f"results/vae/{time.time()}")
    train(autoencoder, dataloader, logger=logger, epochs=100, learning_rate=1e-3)
