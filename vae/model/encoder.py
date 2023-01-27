import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(Encoder, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.linear_mu = nn.Linear(512, latent_dims)
        self.linear_std = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = self.linear(x)
        mu =  self.linear_mu(x)
        # linear_std is calculating the log std
        log_std = self.linear_std(x)
        return mu, log_std
