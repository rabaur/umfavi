import torch
from torch import nn

class RewardEncoder(nn.Module):
    """Variational, amortized approximation of the reward posterior."""

    def __init__(self, feature_module: nn.Module):
        super().__init__()
        self.features = feature_module
        self.mean_head = nn.Linear(feature_module.out_dim, 1)
        self.logvar_head = nn.Sequential(nn.Linear(feature_module.out_dim, 1), nn.Softplus())

    def forward(self, obs: torch.Tensor, acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(obs, acts)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        return mean, logvar
    
    def sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps