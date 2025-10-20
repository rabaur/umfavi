import torch
from torch import nn

class MLPFeatureModule(nn.Module):
    """Module for encoding features using a MLP."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int]):
        super().__init__()
        hidden_sizes = [obs_dim + act_dim] + hidden_sizes
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.SiLU())
        self.features = nn.Sequential(*layers)
        self.out_dim = hidden_sizes[-1]
    
    def forward(self, obs: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        return self.features(torch.cat([obs, acts], dim=-1))