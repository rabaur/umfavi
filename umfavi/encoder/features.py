from typing import Optional
import torch
from torch import nn

class MLPFeatureModule(nn.Module):
    """Module for encoding features using a MLP."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int],
        reward_domain: str = 's',
        activate_last_layer: bool = True
    ):
        super().__init__()
        if reward_domain == 's':
            input_dim = obs_dim
        elif reward_domain == 'sa':
            input_dim = obs_dim + act_dim
        elif reward_domain == 'sas':
            input_dim = obs_dim + act_dim + obs_dim
        else:
            raise Exception(f"Unsupported reward domain '{reward_domain}'")
        hidden_sizes = [input_dim] + hidden_sizes
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if i < len(hidden_sizes) - 2 or activate_last_layer:
                layers.append(nn.LeakyReLU())
        self.features = nn.Sequential(*layers)
        self.out_dim = hidden_sizes[-1]
        self.reward_domain = reward_domain
    
    def forward(
        self,
        obs: torch.Tensor,
        acts: Optional[torch.Tensor] = None,
        next_obs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.reward_domain == 's':
            feats = self.features(obs)
        elif self.reward_domain == 'sa':
            feats = self.features(torch.cat([obs, acts], dim=-1))
        else:
            feats = self.features(torch.cat([obs, acts, next_obs], dim=-1))
        return feats

class QValueModel(nn.Module):
    """Model for estimating Q-values."""

    def __init__(self, obs_dim: int, hidden_sizes: list[int]):
        super().__init__()
        hidden_sizes = [obs_dim] + hidden_sizes
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if i < len(hidden_sizes) - 2:
                layers.append(nn.SiLU())
            else:
                layers.append(nn.Identity())
        self.features = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.features(obs)