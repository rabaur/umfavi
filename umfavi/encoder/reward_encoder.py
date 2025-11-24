import torch
from torch import nn
from umfavi.utils.math import log_var_to_std
from abc import ABC, abstractmethod

class BaseRewardEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, state_features: torch.Tensor, action_features: torch.Tensor, next_state_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...
    @abstractmethod
    def sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        ...
    @abstractmethod
    def predict_and_sample(self, state_features: torch.Tensor, action_features: torch.Tensor, next_state_features: torch.Tensor) -> torch.Tensor:
        ...

class RewardEncoder(BaseRewardEncoder):
    """Variational, amortized approximation of the reward posterior."""

    def __init__(self, feature_module: nn.Module):
        super().__init__()
        self.features = feature_module
        self.mean_head = nn.Linear(feature_module.out_dim, 1)
        self.logvar_head = nn.Linear(feature_module.out_dim, 1)

    def forward(self, state_features: torch.Tensor, action_features: torch.Tensor, next_state_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(state_features, action_features, next_state_features)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        return mean, logvar
    
    def sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = log_var_to_std(logvar)
        eps = torch.randn_like(std)
        return mean + std * eps
    
    def predict_and_sample(self, state_features: torch.Tensor, action_features: torch.Tensor, next_state_features: torch.Tensor) -> float:
        with torch.no_grad():
            mean, logvar = self.forward(state_features, action_features, next_state_features)
            return self.sample(mean, logvar).item()
