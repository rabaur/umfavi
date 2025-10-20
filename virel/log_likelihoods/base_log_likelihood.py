import torch
from abc import ABC, abstractmethod
from torch import nn

class BaseLogLikelihood(nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(self, reward_samples: torch.Tensor, rationality: float = 1.0) -> torch.Tensor:
        ...
    
    @abstractmethod
    def nll(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...