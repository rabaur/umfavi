import numpy as np
from numpy.typing import NDArray
import torch

def sigmoid(x: float) -> float:
    """
    Compute numerically stable sigmoid function.
    
    Args:
        x: Input value
        
    Returns:
        Sigmoid of x
    """
    if x < 0:
        z = np.exp(x)
        return z / (1 + z)
    else:
        return 1.0 / (1.0 + np.exp(-x))


def kl_divergence_normal(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between q(reward|x) and p(reward) = N(0, 1).
    
    Args:
        mean: Mean of the approximate posterior reward distribution
        log_var: Log variance of the approximate posterior reward distribution
        
    Returns:
        KL divergence loss
    """
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return kl_loss


def softmax(X: NDArray, dims=tuple[int]) -> NDArray:
    X -= np.max(X)
    return np.exp(X) / np.sum(np.exp(X), axis=dims, keepdims=True)