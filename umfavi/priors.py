import torch

def kl_divergence_std_normal(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between q(reward|x) and p(reward) = N(0, 1).
    
    Args:
        mean: Mean of the approximate posterior reward distribution
        log_var: Log variance of the approximate posterior reward distribution
        
    Returns:
        KL divergence loss
    """
    kl_loss = 0.5 * (mean.pow(2) + log_var.exp() - log_var - 1.0)
    return kl_loss.mean()