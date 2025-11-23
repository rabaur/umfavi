import torch

def kl_divergence_std_normal(mean: torch.Tensor, log_var: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between q(reward|x) and p(reward) = N(0, 1).
    
    Args:
        mean: Mean of the approximate posterior reward distribution
        log_var: Log variance of the approximate posterior reward distribution
        dones: Boolean mask indicating terminal/invalid states
        
    Returns:
        KL divergence loss (averaged over valid timesteps only)
    """
    # Compute KL divergence per element
    kl_loss = 0.5 * (mean.pow(2) + log_var.exp() - log_var - 1.0)
    
    # Mask out invalid timesteps (where done=True)
    valid_mask = ~dones.bool()  # True for valid timesteps
    
    # Only compute mean over valid timesteps
    valid_kl = kl_loss[valid_mask]
    return valid_kl.mean()