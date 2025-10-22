import torch

def elbo_loss(negative_log_likelihood: torch.Tensor, kl_divergence: torch.Tensor, kl_weight: float = 1.0) -> torch.Tensor:
    """
    Compute ELBO loss.
    
    Args:
        negative_log_likelihood: Negative log likelihood of the model (already a loss)
        kl_divergence: KL divergence between the approximate posterior and the prior
        kl_weight: Weight for the KL divergence
    Returns:
        ELBO loss (to be minimized)
    """
    return negative_log_likelihood + kl_weight * kl_divergence