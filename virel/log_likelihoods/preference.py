import torch
from torch import nn
from virel.log_likelihoods.base_log_likelihood import BaseLogLikelihood

class PreferenceDecoder(BaseLogLikelihood):
    """Head for predicting preferences."""

    def __init__(self):
        super().__init__()

    def forward(self, reward_samples: torch.Tensor, rationality: float = 1.0) -> torch.Tensor:
        """
        Args:
            reward_samples: Float tensor of shape (batch_size * 2, num_steps).
                The first batch_size rewards correspond to the first trajectory, the second to the second trajectory.
            rationality: Rationality coefficient.
        Returns:
            Preference tensor of shape (batch_size)
        """
        assert reward_samples.shape[0] % 2 == 0
        batch_size = reward_samples.shape[0] // 2
        cum_rews_per_traj = torch.sum(reward_samples, dim=1)
        cum_rews1 = cum_rews_per_traj[:batch_size]
        cum_rews2 = cum_rews_per_traj[batch_size:]
        return torch.sigmoid(rationality * (cum_rews1 - cum_rews2))
    
    def nll(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Preference predictions of shape (batch_size)
            targets: Preference targets of shape (batch_size)
        Returns:
            Loss tensor.
        """
        return nn.functional.binary_cross_entropy(preds, targets, reduction='mean')