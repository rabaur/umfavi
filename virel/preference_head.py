import torch
from torch import nn

class PreferenceHead(nn.Module):
    """Head for predicting preferences."""

    def __init__(self):
        super().__init__()

    def forward(self, rews_A: torch.Tensor, rews_B: torch.Tensor, rationality: float = 1.0) -> torch.Tensor:
        """
        Args:
            rews_A: Reward tensor of shape (batch_size, T)
            rews_B: Reward tensor of shape (batch_size, T)
            rationality: Rationality parameter
        Returns:
            Preference tensor of shape (batch_size)
        """
        rews_A_per_traj = torch.sum(rews_A, dim=1)
        rews_B_per_traj = torch.sum(rews_B, dim=1)
        return torch.sigmoid(rationality * (rews_B_per_traj - rews_A_per_traj))
    
    def compute_loss(self, rews_A: torch.Tensor, rews_B: torch.Tensor, prefs: torch.Tensor, rationality: float = 1.0) -> torch.Tensor:
        """
        Args:
            rews_A: Reward tensor of shape (batch_size, T)
            rews_B: Reward tensor of shape (batch_size, T)
            prefs: Preference tensor of shape (batch_size)
            rationality: Rationality parameter
        Returns:
            Loss tensor.
        """
        prefs_pred = self.forward(rews_A, rews_B, rationality)
        return nn.functional.binary_cross_entropy(prefs_pred, prefs, reduction='mean')