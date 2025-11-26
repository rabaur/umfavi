import torch
from torch import nn
from umfavi.loglikelihoods.base import BaseLogLikelihood
from umfavi.types import SampleKey

class DemonstrationsDecoder(BaseLogLikelihood):

    def __init__(self):
        """
        Args:
            Q_value_model: Q-value model, mapping states to Q-value estimates for all actions, i.e., Q : S -> R^{n_actions}.
        """
        super().__init__()

    def forward(self, reward_samples: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The likelihood of the AVRIL demonstrations has two terms:
        - The likelihood of the demonstrations under the Boltzmann-rational expert policy:
            ∑_{s, a} log π(a | s) = ∑_{s, a} log (exp(βQ(s, a)) / ∑_{a'} exp(βQ(s, a')))
        Args:
            reward_samples: Float tensor of shape (batch_size, num_steps).
            **kwargs: Additional arguments including:
                - obs: Float tensor of shape (batch_size, num_steps, obs_dim).
                - acts: Float tensor of shape (batch_size, num_steps, act_dim).
                - reward_mean: Float tensor of shape (batch_size, num_steps) - mean of reward distribution.
                - reward_log_var: Float tensor of shape (batch_size, num_steps) - log variance of reward distribution.
                - rationality: Rationality coefficient (default: 1.0).
                - gamma: Discount factor (default: 0.99).
                - td_error_weight: Weight for the TD-error constraint (default: 1.0).
        Returns:
            ...
        """
        # Extract parameters from kwargs
        acts = kwargs[SampleKey.ACTS].long()

        # Get the Q-value estimates
        q_curr = kwargs["q_curr"]  # (batch_size, n_actions)

        # ------------------------------------------------------------------------------------------------
        # Boltzmann-rational expert policy likelihood
        # ------------------------------------------------------------------------------------------------

        # Compute the log-likelihood of the demonstrations under the Boltzmann-rational expert policy
        demonstrations_nll = nn.functional.cross_entropy(q_curr, acts.squeeze(), reduction='none').mean()

        return demonstrations_nll, {}