import torch
import torch.nn.functional as F
from torch import nn
from umfavi.loglikelihoods.base import BaseLogLikelihood
from umfavi.utils.math import log_var_to_std

class DemonstrationsDecoder(BaseLogLikelihood):

    def __init__(self, Q_value_model: nn.Module):
        """
        Args:
            Q_value_model: Q-value model, mapping states to Q-value estimates for all actions, i.e., Q : S -> R^{n_actions}.
        """
        super().__init__()
        self.Q_value_model = Q_value_model

    def forward(self, reward_samples: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The likelihood of the AVRIL demonstrations has two terms:
        - The likelihood of the demonstrations under the Boltzmann-rational expert policy:
            ∑_{s, a} log π(a | s) = ∑_{s, a} log (exp(βQ(s, a)) / ∑_{a'} exp(βQ(s, a')))
        - A constraint on the TD-error of the Q-value estimates, enforcing that Q(s', a') - Q(s, a) ≈ R(s, a) = q_θ(s, a)
            ∑_{s, a, s', a'} log q_θ(Q(s', a') - Q(s, a) | s, a)
        The constraint is weighted by `td_error_weight`.
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
        state_feats = kwargs["state_features"]
        acts = kwargs["actions"].long()
        reward_means = kwargs["reward_mean"]
        reward_log_vars = kwargs["reward_log_var"]
        rationality = kwargs["rationality"][0].item()
        gamma = kwargs["gamma"][0].item()
        td_error_weight = kwargs["td_error_weight"][0].item()

        # Get the Q-value estimates
        q_values = self.Q_value_model(state_feats)  # (batch_size, num_steps, n_actions)

        # ------------------------------------------------------------------------------------------------
        # TD-error constraint
        # ------------------------------------------------------------------------------------------------

        # Compute the TD-error: R(s_t, a_t) = E_{s',a'~pi,T}[Q(s_t, a_t) - γ * Q(s', a')]
        
        # current and next state-action pairs
        acts_curr = acts[:, :-1]  # (batch_size, num_steps - 1) - actions at time t
        acts_next = acts[:, 1:]   # (batch_size, num_steps - 1) - actions at time t+1

        # Select Q(s_t, a_t) for current state-action pairs
        q_curr = torch.gather(q_values[:, :-1, :], dim=2, index=acts_curr.unsqueeze(-1))  # (batch_size, num_steps - 1, 1)
        
        # Select Q(s_{t+1}, a_{t+1}) for next state-action pairs
        q_next = torch.gather(q_values[:, 1:, :], dim=2, index=acts_next.unsqueeze(-1))  # (batch_size, num_steps - 1, 1)
        
        # Compute TD-error (which should equal the reward)
        td_error_selected = q_curr - gamma * q_next  # (batch_size, num_steps - 1, 1)
        td_error_selected = td_error_selected.squeeze(-1)  # (batch_size, num_steps - 1)

        # Compute the log-likelihood of observing the TD-error under the approximate posterior reward distribution
        reward_means_sliced = reward_means[:, :-1]  # (batch_size, num_steps - 1)
        reward_log_vars_sliced = reward_log_vars[:, :-1]  # (batch_size, num_steps - 1)
        
        # Clamp log_var to prevent numerical instability
        reward_log_vars_sliced = torch.clamp(reward_log_vars_sliced, min=-1.5, max=3)
        reward_vars_sliced = reward_log_vars_sliced.exp()

        td_error_nll = F.gaussian_nll_loss(reward_means_sliced, td_error_selected, reward_vars_sliced, reduction='none').mean()

        # ------------------------------------------------------------------------------------------------
        # Boltzmann-rational expert policy likelihood
        # ------------------------------------------------------------------------------------------------

        # Compute the log-likelihood of the demonstrations under the Boltzmann-rational expert policy
        logits = rationality *q_values  # (batch_size, num_steps, n_actions)

        # Shuffle logits to (batch_size, n_actions, num_steps) since expects (N, C, d1, d2, ...) shape
        logits = logits.permute(0, 2, 1)
        demonstrations_nll = nn.functional.cross_entropy(logits, acts.squeeze(), reduction='none').mean()

        # Compute Q-value statistics for logging
        q_value_max = q_values.max().item()
        q_value_min = q_values.min().item()
        
        metrics = {
            "q_value_max": q_value_max,
            "q_value_min": q_value_min,
        }

        return demonstrations_nll + td_error_nll * td_error_weight, metrics