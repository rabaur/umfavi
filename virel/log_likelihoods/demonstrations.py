import torch
from torch import nn
from virel.log_likelihoods.base_log_likelihood import BaseLogLikelihood
from virel.utils.math import log_var_to_std

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
        obs = kwargs["obs"]
        acts = kwargs["acts"]
        reward_means = kwargs["reward_mean"]
        reward_log_vars = kwargs["reward_log_var"]
        rationality = kwargs["rationality"][0].item()
        gamma = kwargs["gamma"][0].item()
        td_error_weight = kwargs.get("td_error_weight", 1.0)

        # Get obs shape for reshaping
        batch_size, num_steps, obs_dim = obs.shape
        
        # Flatten obs to (batch_size * num_steps, obs_dim) for Q-value model
        obs_flat = obs.reshape(batch_size * num_steps, obs_dim)
        
        # Get the Q-value estimates
        q_values_flat = self.Q_value_model(obs_flat)  # (batch_size * num_steps, n_actions)
        
        # Reshape back to (batch_size, num_steps, n_actions)
        q_values = q_values_flat.reshape(batch_size, num_steps, -1)

        # ------------------------------------------------------------------------------------------------
        # TD-error constraint
        # ------------------------------------------------------------------------------------------------

        # Compute the TD-error: R(s_t, a_t) = Q(s_t, a_t) - γ * Q(s_{t+1}, a_{t+1})
        
        # Get action indices
        act_integer = torch.argmax(acts, dim=-1).unsqueeze(-1)  # (batch_size, num_steps, 1)
        act_integer_curr = act_integer[:, :-1]  # (batch_size, num_steps - 1) - actions at time t
        act_integer_next = act_integer[:, 1:]   # (batch_size, num_steps - 1) - actions at time t+1
        
        # Select Q(s_t, a_t) for current state-action pairs
        q_curr = torch.gather(q_values[:, :-1, :], dim=2, index=act_integer_curr).squeeze(-1)  # (batch_size, num_steps - 1)
        
        # Select Q(s_{t+1}, a_{t+1}) for next state-action pairs
        q_next = torch.gather(q_values[:, 1:, :], dim=2, index=act_integer_next).squeeze(-1)  # (batch_size, num_steps - 1)
        
        # Compute TD-error (which should equal the reward)
        td_error_selected = q_curr - gamma * q_next  # (batch_size, num_steps - 1)

        # Compute the log-likelihood of observing the TD-error under the approximate posterior reward distribution
        reward_means = reward_means.squeeze(-1)[..., :-1]  # (batch_size, num_steps - 1)
        reward_log_vars = reward_log_vars.squeeze(-1)[..., :-1]  # (batch_size, num_steps - 1)
        reward_stds = log_var_to_std(reward_log_vars)
        
        q_theta = torch.distributions.Normal(reward_means, reward_stds)
        td_error_nll = -q_theta.log_prob(td_error_selected).sum()

        # ------------------------------------------------------------------------------------------------
        # Boltzmann-rational expert policy likelihood
        # ------------------------------------------------------------------------------------------------

        # Compute the log-likelihood of the demonstrations under the Boltzmann-rational expert policy
        log_probs = torch.log_softmax(rationality * q_values, dim=-1)  # (batch_size, num_steps, n_actions)
        
        # Gather log probabilities for the actions taken
        demonstrations_nll = -log_probs.gather(dim=2, index=act_integer).squeeze(-1).sum()  # (batch_size,)

        return demonstrations_nll + td_error_nll * td_error_weight