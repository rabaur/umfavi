import torch
from torch import nn
from virel.log_likelihoods.base_log_likelihood import BaseLogLikelihood

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
                - rationality: Rationality coefficient (default: 1.0).
                - gamma: Discount factor (default: 0.99).
                - td_error_weight: Weight for the TD-error constraint (default: 1.0).
        Returns:
            ...
        """
        # Extract parameters from kwargs
        obs = kwargs["obs"]
        acts = kwargs["acts"]
        rationality = kwargs.get("rationality", 1.0)
        gamma = kwargs.get("gamma", 0.99)
        td_error_weight = kwargs.get("td_error_weight", 1.0)
        
        # For this decoder, we need reward_means and reward_log_vars, but we only have reward_samples
        # We'll use the samples as means and assume zero variance for now
        # This is a simplification - in practice, you might want to pass these separately
        reward_means = reward_samples
        reward_log_vars = torch.zeros_like(reward_samples)
        # Flatten the first two dimensions of obs and acts such that the shape is (batch_size * num_steps, obs_dim) and (batch_size * num_steps, act_dim)
        obs = obs.view(obs.shape[0] * obs.shape[1], -1)
        acts = acts.view(acts.shape[0] * acts.shape[1], -1)

        # Get the Q-value estimates
        q_values = self.Q_value_model(obs)  # (batch_size * num_steps, n_actions)

        # Transform back
        q_values = q_values.view(obs.shape[0], obs.shape[1], -1)  # (batch_size, num_steps, n_actions)

        # ------------------------------------------------------------------------------------------------
        # TD-error constraint
        # ------------------------------------------------------------------------------------------------


        # Compute the TD-error Q(s', a') - Q(s, a)
        td_error = q_values[:, 1:] - gamma * q_values[:, :-1]  # (batch_size, num_steps - 1, n_actions)

        # Select TD-error for observed (s, a) pairs
        act_integer = torch.argmax(acts, dim=-1).squeeze()[:, :-1] # (batch_size, num_steps - 1)
        td_error = td_error[..., act_integer]  # (batch_size, num_steps - 1)

        # Compute the log-likelihood of observing the TD-error under the approximate posterior reward distribution
        q_theta = torch.distributions.Normal(reward_means, torch.exp(reward_log_vars))
        td_error_nll = -q_theta.log_prob(td_error).sum(dim=1)

        # ------------------------------------------------------------------------------------------------
        # Boltzmann-rational expert policy likelihood
        # ------------------------------------------------------------------------------------------------

        # Compute the log-likelihood of the demonstrations under the Boltzmann-rational expert policy
        log_probs = torch.log_softmax(rationality * q_values, dim=1)
        demonstrations_nll = -log_probs.gather(1, act_integer).sum(dim=1)

        return demonstrations_nll + td_error_nll * td_error_weight