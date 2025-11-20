import torch
from umfavi.utils.math import log_var_to_std

def td_error_regularizer(
    q_values: torch.Tensor,
    acts: torch.Tensor,
    reward_mean: torch.Tensor,
    reward_log_var: torch.Tensor,
    gamma: float
) -> torch.Tensor:

    # ------------------------------------------------------------------------------------------------
    # TD-error constraint
    # ------------------------------------------------------------------------------------------------

    # Compute the TD-error: R(s_t, a_t) = E_{s',a'~pi,T}[Q(s, a) - γ * Q(s', a')]
    
    # current and next state-action pairs
    acts_curr = acts[..., :-1]  # (batch_size, num_steps - 1) - actions at time t
    acts_next = acts[..., 1:]   # (batch_size, num_steps - 1) - actions at time t+1

    # Select Q(s_t, a_t) for current state-action pairs
    q_curr = torch.gather(q_values[..., :-1, :], dim=-1, index=acts_curr.unsqueeze(-1)).squeeze(-1)  # (batch_size, ..., num_steps - 1)
    
    # Select Q(s_{t+1}, a_{t+1}) for next state-action pairs
    q_next = torch.gather(q_values[..., 1:, :], dim=-1, index=acts_next.unsqueeze(-1)).squeeze(-1)  # (batch_size, num_steps - 1)
    
    # Compute TD-error (which should equal the reward)
    td_error_selected = q_curr - gamma[0].item() * q_next  # (batch_size, num_steps - 1)

    # Compute the log-likelihood of observing the TD-error under the approximate posterior reward distribution
    reward_mean_selected = reward_mean[..., :-1]  # (batch_size, num_steps - 1)
    reward_log_var_selected = reward_log_var[..., :-1]  # (batch_size, num_steps - 1)
    
    # Clamp log_var to prevent numerical instability
    reward_std_selected = log_var_to_std(reward_log_var_selected)

    normal_dist = torch.distributions.Normal(reward_mean_selected, reward_std_selected)
    td_error_nll = -normal_dist.log_prob(td_error_selected).mean()
    return td_error_nll