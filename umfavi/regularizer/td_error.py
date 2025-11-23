import torch
from umfavi.utils.math import log_var_to_std

def td_error_regularizer(
    q_values: torch.Tensor,
    acts: torch.Tensor,
    reward_mean: torch.Tensor,
    reward_log_var: torch.Tensor,
    gamma: float,
    dones: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        q_values: Q-values of shape (batch_size, num_steps, n_actions)
        acts: Actions of shape (batch_size, num_steps, 1)
        reward_mean: Mean of reward distribution of shape (batch_size, num_steps)
        reward_log_var: Log variance of reward distribution of shape (batch_size, num_steps)
        gamma: Discount factor
        dones: Boolean mask of shape (batch_size, num_steps, 1) indicating terminal/invalid states
    """

    # ------------------------------------------------------------------------------------------------
    # TD-error constraint
    # ------------------------------------------------------------------------------------------------

    # Compute the TD-error: R(s_t, a_t) = E_{s',a'~pi,T}[Q(s, a) - γ * Q(s', a')]
    
    # current and next state-action pairs
    acts = acts.long()
    acts_curr = acts  # (batch_size, num_steps) - actions at time t
    acts_next = acts[..., 1:, :]   # (batch_size, num_steps - 1) - actions at time t+1

    # Select Q(s_t, a_t) for current state-action pairs
    q_curr = torch.gather(q_values, dim=-1, index=acts_curr)  # (batch_size, ..., num_steps, 1)
    
    # Select Q(s_{t+1}, a_{t+1}) for next state-action pairs
    q_next = torch.zeros_like(q_curr)  # (batch_size, ..., num_steps, 1)

    # Pad q_next with 0, since 
    q_next[..., :-1, :] = torch.gather(q_values[..., 1:, :], dim=-1, index=acts_next)  # (batch_size, ..., num_steps - 1, 1)
    
    # Compute TD-error (which should equal the reward)
    td_error = q_curr - gamma[0].item() * q_next  # (batch_size, num_steps)

    # Compute the log-likelihood of observing the TD-error under the approximate posterior reward distribution
    
    # Clamp log_var to prevent numerical instability
    reward_std = log_var_to_std(reward_log_var)

    # Create mask for valid timesteps (not done/padded)
    # dones has shape (batch_size, num_steps, 1), squeeze to (batch_size, num_steps)
    valid_mask = ~dones.squeeze(-1).bool()  # True for valid timesteps
    
    # Only compute loss over valid timesteps (where mask is True)
    valid_reward_mean = reward_mean[valid_mask]
    valid_reward_std = reward_std[valid_mask]
    valid_td_error = td_error[valid_mask]
    
    # Compute NLL only over valid timesteps
    normal_dist = torch.distributions.Normal(valid_reward_mean, valid_reward_std)
    td_error_nll = -normal_dist.log_prob(valid_td_error).mean()
    
    return td_error_nll