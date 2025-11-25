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

    # Compute the TD-error: R(s_t, a_t) = Q(s_t, a_t) - γ * Q(s_{t+1}, a_{t+1})
    
    # Slice Q-values and actions for current and next timesteps
    acts = acts.long()
    
    # Q-values and actions at time t (current)
    q_values_curr = q_values[..., :-1, :]  # (batch_size, num_steps - 1, n_actions)
    acts_curr = acts[..., :-1, :]  # (batch_size, num_steps - 1, 1)
    
    # Q-values and actions at time t+1 (next)
    q_values_next = q_values[..., 1:, :]  # (batch_size, num_steps - 1, n_actions)
    acts_next = acts[..., 1:, :]   # (batch_size, num_steps - 1, 1)

    # Select Q(s_t, a_t) for current state-action pairs
    q_curr = torch.gather(q_values_curr, dim=-1, index=acts_curr).squeeze(-1)  # (batch_size, num_steps - 1)
    
    # Select Q(s_{t+1}, a_{t+1}) for next state-action pairs
    q_next = torch.gather(q_values_next, dim=-1, index=acts_next).squeeze(-1)  # (batch_size, num_steps - 1)

    # Compute TD-error (which should equal the reward)
    # Handle gamma whether it's a tensor or float
    gamma_val = gamma[0].item()
    td_error = q_curr - gamma_val * q_next  # (batch_size, num_steps - 1)

    # Compute the log-likelihood of observing the TD-error under the approximate posterior reward distribution
    reward_std = log_var_to_std(reward_log_var)

    # Create mask for valid timesteps (not done/padded)
    # dones has shape (batch_size, num_steps, 1)
    # We need to slice it to match td_error shape (num_steps - 1)
    valid_mask = ~dones[..., :-1, :].squeeze(-1).bool()  # (batch_size, num_steps - 1)
    
    # Slice rewards to match td_error shape and apply mask
    # reward_mean and reward_log_var have shape (batch_size, num_steps)
    # We need (batch_size, num_steps - 1) to match td_error
    valid_reward_mean = reward_mean[..., :-1][valid_mask]
    valid_reward_std = reward_std[..., :-1][valid_mask]
    valid_td_error = td_error[valid_mask]
    
    # Compute NLL only over valid timesteps
    normal_dist = torch.distributions.Normal(valid_reward_mean, valid_reward_std)
    td_error_nll = -normal_dist.log_prob(valid_td_error).mean()
    
    return td_error_nll