import torch
from umfavi.utils.math import log_var_to_std
from umfavi.types import SampleKey

def td_error_regularizer(**kwargs) -> torch.Tensor:
    """
    Args:
        q_values: Q-values of shape (batch_size, num_steps, n_actions)
        acts: Actions of shape (batch_size, num_steps, 1)
        reward_mean: Mean of reward distribution of shape (batch_size, num_steps)
        reward_log_var: Log variance of reward distribution of shape (batch_size, num_steps)
        gamma: Discount factor
        dones: Boolean mask of shape (batch_size, num_steps, 1) indicating terminal/invalid states
    """

    # Unpack variables
    acts_curr = kwargs[SampleKey.ACTS].long()
    acts_next = kwargs[SampleKey.NEXT_ACTS].long()
    gamma = kwargs[SampleKey.GAMMA][0].item()
    dones = kwargs[SampleKey.DONES]
    q_curr = kwargs["q_curr"]  # (batch_size, num_actions)
    q_next = kwargs["q_next"]  # (batch_size, num_actions)
    reward_mean = kwargs["reward_mean"]
    reward_std = log_var_to_std(kwargs["reward_log_var"])

    # Select Q(s_t, a_t) for current state-action pairs
    q_curr_a = torch.gather(q_curr, dim=-1, index=acts_curr).squeeze(-1)  # (batch_size, num_steps - 1)
    
    # Select Q(s_{t+1}, a_{t+1}) for next state-action pairs
    q_next_a = torch.gather(q_next, dim=-1, index=acts_next).squeeze(-1)  # (batch_size, num_steps - 1)

    # Set Q_next to 0 for invalid timesteps
    q_next_a = q_next_a * (1.0 - dones)

    # Compute TD-error (which should equal the reward)
    # Handle gamma whether it's a tensor or float
    td_error = q_curr_a - gamma * q_next_a  # (batch_size)
    
    # Compute NLL only over valid timesteps
    normal_dist = torch.distributions.Normal(reward_mean, reward_std)
    td_error_nll = -normal_dist.log_prob(td_error).mean()
    
    return td_error_nll