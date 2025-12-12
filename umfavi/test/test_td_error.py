import torch
from umfavi.regularizer.td_error import td_error_regularizer
from umfavi.types import SampleKey

def test_perfect_td_match():
    """When reward_mean == TD_error, loss should be near zero."""
    batch_size = 4
    n_actions = 3
    
    # Create known Q-values
    q_curr = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0],
                           [10.0, 11.0, 12.0]])
    q_next = torch.tensor([[0.5, 1.0, 1.5],
                           [2.0, 2.5, 3.0],
                           [3.5, 4.0, 4.5],
                           [5.0, 5.5, 6.0]])
    
    acts_curr = torch.tensor([[0], [1], [2], [0]])  # Select Q: 1, 5, 9, 10
    acts_next = torch.tensor([[1], [2], [0], [1]])  # Select Q: 1, 3, 3.5, 5.5
    gamma = 0.99
    
    # Compute expected TD errors
    expected_td = torch.tensor([1.0 - 0.99*1.0,   # = 0.01
                                 5.0 - 0.99*3.0,   # = 2.03
                                 9.0 - 0.99*3.5,   # = 5.535
                                 10.0 - 0.99*5.5]) # = 4.555
    
    # Set reward_mean to exactly match TD error
    reward_mean = expected_td
    reward_log_var = torch.full((batch_size,), -10.0)  # Very small variance
    
    kwargs = {
        SampleKey.ACTS: acts_curr,
        SampleKey.NEXT_ACTS: acts_next,
        SampleKey.GAMMA: torch.tensor([gamma]),
        SampleKey.TERMINATED: torch.zeros(batch_size, dtype=torch.bool),
        SampleKey.INVALID: torch.zeros(batch_size, dtype=torch.bool),
        "q_curr": q_curr,
        "q_next": q_next,
        "reward_mean": reward_mean,
        "reward_log_var": reward_log_var,
    }
    
    loss = td_error_regularizer(**kwargs)
    assert loss < 1e-3, f"Expected near-zero loss, got {loss}"


def test_terminal_states():
    """Terminal states should have Q(s',a') = 0 in TD error."""
    batch_size = 2
    n_actions = 2
    
    q_curr = torch.tensor([[5.0, 3.0], [4.0, 6.0]])
    q_next = torch.tensor([[100.0, 200.0], [300.0, 400.0]])  # Should be ignored for terminal
    
    acts_curr = torch.tensor([[0], [1]])  # Select: 5, 6
    acts_next = torch.tensor([[0], [0]])
    gamma = 0.99
    
    # Second transition is terminal
    terminated = torch.tensor([False, True])
    
    # For terminal: TD = Q(s,a) - 0 = Q(s,a) = 6
    # For non-terminal: TD = 5 - 0.99*100 = -94
    expected_td = torch.tensor([-94.0, 6.0])
    
    reward_mean = expected_td
    reward_log_var = torch.full((batch_size,), -10.0)
    
    kwargs = {
        SampleKey.ACTS: acts_curr,
        SampleKey.NEXT_ACTS: acts_next,
        SampleKey.GAMMA: torch.tensor([gamma]),
        SampleKey.TERMINATED: terminated,
        SampleKey.INVALID: torch.zeros(batch_size, dtype=torch.bool),
        "q_curr": q_curr,
        "q_next": q_next,
        "reward_mean": reward_mean,
        "reward_log_var": reward_log_var,
    }
    
    loss = td_error_regularizer(**kwargs)
    assert loss < 1e-3, f"Terminal handling failed, loss = {loss}"


def test_invalid_masking():
    """Invalid transitions should be excluded from loss."""
    batch_size = 4
    n_actions = 2
    
    q_curr = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    q_next = torch.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0], [3.5, 4.0]])
    
    acts_curr = torch.tensor([[0], [0], [0], [0]])
    acts_next = torch.tensor([[0], [0], [0], [0]])
    gamma = 0.99
    
    # Mark indices 1 and 3 as invalid
    invalid = torch.tensor([False, True, False, True])
    
    # Only indices 0, 2 are valid
    # TD[0] = 1.0 - 0.99*0.5 = 0.505
    # TD[2] = 5.0 - 0.99*2.5 = 2.525
    
    # Set correct TD for valid, garbage for invalid
    reward_mean = torch.tensor([0.505, 9999.0, 2.525, 9999.0])
    reward_log_var = torch.full((batch_size,), -10.0)
    
    kwargs = {
        SampleKey.ACTS: acts_curr,
        SampleKey.NEXT_ACTS: acts_next,
        SampleKey.GAMMA: torch.tensor([gamma]),
        SampleKey.TERMINATED: torch.zeros(batch_size, dtype=torch.bool),
        SampleKey.INVALID: invalid,
        "q_curr": q_curr,
        "q_next": q_next,
        "reward_mean": reward_mean,
        "reward_log_var": reward_log_var,
    }
    
    loss = td_error_regularizer(**kwargs)
    # If masking works, the 9999.0 values should be ignored → low loss
    assert loss < 1e-3, f"Invalid masking failed, loss = {loss}"