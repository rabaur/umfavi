import numpy as np
from numpy.typing import NDArray

def q_opt_iteration(
    Q_old: NDArray,
    T: NDArray,
    R: NDArray,
    gamma: float
) -> NDArray:
    """
    Q-value iteration - vectorized version.
    
    Args:
        Q_old: Current Q-values matrix (n_states, n_actions)
        T: Transition matrix (n_states, n_actions, n_states)
        R: Reward matrix (n_states, n_actions)
        gamma: Discount factor
        
    Returns:
        Updated Q-values matrix (n_states, n_actions)
    """
    # Compute V(s) = max_a Q(s,a) for each state
    V = np.max(Q_old, axis=1)  # Shape: (n_states,)
    
    # Compute expected future value: sum_s' P(s'|s,a) * V(s')
    expected_future_value_vectorized = np.sum(T * V, axis=2)
    
    # Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) * V(s')
    Q_new = R + gamma * expected_future_value_vectorized
    
    return Q_new

def q_opt(
    T: NDArray,
    R: NDArray,
    gamma: float,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> NDArray:
    """
    Q-value iteration to convergence using fixed iterations for differentiation.
    
    Args:
        T: Transition matrix (n_states, n_actions, n_states)
        R: Reward matrix (n_states, n_actions)
        gamma: Discount factor
        max_iter: Maximum number of iterations (fixed)
        tol: Convergence tolerance (not used, kept for compatibility)
        
    Returns:
        Optimal Q-values matrix (n_states, n_actions)
    """
    n_states, n_actions = R.shape
    Q = np.zeros((n_states, n_actions))
    
    for _ in range(max_iter):
        Q_new = q_opt_iteration(Q, T, R, gamma)
        if np.max(np.abs(Q_new - Q)) < tol:
            Q = Q_new
            break
        Q = Q_new
    
    return Q