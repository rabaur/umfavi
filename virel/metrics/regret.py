import numpy as np
from numpy.typing import NDArray
from typing import Optional
from virel.utils.tabular import q_opt


def evaluate_regret(
    R_true: NDArray,
    R_est: NDArray,
    P: NDArray,
    gamma: float,
    initial_state_dist: Optional[NDArray] = None,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> float:
    """
    Computes expected regret over states.
    """
    
    # Compute optimal Q-values for the true reward
    Q_true_opt = q_opt(P, R_true, gamma, max_iter=max_iter, tol=tol)
    
    # Compute optimal Q-values for the estimated reward
    Q_est_opt = q_opt(P, R_est, gamma, max_iter=max_iter, tol=tol)
    
    # Extract optimal policies (greedy w.r.t. Q-values)
    pi_true_opt = np.argmax(Q_true_opt, axis=1, keepdims=True)  # Shape: (n_states,)
    pi_est = np.argmax(Q_est_opt, axis=1, keepdims=True)    # Shape: (n_states,)

    # Compute the value functions
    V_true_opt = np.take_along_axis(Q_true_opt, pi_true_opt, axis=-1)
    V_est_opt = np.take_along_axis(Q_est_opt, pi_est, axis=-1)

    # Regret (normalized by the number of states)
    regret = (V_true_opt - V_est_opt).mean()
    return regret

