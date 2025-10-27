import numpy as np
from numpy.typing import NDArray
from virel.utils.tabular import q_opt


def value_under_policy(P, R_true, gamma, pi):
    S, A, Sp = P.shape
    assert Sp == S
    # r_pi[s] = E[R(s, pi[s], S')]
    if R_true.ndim == 2:      # R(s,a)
        r_pi = R_true[np.arange(S), pi]
    else:                     # R(s,a,s')
        r_sa = R_true[np.arange(S), pi, :]          # (S, S)
        r_pi = np.sum(P[np.arange(S), pi, :] * r_sa, axis=1)

    # P_pi[s, s'] = P(s'|s, pi[s])
    P_pi = P[np.arange(S), pi, :]                   # (S, S)

    # Solve (I - gamma P_pi) V = r_pi
    I = np.eye(S)
    V = np.linalg.solve(I - gamma * P_pi, r_pi)
    return V  # shape (S,)


def evaluate_regret(
    R_true: NDArray,
    R_est: NDArray,
    P: NDArray,
    gamma: float,
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
    V_true_star = np.max(Q_true_opt, axis=1)           # (S,)
    pi_est = np.argmax(Q_est_opt, axis=1)              # (S,)
    V_true_pi = value_under_policy(P, R_true, gamma, pi_est)

    regret = float(np.mean(V_true_star - V_true_pi))
    return regret

