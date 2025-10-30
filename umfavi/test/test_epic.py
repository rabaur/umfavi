import numpy as np
from umfavi.metrics.epic import canonically_shaped_reward, epic_distance

def test_random_potential():

    # Generate random reward function
    n_S, n_A = 100, 50
    gamma = 0.99
    shaping_factor = 0.75
    R_true = np.random.randn(n_S, n_A, n_S)

    # Create state-wise bounded function f(s)
    f = np.random.randn(n_S)

    # Create potential function p(s) = gamma * f(s') - f(s)
    p = gamma * f[None, None, :] - f[:, None, None]

    # Create shaped reward
    R_shaped = R_true + p

    # Compute the canonical rewards
    R_true_canon = canonically_shaped_reward(R_true, gamma)
    R_shaped_canon = canonically_shaped_reward(R_shaped, gamma)

    # Canonicalized versions should be equivalent
    assert np.allclose(R_true_canon, R_shaped_canon), "Canonicalized reward are not the same"

    # Shape and scale reward
    R_shaped_and_scaled = shaping_factor * R_true + p

    # Epic distance should be 0
    dist = epic_distance(R_true, R_shaped_and_scaled, gamma)
    assert np.allclose(dist, 0, atol=1e-6), f"Epic distance {dist} != 0"