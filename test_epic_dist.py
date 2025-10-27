import numpy as np
from virel.envs.dct_grid_env import DCTGridEnv
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from virel.envs.dct_grid_env import Action
from virel.metrics.epic import canonically_shaped_reward, epic_distance
from virel.utils.reward import Rsa_to_Rsas

ACTION_SYMBOLS = {
    Action.RIGHT: "→",
    Action.UP: "↑",
    Action.LEFT: "←",
    Action.DOWN: "↓",
    Action.STAY: "⊙",
}

def compare_sparse_and_dense_rewards():

    grid_size = 5
    S = grid_size * grid_size
    A = 5
    gamma = 0.99

    env_sparse = DCTGridEnv(grid_size, 8, "sparse", 0.0)
    env_dense = DCTGridEnv(grid_size, 8, "dense", 0.0)

    Rsa_sparse = env_sparse.R
    Rsa_dense = env_dense.R

    Rsas_sparse = Rsa_to_Rsas(Rsa_sparse)
    Rsas_dense = Rsa_to_Rsas(Rsa_dense)

    dist = epic_distance(Rsas_sparse, Rsas_dense, gamma)
    assert dist == 0

    print(dist)

def remove_potential():

    # Sample random R(s,a,s')
    S, A, Sp = 5, 3, 5
    gamma = 0.99
    R1 = np.random.randn(S, A, Sp)

    # Define random scalar function
    F = np.random.randn(S)
    
    # Derive potential
    P = gamma * F[None, :] - F[:, None]

    # Create potential shaped reward
    R2 = R1 + P[:, None, :]

    # Deshape reward
    R1_deshaped = canonically_shaped_reward(R1, gamma)
    R2_deshaped = canonically_shaped_reward(R2, gamma)

    dist = epic_distance(R1_deshaped, R2_deshaped, gamma)
    assert dist == 0

    # Check if deshaped and original reward are 
    assert np.allclose(R1_deshaped, R2_deshaped)


if __name__ == "__main__":
    remove_potential()
    compare_sparse_and_dense_rewards()