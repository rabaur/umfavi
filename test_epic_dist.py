import numpy as np
from torch.ao import ns
from virel.metrics.metrics import epic_distance, singleton_shaping_canonical_reward, fully_connected_random_canonical_reward, pearson_distance
from virel.envs.dct_grid_env import DCTGridEnv
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from virel.envs.dct_grid_env import Action

ACTION_SYMBOLS = {
    Action.RIGHT: "→",
    Action.UP: "↑",
    Action.LEFT: "←",
    Action.DOWN: "↓",
    Action.STAY: "⊙",
}

def sa_to_sas(rew: NDArray):
    ns = rew.shape[0]
    return np.repeat(rew[..., np.newaxis], ns, axis=2)


def compare_sparse_and_dense_rewards():

    grid_size = 8
    n_a = 5
    sparse_env = DCTGridEnv(grid_size=grid_size, n_dct_basis_fns=8, reward_type="sparse", p_rand=0.0)
    dense_env = DCTGridEnv(grid_size=grid_size, n_dct_basis_fns=8, reward_type="dense", p_rand=0.0)

    sparse_rew = sparse_env.R
    dense_rew = dense_env.R

    # our rewards (s, a) only, but epic distance needs (s, a, s').
    sparse_rew = sa_to_sas(sparse_rew)
    dense_rew = sa_to_sas(dense_rew)

    dense_deshaped = fully_connected_random_canonical_reward(dense_rew, 0.999)
    sparse_deshaped = fully_connected_random_canonical_reward(sparse_rew, 0.999)
    dense_reshaped = np.reshape(dense_rew[..., 0], (grid_size, grid_size, -1))
    dense_deshaped_reshaped = np.reshape(dense_deshaped[..., 0], (grid_size, grid_size, -1))
    sparse_deshaped_reshaped = np.reshape(sparse_deshaped[..., 0], (grid_size, grid_size, -1))

    max_dense_deshaped = dense_deshaped.max()
    max_sparse_deshaped = sparse_deshaped.max()

    print(pearson_distance(dense_deshaped, sparse_deshaped))



    fig, axs = plt.subplots(3, 5, figsize=(15, 9))
    
    # Set consistent color scales for better comparison
    vmin_dense = dense_reshaped.min()
    vmax_dense = dense_reshaped.max()
    vmin_deshaped = min(dense_deshaped_reshaped.min(), sparse_deshaped_reshaped.min())
    vmax_deshaped = max(dense_deshaped_reshaped.max(), sparse_deshaped_reshaped.max())
    
    for a in range(n_a):
        im0 = axs[0, a].imshow(dense_reshaped[..., a], vmin=vmin_dense, vmax=vmax_dense, cmap='viridis')
        axs[0, a].set_title(f'Dense (a={ACTION_SYMBOLS[a]})')
        axs[0, a].axis('off')
        
        im1 = axs[1, a].imshow(dense_deshaped_reshaped[..., a], vmin=vmin_deshaped, vmax=vmax_deshaped, cmap='viridis')
        axs[1, a].set_title(f'Dense Deshaped (a={ACTION_SYMBOLS[a]})')
        axs[1, a].axis('off')
        
        im2 = axs[2, a].imshow(sparse_deshaped_reshaped[..., a], vmin=vmin_deshaped, vmax=vmax_deshaped, cmap='viridis')
        axs[2, a].set_title(f'Sparse Deshaped (a={ACTION_SYMBOLS[a]})')
        axs[2, a].axis('off')
    
    # Add colorbars for each row
    fig.colorbar(im0, ax=axs[0, :], orientation='vertical', pad=0.05, label='Dense Reward')
    fig.colorbar(im1, ax=axs[1, :], orientation='vertical', pad=0.05, label='Dense Deshaped')
    fig.colorbar(im2, ax=axs[2, :], orientation='vertical', pad=0.05, label='Sparse Deshaped')
    
    plt.tight_layout()
    plt.show()

    dist = epic_distance(sparse_rew, dense_rew, 0.999)

    print(dist)


if __name__ == "__main__":
    compare_sparse_and_dense_rewards()