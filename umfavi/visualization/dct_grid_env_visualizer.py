import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from vsup import VSUP
from umfavi.envs.grid_env.env import GridEnv
from umfavi.multi_fb_model import MultiFeedbackTypeModel
from umfavi.utils.math import log_var_to_std
from umfavi.metrics.epic import canonically_shaped_reward
from umfavi.utils.reward import Rsa_to_Rsas
from umfavi.envs.grid_env.actions import Action
from torch.utils.data import DataLoader

from umfavi.utils.torch import to_numpy

# Action symbols for visualization
ACTION_SYMBOLS = {
    Action.RIGHT: "→",
    Action.UP:    "↑",
    Action.LEFT:  "←",
    Action.DOWN:  "↓",
    Action.STAY:  "⊙",
}

def visualize_state_action_dist(
    env: GridEnv,
    dataloader: DataLoader
):
    N = env.grid_size
    for batch in dataloader:
        visualize_batch(batch, N)

def visualize_batch(batch: dict, N: int):
    counts = np.zeros((N, N))
    states = batch["states"]
    for traj in states:
        for cell in traj:
            cell = to_numpy(cell)
            i, j = int(cell[0]), int(cell[1])
            counts[i, j] += 1 
    counts = np.log(counts + 1)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(counts)
    plt.show()


def visualize_rewards(
    env: GridEnv,
    act_transform: Callable,
    fb_model: MultiFeedbackTypeModel,
    device: torch.device,
    gamma: float = 0.99):
    """
    Visualizes the rewards for a DCT grid environment using VSUP 
    (Value-Suppressing Uncertainty Palette) to combine mean and uncertainty.
    Also displays the canonicalized mean reward.
    """
    print({a.value: a.name for a in Action})
    grid_size = env.grid_size
    gt_rewards = np.reshape(env.R, (grid_size, grid_size, -1))
    n_actions = env.action_space.n
    
    # Create figure with 3 columns: ground truth, VSUP, and canonicalized
    _, axs = plt.subplots(
        nrows=n_actions,
        ncols=3  # gt, vsup (mean + uncertainty combined), canonicalized
    )
    
    # Handle case where n_actions == 1 (axs would be 1D)
    if n_actions == 1:
        axs = axs.reshape(1, -1)
    
    # Set column titles
    axs[0, 0].set_title("Ground Truth", fontsize=14, fontweight='bold')
    axs[0, 1].set_title("Learned (Mean + Uncertainty)", fontsize=14, fontweight='bold')
    axs[0, 2].set_title("Canonicalized Mean", fontsize=14, fontweight='bold')

    # Collect all data first to compute shared color scales
    all_gt_rewards = []
    all_learned_means = []
    all_learned_stds = []
    
    # Get reward mean and logvar for each state-action combination
    state_feats_flat = torch.tensor(env.S, dtype=torch.float32).to(device=device)

    row_order = [Action.RIGHT, Action.UP, Action.LEFT, Action.DOWN, Action.STAY]
    
    # Store means in (S, A) format for canonicalization
    learned_means_sa = np.zeros((grid_size**2, n_actions))

    for row_idx, act in enumerate(row_order):
        
        # Convert action to features
        a = act.value
        a_feat = act_transform(a, n_actions=n_actions)

        # Repeat to match features
        a_feat_reps = torch.tile(torch.tensor(a_feat, dtype=torch.float32).to(device), (grid_size**2, 1))

        # Predict mean and logvar
        mean, log_var = fb_model.encoder(state_feats_flat, a_feat_reps, state_feats_flat)

        # Reshape the mean and the logvar
        mean = np.array(mean.detach().cpu())
        std = log_var_to_std(log_var).detach().cpu().numpy()
        mean_flat = mean.squeeze()
        mean_grid = mean_flat.reshape(grid_size, grid_size)
        std = std.reshape(grid_size, grid_size).squeeze()
        
        # Store flat mean for canonicalization
        learned_means_sa[:, a] = mean_flat
        
        # Store data for shared color scale computation
        all_gt_rewards.append(gt_rewards[..., a])
        all_learned_means.append(mean_grid)
        all_learned_stds.append(std)
    
    # Convert learned means to Rsas format and apply canonicalization
    learned_means_sas = Rsa_to_Rsas(learned_means_sa)
    true_means_sas = Rsa_to_Rsas(env.R)
    canonicalized_means_sas = canonically_shaped_reward(true_means_sas, gamma)
    
    # Store canonicalized means per action
    all_canonicalized_means = []
    for a in range(n_actions):
        # Extract mean over s' for each s,a pair
        canon_mean = canonicalized_means_sas[:, a, :].mean(axis=1)
        canon_grid = canon_mean.reshape(grid_size, grid_size)
        all_canonicalized_means.append(canon_grid)
    
    # Compute shared color scales
    all_gt_rewards_flat = np.concatenate([r.flatten() for r in all_gt_rewards])
    all_learned_means_flat = np.concatenate([r.flatten() for r in all_learned_means])
    all_learned_stds_flat = np.concatenate([r.flatten() for r in all_learned_stds])
    all_canonicalized_means_flat = np.concatenate([r.flatten() for r in all_canonicalized_means])
    
    vmin_gt, vmax_gt = np.min(all_gt_rewards_flat), np.max(all_gt_rewards_flat)
    vmin_mean, vmax_mean = np.min(all_learned_means_flat), np.max(all_learned_means_flat)
    vmin_std, vmax_std = np.min(all_learned_stds_flat), np.max(all_learned_stds_flat)
    vmin_canon, vmax_canon = np.min(all_canonicalized_means_flat), np.max(all_canonicalized_means_flat)

    # Initialize VSUP with desired parameters
    # mode='usl' = uncertainty-suppressing lightness
    # palette='viridis' uses the viridis colormap as base
    vsup = VSUP(palette='viridis', mode='usl')

    # Now plot with shared color scales
    for row_idx, act in enumerate(row_order):
        a = act.value
        # Get data for this action
        mean_grid = all_learned_means[a]
        std_grid = all_learned_stds[a]
        canon_grid = all_canonicalized_means[a]
        
        # Normalize mean and variance to [0, 1] range for VSUP
        mean_normalized = (mean_grid - vmin_mean) / (vmax_mean - vmin_mean + 1e-8)
        std_normalized = (std_grid - vmin_std) / (vmax_std - vmin_std + 1e-8)
        
        # Generate VSUP colors (calling vsup directly returns RGB array)
        vsup_colors = vsup(mean_normalized, std_normalized)
        
        # Plot ground truth
        axs[row_idx, 0].imshow(gt_rewards[..., a], cmap="viridis", vmin=vmin_gt, vmax=vmax_gt)
        
        # Plot VSUP visualization (combined mean + uncertainty)
        axs[row_idx, 1].imshow(vsup_colors)
        
        # Plot canonicalized mean
        axs[row_idx, 2].imshow(canon_grid, cmap="viridis", vmin=vmin_canon, vmax=vmax_canon)
        
        # Set row label (action symbol) on the leftmost subplot
        axs[a, 0].set_ylabel(
            f"{ACTION_SYMBOLS[a]}",
            fontsize=16,
            labelpad=20,
            rotation=0,
            va='center')
        
        # Remove individual subplot titles and axis labels
        for col in range(3):
            axs[a, col].set_xticks([])
            axs[a, col].set_yticks([])
    
    # Add colorbar for ground truth
    # plt.colorbar(im1, ax=axs[:, 0], label="Reward Value")
    
    plt.show()



    

    
    