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
    dataloader: DataLoader,
    ax: plt.Axes
):
    N = env.grid_size
    counts = np.zeros((N, N))
    for batch in dataloader:
        states = batch["states"]
        for traj in states:
            for cell in traj:
                cell = to_numpy(cell)
                i, j = int(cell[0]), int(cell[1])
                counts[i, j] += 1
    ax.imshow(np.log(counts + 1))

def visualize_rewards(
    env: GridEnv,
    fb_model: MultiFeedbackTypeModel,
    device: torch.device,
    dataloader: DataLoader):
    """
    Visualizes the rewards for a DCT grid environment using VSUP 
    (Value-Suppressing Uncertainty Palette) to combine mean and uncertainty.
    Also displays the canonicalized mean reward.
    """
    grid_size = env.grid_size
    gt_rewards = np.reshape(env.R, (grid_size, grid_size, -1))
    gt_rewards = np.max(gt_rewards, axis=-1)
    
    # Create figure with 3 columns: ground truth, state-action distribution, VSUP
    _, axs = plt.subplots(
        nrows=1,
        ncols=3
    )
    
    # Set column titles
    axs[0].set_title("Ground Truth", fontsize=14, fontweight='bold')
    axs[1].set_title("log(Occupancy)", fontsize=14, fontweight='bold')
    axs[2].set_title("Inferred Reward", fontsize=14, fontweight='bold')
    
    # Get reward mean and logvar for each state-action combination
    state_feats_flat = torch.tensor(env.S).to(device=device)

    # Predict mean and logvar
    mean, log_var = fb_model.encoder(state_feats_flat, None, None)

    # Reshape the mean and the logvar
    mean = to_numpy(mean).squeeze()
    std = to_numpy(log_var_to_std(log_var)).squeeze()
    mean_grid = mean.reshape(grid_size, grid_size)
    std_grid = std.reshape(grid_size, grid_size)
    
    vmin_gt, vmax_gt = np.min(gt_rewards), np.max(gt_rewards)
    vmin_mean, vmax_mean = np.min(mean), np.max(mean)
    vmin_std, vmax_std = np.min(std), np.max(std)

    # Initialize VSUP with desired parameters
    # mode='usl' = uncertainty-suppressing lightness
    # palette='viridis' uses the viridis colormap as base
    vsup = VSUP(palette='viridis', mode='usl')
        
    # Normalize mean and variance to [0, 1] range for VSUP
    mean_normalized = (mean_grid - vmin_mean) / (vmax_mean - vmin_mean + 1e-8)
    std_normalized = (std_grid - vmin_std) / (vmax_std - vmin_std + 1e-8)
        
    # Generate VSUP colors (calling vsup directly returns RGB array)
    vsup_colors = vsup(mean_normalized, std_normalized)
        
    # Plot ground truth
    axs[0].imshow(gt_rewards, cmap="viridis", vmin=vmin_gt, vmax=vmax_gt)

    # Plot ground truth
    visualize_state_action_dist(env, dataloader, axs[1])
        
    # Plot VSUP visualization (combined mean + uncertainty)
    axs[2].imshow(vsup_colors)
         
    # Remove individual subplot titles and axis labels
    for col in range(3):
        axs[col].set_xticks([])
        axs[col].set_yticks([])
    
    # Add colorbar for ground truth
    # plt.colorbar(im1, ax=axs[:, 0], label="Reward Value")
    
    plt.show()



    

    
    