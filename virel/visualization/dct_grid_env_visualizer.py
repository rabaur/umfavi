import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from vsup import VSUP
from virel.envs.dct_grid_env import DCTGridEnv, Action
from virel.multi_fb_model import MultiFeedbackTypeModel
from virel.utils.math import log_var_to_std

# Action symbols for visualization
ACTION_SYMBOLS = {
    Action.RIGHT: "→",
    Action.UP:    "↑",
    Action.LEFT:  "←",
    Action.DOWN:  "↓",
    Action.STAY:  "⊙",
}

def visualize_rewards(
    env: DCTGridEnv,
    act_transform: Callable,
    fb_model: MultiFeedbackTypeModel,
    device: torch.device):
    """
    Visualizes the rewards for a DCT grid environment using VSUP 
    (Value-Suppressing Uncertainty Palette) to combine mean and uncertainty.
    """
    print({a.value: a.name for a in Action})
    grid_size = env.grid_size
    gt_rewards = np.reshape(env.R, (grid_size, grid_size, -1))
    n_actions = env.action_space.n
    
    # Create figure with 2 columns: ground truth and VSUP visualization
    _, axs = plt.subplots(
        nrows=n_actions,
        ncols=2  # gt, vsup (mean + uncertainty combined)
    )
    
    # Handle case where n_actions == 1 (axs would be 1D)
    if n_actions == 1:
        axs = axs.reshape(1, -1)
    
    # Set column titles
    axs[0, 0].set_title("Ground Truth", fontsize=14, fontweight='bold')
    axs[0, 1].set_title("Learned (Mean + Uncertainty)", fontsize=14, fontweight='bold')

    # Collect all data first to compute shared color scales
    all_gt_rewards = []
    all_learned_means = []
    all_learned_stds = []
    
    # Get reward mean and logvar for each state-action combination
    state_feats_flat = torch.tensor(env.S, dtype=torch.float32).to(device=device)

    row_order = [Action.RIGHT, Action.UP, Action.LEFT, Action.DOWN, Action.STAY]

    for row_idx, act in enumerate(row_order):
        
        # Convert action to features
        a = act.value
        a_feat = act_transform(a)

        # Repeat to match features
        a_feat_reps = torch.tile(torch.tensor(a_feat, dtype=torch.float32).to(device), (grid_size**2, 1))

        # Predict mean and logvar
        mean, log_var = fb_model.encoder(state_feats_flat, a_feat_reps)

        # Reshape the mean and the logvar
        mean = np.array(mean.detach().cpu())
        std = log_var_to_std(log_var).detach().cpu().numpy()
        mean = mean.reshape(grid_size, grid_size).squeeze()
        std = std.reshape(grid_size, grid_size).squeeze()
        
        # Store data for shared color scale computation
        all_gt_rewards.append(gt_rewards[..., a])
        all_learned_means.append(mean)
        all_learned_stds.append(std)
    
    # Compute shared color scales
    all_gt_rewards_flat = np.concatenate([r.flatten() for r in all_gt_rewards])
    all_learned_means_flat = np.concatenate([r.flatten() for r in all_learned_means])
    all_learned_stds_flat = np.concatenate([r.flatten() for r in all_learned_stds])
    
    vmin_gt, vmax_gt = np.min(all_gt_rewards_flat), np.max(all_gt_rewards_flat)
    vmin_mean, vmax_mean = np.min(all_learned_means_flat), np.max(all_learned_means_flat)
    vmin_std, vmax_std = np.min(all_learned_stds_flat), np.max(all_learned_stds_flat)

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
        
        # Normalize mean and variance to [0, 1] range for VSUP
        mean_normalized = (mean_grid - vmin_mean) / (vmax_mean - vmin_mean + 1e-8)
        std_normalized = (std_grid - vmin_std) / (vmax_std - vmin_std + 1e-8)
        
        # Generate VSUP colors (calling vsup directly returns RGB array)
        vsup_colors = vsup(mean_normalized, std_normalized)
        
        # Plot ground truth
        axs[row_idx, 0].imshow(gt_rewards[..., a], cmap="viridis", vmin=vmin_gt, vmax=vmax_gt)
        
        # Plot VSUP visualization (combined mean + uncertainty)
        axs[row_idx, 1].imshow(vsup_colors)
        
        # Set row label (action symbol) on the leftmost subplot
        action_enum = Action(a)
        print(f"Action {a} ({action_enum}) symbol: {ACTION_SYMBOLS[a]}")
        axs[a, 0].set_ylabel(
            f"{ACTION_SYMBOLS[a]}",
            fontsize=16,
            labelpad=20,
            rotation=0,
            va='center')
        
        # Remove individual subplot titles and axis labels
        for col in range(2):
            axs[a, col].set_xticks([])
            axs[a, col].set_yticks([])
    
    # Add colorbar for ground truth
    # plt.colorbar(im1, ax=axs[:, 0], label="Reward Value")
    
    plt.show()



    

    
    