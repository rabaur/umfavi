import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from virel.envs.dct_grid_env import DCTGridEnv, dct_features, reward_factory

from virel.single_fb_mod_model import SingleFeedbackTypeModel

def visualize_rewards(env: DCTGridEnv, obs_transform: Callable, act_transform: Callable, fb_model: SingleFeedbackTypeModel, device: torch.device):
    """
    Visualizes the rewards for a DCT grid environment.
    """
    grid_size = env.grid_size
    gt_rewards = np.reshape(env.R, (grid_size, grid_size, -1))
    dct_features_flat = torch.tensor(env.S, dtype=torch.float32).to(device)  # (N^2, K^2)
    n_samples = dct_features_flat.shape[0]

    n_actions = env.action_space.n

    fig, axs = plt.subplots(
        n_actions,
        3, # gt, mean, var
        constrained_layout=True
    )
    
    # Set column titles
    axs[0, 0].set_title("Ground Truth", fontsize=14, fontweight='bold')
    axs[0, 1].set_title("Learned Mean", fontsize=14, fontweight='bold')
    axs[0, 2].set_title("Learned Variance", fontsize=14, fontweight='bold')

    # Collect all data first to compute shared color scales
    all_gt_rewards = []
    all_learned_means = []
    all_learned_vars = []
    
    # Get reward mean and logvar for each state-action combination
    for a in range(env.action_space.n):
        
        # Convert action to features
        a_feat = act_transform(a)

        # Repeat to match features
        a_feat_reps = torch.tile(torch.tensor(a_feat, dtype=torch.float32).to(device), (n_samples, 1))

        # Predict mean and logvar
        mean, log_var = fb_model.encoder(dct_features_flat, a_feat_reps)

        # Reshape the mean and the logvar
        mean = mean.detach().cpu().numpy()
        var = torch.exp(0.5 * log_var).detach().cpu().numpy()
        mean = mean.reshape(grid_size, grid_size)
        var = var.reshape(grid_size, grid_size)
        
        # Store data for shared color scale computation
        all_gt_rewards.append(gt_rewards[..., a])
        all_learned_means.append(mean)
        all_learned_vars.append(var)
    
    # Compute shared color scales
    all_gt_rewards = np.concatenate([r.flatten() for r in all_gt_rewards])
    all_learned_means = np.concatenate([r.flatten() for r in all_learned_means])
    all_learned_vars = np.concatenate([r.flatten() for r in all_learned_vars])
    
    vmin_gt, vmax_gt = np.min(all_gt_rewards), np.max(all_gt_rewards)
    vmin_mean, vmax_mean = np.min(all_learned_means), np.max(all_learned_means)
    vmin_var, vmax_var = np.min(all_learned_vars), np.max(all_learned_vars)

    # Now plot with shared color scales
    for a in range(env.action_space.n):
        # Plot images with shared color scales
        im1 = axs[a, 0].imshow(gt_rewards[..., a], cmap="viridis", vmin=vmin_gt, vmax=vmax_gt)
        im2 = axs[a, 1].imshow(all_learned_means[a * grid_size * grid_size:(a + 1) * grid_size * grid_size].reshape(grid_size, grid_size), 
                               cmap="viridis", vmin=vmin_mean, vmax=vmax_mean)
        im3 = axs[a, 2].imshow(all_learned_vars[a * grid_size * grid_size:(a + 1) * grid_size * grid_size].reshape(grid_size, grid_size), 
                               cmap="viridis", vmin=vmin_var, vmax=vmax_var)
        
        # Set row label (action name) on the leftmost subplot
        axs[a, 0].set_ylabel(f"Action {a}", fontsize=12)
        
        # Remove individual subplot titles and axis labels
        for col in range(3):
            axs[a, col].set_xticks([])
            axs[a, col].set_yticks([])
    
    # Add colorbars for each column
    plt.colorbar(im1, ax=axs[:, 0], label="Reward Value")
    plt.colorbar(im2, ax=axs[:, 1], label="Reward Value") 
    plt.colorbar(im3, ax=axs[:, 2], label="Uncertainty")
    
    plt.show()



    

    
    