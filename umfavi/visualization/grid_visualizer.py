import torch
import numpy as np
import matplotlib.pyplot as plt
from umfavi.envs.grid_env.env import GridEnv
from umfavi.utils.math import log_var_to_std
from umfavi.envs.grid_env.actions import Action
from umfavi.utils.torch import to_numpy
from umfavi.utils.feature_transforms import get_feature_combinations
from umfavi.multi_fb_model import MultiFeedbackTypeModel


# Action symbols for visualization
ACTION_SYMBOLS = {
    Action.RIGHT: "→",
    Action.UP:    "↑",
    Action.LEFT:  "←",
    Action.DOWN:  "↓",
    Action.STAY:  "⊙",
}


def vis_grid_env(
    env: GridEnv,
    fb_model: MultiFeedbackTypeModel
):
    """
    Visualizes the estimated rewards for a grid environment.
    
    Returns:
        fig: matplotlib figure object that can be logged to wandb
    """
    encoder = fb_model.encoder
    grid_size = env.grid_size
    gt_rewards = np.reshape(env._R, (grid_size, grid_size, -1))
    gt_rewards = np.max(gt_rewards, axis=-1)
    reward_domain = encoder.features.reward_domain
    if reward_domain == 'sas':
        raise NotImplementedError("Visualization of s,a,s' rewards is not implemented")
    
    # Construct one-hot features for all states and actions
    n_states = grid_size * grid_size
    n_actions = env.action_space.n
    device = next(encoder.parameters()).device
    
    # One-hot state features: identity matrix of shape (n_states, n_states)
    all_obs_features = torch.eye(n_states, device=device)
    # One-hot action features: identity matrix of shape (n_actions, n_actions)
    all_act_features = torch.eye(n_actions, device=device)
    
    # Construct all state-action-next_state features to compute the estimated reward matrix
    batch_state_features, batch_action_features, batch_next_state_features = get_feature_combinations(reward_domain, all_obs_features, all_act_features)

    # Predict mean and logvar
    mean, log_var = encoder(batch_state_features, batch_action_features, batch_next_state_features)

    if reward_domain == 's':
        # Create figure with 1 row, 3 columns:
        # ground truth, mean rewards, std rewards
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(15, 5)
        )
        # Reshape the mean and the logvar
        mean = to_numpy(mean).squeeze()
        std = to_numpy(log_var_to_std(log_var)).squeeze()
        mean_grid = mean.reshape(grid_size, grid_size)
        std_grid = std.reshape(grid_size, grid_size)
        
        vmin_gt, vmax_gt = np.min(gt_rewards), np.max(gt_rewards)
        vmin_mean, vmax_mean = np.min(mean), np.max(mean)
        vmin_std, vmax_std = np.min(std), np.max(std)
        
        # Plot ground truth
        im1 = axs[0].imshow(gt_rewards, vmin=vmin_gt, vmax=vmax_gt)
        axs[0].set_title("Ground Truth", fontsize=14)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
        
        # Plot mean
        im2 = axs[1].imshow(mean_grid, vmin=vmin_mean, vmax=vmax_mean)
        axs[1].set_title(r"$\mu$", fontsize=14)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
        
        # Plot std
        im3 = axs[2].imshow(std_grid, vmin=vmin_std, vmax=vmax_std)
        axs[2].set_title(r"$\sigma$", fontsize=14)
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        plt.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig

    elif reward_domain == 'sa':
        # Create figure with 1 + num_actions rows: 
        # Row 0: ground truth (empty second column)
        # Remaining rows: mean rewards, std rewards for each action
        num_actions = env.action_space.n
        fig, axs = plt.subplots(
            nrows=1 + num_actions,
            ncols=2,
            figsize=(12, 4 * (1 + num_actions))
        )
        
        # Reshape the mean and the logvar
        # Shape: (num_states * num_actions,) -> (num_states, num_actions) -> (num_actions, grid_size, grid_size)
        mean = to_numpy(mean).squeeze()
        std = to_numpy(log_var_to_std(log_var)).squeeze()
        
        # Reshape to (num_states, num_actions)
        num_states = grid_size * grid_size
        mean_sa = mean.reshape(num_states, num_actions)  # (S, A)
        std_sa = std.reshape(num_states, num_actions)    # (S, A)
        
        # Compute global vmin/vmax for consistent colormaps
        vmin_gt, vmax_gt = np.min(gt_rewards), np.max(gt_rewards)
        vmin_mean, vmax_mean = np.min(mean), np.max(mean)
        vmin_std, vmax_std = np.min(std), np.max(std)
        
        # Plot ground truth in first row
        im1 = axs[0, 0].imshow(gt_rewards, vmin=vmin_gt, vmax=vmax_gt)
        axs[0, 0].set_title("Ground Truth", fontsize=14)
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        plt.colorbar(im1, ax=axs[0, 0], fraction=0.046, pad=0.04)
        
        # Hide the second column in the first row
        axs[0, 1].axis('off')
        
        # Plot mean and std for each action
        for action_idx in range(num_actions):
            # Extract rewards for this action and reshape to grid
            mean_grid = mean_sa[:, action_idx].reshape(grid_size, grid_size)
            std_grid = std_sa[:, action_idx].reshape(grid_size, grid_size)
            
            # Plot mean
            im_mean = axs[action_idx + 1, 0].imshow(mean_grid, vmin=vmin_mean, vmax=vmax_mean)
            # Plot std
            im_std = axs[action_idx + 1, 1].imshow(std_grid, vmin=vmin_std, vmax=vmax_std)
            
            # Set titles with action symbol if available
            action_enum = Action(action_idx)
            action_symbol = ACTION_SYMBOLS.get(action_enum, str(action_idx))
            axs[action_idx + 1, 0].set_title(f"$\mu$ ({action_symbol})", fontsize=14)
            axs[action_idx + 1, 1].set_title(f"$\sigma$ ({action_symbol})", fontsize=14)
            
            # Remove axis ticks
            axs[action_idx + 1, 0].set_xticks([])
            axs[action_idx + 1, 0].set_yticks([])
            axs[action_idx + 1, 1].set_xticks([])
            axs[action_idx + 1, 1].set_yticks([])
            
            # Add colorbars
            plt.colorbar(im_mean, ax=axs[action_idx + 1, 0], fraction=0.046, pad=0.04)
            plt.colorbar(im_std, ax=axs[action_idx + 1, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown reward domain: {reward_domain}")
