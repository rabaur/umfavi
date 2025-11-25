import torch
import numpy as np
import matplotlib.pyplot as plt
from umfavi.envs.grid_env.env import GridEnv
from umfavi.utils.math import log_var_to_std
from umfavi.envs.grid_env.actions import Action
from torch.utils.data import DataLoader
from umfavi.utils.torch import to_numpy
from umfavi.utils.feature_transforms import get_feature_combinations
from umfavi.encoder.reward_encoder import RewardEncoder
from umfavi.types import SampleKey


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
    device = None
    counts_flat = None  # will hold counts over N*N cells

    for batch in dataloader:
        states = batch[SampleKey.STATES].long()          # shape: (..., 2) where last dim is (row, col)
        # ensure tensor
        if not isinstance(states, torch.Tensor):
            states = torch.as_tensor(states)

        if device is None:
            device = states.device
            counts_flat = torch.zeros(N * N, dtype=torch.long, device=device)

        # collapse all leading dims, idx is already flattened
        # works for (T, 2), (2, T, 2), (B, T, 2), (B, 2, T, 2), etc.
        idxs = states.reshape(-1).long()

        # count occurrences in this batch
        batch_counts = torch.bincount(idxs, minlength=N * N)

        # accumulate
        counts_flat += batch_counts

    # reshape to 2D and move to numpy for plotting
    counts = counts_flat.view(N, N).cpu().numpy()
    im = ax.imshow(np.log(counts + 1))
    return im

def visualize_rewards(
    env: GridEnv,
    encoder: RewardEncoder,
    dataloader: DataLoader,
    all_obs_features: torch.Tensor,
    all_act_features: torch.Tensor
):
    """
    Visualizes the estimated rewards for a grid environment.
    
    Returns:
        fig: matplotlib figure object that can be logged to wandb
    """
    grid_size = env.grid_size
    gt_rewards = np.reshape(env._R, (grid_size, grid_size, -1))
    gt_rewards = np.max(gt_rewards, axis=-1)
    reward_domain = encoder.features.reward_domain
    if reward_domain == 'sas':
        raise NotImplementedError("Visualization of s,a,s' rewards is not implemented")
    
    # Construct all state-action-next_state features to compute the estimated reward matrix
    batch_state_features, batch_action_features, batch_next_state_features = get_feature_combinations(reward_domain, all_obs_features, all_act_features)

    # Predict mean and logvar
    mean, log_var = encoder(batch_state_features, batch_action_features, batch_next_state_features)

    if reward_domain == 's':
        # Create figure with 2 rows: 
        # Row 0: ground truth, log occupancy
        # Row 1: mean rewards, std rewards
        fig, axs = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(12, 14)
        )
        # Reshape the mean and the logvar
        mean = to_numpy(mean).squeeze()
        std = to_numpy(log_var_to_std(log_var)).squeeze()
        mean_grid = mean.reshape(grid_size, grid_size)
        std_grid = std.reshape(grid_size, grid_size)
        
        vmin_gt, vmax_gt = np.min(gt_rewards), np.max(gt_rewards)
        vmin_mean, vmax_mean = np.min(mean), np.max(mean)
        vmin_std, vmax_std = np.min(std), np.max(std)
        
        # Plot mean and std
        im3 = axs[1, 0].imshow(mean_grid, vmin=vmin_mean, vmax=vmax_mean)
        im4 = axs[1, 1].imshow(std_grid, vmin=vmin_std, vmax=vmax_std)

        # Set titles for last two rows
        axs[1, 0].set_title(r"$\mu$", fontsize=14)
        axs[1, 1].set_title(r"$\sigma$", fontsize=14)

        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        plt.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)
        plt.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

    elif reward_domain == 'sa':
        # Create figure with 1 + num_actions rows: 
        # Row 0: ground truth, log occupancy
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

    else:
        raise ValueError(f"Unknown reward domain: {reward_domain}")
    
    # Set titles for first row
    axs[0, 0].set_title("Ground Truth", fontsize=14)
    axs[0, 1].set_title("log(Occupancy)", fontsize=14)
        
    # Plot ground truth
    im1 = axs[0, 0].imshow(gt_rewards, vmin=vmin_gt, vmax=vmax_gt)

    # Plot log occupancy
    im2 = visualize_state_action_dist(env, dataloader, axs[0, 1])
    
    # Remove axis ticks for first row
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    
    # Add colorbars for first row
    plt.colorbar(im1, ax=axs[0, 0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axs[0, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig
