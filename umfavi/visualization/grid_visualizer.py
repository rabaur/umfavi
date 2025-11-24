import torch
import numpy as np
import matplotlib.pyplot as plt
from umfavi.envs.grid_env.env import GridEnv
from umfavi.multi_fb_model import MultiFeedbackTypeModel
from umfavi.utils.math import log_var_to_std
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
    device = None
    counts_flat = None  # will hold counts over N*N cells

    for batch in dataloader:
        states = batch["states"]          # shape: (..., 2) where last dim is (row, col)
        # ensure tensor
        if not isinstance(states, torch.Tensor):
            states = torch.as_tensor(states)

        if device is None:
            device = states.device
            counts_flat = torch.zeros(N * N, dtype=torch.long, device=device)

        # collapse all leading dims, keep last dim 2
        # works for (T, 2), (2, T, 2), (B, T, 2), (B, 2, T, 2), etc.
        idxs = states.reshape(-1, 2)

        rows = idxs[:, 0].long()
        cols = idxs[:, 1].long()

        # optional safety in case something is slightly out of bounds
        rows = rows.clamp(0, N - 1)
        cols = cols.clamp(0, N - 1)

        # map 2D indices to flat indices
        flat_idx = rows * N + cols

        # count occurrences in this batch
        batch_counts = torch.bincount(flat_idx, minlength=N * N)

        # accumulate
        counts_flat += batch_counts

    # reshape to 2D and move to numpy for plotting
    counts = counts_flat.view(N, N).cpu().numpy()
    im = ax.imshow(np.log(counts + 1))
    return im

def visualize_rewards(
    env: GridEnv,
    fb_model: MultiFeedbackTypeModel,
    device: torch.device,
    dataloader: DataLoader):
    """
    Visualizes the rewards for a DCT grid environment using VSUP 
    (Value-Suppressing Uncertainty Palette) to combine mean and uncertainty.
    Also displays the canonicalized mean reward and Q-values per action.
    
    Returns:
        fig: matplotlib figure object that can be logged to wandb
    """
    grid_size = env.grid_size
    gt_rewards = np.reshape(env._R, (grid_size, grid_size, -1))
    gt_rewards = np.max(gt_rewards, axis=-1)
    
    # Create figure with 3 rows: 
    # Row 0: ground truth, log occupancy
    # Row 1: mean rewards, std rewards
    # Row 2-3: Q-values for each action (5 actions in 2 rows)
    fig, axs = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(12, 14)
    )
    
    # Set titles for first two rows
    axs[0, 0].set_title("Ground Truth", fontsize=14)
    axs[0, 1].set_title("log(Occupancy)", fontsize=14)
    axs[1, 0].set_title(r"$\mu$", fontsize=14)
    axs[1, 1].set_title(r"$\sigma$", fontsize=14)
    
    # Get reward mean and logvar for each state-action combination
    state_feats_flat = torch.tensor(env._S).to(device=device)

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
        
    # Plot ground truth
    im1 = axs[0, 0].imshow(gt_rewards, vmin=vmin_gt, vmax=vmax_gt)

    # Plot log occupancy
    im2 = visualize_state_action_dist(env, dataloader, axs[0, 1])
        
    # Plot mean
    im3 = axs[1, 0].imshow(mean_grid, vmin=vmin_mean, vmax=vmax_mean)

    # Plot std
    im4 = axs[1, 1].imshow(std_grid, vmin=vmin_std, vmax=vmax_std)
    
    # Get Q-values for all states
    q_values = fb_model.Q_value_model(state_feats_flat)  # shape: (n_states, n_actions)
    q_values_np = to_numpy(q_values)  # shape: (grid_size^2, 5)
    
    # Compute global vmin/vmax for Q-values to use same scale
    vmin_q, vmax_q = np.min(q_values_np), np.max(q_values_np)
    
    # Plot Q-values for each action
    action_list = [Action.RIGHT, Action.UP, Action.LEFT, Action.DOWN, Action.STAY]
    q_images = []
    
    for idx, action in enumerate(action_list):
        row = 2 + idx // 3  # rows 2 and 3
        col = idx % 3        # columns 0, 1, 2
        
        # Get Q-values for this action and reshape to grid
        q_action = q_values_np[:, action].reshape(grid_size, grid_size)
        
        # Plot with consistent colormap scale
        im_q = axs[row, col].imshow(q_action, vmin=vmin_q, vmax=vmax_q)
        axs[row, col].set_title(f"Q({ACTION_SYMBOLS[action]})", fontsize=14)
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])
        
        q_images.append(im_q)
    
    # Hide unused subplots in first two rows
    axs[0, 2].axis('off')
    axs[1, 2].axis('off')
         
    # Remove axis labels for first two rows
    for row in range(2):
        for col in range(2):
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
    
    # Add individual colorbars for each subplot
    plt.colorbar(im1, ax=axs[0, 0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axs[0, 1], fraction=0.046, pad=0.04)
    plt.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)
    plt.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)
    
    # Add colorbars for Q-value plots
    for idx, im_q in enumerate(q_images):
        row = 2 + idx // 3
        col = idx % 3
        plt.colorbar(im_q, ax=axs[row, col], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig
