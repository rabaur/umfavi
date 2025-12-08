import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from umfavi.multi_fb_model import MultiFeedbackTypeModel
from umfavi.utils.math import log_var_to_std, softmax
from umfavi.utils.torch import to_numpy
from umfavi.types import SampleKey, TrajKeys
from umfavi.utils.policies import ExpertPolicy, QValueModel
from umfavi.utils.gym import rollout, unpack_trajectory


class LearnedQValueModel(QValueModel):
    """
    Q-value model wrapper for the learned Q-value network in MultiFeedbackTypeModel.
    """
    
    def __init__(self, q_net: torch.nn.Module, device: torch.device, gamma: float = 0.99):
        """
        Initialize learned Q-value model wrapper.
        
        Args:
            q_net: The learned Q-value network (fb_model.Q_value_model)
            device: PyTorch device
            gamma: Discount factor
        """
        self.q_net = q_net
        self.device = device
        self._gamma = gamma
    
    def get_q_values(self, observation) -> np.ndarray:
        """Get Q-values for an observation."""
        # Handle reset return format (obs, info)
        if isinstance(observation, tuple):
            observation = observation[0]
        
        # Convert observation to tensor
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        
        # Convert to numpy
        return q_values.cpu().numpy().flatten()
    
    @property
    def gamma(self) -> float:
        return self._gamma


class LearnedQExpertPolicy(ExpertPolicy):
    """
    Expert policy using the learned Q-value model from MultiFeedbackTypeModel.
    """
    
    def __init__(self, q_model: LearnedQValueModel, rationality: float = 1.0):
        """
        Initialize learned Q expert policy.
        
        Args:
            q_model: LearnedQValueModel for Q-value computation
            rationality: Rationality parameter (β) for softmax policy
        """
        super().__init__(q_model, rationality)
    
    def __call__(self, observation):
        """
        Sample action from the expert policy using softmax over Q-values.
        
        Args:
            observation: Observation from environment
            
        Returns:
            Action sampled from the expert policy
        """
        # Get Q-values for this observation
        q_values = self.q_model.get_q_values(observation)
        
        # Apply softmax with rationality parameter
        if self.rationality != float('inf'):
            action_probs = softmax(self.rationality * q_values, dims=0)
            # Sample action from the distribution
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            # Deterministic policy
            action = np.argmax(q_values)
        
        return action


def visualize_lunarlander_rewards(
    fb_model: MultiFeedbackTypeModel,
    device: torch.device,
    dataloader=None,
    resolution=30,
    num_samples=5,
    est_expert_policy: ExpertPolicy = None,
    env: gym.Env = None,
    num_trajectories: int = 50,
    max_traj_steps: int = 100
):
    """
    Visualizes learned rewards, uncertainties, and Q-values for LunarLander-v3 environment.
    
    Creates a 4x4 grid showing:
    - Row 1: 4 action reward heatmaps over x,y position space
    - Row 2: 4 action uncertainty heatmaps over x,y position space
    - Row 3: 4 action Q-value heatmaps over x,y position space
    - Row 4: Visitation distribution + trajectories (with est_expert_policy) colored by predicted reward + 
             trajectories (with learned Q-value policy) colored by estimated Q-value
    
    Args:
        fb_model: The multi-feedback model
        device: PyTorch device
        dataloader: Optional dataloader for visitation visualization
        resolution: Resolution of x,y heatmaps
        num_samples: Number of sample points per dimension for averaging
        est_expert_policy: Optional estimated expert policy for trajectory sampling (used for reward trajectories)
        env: Optional environment for rolling out trajectories
        num_trajectories: Number of trajectories to sample on one plot (default: 50)
        max_traj_steps: Maximum steps per trajectory (default: 100)
    
    Returns:
        matplotlib Figure object
    """
    fig, axs = plt.subplots(4, 4, figsize=(24, 24))
    axs = axs.flatten()
    
    # LunarLander-v3 observation space bounds
    x_range = (-1, 1)
    y_range = (-0.25, 1.75)
    
    # Define sampling ranges for other dimensions
    # vx, vy, angle, angular_velocity, leg1_contact, leg2_contact
    other_dims_ranges = [
        (-10.0, 10.0),  # vx
        (-10.0, 10.0),  # vy
        (-6.28, 6.28),  # angle (≈ -2π to 2π)
        (-10.0, 10.0),  # angular velocity
        (0.0, 1.0),     # leg1 contact (boolean)
        (0.0, 1.0),     # leg2 contact (boolean)
    ]
    
    # Create sample points for the other dimensions
    sample_points = [np.linspace(low, high, num_samples) for low, high in other_dims_ranges]
    
    action_names = ["Do Nothing", "Fire Left", "Fire Main", "Fire Right"]
    
    for action_idx in range(4):
        # Compute average reward and uncertainty over x,y grid
        reward_grid, uncertainty_grid = compute_action_reward_grid(
            fb_model, device, action_idx,
            x_range, y_range, resolution,
            sample_points
        )
        
        # Compute average Q-values over x,y grid
        qvalue_grid = compute_action_qvalue_grid(
            fb_model, device, action_idx,
            x_range, y_range, resolution,
            sample_points
        )
        
        # Plot reward heatmap (Row 1)
        im_reward = axs[action_idx].imshow(
            reward_grid.T,  # Transpose for correct orientation
            origin='lower',
            aspect='auto',
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            cmap='RdYlGn'  # Red-Yellow-Green colormap (red=bad, green=good)
        )
        
        axs[action_idx].set_xlabel('X Position')
        axs[action_idx].set_ylabel('Y Position')
        axs[action_idx].set_title(f'Reward - {action_names[action_idx]}')
        
        # Add landing pad marker at (0, 0)
        axs[action_idx].plot(0, 0, 'b*', markersize=15, label='Landing Pad')
        axs[action_idx].legend(loc='upper right')
        
        plt.colorbar(im_reward, ax=axs[action_idx], fraction=0.046, pad=0.04, label='Avg Reward')
        
        # Plot uncertainty heatmap (Row 2)
        im_uncertainty = axs[action_idx + 4].imshow(
            uncertainty_grid.T,  # Transpose for correct orientation
            origin='lower',
            aspect='auto',
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            cmap='plasma'  # Plasma colormap for uncertainty
        )
        
        axs[action_idx + 4].set_xlabel('X Position')
        axs[action_idx + 4].set_ylabel('Y Position')
        axs[action_idx + 4].set_title(f'Uncertainty - {action_names[action_idx]}')
        
        # Add landing pad marker
        axs[action_idx + 4].plot(0, 0, 'b*', markersize=15, label='Landing Pad')
        axs[action_idx + 4].legend(loc='upper right')
        
        plt.colorbar(im_uncertainty, ax=axs[action_idx + 4], fraction=0.046, pad=0.04, label='Std Dev')
        
        # Plot Q-value heatmap (Row 3)
        im_qvalue = axs[action_idx + 8].imshow(
            qvalue_grid.T,  # Transpose for correct orientation
            origin='lower',
            aspect='auto',
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            cmap='RdYlGn'  # Same as rewards
        )
        
        axs[action_idx + 8].set_xlabel('X Position')
        axs[action_idx + 8].set_ylabel('Y Position')
        axs[action_idx + 8].set_title(f'Q-Value - {action_names[action_idx]}')
        
        # Add landing pad marker
        axs[action_idx + 8].plot(0, 0, 'b*', markersize=15, label='Landing Pad')
        axs[action_idx + 8].legend(loc='upper right')
        
        plt.colorbar(im_qvalue, ax=axs[action_idx + 8], fraction=0.046, pad=0.04, label='Avg Q-Value')
    
    # Plot visitation distribution if dataloader provided (Row 4, first subplot)
    if dataloader is not None:
        plot_visitation_distribution(axs[12], dataloader, x_range, y_range, resolution)
    else:
        axs[12].text(0.5, 0.5, 'No dataset provided', 
                   ha='center', va='center', transform=axs[12].transAxes)
        axs[12].axis('off')
    
    # Plot multiple trajectories colored by reward (Row 4, second subplot)
    if est_expert_policy is not None and env is not None:
        plot_multiple_trajectories_with_reward(
            axs[13], 
            fb_model, 
            device, 
            est_expert_policy, 
            env,
            x_range, 
            y_range,
            max_traj_steps,
            num_trajectories=num_trajectories
        )
    else:
        axs[13].text(0.5, 0.5, 'No policy/env provided', 
                   ha='center', va='center', transform=axs[13].transAxes)
        axs[13].axis('off')
    
    # Plot multiple trajectories colored by Q-value using policy derived from learned Q-value model (Row 4, third subplot)
    if env is not None:
        # Create expert policy from the learned Q-value model
        learned_q_model = LearnedQValueModel(fb_model.Q_value_model, device, gamma=0.99)
        learned_q_policy = LearnedQExpertPolicy(learned_q_model, rationality=10)
        
        plot_multiple_trajectories_with_qvalue(
            axs[14], 
            fb_model, 
            device, 
            learned_q_policy, 
            env,
            x_range, 
            y_range,
            max_traj_steps,
            num_trajectories=num_trajectories
        )
    else:
        axs[14].text(0.5, 0.5, 'No env provided', 
                   ha='center', va='center', transform=axs[14].transAxes)
        axs[14].axis('off')
    
    # Hide the remaining subplot in row 4
    axs[15].axis('off')
    
    plt.tight_layout()
    return fig


def compute_action_reward_grid(
    fb_model, device, action_idx,
    x_range, y_range, resolution,
    sample_points
):
    """
    Computes average reward and uncertainty for a given action over x,y grid.
    
    Args:
        fb_model: The multi-feedback model
        action_idx: Index of the action (0-3)
        x_range: Tuple of (min, max) for x coordinate
        y_range: Tuple of (min, max) for y coordinate
        resolution: Grid resolution for x,y
        sample_points: List of arrays with sample values for other dimensions
    
    Returns:
        Tuple of two 2D numpy arrays of shape (resolution, resolution):
        - Average rewards
        - Average uncertainties (standard deviation)
    """
    # Create x,y grid
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    
    # Create meshgrids for ALL dimensions at once
    grids = np.meshgrid(
        x, y,
        sample_points[0],  # vx
        sample_points[1],  # vy
        sample_points[2],  # angle
        sample_points[3],  # angular velocity
        sample_points[4],  # leg1
        sample_points[5],  # leg2
        indexing='ij'
    )
    
    # Stack into state array: [x, y, vx, vy, angle, ang_vel, leg1, leg2]
    # Shape: (resolution, resolution, n_vx, n_vy, n_angle, n_ang_vel, n_leg1, n_leg2, 8)
    states = np.stack(grids, axis=-1)
    
    # Flatten to (n_total_states, 8) for batch prediction
    original_shape = states.shape[:-1]  # Save for reshaping later
    states_flat = states.reshape(-1, 8)
    
    # Predict rewards and uncertainties for this action in one batch
    mean_flat, std_flat = predict_reward_for_action(
        fb_model, states_flat, device, action_idx
    )
    
    # Reshape back to original grid structure
    mean_grid = mean_flat.reshape(original_shape)
    std_grid = std_flat.reshape(original_shape)
    
    # Average over all dimensions except x,y (first two dimensions)
    # Shape: (resolution, resolution, n_vx, n_vy, n_angle, n_ang_vel, n_leg1, n_leg2)
    # -> (resolution, resolution)
    reward_avg = mean_grid.max(axis=tuple(range(2, len(original_shape))))
    uncertainty_avg = std_grid.max(axis=tuple(range(2, len(original_shape))))
    
    return reward_avg, uncertainty_avg


def predict_reward_for_action(fb_model, states, device, action_idx):
    """
    Predicts reward and uncertainty for a batch of states and a specific action.
    
    Args:
        fb_model: The multi-feedback model
        states: Numpy array of states (n_states, state_dim)
        device: PyTorch device
        action_idx: Index of the action
    
    Returns:
        Tuple of numpy arrays (mean, std) of predicted rewards and uncertainties
    """
    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    n_states = states_tensor.shape[0]
    
    # Get reward domain from feature module
    reward_domain = fb_model.encoder.features.reward_domain
    
    with torch.no_grad():
        if reward_domain == 's':
            # State-only reward (ignore action)
            mean, var = fb_model.encoder(states_tensor, None, None)
        elif reward_domain == 'sa':
            # State-action reward
            # Create one-hot encoded action (LunarLander has 4 discrete actions)
            actions = torch.zeros(n_states, 4, device=device)
            actions[:, action_idx] = 1.0
            mean, var = fb_model.encoder(states_tensor, actions, None)
        elif reward_domain == 'sas':
            # State-action-next_state reward
            # For visualization, use same state as next state
            actions = torch.zeros(n_states, 4, device=device)
            actions[:, action_idx] = 1.0
            mean, var = fb_model.encoder(states_tensor, actions, states_tensor)
        else:
            raise ValueError(f"Unknown reward_domain: {reward_domain}")
    
    # Convert variance to standard deviation
    std = log_var_to_std(var)
    
    return to_numpy(mean.squeeze()), to_numpy(std.squeeze())


def compute_action_qvalue_grid(
    fb_model, device, action_idx,
    x_range, y_range, resolution,
    sample_points
):
    """
    Computes average Q-value for a given action over x,y grid.
    
    Args:
        fb_model: The multi-feedback model
        action_idx: Index of the action (0-3)
        x_range: Tuple of (min, max) for x coordinate
        y_range: Tuple of (min, max) for y coordinate
        resolution: Grid resolution for x,y
        sample_points: List of arrays with sample values for other dimensions
    
    Returns:
        2D numpy array of shape (resolution, resolution) with average Q-values
    """
    # Create x,y grid
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    
    # Create meshgrids for ALL dimensions at once
    grids = np.meshgrid(
        x, y,
        sample_points[0],  # vx
        sample_points[1],  # vy
        sample_points[2],  # angle
        sample_points[3],  # angular velocity
        sample_points[4],  # leg1
        sample_points[5],  # leg2
        indexing='ij'
    )
    
    # Stack into state array: [x, y, vx, vy, angle, ang_vel, leg1, leg2]
    # Shape: (resolution, resolution, n_vx, n_vy, n_angle, n_ang_vel, n_leg1, n_leg2, 8)
    states = np.stack(grids, axis=-1)
    
    # Flatten to (n_total_states, 8) for batch prediction
    original_shape = states.shape[:-1]  # Save for reshaping later
    states_flat = states.reshape(-1, 8)
    
    # Predict Q-values for this action in one batch
    qvalue_flat = predict_qvalue_for_action(
        fb_model, states_flat, device, action_idx
    )
    
    # Reshape back to original grid structure
    qvalue_grid = qvalue_flat.reshape(original_shape)
    
    # Average over all dimensions except x,y (first two dimensions)
    qvalue_avg = qvalue_grid.max(axis=tuple(range(2, len(original_shape))))
    
    return qvalue_avg


def predict_qvalue_for_action(fb_model, states, device, action_idx):
    """
    Predicts Q-value for a batch of states and a specific action.
    
    Args:
        fb_model: The multi-feedback model
        states: Numpy array of states (n_states, state_dim)
        device: PyTorch device
        action_idx: Index of the action
    
    Returns:
        Numpy array of predicted Q-values for the specified action
    """
    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # Get Q-values for all actions
        q_values = fb_model.Q_value_model(states_tensor)  # Shape: (n_states, n_actions)
        
        # Extract Q-values for the specified action
        q_values_for_action = q_values[:, action_idx]
    
    return to_numpy(q_values_for_action)


def plot_visitation_distribution(ax, dataloader, x_range, y_range, resolution):
    """
    Plots the visitation distribution over x,y coordinates from the dataset.
    
    Args:
        ax: Matplotlib axis to plot on
        dataloader: DataLoader containing trajectory data
        x_range: Tuple of (min, max) for x coordinate
        y_range: Tuple of (min, max) for y coordinate
        resolution: Resolution for 2D histogram bins
    """
    all_x_coords = []
    all_y_coords = []
    
    # Collect all x,y coordinates from the dataloader
    for batch in dataloader:
        # Get observations (states or obs) from batch
        if SampleKey.STATES in batch:
            states = batch[SampleKey.STATES]
        elif SampleKey.OBS in batch:
            states = batch[SampleKey.OBS]
        else:
            continue
        
        # Convert to numpy if needed
        if isinstance(states, torch.Tensor):
            states = to_numpy(states)
        
        # Handle different batch structures:
        # - Preference dataset: (batch_size, 2, T, state_dim)
        # - Demonstration dataset: (batch_size, T, state_dim)
        # - Flattened: (T, state_dim)
        
        # Flatten all dimensions except the last (state_dim)
        states_flat = states.reshape(-1, states.shape[-1])
        
        # Extract x (dim 0) and y (dim 1) coordinates
        all_x_coords.append(states_flat[:, 0])
        all_y_coords.append(states_flat[:, 1])
    
    if not all_x_coords:
        ax.text(0.5, 0.5, 'No state data in dataloader', 
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return
    
    # Concatenate all coordinates
    x_coords = np.concatenate(all_x_coords)
    y_coords = np.concatenate(all_y_coords)
    
    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(
        x_coords, y_coords,
        bins=resolution,
        range=[[x_range[0], x_range[1]], [y_range[0], y_range[1]]]
    )
    
    # Plot log(1 + count) for better visualization
    im = ax.imshow(
        np.log1p(hist.T),  # Transpose for correct orientation
        origin='lower',
        aspect='auto',
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        cmap='viridis'
    )
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Dataset Visitation (log scale)')
    
    # Add landing pad marker
    ax.plot(0, 0, 'r*', markersize=15, label='Landing Pad')
    ax.legend(loc='upper right')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log(1+count)')


def plot_multiple_trajectories_with_reward(
    ax, 
    fb_model, 
    device, 
    policy: ExpertPolicy, 
    env: gym.Env,
    x_range, 
    y_range,
    max_steps: int = 100,
    num_trajectories: int = 50
):
    """
    Plots multiple trajectories sampled from the policy, colored by predicted reward.
    
    Args:
        ax: Matplotlib axis to plot on
        fb_model: The multi-feedback model
        device: PyTorch device
        policy: Policy to sample trajectory from
        env: Environment to roll out in
        x_range: Tuple of (min, max) for x coordinate
        y_range: Tuple of (min, max) for y coordinate
        max_steps: Maximum number of steps in trajectory
        num_trajectories: Number of trajectories to sample
    """
    all_x_coords = []
    all_y_coords = []
    all_rewards = []
    
    # Get reward domain from feature module
    reward_domain = fb_model.encoder.features.reward_domain
    
    # Sample multiple trajectories
    for seed in range(num_trajectories):
        # Sample a trajectory using the policy
        trajectory = rollout(env, policy, num_steps=max_steps, seed=seed)
        
        # Unpack trajectory
        trajectory_data = unpack_trajectory(trajectory)
        
        # Extract states from trajectory
        states = trajectory_data[TrajKeys.OBS]  # Shape: (T, state_dim)
        
        # Extract x,y coordinates
        x_coords = states[:, 0]
        y_coords = states[:, 1]
        
        # Get predicted rewards for each state along the trajectory
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            if reward_domain == 's':
                # State-only reward
                mean_rewards, _ = fb_model.encoder(states_tensor, None, None)
            elif reward_domain == 'sa':
                # State-action reward - use actual actions from trajectory
                actions = trajectory_data[TrajKeys.ACTS]  # Shape: (T,)
                n_states = states_tensor.shape[0]
                # Create one-hot encoded actions (LunarLander has 4 discrete actions)
                actions_onehot = torch.zeros(n_states, 4, device=device)
                for i, action in enumerate(actions):
                    actions_onehot[i, action] = 1.0
                mean_rewards, _ = fb_model.encoder(states_tensor, actions_onehot, None)
            elif reward_domain == 'sas':
                # State-action-next_state reward
                actions = trajectory_data[TrajKeys.ACTS]  # Shape: (T,)
                next_states = trajectory_data[TrajKeys.NEXT_OBS]  # Shape: (T, state_dim)
                next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
                n_states = states_tensor.shape[0]
                # Create one-hot encoded actions
                actions_onehot = torch.zeros(n_states, 4, device=device)
                for i, action in enumerate(actions):
                    actions_onehot[i, action] = 1.0
                mean_rewards, _ = fb_model.encoder(states_tensor, actions_onehot, next_states_tensor)
            else:
                raise ValueError(f"Unknown reward_domain: {reward_domain}")
        
        predicted_rewards = to_numpy(mean_rewards.squeeze())
        
        # Draw trajectory line connecting the points
        ax.plot(x_coords, y_coords, 'k-', alpha=0.1, linewidth=0.5)
        
        # Mark start and end points
        if seed == 0:
            ax.plot(x_coords[0], y_coords[0], 'bo', markersize=6, label='Start', alpha=0.6, zorder=3)
            ax.plot(x_coords[-1], y_coords[-1], 'rs', markersize=6, label='End', alpha=0.6, zorder=3)
        else:
            ax.plot(x_coords[0], y_coords[0], 'bo', markersize=6, alpha=0.6, zorder=3)
            ax.plot(x_coords[-1], y_coords[-1], 'rs', markersize=6, alpha=0.6, zorder=3)
        
        # Collect all coordinates and rewards
        all_x_coords.extend(x_coords)
        all_y_coords.extend(y_coords)
        all_rewards.extend(predicted_rewards)
    
    # Create scatter plot with all points colored by predicted reward
    scatter = ax.scatter(
        all_x_coords, 
        all_y_coords, 
        c=all_rewards,
        cmap='RdYlGn',  # Same colormap as reward heatmaps
        s=20,
        alpha=0.6,
        edgecolors='none'
    )
    
    # Add landing pad marker
    ax.plot(0, 0, 'g*', markersize=20, label='Landing Pad', zorder=10)
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'{num_trajectories} Trajectories (colored by predicted reward)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label='Predicted Reward')


def plot_multiple_trajectories_with_qvalue(
    ax, 
    fb_model, 
    device, 
    policy: ExpertPolicy, 
    env: gym.Env,
    x_range, 
    y_range,
    max_steps: int = None,
    num_trajectories: int = 50
):
    """
    Plots multiple trajectories sampled from an expert policy derived from the learned Q-value model,
    colored by estimated Q-value.
    
    Args:
        ax: Matplotlib axis to plot on
        fb_model: The multi-feedback model (with Q_value_model)
        device: PyTorch device
        policy: Expert policy derived from fb_model.Q_value_model (LearnedQExpertPolicy)
        env: Environment to roll out in
        x_range: Tuple of (min, max) for x coordinate
        y_range: Tuple of (min, max) for y coordinate
        max_steps: Maximum number of steps in trajectory
        num_trajectories: Number of trajectories to sample
    """
    all_x_coords = []
    all_y_coords = []
    all_qvalues = []
    
    # Sample multiple trajectories
    for seed in range(num_trajectories):
        # Sample a trajectory using the expert policy derived from Q-value model
        trajectory = rollout(env, policy, num_steps=max_steps, seed=seed)
        
        # Unpack trajectory
        trajectory_data = unpack_trajectory(trajectory)
        
        # Extract states from trajectory
        states = trajectory_data[TrajKeys.OBS]  # Shape: (T, state_dim)
        
        # Extract x,y coordinates
        x_coords = states[:, 0]
        y_coords = states[:, 1]
        
        # Get estimated Q-values for each state along the trajectory
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            # Get Q-values from the learned Q-value model
            q_values = fb_model.Q_value_model(states_tensor)  # Shape: (T, num_actions)
            
            # Take max Q-value for each state (V(s) = max_a Q(s,a))
            max_q_values = torch.max(q_values, dim=1)[0]  # Shape: (T,)
        
        estimated_qvalues = to_numpy(max_q_values)
        
        # Draw trajectory line connecting the points
        ax.plot(x_coords, y_coords, 'k-', alpha=0.1, linewidth=0.5)
        
        # Mark start and end points
        if seed == 0:
            ax.plot(x_coords[0], y_coords[0], 'bo', markersize=6, label='Start', alpha=0.6, zorder=3)
            ax.plot(x_coords[-1], y_coords[-1], 'rs', markersize=6, label='End', alpha=0.6, zorder=3)
        else:
            ax.plot(x_coords[0], y_coords[0], 'bo', markersize=6, alpha=0.6, zorder=3)
            ax.plot(x_coords[-1], y_coords[-1], 'rs', markersize=6, alpha=0.6, zorder=3)
        
        # Collect all coordinates and Q-values
        all_x_coords.extend(x_coords)
        all_y_coords.extend(y_coords)
        all_qvalues.extend(estimated_qvalues)
    
    # Create scatter plot with all points colored by estimated Q-value
    scatter = ax.scatter(
        all_x_coords, 
        all_y_coords, 
        c=all_qvalues,
        cmap='viridis',  # Good colormap for Q-values
        s=20,
        alpha=0.6,
        edgecolors='none'
    )
    
    # Add landing pad marker
    ax.plot(0, 0, 'g*', markersize=20, label='Landing Pad', zorder=10)
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'{num_trajectories} Trajectories from Learned Q-Policy (colored by Q-value)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label='Estimated Q-value')

