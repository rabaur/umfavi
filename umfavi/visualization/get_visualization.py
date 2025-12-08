import gymnasium as gym
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from umfavi.envs.grid_env.env import GridEnv
from umfavi.multi_fb_model import MultiFeedbackTypeModel
from umfavi.utils.policies import ExpertPolicy
from umfavi.visualization.cartpole_visualizer import vis_cartpole
from umfavi.visualization.grid_visualizer import vis_grid_env
from umfavi.visualization.lunarlander_visualizer import vis_lunarlander
from umfavi.utils.gym import get_env_name

def get_visualization(
    env: gym.Env,
    fb_model: MultiFeedbackTypeModel,
    dataloader: DataLoader,
    all_obs_features: NDArray,
    all_state_features: NDArray,
    est_expert_policy: ExpertPolicy
):
    if isinstance(env, GridEnv):
        fig = vis_grid_env(env, fb_model, all_obs_features, all_state_features)
    elif get_env_name(env) == "LunarLander-v3":
        fig = vis_lunarlander(
            fb_model,
            dataloader=dataloader,
            resolution=30,
            num_samples=5,
            est_expert_policy=est_expert_policy,
            env=env,
            num_trajectories=50,
            max_traj_steps=100
        )
    elif get_env_name(env) == "CartPole-v1":
        fig = vis_cartpole(
            fb_model,

        )