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
    fb_model: MultiFeedbackTypeModel
):
    if isinstance(env, GridEnv):
        fig = vis_grid_env(
            env,
            fb_model
        )
    elif get_env_name(env) == "LunarLander-v3":
        fig = vis_lunarlander(
            env=env,
            fb_model=fb_model,
        )
    else: 
        raise NotImplementedError(f"Visualization for environment {get_env_name(env)} not implemented")
    return fig