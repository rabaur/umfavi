import gymnasium as gym
from umfavi.envs.grid_env.env import GridEnv
from umfavi.utils.gym import is_registered_gym_env


def get_env(**kwargs) -> gym.Env:
    env_name = kwargs["env_name"]
    if is_registered_gym_env(env_name):
        return gym.make(env_name)
    elif env_name.startswith("grid"):
        rew_type = env_name.split("_")[1]
        kwargs["reward_type"] = rew_type
        return GridEnv(**kwargs)
    else:
        raise NotImplementedError(f"Uknown environment {env_name}")
