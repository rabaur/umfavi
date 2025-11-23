import torch
import gymnasium as gym
from typing import Callable, Optional
from umfavi.encoder.reward_encoder import BaseRewardEncoder
from umfavi.types import ActType, ObsType

class LearnedRewardWrapper(gym.Wrapper):
    
    def __init__(
        self,
        env,
        reward_encoder: BaseRewardEncoder,
        act_transform: Optional[Callable[ActType, ActType]] = None,
        obs_transform: Optional[Callable[ObsType, ObsType]] = None
    ):
        """
        Args:
            env: Base-environment to be wrapped.
            reward_encoder: The reward encoder to replace ground-truth reward.
            act_transform: Optional transform applied to actions before supplying them to reward_encoder.
            obs_transform: Optional transform applied to observations before supplying them to reward_encoder.
        """
        super().__init__(env)
        self.reward_encoder = reward_encoder
        self._last_obs = None
        self.act_transform = act_transform
        self.obs_transform = obs_transform
        self.device = next(iter(reward_encoder.parameters())).get_device()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info
    
    def _transform_action(self, act: ActType):
        if self.act_transform:
            act = self.act_transform(act)
        return torch.tensor(act).to(device=self.device)
    
    def _transform_obs(self, obs: ObsType):
        if self.obs_transform:
            obs = self.obs_transform(obs)
        return torch.tensor(obs).to(device=self.device)

    def step(self, action):
        # Step the real env
        next_obs, true_r, terminated, truncated, info = self.env.step(action)

        last_obs_tensor = self._transform_obs(self._last_obs)
        action_tensor = self._transform_action(action)
        next_obs_tensor = self._transform_obs(next_obs)
        
        # Compute learned reward
        learned_r = self.reward_encoder.predict_and_sample(last_obs_tensor, action_tensor, next_obs_tensor).item()

        # Store for next call
        self._last_obs = next_obs

        # Keep true reward for analysis
        info["true_reward"] = true_r
        info["learned_reward"] = learned_r

        # Return learned reward to the agent
        return next_obs, learned_r, terminated, truncated, info