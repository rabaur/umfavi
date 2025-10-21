import torch
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from typing import Callable
import gymnasium as gym
from virel.utils.gym import rollout, get_obs_act_pairs, get_rewards
from virel.utils.math import sigmoid

class DemonstrationDataset(Dataset):
    pass