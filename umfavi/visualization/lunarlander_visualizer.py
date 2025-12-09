import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from umfavi.multi_fb_model import MultiFeedbackTypeModel
from umfavi.utils.torch import get_model_device, to_numpy


def vis_lunarlander(
    env: gym.Env,
    fb_model: MultiFeedbackTypeModel,
    resolution: int = 64,
    batch_size: int = 1024
):
    obs_space = env.observation_space
    x_range = np.linspace(obs_space.low[0], obs_space.high[0], resolution)
    y_range = np.linspace(obs_space.low[1], obs_space.high[1], resolution)
    xs, ys = np.meshgrid(x_range, y_range)
    xys = np.stack([xs, ys], axis=-1)
    xys_flat = np.reshape(xys, (-1, 2))
    num_data = xys_flat.shape[0]
    other_feats = np.full((num_data, 4), 0.0)
    all_feats = np.concatenate([xys_flat, other_feats], axis=1)
    model_device = get_model_device(fb_model)
    all_feats_torch = torch.tensor(all_feats, device=model_device, dtype=torch.float32)

    # Predict Q-values
    est_q_vals = np.empty((num_data, 4))
    for i in range(start=0, stop=num_data, step=batch_size):
        batch = all_feats_torch[i:i+batch_size]
        q_vals_batch = to_numpy(fb_model.Q_value_model(batch))
        est_q_vals[i:i+batch_size] = q_vals_batch
    
    # Reshape Q-values
    est_q_vals_resh = np.reshape(est_q_vals, (resolution, resolution, 4))

    # Create 4 plots, one per action
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0, 0].imshow(est_q_vals_resh[..., 0])
    axs[1, 0].imshow(est_q_vals_resh[..., 1])
    axs[0, 1].imshow(est_q_vals_resh[..., 2])
    axs[1, 1].imshow(est_q_vals_resh[..., 3])

    return fig



    

