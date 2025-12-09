import torch
from numpy.typing import NDArray

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda" # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return "mps" # Apple GPU
    else:
        return "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

def to_numpy(t: torch.Tensor) -> NDArray:
    return t.detach().cpu().numpy()

def get_model_device(model: torch.nn.Module):
    return next(iter(model.parameters())).device