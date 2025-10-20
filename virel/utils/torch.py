import torch

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda" # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return "mps" # Apple GPU
    else:
        return "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available