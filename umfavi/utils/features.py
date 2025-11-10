import numpy as np

def one_hot_features(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.float32)

def discrete_features(n: int) -> np.ndarray:
    """
    When features are learned via embedding, we simply pass the state index as the feature.
    """
    return np.arange(n, dtype=np.int32)