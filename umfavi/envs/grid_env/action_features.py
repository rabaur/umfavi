import numpy as np

def action_feature_factory(feature_type: str, n_actions: int):
    if feature_type == "one_hot":
        feats = np.eye(n_actions, dtype=np.float32)  # shape (n_actions, n_actions)
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")

    assert feats.shape[0] == n_actions, f"Action-feature matrix has {feats.shape[0]} instead of {n_actions=} rows"
    return feats


