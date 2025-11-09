from umfavi.envs.grid_env.state_features import discrete_features
from umfavi.utils.features import one_hot_features


def action_feature_factory(feature_type: str, n_actions: int):
    """
    Creates action features.

    Returns:
        (n_actions, action_feature_dim) tensor where row i corresponds to features of action i.
    """
    if feature_type == "one_hot":
        feats = one_hot_features(n_actions)
    elif feature_type == "embedding":
        feats = discrete_features(n_actions)
    assert feats.shape[0] == n_actions, f"Action-feature matrix has {feats.shape[0]} instead of {n_actions=} rows"
    return feats

