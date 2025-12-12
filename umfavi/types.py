from typing import Any
from enum import Enum

# Type aliases
ObsType = Any
ActType = Any
TrajectoryType = list[tuple[ObsType, ActType, float, ObsType, bool, dict[str, Any]]]


# String Enums for consistent key naming
class TrajKeys(str, Enum):
    """Keys for unpacked trajectory dictionaries returned by unpack_trajectory()."""
    OBS = "observations"
    ACTS = "actions"
    REWS = "rewards"
    NEXT_OBS = "next_observations"
    TERMINATED = "terminated"
    TRUNCATED = "truncated"
    INVALID = "invalid"


class SampleKey(str, Enum):
    """Keys for dataset sample dictionaries returned by Dataset.__getitem__()."""
    FEEDBACK_TYPE = "feedback_type"

    # States
    STATES = "states"
    NEXT_STATES = "next_states"

    # Observations (state-features).
    # Equal to states if state == observation
    OBS = "observations"
    NEXT_OBS = "next_observations"

    # Actions
    ACTS = "actions"
    NEXT_ACTS = "next_actions"

    # Action features (action features are to actions like observations to states)
    # Equal to acts if actions == action features
    ACT_FEATS = "action_features"
    NEXT_ACT_FEATS = "next_action_features"
    INVALID = "invalid"
    TERMINATED = "terminated"
    TRUNCATED = "truncated"
    TARGETS = "targets"
    RATIONALITY = "rationality"
    GAMMA = "gamma"
    PREFERENCE = "preference"
    TD_ERROR_WEIGHT = "td_error_weight"


class FeedbackType(str, Enum):
    """Types of feedback for multi-feedback learning."""
    PREFERENCE = "preference"
    DEMONSTRATION = "demonstration"