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
    DONES = "dones"


class SampleKey(str, Enum):
    """Keys for dataset sample dictionaries returned by Dataset.__getitem__()."""
    FEEDBACK_TYPE = "feedback_type"
    STATES = "states"
    NEXT_STATES = "next_states"
    OBS = "obs"
    NEXT_OBS = "next_obs"
    ACTS = "actions"
    ACT_FEATS = "action_features"
    DONES = "dones"
    TARGETS = "targets"
    RATIONALITY = "rationality"
    GAMMA = "gamma"
    PREFERENCE = "preference"
    TD_ERROR_WEIGHT = "td_error_weight"


class FeedbackType(str, Enum):
    """Types of feedback for multi-feedback learning."""
    PREFERENCE = "preference"
    DEMONSTRATION = "demonstration"