from umfavi.types import FeedbackType
from umfavi.loglikelihoods.preference import PreferenceDecoder
from umfavi.loglikelihoods.demonstrations import DemonstrationsDecoder

def get_nll(fb_type: FeedbackType):
    if fb_type == FeedbackType.PREFERENCE:
        return PreferenceDecoder()
    elif fb_type == FeedbackType.DEMONSTRATION:
        return DemonstrationsDecoder()
    else:
        raise ValueError(f"Invalid feedback type: {fb_type.value}")