import numpy as np
from numpy.typing import NDArray

def Rsa_to_Rsas(Rsa: NDArray) -> NDArray:
    """
    Converts a reward function R(s,a) to a reward function R(s,a,s').
    """
    R_sa = np.asarray(Rsa)
    S, A = R_sa.shape
    R_sas = np.broadcast_to(R_sa[..., None], (S, A, S))
    return R_sas