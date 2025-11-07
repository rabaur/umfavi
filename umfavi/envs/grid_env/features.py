import numpy as np

def dct_features(grid_size: int, n_dct_basis_fns: int) -> np.ndarray:
    """
    Creates 2D DCT-II features on an N×N grid over [0,1]^2
    using the first n_dct_basis_fns in each dimension.
    """
    N = grid_size
    K = n_dct_basis_fns

    # grid coordinates
    x = np.arange(N)
    y = np.arange(N)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    # DCT-II normalization factors
    def alpha(p):
        return np.where(p == 0, np.sqrt(1.0 / N), np.sqrt(2.0 / N))

    # compute basis functions
    feats = []
    for u in range(K):
        au = alpha(u)
        cos_u = np.cos(np.pi * (2 * xv + 1) * u / (2 * N))
        for v in range(K):
            av = alpha(v)
            cos_v = np.cos(np.pi * (2 * yv + 1) * v / (2 * N))
            feats.append(au * av * cos_u * cos_v)

    # stack into (N, N, K*K)
    return np.stack(feats, axis=-1)

def coordinate_features(grid_size: int):
    """
    Creates coordinate features on an N×N grid scaled within [0,1]^2
    """
    N = grid_size

    # grid coordinates
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    return np.stack([xv, yv], axis=-1)