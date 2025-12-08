import numpy as np
import matplotlib.pyplot as plt

# ============ CONFIGURATION ============
CURVE_TYPE = "hilbert"  # Options: "hilbert", "morton"
resolution = 16
# =======================================

def morton_encode_2d(coords):
    """
    Encode 2D coordinates to Morton codes (Z-order curve).
    coords: array of shape (n, 2) with integer coordinates
    Returns: array of shape (n,) with Morton codes
    """
    x = coords[:, 0].astype(np.uint64)
    y = coords[:, 1].astype(np.uint64)
    
    # Interleave bits of x and y
    morton = np.zeros(len(x), dtype=np.uint64)
    for i in range(32):  # Support up to 32 bits per coordinate
        morton |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
    
    return morton

def hilbert_encode_2d(coords, resolution):
    """
    Encode 2D coordinates using Hilbert curve.
    coords: array of shape (n, 2) with integer coordinates
    resolution: the size of the grid (must be power of 2)
    Returns: array of indices sorted by Hilbert curve order
    """
    from hilbert import encode
    num_bits = int(np.log2(resolution))
    hilbert_codes = encode(coords, 2, num_bits)
    return hilbert_codes

def get_curve_ordering(coords, curve_type, resolution):
    """
    Get the ordering of coordinates according to specified space-filling curve.
    Returns indices that sort the coordinates by curve order.
    """
    if curve_type == "hilbert":
        return hilbert_encode_2d(coords, resolution)
    elif curve_type == "morton":
        morton_codes = morton_encode_2d(coords)
        return np.argsort(morton_codes)
    else:
        raise ValueError(f"Unknown curve type: {curve_type}. Use 'hilbert' or 'morton'.")

# Generate RGB cube
r, g, b = np.meshgrid(np.linspace(0, 1, resolution), np.linspace(0, 1, resolution), np.linspace(0, 1, resolution))
rgb_cube = np.stack([r, g, b], axis=-1)

# Generate 2D coordinate grid
ii, jj = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing='ij')
idxs = np.stack([ii, jj], axis=-1, dtype=np.int64)
idxs_flat = np.reshape(idxs, (-1, 2))

# Get ordering using selected curve
idxs_flat_encoded = get_curve_ordering(idxs_flat, CURVE_TYPE, resolution)
idxs_shuffled = idxs_flat[idxs_flat_encoded]

# Unfold RGB cube according to curve ordering
rgb_cube_unfolded_rg = rgb_cube[idxs_shuffled[:, 0], idxs_shuffled[:, 1], :]
rgb_cube_unfolded_rb = rgb_cube[idxs_shuffled[:, 0], :, idxs_shuffled[:, 1]]
rgb_cube_unfolded_gb = rgb_cube[:, idxs_shuffled[:, 0], idxs_shuffled[:, 1]]

# Plot results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(rgb_cube_unfolded_rg, aspect='auto')
axs[1].imshow(rgb_cube_unfolded_rb, aspect='auto')
axs[2].imshow(rgb_cube_unfolded_gb, aspect='auto')
axs[0].set_title(f'RG ({CURVE_TYPE.capitalize()})')
axs[1].set_title(f'RB ({CURVE_TYPE.capitalize()})')
axs[2].set_title(f'GB ({CURVE_TYPE.capitalize()})')
plt.tight_layout()
plt.show()