from umfavi.visualization.unfold_tensor import unfold_tensor
import numpy as np
import matplotlib.pyplot as plt
from hilbert import encode

# Generate RGB cube
resolution = 16
r, g, b = np.meshgrid(np.linspace(0, 1, resolution), np.linspace(0, 1, resolution), np.linspace(0, 1, resolution))
rgb_cube = np.stack([r, g, b], axis=-1)

unfolded_rg_cube = unfold_tensor(rgb_cube, "morton", [0, 1], 2)
unfolded_rb_cube = unfold_tensor(rgb_cube, "morton", [0, 2], 1)
unfolded_gb_cube = unfold_tensor(rgb_cube, "morton", [1, 2], 0)

unfolded_rg_cube_hilbert = unfold_tensor(rgb_cube, "hilbert", [0, 1], 2)
unfolded_rb_cube_hilbert = unfold_tensor(rgb_cube, "hilbert", [0, 2], 1)
unfolded_gb_cube_hilbert = unfold_tensor(rgb_cube, "hilbert", [1, 2], 0)
# Plot results
fig, axs = plt.subplots(2, 3, figsize=(15, 5))
axs[0, 0].imshow(unfolded_rg_cube, aspect='auto')
axs[0, 1].imshow(unfolded_rb_cube, aspect='auto')
axs[0, 2].imshow(unfolded_gb_cube, aspect='auto')
axs[1, 0].imshow(unfolded_rg_cube_hilbert, aspect='auto')
axs[1, 1].imshow(unfolded_rb_cube_hilbert, aspect='auto')
axs[1, 2].imshow(unfolded_gb_cube_hilbert, aspect='auto')
axs[0, 0].set_title('RG (Morton)')
axs[0, 1].set_title('RB (Morton)')
axs[0, 2].set_title('GB (Morton)')
axs[1, 0].set_title('RG (Hilbert)')
axs[1, 1].set_title('RB (Hilbert)')
axs[1, 2].set_title('GB (Hilbert)')
plt.tight_layout()
plt.show()