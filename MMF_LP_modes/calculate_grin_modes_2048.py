"""
Calculate LP modes for GRIN fiber
- Grid: 2048x2048
- Window: 4*radius x 4*radius (100um x 100um)
- Fiber radius: 25 um
- Wavelength: 1030 nm
- n_clad: 1.4504, n_core: 1.4641
- Calculate first 121 modes, plot 11x11, save as numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from mmfsim.grid import Grid
from mmfsim.fiber import GrinFiber

# Parameters
radius = 25e-6  # 25 um
wavelength = 1030e-9  # 1030 nm
n_core = 1.4641  # n_max
n_clad = 1.4504  # n_min

# Grid parameters
pixel_numbers = (2048, 2048)
window_size = 4 * radius  # 100 um
pixel_size = window_size / pixel_numbers[0]  # ~48.83 nm

print(f"Pixel size: {pixel_size * 1e9:.2f} nm")
print(f"Window size: {window_size * 1e6:.2f} um x {window_size * 1e6:.2f} um")

# Create grid and fiber
grid = Grid(pixel_size=pixel_size, pixel_numbers=pixel_numbers)
fiber = GrinFiber(radius=radius, wavelength=wavelength, n1=n_core, n2=n_clad)

print(fiber)

# Check how many modes are supported
print(f"\nNumber of LP modes (non-degenerate): {fiber._N_modes}")
print(f"Number of LP modes (counting degenerates): {fiber._N_modes_degen}")

# Compute mode fields
print("\nComputing mode fields...")
fiber.compute_modes_fields(grid, verbose=True)

# fiber._modes has shape (2048, 2048, 2, N_modes)
# The third dimension (2) contains the two degenerate orientations
# For n=0 modes: both orientations are identical
# For n>0 modes: cos and sin orientations

# Extract 121 modes (including degenerate orientations)
# We'll take each LP mode and include both orientations for degenerate modes
modes_list = []
mode_labels = []

for i in range(fiber._N_modes):
    n = int(fiber._neff_hnm[i, 2])
    m = int(fiber._neff_hnm[i, 3])

    if n == 0:
        # Centrosymmetric mode - only one orientation
        modes_list.append(fiber._modes[:, :, 0, i])
        mode_labels.append(f"LP({n},{m})")
    else:
        # Non-centrosymmetric mode - two orientations (cos and sin)
        modes_list.append(fiber._modes[:, :, 0, i])  # cos orientation
        mode_labels.append(f"LP({n},{m})a")
        modes_list.append(fiber._modes[:, :, 1, i])  # sin orientation
        mode_labels.append(f"LP({n},{m})b")

    if len(modes_list) >= 121:
        break

n_modes_requested = 121
n_modes_available = len(modes_list)

if n_modes_available < n_modes_requested:
    print(f"\nWarning: Only {n_modes_available} modes available (requested {n_modes_requested})")
    print("Proceeding with available modes...")
else:
    modes_list = modes_list[:n_modes_requested]
    mode_labels = mode_labels[:n_modes_requested]

n_modes = len(modes_list)
print(f"\nTotal modes extracted: {n_modes}")

# Stack into array (N_modes, 2048, 2048)
modes_array = np.stack(modes_list, axis=0)
print(f"Modes array shape: {modes_array.shape}")

# Save as numpy file
output_file = f"grin_modes_2048x2048_{n_modes}modes.npy"
np.save(output_file, modes_array)
print(f"\nSaved modes to {output_file}")

# Determine grid size for plotting
n_cols = int(np.ceil(np.sqrt(n_modes)))
n_rows = int(np.ceil(n_modes / n_cols))


print(f"\nPlotting {n_modes} modes in {n_rows}x{n_cols} grid...")

fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
extent = np.array([-window_size/2, window_size/2, -window_size/2, window_size/2]) * 1e6  # in um

for idx in range(n_rows * n_cols):
    row = idx // n_cols
    col = idx % n_cols
    ax = axes[row, col]

    if idx < n_modes:
        mode_field = modes_array[idx]
        vmax = np.max(np.abs(mode_field))

        ax.imshow(mode_field, extent=extent, cmap='bwr', vmin=-vmax, vmax=vmax)
        ax.set_title(mode_labels[idx], fontsize=6)

        # Draw fiber core boundary
        circle = plt.Circle((0, 0), radius * 1e6, fill=False, edgecolor='k', linestyle='--', linewidth=0.5)
        ax.add_patch(circle)
    else:
        ax.axis('off')

    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig(f"grin_modes_2048x2048_{n_modes}modes.png", dpi=150)
plt.show()

print("\nDone!")
