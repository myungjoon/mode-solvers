"""
Calculate LP modes for GRIN fiber (memory-efficient version)
- Grid: 4096x4096 (calculation), downsample 16x to 256x256 (output)
- Window: 4*radius x 4*radius (100um x 100um)
- Fiber radius: 25 um
- Wavelength: 1030 nm
- n_clad: 1.4504, n_core: 1.4641
- Calculate first 121 modes, plot 11x11, save as numpy
- Memory optimization: downsample and free memory after each mode
"""

import gc
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from mmfsim.grid import Grid
from mmfsim.fiber import GrinFiber
from mmfsim.modes import GrinLPMode

# Parameters
radius = 25e-6  # 25 um
wavelength = 1030e-9  # 1030 nm
n_core = 1.4641  # n_max
n_clad = 1.4504  # n_min

# Grid parameters
pixel_numbers = (4096*2, 4096*2)
window_size = 4 * radius  # 100 um
pixel_size = window_size / pixel_numbers[0]  # ~24.41 nm

# Downsample parameters
downsample_factor = 16
output_size = pixel_numbers[0] // downsample_factor  # 256

print(f"Pixel size: {pixel_size * 1e9:.2f} nm")
print(f"Window size: {window_size * 1e6:.2f} um x {window_size * 1e6:.2f} um")
print(f"Calculation grid: {pixel_numbers[0]}x{pixel_numbers[1]}")
print(f"Output grid (16x downsampled): {output_size}x{output_size}")

# Create grid and fiber
grid = Grid(pixel_size=pixel_size, pixel_numbers=pixel_numbers)
fiber = GrinFiber(radius=radius, wavelength=wavelength, n1=n_core, n2=n_clad)

print(fiber)
print(f"\nV number: {fiber._V:.2f}")
print(f"Guided LP modes (non-degen): {fiber._N_modes}")
print(f"Guided LP modes (with degen): {fiber._N_modes_degen}")

# Calculate GRIN index profile
# n(r) = n_core * sqrt(1 - 2*delta*(r/a)^2) for r < a
# n(r) = n_clad for r >= a
# where delta = (n_core^2 - n_clad^2) / (2*n_core^2)
print("\nCalculating index profile...")
delta = (n_core**2 - n_clad**2) / (2 * n_core**2)
R = grid.R  # radial distance from center

index_profile = np.where(
    R <= radius,
    n_core * np.sqrt(1 - 2 * delta * (R / radius)**2),
    n_clad
)

print(f"Index profile shape: {index_profile.shape}")
print(f"Index range: {index_profile.min():.6f} to {index_profile.max():.6f}")

# Downsample and save index profile
index_profile_ds = zoom(index_profile, 1/downsample_factor, order=1)
np.save(f"grin_index_profile_{output_size}x{output_size}.npy", index_profile_ds)
print(f"Saved downsampled index profile to grin_index_profile_{output_size}x{output_size}.npy")
print(f"Downsampled index profile shape: {index_profile_ds.shape}")

# Free high-res index profile memory
del index_profile
del R
gc.collect()

# Plot index profile (downsampled)
# fig_idx, ax_idx = plt.subplots(figsize=(8, 6))
# extent = np.array([-window_size/2, window_size/2, -window_size/2, window_size/2]) * 1e6
# im = ax_idx.imshow(index_profile_ds, extent=extent, cmap='viridis')
# ax_idx.set_xlabel('x [μm]')
# ax_idx.set_ylabel('y [μm]')
# ax_idx.set_title('GRIN Fiber Index Profile')
# circle = plt.Circle((0, 0), radius * 1e6, fill=False, edgecolor='white', linestyle='--', linewidth=1)
# ax_idx.add_patch(circle)
# plt.colorbar(im, ax=ax_idx, label='Refractive Index')
# plt.tight_layout()
# plt.savefig(f"grin_index_profile_{output_size}x{output_size}.png", dpi=300)
# plt.show()

# Generate LP mode indices (n, m) sorted by mode group number h = 2n + m - 1
# This ensures modes are ordered by decreasing effective index
def generate_mode_indices(n_modes_needed):
    """Generate LP mode indices ordered by mode group number h = 2n + m - 1"""
    modes = []
    h = 0
    while len(modes) < n_modes_needed:
        # For each h value, find all valid (n, m) pairs where 2n + m - 1 = h
        # m >= 1, n >= 0
        for n in range(h + 1):
            m = h - 2 * n + 1
            if m >= 1:
                if n == 0:
                    # Centrosymmetric mode - single orientation
                    modes.append((n, m, 'a'))
                else:
                    # Non-centrosymmetric - two orientations
                    modes.append((n, m, 'a'))  # cos
                    modes.append((n, m, 'b'))  # sin
                if len(modes) >= n_modes_needed:
                    break
        h += 1
    return modes[:n_modes_needed]

# Generate exactly 121 modes
n_modes_requested = 50
mode_indices = generate_mode_indices(n_modes_requested)
print(f"\nComputing {len(mode_indices)} modes...")

# Compute each mode and downsample (memory-efficient: free memory after each mode)
mode_labels = []

# Pre-allocate output array to avoid memory fragmentation
modes_array = np.zeros((len(mode_indices), output_size, output_size), dtype=np.float64)

for idx, (n, m, orientation) in enumerate(mode_indices):
    mode = GrinLPMode(n, m)
    mode.compute(fiber, grid)

    if orientation == 'a':
        field = mode._fields[:, :, 0].copy()  # Copy to release reference to mode._fields
    else:
        field = mode._fields[:, :, 1].copy()

    # Explicitly delete mode to free memory
    del mode

    # Downsample the mode field directly into pre-allocated array
    modes_array[idx] = zoom(field, 1/downsample_factor, order=1)

    # Free the high-res field
    del field

    label = f"LP({n},{m})" if n == 0 else f"LP({n},{m}){orientation}"
    mode_labels.append(label)

    if (idx + 1) % 10 == 0 or idx == len(mode_indices) - 1:
        print(f"  {idx + 1} / {len(mode_indices)}")
        # Force garbage collection every 10 modes
        gc.collect()

print(f"\nTotal modes computed: {len(mode_labels)}")
print(f"Modes array shape: {modes_array.shape}")

# Save as numpy file
output_file = f"grin_modes_{output_size}x{output_size}_{len(mode_labels)}modes.npy"
np.save(output_file, modes_array)
print(f"\nSaved modes to {output_file}")

# Plot 11x11 grid
print("\nPlotting 11x11 mode grid...")
fig, axes = plt.subplots(11, 11, figsize=(16, 10))
extent = np.array([-window_size/2, window_size/2, -window_size/2, window_size/2]) * 1e6  # in um

# for idx in range(n_modes_requested):
#     row = idx // 11
#     col = idx % 11
#     ax = axes[row, col]

#     mode_field = modes_array[idx]
#     vmax = np.max(np.abs(mode_field))

#     ax.imshow(mode_field, extent=extent, cmap='bwr', vmin=-vmax, vmax=vmax)
#     ax.set_title(mode_labels[idx], fontsize=5)
#     ax.set_xticks([])
#     ax.set_yticks([])

#     # Draw fiber core boundary
#     circle = plt.Circle((0, 0), radius * 1e6, fill=False, edgecolor='k', linestyle='--', linewidth=0.3)
#     ax.add_patch(circle)

# plt.tight_layout()
# plt.savefig(f"grin_modes_{output_size}x{output_size}_{len(mode_labels)}modes.png", dpi=300)
# plt.show()

print("\nDone!")
