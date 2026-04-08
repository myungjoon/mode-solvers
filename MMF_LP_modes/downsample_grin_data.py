"""
Downsample GRIN fiber data by 16x in each spatial dimension
- Input: 2048x2048 -> Output: 128x128
"""

import numpy as np
from scipy.ndimage import zoom

# Downsample factor
downsample_factor = 8
output_size = 2048 // downsample_factor  # 128

print(f"Downsampling from 2048x2048 to {output_size}x{output_size}")

# Load index profile
print("\nLoading index profile...")
index_profile = np.load("grin_index_profile_2048x2048.npy")
print(f"Original shape: {index_profile.shape}")

# Downsample index profile using zoom (anti-aliased)
index_profile_ds = zoom(index_profile, 1/downsample_factor, order=1)
print(f"Downsampled shape: {index_profile_ds.shape}")

# Save downsampled index profile
np.save(f"grin_index_profile_{output_size}x{output_size}.npy", index_profile_ds)
print(f"Saved to grin_index_profile_{output_size}x{output_size}.npy")

# Load modes
print("\nLoading modes...")
modes = np.load("grin_modes_2048x2048_121modes.npy")
print(f"Original shape: {modes.shape}")

# Downsample each mode
n_modes = modes.shape[0]
modes_ds = np.zeros((n_modes, output_size, output_size), dtype=modes.dtype)

for i in range(n_modes):
    modes_ds[i] = zoom(modes[i], 1/downsample_factor, order=1)
    if (i + 1) % 20 == 0 or i == n_modes - 1:
        print(f"  Downsampled {i + 1} / {n_modes} modes")

print(f"Downsampled modes shape: {modes_ds.shape}")

# Save downsampled modes
np.save(f"grin_modes_{output_size}x{output_size}_121modes.npy", modes_ds)
print(f"Saved to grin_modes_{output_size}x{output_size}_121modes.npy")

print("\nDone!")
