import numpy as np

# Load the high-resolution data
n = np.load('n_4096_GRIN_1030.npy')
print(f"Original shape: {n.shape}")

print(n.min(), n.max())

# Downsample 16x in each dimension
n_downsampled = n[::16, ::16]
print(f"Downsampled shape: {n_downsampled.shape}")

# Save the downsampled data
np.save('n_256_GRIN_1030.npy', n_downsampled)
print("Saved n_256_GRIN_1030.npy")
