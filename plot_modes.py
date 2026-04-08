import numpy as np
import matplotlib.pyplot as plt

# Load the modes data
modes = np.load('modes_256x256_GRIN_1030.npy')
print(f"Loaded modes shape: {modes.shape}")

# Create 11x11 subplots
fig, axes = plt.subplots(11, 11, figsize=(16, 10))

# Plot each mode
for i in range(121):
    row = i // 11
    col = i % 11
    ax = axes[row, col]

    # Plot the mode intensity (absolute value squared for complex modes)
    if np.iscomplexobj(modes):
        im = ax.imshow(np.abs(modes[i])**2, cmap='turbo')
    else:
        im = ax.imshow(modes[i], cmap='hot')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{i}', fontsize=6)

plt.tight_layout()
plt.savefig('modes_plot.png', dpi=200)
plt.show()
