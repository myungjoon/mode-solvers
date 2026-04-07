"""
Test script for the semivectorial mode solver (svmodes).
Tests with a 4096x4096 grid using a GRIN fiber geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.insert(0, './modesolver')
from svmodes import svmodes


def test_GRIN_fiber():
    """
    Test the mode solver with a GRIN fiber using loaded index profile.
    """
    print("=" * 60)
    print("GRIN Fiber Mode Solver Test")
    print("=" * 60)

    # Load index profile from file
    print("\nLoading GRIN fiber index profile...")
    n_profile = np.load('./fibers/GRIN_1030_120modes/n_2048_GRIN_1030.npy')

    # Grid parameters (from loaded profile)
    nx, ny = n_profile.shape

    # Fiber parameters
    wavelength = 1.03  # 1030 nm in microns

    # Computational domain (must match the index profile generation)
    core_radius = 25.0  # microns
    domain_size = 4 * core_radius  # 100 microns
    x_range = (-domain_size/2, domain_size/2)
    y_range = (-domain_size/2, domain_size/2)

    # Grid spacing
    dx = domain_size / nx
    dy = domain_size / ny

    # Calculate permittivity from refractive index
    eps = n_profile**2

    # Create coordinate arrays
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)

    # Get index range for display
    n_core = n_profile.max()
    n_clad = n_profile.min()

    print(f"\nGrid size: {nx} x {ny}")
    print(f"Domain size: {domain_size} x {domain_size} microns")
    print(f"Grid spacing: dx = {dx:.4f} um, dy = {dy:.4f} um")
    print(f"\nFiber parameters:")
    print(f"  Wavelength: {wavelength} um (1030 nm)")
    print(f"  Core radius: {core_radius} um")
    print(f"  n_max (core): {n_core:.6f}")
    print(f"  n_min (clad): {n_clad:.6f}")
    print(f"  Index profile loaded from: ./fibers/GRIN_1030_120modes/n_4096_GRIN_1030.npy")

    # Plot index profile
    print("\nPlotting index profile...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(n_profile, extent=[x[0], x[-1], y[0], y[-1]],
                   cmap='turbo', origin='lower')
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_title(f'Refractive Index Profile ({nx}x{ny})')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='n')
    plt.tight_layout()
    plt.savefig('./fibers/GRIN_1030_120modes/index_profile.png', dpi=150)
    print(f"  Saved: ./fibers/GRIN_1030_120modes/index_profile.png")
    plt.show()

    # Mode solver parameters
    guess = (n_core + n_clad) / 2 + 1e-6  # Initial guess for effective index
    nmodes = 121  # Number of modes to find
    boundary = '0000'  # Zero boundary conditions (absorbing)

    # Use scalar field for LP modes (valid for weakly guiding GRIN fibers)
    field = 'scalar'
    print(f"\n{'-'*60}")
    print(f"Solving for LP modes (scalar)...")
    print(f"{'-'*60}")

    start_time = time.time()
    try:
        phi, neff = svmodes(wavelength, guess, nmodes, dx, dy, eps, boundary, field)
        elapsed = time.time() - start_time

        print(f"Mode solving time: {elapsed:.2f} seconds ({elapsed/nmodes:.2f} sec/mode)")
        print(f"\nEffective indices (neff):")
        for i, n_eff in enumerate(neff):
            if np.iscomplex(n_eff):
                print(f"  Mode {i+1}: {n_eff.real:.6f} + {n_eff.imag:.2e}j")
            else:
                print(f"  Mode {i+1}: {n_eff:.6f}")

        # Verify results
        print(f"\nVerification:")
        print(f"  All neff between n_clad and n_core: ", end="")
        neff_real = np.real(neff)
        if np.all((neff_real >= n_clad - 0.001) & (neff_real <= n_core + 0.001)):
            print("PASS")
        else:
            print("WARNING - some neff out of expected range")

        # Plot first 6 modes (2x3 grid)
        nplot = min(6, nmodes)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'LP Modes - GRIN Fiber ({nx}x{ny} grid, showing {nplot}/{nmodes})', fontsize=14)

        for i in range(nplot):
            ax = axes[i // 3, i % 3]
            mode_field = np.real(phi[:, :, i])

            # Plot mode intensity
            im = ax.imshow(np.abs(mode_field)**2,
                          extent=[x[0], x[-1], y[0], y[-1]],
                          cmap='turbo', origin='lower')

            ax.set_xlabel('x (μm)')
            ax.set_ylabel('y (μm)')
            ax.set_title(f'LP Mode {i+1}: neff = {np.real(neff[i]):.6f}')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax, label='|E|²')

        plt.tight_layout()
        plt.savefig('./fibers/GRIN_1030_120modes/mode_solver_test_LP.png', dpi=150)
        print(f"  Plot saved: ./fibers/GRIN_1030_120modes/mode_solver_test_LP.png")

        # Reshape modes to (M, Nx, Ny) format
        phi = np.transpose(phi, (2, 0, 1))  # (Nx, Ny, M) -> (M, Nx, Ny)

        # Downsample by 16x in spatial dimensions
        ds = 16
        phi_ds = phi[:, ::ds, ::ds]
        n_profile_ds = n_profile[::ds, ::ds]

        # Save downsampled modes and index profile
        save_dir = './fibers/GRIN_1030_120modes'
        nx_ds, ny_ds = n_profile_ds.shape

        modes_path = f'{save_dir}/modes_{nx_ds}x{ny_ds}_GRIN_1030_py.npy'
        np.save(modes_path, phi_ds)
        print(f"\nModes saved to: {modes_path}")
        print(f"  Shape: {phi_ds.shape} (downsampled {ds}x from {phi.shape})")

        # n_path = f'{save_dir}/n_{nx_ds}_GRIN_1030.npy'
        # np.save(n_path, n_profile_ds)
        # print(f"Index profile saved to: {n_path}")
        # print(f"  Shape: {n_profile_ds.shape} (downsampled {ds}x from {n_profile.shape})")
        # print(f"  Reference modes shape: {phi_ref.shape}")

        # # Compute overlap (correlation) for each mode
        # print(f"\nMode comparison:")
        # print(f"  Overlap = |<py, ref>| / (||py|| * ||ref||), where 1.0 = perfect match")
        # print(f"  {'-'*50}")
        # for i in range(nmodes):
        #     mode_py = phi[i, :, :].flatten()
        #     mode_ref = phi_ref[i, :, :].flatten()
        #
        #     # Compute overlap (accounts for sign flip)
        #     overlap = np.abs(np.vdot(mode_py, mode_ref)) / (np.linalg.norm(mode_py) * np.linalg.norm(mode_ref))
        #
        #     print(f"  Mode {i+1}: overlap = {overlap:.6f} ({overlap*100:.4f}%)")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"ERROR after {elapsed:.2f} seconds: {e}")
        import traceback
        traceback.print_exc()

    plt.show()



if __name__ == '__main__':
    print("Mode Solver Test Suite")
    print("=" * 60)
    print("Testing svmodes with GRIN fiber (1030 nm)\n")

    # Run tests
    test_GRIN_fiber()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
