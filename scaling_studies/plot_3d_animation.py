"""
plot_3d_animation.py — 3D Spiral Wave Visualisation with Energy Audit
======================================================================

Paper reference
---------------
  Supplementary material for Figure 2 (3D ventricular wedge scaling).

Purpose
-------
  Reads the VTK output files from run_3d_experiment_spiral.py and
  creates an animated visualisation of the 3D spiral wave across three
  orthogonal cross-sections, showing:
    - Voltage u(x,y,z)
    - Metabolic stiffness k_eff(x,y,z)
    - Diffusive dissipation power P_diff(x,y,z)

Usage
-----
  python scaling_studies/plot_3d_animation.py

  Requires ffmpeg for MP4 output.

Input
-----
  threeD_spiral_audit/audit_3d_*.vtk

Output
------
  threeD_spiral_audit.mp4 — animated visualisation

Dependencies
------------
  numpy, matplotlib, ffmpeg (system)

Author
------
  Jagir R. Hussan
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# =========================================================================
# 0. CONFIGURATION
# =========================================================================

DATA_DIR    = "threeD_spiral_audit"
FILE_PATTERN = "audit_3d_*.vtk"

# Slice positions (fractional: 0.5 = middle of the volume)
SLICE_X_RATIO = 0.5
SLICE_Y_RATIO = 0.5
SLICE_Z_RATIO = 0.5

FPS = 10

# Voltage visualisation
V_MIN, V_MAX = 0.0, 1.0
CMAP_V = "inferno"

# Metabolic stiffness visualisation
K_MIN, K_MAX = 8.0, 20.0
CMAP_K = "viridis"

# Diffusive dissipation visualisation
P_MIN, P_MAX = 0.0, 2.0
CMAP_P = "magma"


# =========================================================================
# 1. VTK PARSING
# =========================================================================

def parse_vtk_dimensions(filepath):
    """Read the DIMENSIONS header from an ASCII VTK file."""
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("DIMENSIONS"):
                parts = line.split()
                return (int(parts[1]), int(parts[2]), int(parts[3]))
    return 0, 0, 0


def load_vtk_multifield(filepath, nx, ny, nz):
    """
    Parse multi-scalar ASCII VTK STRUCTURED_POINTS.

    Returns a dict mapping field names to 3D numpy arrays (Fortran order).
    """
    data_dict = {}

    with open(filepath, 'r') as f:
        content = f.read().split()
        iterator = iter(content)

        try:
            while True:
                token = next(iterator)

                if token == "SCALARS":
                    field_name = next(iterator)
                    _ = next(iterator)  # data type
                    _ = next(iterator)  # num components

                    # Skip LOOKUP_TABLE default
                    next_token = next(iterator)
                    if next_token == "LOOKUP_TABLE":
                        _ = next(iterator)  # "default"

                    # Read n_points floats
                    n_points = nx * ny * nz
                    values = []
                    for _ in range(n_points):
                        values.append(float(next(iterator)))

                    arr = np.array(values).reshape((nx, ny, nz), order='F')
                    data_dict[field_name] = arr

        except StopIteration:
            pass

    return data_dict


def get_sorted_files():
    """Return VTK files sorted by step number."""
    files = glob.glob(os.path.join(DATA_DIR, FILE_PATTERN))
    if not files:
        # Fallback to legacy naming
        files = glob.glob(os.path.join(DATA_DIR, "wedge_*.vtk"))

    def extract_step(fname):
        match = re.search(r"_(\d+)\.vtk", fname)
        return int(match.group(1)) if match else 0

    return sorted(files, key=extract_step)


# =========================================================================
# 2. MAIN
# =========================================================================

def main():
    files = get_sorted_files()
    if not files:
        print(f"No VTK files found in {DATA_DIR} matching "
              f"'{FILE_PATTERN}'.")
        print("Run run_3d_experiment_spiral.py with VTK output enabled.")
        return

    print(f"Found {len(files)} frames. "
          f"Loading metadata from {os.path.basename(files[0])}...")
    nx, ny, nz = parse_vtk_dimensions(files[0])
    print(f"Grid: {nx} × {ny} × {nz}")

    # Slice indices
    idx_x = int(nx * SLICE_X_RATIO)
    idx_y = int(ny * SLICE_Y_RATIO)
    idx_z = int(nz * SLICE_Z_RATIO)

    # --- Setup figure: 3 rows × 3 columns ---
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle("Thermodynamic Audit: 3D Spiral Wave", fontsize=16)

    rows   = ["Voltage ($u$)", "Metabolic ($k_{eff}$)", "Diffusive Cost (W)"]
    cmaps  = [CMAP_V, CMAP_K, CMAP_P]
    vmins  = [V_MIN, K_MIN, P_MIN]
    vmaxs  = [V_MAX, K_MAX, P_MAX]
    fields = ["Voltage_u", "Metabolic_Stiffness_k", "Dissipation_Power_Pdiff"]

    # Initial data
    data0 = load_vtk_multifield(files[0], nx, ny, nz)

    images = []  # (row, col) → imshow artist

    for r in range(3):
        field_name = fields[r]
        field_data = data0.get(field_name, np.zeros((nx, ny, nz)))

        # XY plane (Z slice)
        ax_xy = axes[r, 0]
        im_xy = ax_xy.imshow(field_data[:, :, idx_z].T, origin='lower',
                             cmap=cmaps[r], vmin=vmins[r], vmax=vmaxs[r])
        ax_xy.set_ylabel(rows[r], fontsize=12, fontweight='bold')
        if r == 0:
            ax_xy.set_title(f"XY Plane (Z={idx_z})")

        # XZ plane (Y slice)
        ax_xz = axes[r, 1]
        im_xz = ax_xz.imshow(field_data[:, idx_y, :].T, origin='lower',
                             cmap=cmaps[r], vmin=vmins[r], vmax=vmaxs[r])
        if r == 0:
            ax_xz.set_title(f"XZ Plane (Y={idx_y})")

        # YZ plane (X slice)
        ax_yz = axes[r, 2]
        im_yz = ax_yz.imshow(field_data[idx_x, :, :].T, origin='lower',
                             cmap=cmaps[r], vmin=vmins[r], vmax=vmaxs[r])
        if r == 0:
            ax_yz.set_title(f"YZ Plane (X={idx_x})")

        # Turn off ticks
        for ax in [ax_xy, ax_xz, ax_yz]:
            ax.set_xticks([])
            ax.set_yticks([])

        images.append([im_xy, im_xz, im_yz])

    # Time annotation
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)

    def update(frame_idx):
        fname = files[frame_idx]
        data = load_vtk_multifield(fname, nx, ny, nz)

        step_num = 0
        match = re.search(r"_(\d+)\.vtk", fname)
        if match:
            step_num = match.group(1)

        for r in range(3):
            field_name = fields[r]
            if field_name in data:
                d = data[field_name]
                images[r][0].set_data(d[:, :, idx_z].T)
                images[r][1].set_data(d[:, idx_y, :].T)
                images[r][2].set_data(d[idx_x, :, :].T)

        time_text.set_text(f"Simulation Step: {step_num}")

        all_artists = [img for row in images for img in row] + [time_text]
        return all_artists

    print("Rendering animation...")
    ani = animation.FuncAnimation(
        fig, update, frames=len(files),
        interval=1000 / FPS, blit=True)

    out_file = os.path.join(DATA_DIR, "threeD_spiral_audit.mp4")
    print(f"Saving to {out_file}...")
    ani.save(out_file, writer='ffmpeg', fps=FPS, dpi=100)
    print("Done.")


if __name__ == "__main__":
    main()