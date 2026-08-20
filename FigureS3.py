"""
FigureS3.py — Metabolic Masking: Spatial Evidence at t = 87 ms
===============================================================

Paper reference
---------------
  Supplementary Figure S3: Spatial decomposition of the metabolic
  masking phenomenon described in Figure 3.

Purpose
-------
  At the specific time point t = 87 ms (post-S2, during the spiral-wave
  transient), this script extracts:
    1. Spatial maps of membrane potential u(x,y) and metabolic recruitment
       k_eff(x,y) for both healthy and fibrotic tissue.
    2. A bar-chart comparison of global diffusive power P_diff vs. mechanical
       viscous power P_mech, quantifying the thermodynamic origin of the
       metabolic masking.

  The k_eff map is reconstructed from the instantaneous voltage field
  using the same harmonic-mean stencil and metabolic-autoregulation logic
  as the phs_hetero_core kernel, ensuring consistency with the simulation.

Input
-----
  results_thermo_audit/HEALTHY_SPIRAL_*/results.npz
  results_thermo_audit/PATCHY_SPIRAL_*/results.npz

Output
------
  figures/S3.pdf — hybrid figure: 2×2 spatial maps + bar chart

Dependencies
------------
  numpy, matplotlib, numba

Author
------
  Jagir R. Hussan
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from numba import njit


# =========================================================================
# 1. RECONSTRUCTION KERNEL — Instantaneous k_eff from voltage snapshot
# =========================================================================

@njit(fastmath=True)
def reconstruct_instantaneous_keff(u_frame, N, SIG_L, SIG_T):
    """
    Reconstruct the instantaneous metabolic recruitment map k_eff(x,y)
    from a single voltage snapshot, using the same harmonic-mean
    diffusion stencil and autoregulation logic as phs_hetero_core.py.

    Parameters
    ----------
    u_frame : ndarray (N²,) — flat voltage at a single time point
    N       : int           — grid dimension
    SIG_L   : ndarray (N²,) — longitudinal conductivity map
    SIG_T   : ndarray (N²,) — transverse conductivity map

    Returns
    -------
    k_eff_map : ndarray (N²,) — reconstructed k_eff per node
    """
    # Metabolic parameters (matching phs_hetero_core / run_experiment_ensemble)
    k_base    = 8.0
    k_recruit = 5.0
    k_max     = 20.0
    div_eps   = 1e-16

    n_nodes = N * N
    k_eff_map = np.zeros(n_nodes, dtype=np.float64)

    for i in range(n_nodes):
        r, c = i // N, i % N
        g_curr_L = max(SIG_L[i], 1e-9)
        g_curr_T = max(SIG_T[i], 1e-9)
        diff_flux = 0.0

        # X-direction: left neighbour
        if c > 0:
            g_prev = max(SIG_L[i-1], 1e-9)
            g_eff = (2.0 * g_curr_L * g_prev) / (g_curr_L + g_prev + div_eps)
            diff_flux += g_eff * (u_frame[i-1] - u_frame[i])

        # X-direction: right neighbour
        if c < N - 1:
            g_next = max(SIG_L[i+1], 1e-9)
            g_eff = (2.0 * g_curr_L * g_next) / (g_curr_L + g_next + div_eps)
            diff_flux += g_eff * (u_frame[i+1] - u_frame[i])

        # Y-direction: top neighbour
        if r > 0:
            g_prev = max(SIG_T[i-N], 1e-9)
            g_eff = (2.0 * g_curr_T * g_prev) / (g_curr_T + g_prev + div_eps)
            diff_flux += g_eff * (u_frame[i-N] - u_frame[i])

        # Y-direction: bottom neighbour
        if r < N - 1:
            g_next = max(SIG_T[i+N], 1e-9)
            g_eff = (2.0 * g_curr_T * g_next) / (g_curr_T + g_next + div_eps)
            diff_flux += g_eff * (u_frame[i+N] - u_frame[i])

        # Metabolic autoregulation: recruit k_eff based on outward flux
        load = -diff_flux if diff_flux < 0 else 0.0
        k_eff_map[i] = min(k_base + (k_recruit * load), k_max)

    return k_eff_map


# =========================================================================
# 2. DATA LOADING
# =========================================================================

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(REPO_ROOT, "results_thermo_audit")
PLOTS_DIR  = os.path.join(REPO_ROOT, "figures")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Locate the first healthy and patchy spiral runs
h_files = glob.glob(os.path.join(OUTPUT_DIR, "HEALTHY_SPIRAL_*", "results.npz"))
p_files = glob.glob(os.path.join(OUTPUT_DIR, "PATCHY_SPIRAL_*", "results.npz"))

if not h_files or not p_files:
    print("Error: No simulation data found.")
    print("Run 'run_experiment_ensemble.py' first.")
    exit(1)

with np.load(h_files[0]) as data:
    U_h, SL_h, ST_h     = data['U'], data['SL'], data['ST']
    E_diff_h, E_mech_h  = data['E_diff'], data['E_mech']

with np.load(p_files[0]) as data:
    U_p, SL_p, ST_p     = data['U'], data['SL'], data['ST']
    E_diff_p, E_mech_p  = data['E_diff'], data['E_mech']

N = int(np.sqrt(U_h.shape[1]))
t_eval = np.linspace(0, 120.0, U_h.shape[0])


# =========================================================================
# 3. EXTRACT DATA AT target_time = 87 ms
# =========================================================================

target_time = 87.0
frame_idx = np.argmin(np.abs(t_eval - target_time))

# Voltage maps
u_h_map = U_h[frame_idx].reshape((N, N))
u_p_map = U_p[frame_idx].reshape((N, N))

# Reconstructed k_eff maps
k_h_map = reconstruct_instantaneous_keff(
    U_h[frame_idx], N, SL_h, ST_h).reshape((N, N))
k_p_map = reconstruct_instantaneous_keff(
    U_p[frame_idx], N, SL_p, ST_p).reshape((N, N))

# Instantaneous power (finite-difference of cumulative energy)
dt = t_eval[frame_idx] - t_eval[frame_idx - 1]
P_diff_h = (E_diff_h[frame_idx] - E_diff_h[frame_idx - 1]) / dt
P_mech_h = (E_mech_h[frame_idx] - E_mech_h[frame_idx - 1]) / dt
P_diff_p = (E_diff_p[frame_idx] - E_diff_p[frame_idx - 1]) / dt
P_mech_p = (E_mech_p[frame_idx] - E_mech_p[frame_idx - 1]) / dt


# =========================================================================
# 4. PLOTTING
# =========================================================================

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

fig = plt.figure(figsize=(16, 8), constrained_layout=True)

# Main grid: left (2×2 spatial maps) | right (bar chart)
gs_main = GridSpec(1, 2, figure=fig, width_ratios=[2, 1.2])

# Nested grid for maps: 2 data rows + 1 colourbar row
gs_left = gs_main[0].subgridspec(3, 2, height_ratios=[1, 1, 0.08])

# --- Spatial map axes ---
ax_uh = fig.add_subplot(gs_left[0, 0])
ax_kh = fig.add_subplot(gs_left[0, 1])
ax_up = fig.add_subplot(gs_left[1, 0])
ax_kp = fig.add_subplot(gs_left[1, 1])

# Colourbar axes
cax_u = fig.add_subplot(gs_left[2, 0])
cax_k = fig.add_subplot(gs_left[2, 1])

# Turn off map axes ticks
for ax in [ax_uh, ax_kh, ax_up, ax_kp]:
    ax.axis('off')

# Row labels
ax_uh.set_title("Membrane potential ($u$)", fontweight='bold')
ax_kh.set_title("Metabolic demand ($k_{eff}$)", fontweight='bold')
ax_uh.text(-0.05, 0.5, "Healthy", transform=ax_uh.transAxes,
           rotation=90, va='center', ha='right',
           fontweight='bold', fontsize=14)
ax_up.text(-0.05, 0.5, "Fibrotic", transform=ax_up.transAxes,
           rotation=90, va='center', ha='right',
           fontweight='bold', fontsize=14)

# Plot maps
im_u = ax_uh.imshow(u_h_map, cmap='plasma', vmin=-0.1, vmax=1.2,
                    origin='lower')
im_k = ax_kh.imshow(k_h_map, cmap='binary', vmin=8.0, vmax=12.0,
                    origin='lower')
ax_up.imshow(u_p_map, cmap='plasma', vmin=-0.1, vmax=1.2,
             origin='lower')
ax_kp.imshow(k_p_map, cmap='binary', vmin=8.0, vmax=12.0,
             origin='lower')

# Horizontal colourbars
cbar_u = fig.colorbar(im_u, cax=cax_u, orientation='horizontal')
cbar_u.set_label("Voltage (dimensionless)", fontweight='bold')

cbar_k = fig.colorbar(im_k, cax=cax_k, orientation='horizontal')
cbar_k.set_label("ATP recruitment rate ($k_{eff}$)", fontweight='bold')

# --- Bar chart: thermodynamic origin of metabolic masking ---
ax_bar = fig.add_subplot(gs_main[1])

labels  = ['Healthy tissue', 'Fibrotic tissue']
p_diffs = [P_diff_h, P_diff_p]
p_mechs = [P_mech_h, P_mech_p]

x     = np.arange(len(labels))
width = 0.35

ax_bar.bar(x - width / 2, p_diffs, width,
           label='Diffusive power ($P_{diff}$)',
           color='#0072B2', edgecolor='black')
ax_bar.bar(x + width / 2, p_mechs, width,
           label='Mechanical viscous power ($P_{mech}$)',
           color='#D55E00', edgecolor='black')

ax_bar.set_ylabel('Global dissipated power (W)', fontweight='bold')
ax_bar.set_title("Thermodynamic origin of metabolic masking",
                 fontweight='bold')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, fontweight='bold')
ax_bar.legend()
ax_bar.grid(True, alpha=0.3, axis='y', linestyle='--')

# Annotation: scar blocks diffusion → diffusive power drops
ax_bar.annotate('Scar blocks diffusion:\nNumerator drops.',
                xy=(1 - width / 2, P_diff_p),
                xytext=(0.45, P_diff_p * 1.1),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=2),
                fontsize=12, fontweight='bold')

fig.suptitle(f"Time: {target_time:.1f} ms", fontsize=18, fontweight='bold')

out_path = os.path.join(PLOTS_DIR, "S3.pdf")
fig.savefig(out_path, dpi=300)
print(f"Saved {out_path}")