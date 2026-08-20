"""
FigureS2.py — Fibrotic Tissue Structure Visualisation
======================================================

Paper reference
---------------
  Supplementary Figure S2: Comparison of tissue topology models used
  in the fibrosis study (Figure 3).

Purpose
-------
  Generates a side-by-side comparison of the three tissue topologies
  used in the fibrosis ensemble simulations:
    1. Homogeneous (Healthy) — uniform conductivity σ = 1.0
    2. Diffuse fibrosis — uncorrelated random scar (40% burden)
    3. Patchy fibrosis  — spatially correlated scar (σ_filter = 2.0)

  The maps are generated using the EXACT same logic as
  run_experiment_ensemble.py.generate_maps() to guarantee that the
  visualisation matches the actual simulation inputs.

Output
------
  figures/S2.pdf — three-panel figure with binary colour legend

Dependencies
------------
  numpy, scipy, matplotlib

Author
------
  Jagir R. Hussan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors

# =========================================================================
# 0. CONFIGURATION
# =========================================================================

N   = 80   # Grid size (matching run_experiment_ensemble.py)
DPI = 300  # Publication-quality resolution

# Fibrosis burden: fraction of nodes below this noise percentile
# are classified as fibrotic (matching run_experiment_ensemble.py)
FIBROSIS_THRESHOLD = 0.4

# Fibrotic tissue properties (matching run_experiment_ensemble.py)
SIGMA_FIBROTIC = 0.1   # Conductivity in fibrotic regions (10% of baseline)
SIGMA_HEALTHY  = 1.0   # Conductivity in healthy myocardium


# =========================================================================
# 1. TISSUE MAP GENERATION (mirrors run_experiment_ensemble.py)
# =========================================================================

def generate_maps(topology_type, N, seed=None):
    """
    Generate longitudinal conductivity map for a given topology.

    This function replicates the logic of
    run_experiment_ensemble.py.generate_maps() to ensure that the
    visualised maps are identical to the simulation inputs.

    Parameters
    ----------
    topology_type : str — "HEALTHY", "DIFFUSE", or "PATCHY"
    N    : int — grid dimension
    seed : int or None — random seed

    Returns
    -------
    SIG_L : ndarray (N, N) — longitudinal conductivity map
    """
    n_nodes = N * N
    np.random.seed(seed)

    # Default: uniform healthy tissue
    SIG_L = np.full(n_nodes, SIGMA_HEALTHY)

    if topology_type == "HEALTHY":
        pass   # no modifications needed

    elif topology_type == "DIFFUSE":
        # Uncorrelated random noise → independent Bernoulli at each node
        raw_noise = np.random.rand(N, N)
        noise = raw_noise.flatten()
        # Normalise
        noise = (noise - noise.min()) / (noise.max() - noise.min())

    else:  # "PATCHY"
        # Gaussian-filtered noise → spatially correlated scar clusters
        raw_noise = np.random.rand(N, N)
        # σ = 2.0 matches the run_experiment_ensemble.py configuration
        noise = gaussian_filter(raw_noise, sigma=2.0).flatten()
        noise = (noise - noise.min()) / (noise.max() - noise.min())

    # Apply fibrosis threshold
    if topology_type != "HEALTHY":
        fibro_mask = noise < FIBROSIS_THRESHOLD
        SIG_L[fibro_mask] = SIGMA_FIBROTIC

    return SIG_L.reshape(N, N)


# =========================================================================
# 2. PLOTTING
# =========================================================================

def main():
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'figure.dpi': DPI,
    })

    # Topologies to visualise
    scenarios = [
        ("HEALTHY", "Homogeneous\n(Healthy)",      1),
        ("DIFFUSE", "Diffuse fibrosis\n(Random)",  100),
        ("PATCHY",  "Patchy fibrosis\n(Correlated)", 100),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5),
                             constrained_layout=True)

    # Binary colormap: dark grey = fibrosis, white = healthy
    cmap = mcolors.ListedColormap(['#4a4a4a', '#f5f5f5'])
    bounds = [0.0, 0.5, 1.1]
    norm  = mcolors.BoundaryNorm(bounds, cmap.N)

    for ax, (topo_key, title, seed) in zip(axes, scenarios):
        data = generate_maps(topo_key, N, seed=seed)
        ax.imshow(data, cmap=cmap, norm=norm,
                  interpolation='nearest', origin='lower')

        ax.set_title(title, pad=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # Border frame
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    # Custom legend (binary states)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#f5f5f5', edgecolor='black',
              label=r'Myocardium ($\sigma=1.0$)'),
        Patch(facecolor='#4a4a4a', edgecolor='black',
              label=r'Fibrosis ($\sigma=0.1$)'),
    ]

    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False,
               fontsize=16)

    plt.savefig("figures/S2.pdf", dpi=DPI,
                bbox_inches='tight', pad_inches=0.15)
    print("Saved figures/S2.pdf")


if __name__ == "__main__":
    main()