"""
run_experiment_ensemble.py — Ensemble 2D Tissue Simulation Runner
==================================================================

Paper reference
---------------
  Generates raw data for Figures 3 and S3 (thermodynamic profiling of
  myocardial fibrosis).  Implements the protocol described in Methods
  § "Modelling of myocardial fibrosis".

Purpose
-------
  Runs a Monte Carlo ensemble of 2D electromechanical tissue simulations
  across three tissue topologies (Healthy, Diffuse fibrosis, Patchy fibrosis)
  and two stimulation protocols (Planar wave, Spiral wave).  Each simulation
  produces a results.npz file containing:
    - Subsample voltage snapshots  U(t)
    - Electrical/mechanical stored energy histories
    - Cumulative dissipated energy histories (diffusive, reactive, mechanical)
    - Spatial energy dissipation maps
    - Spatial metabolic recruitment map (k_eff)

  The ensemble size is 1 for Healthy (deterministic) and 50 for fibrotic
  topologies (to capture statistical variance from random scar placement).

Usage
-----
  python run_experiment_ensemble.py

  ⚠ Warning: This is computationally expensive.  On a typical workstation
  with 8–16 cores, the full ensemble (152 simulations, 80×80 grid, 120 ms
  simulation time) takes approximately 2–6 hours.  The script uses
  multiprocessing with N/2 worker processes.

Output directory
----------------
  results_thermo_audit/
    HEALTHY_PLANAR_s00/    results.npz
    HEALTHY_SPIRAL_s00/    results.npz
    DIFFUSE_PLANAR_s00..49/results.npz
    DIFFUSE_SPIRAL_s00..49/results.npz
    PATCHY_PLANAR_s00..49/results.npz
    PATCHY_SPIRAL_s00..49/results.npz

Post-processing
---------------
  Figure3.py  — reads results_thermo_audit/*/results.npz → Figure 3 panels
  FigureS3.py — reads results_thermo_audit/*/results.npz → Figure S3

Dependencies
------------
  numpy, scipy, multiprocessing, phs_hetero_core (local module)

Author
------
  Jagir R. Hussan
"""

import os
# Ensure Numba uses single-threaded execution per worker
# (parallelism is handled by multiprocessing at the simulation level)
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import multiprocessing
import phs_hetero_core
from scipy.ndimage import gaussian_filter

from phs_models import PARAMS_2D_FIBROSIS


# =========================================================================
# 0. CONFIGURATION
# =========================================================================

# --- Grid and time domain ---
N      = 80          # Grid dimension (N×N nodes)
T_END  = 120.0       # Total simulation time [ms]
T_EVAL = np.linspace(0, T_END, 1200)  # Output sampling points

# --- Output directory (relative to repository root) ---
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results_thermo_audit")

# --- Model parameters (drawn from PARAMS_2D_FIBROSIS in phs_models.py) ---
# Tuple format for Numba kernel:
#   (k_base, a, eps_ap, mu1, mu2, c0, alpha_fb, k_recruit, k_max, unused)
P = PARAMS_2D_FIBROSIS
PARAMS_MATCHED = (P['k'],          # k_base    — baseline reaction rate
                  P['a'],          # a         — excitation threshold
                  P['eps'],        # eps_ap    — base recovery rate
                  P['mu1'],        # mu1       — APD restitution
                  P['mu2'],        # mu2       — APD restitution
                  P['c0'],         # c0        — cross-bridge cycling rate
                  P['alpha_fb'],   # alpha_fb  — gyrator coupling strength
                  P['k_recruit'],  # k_recruit — metabolic recruitment gain
                  P['k_max'],      # k_max     — metabolic ceiling
                  0.0)             # (unused placeholder)


# =========================================================================
# 1. TISSUE TOPOLOGY GENERATION
# =========================================================================

def generate_maps(topology_type, N, seed=None):
    """
    Generate spatial maps for conductivity (SIG_L, SIG_T) and
    mechanical viscosity (K_VISC) for a given tissue topology.

    Topology types
    --------------
    HEALTHY  : Uniform tissue with default parameters everywhere.
    DIFFUSE  : Spatially uncorrelated random fibrosis — each node has
               an independent 40% probability of being fibrotic.
    PATCHY   : Spatially correlated fibrosis — Gaussian-filtered noise
               (σ = 2.0) produces clustered scar regions mimicking
               the histological pattern of replacement fibrosis.

    Fibrotic node properties
    ------------------------
      Conductivity   : reduced to 10% of baseline (σ_l = σ_t = 0.1)
      Viscosity      : increased 5× (k_visc = 0.25) to represent
                        collagen deposition and increased passive stiffness
                        [Nezlobinsky et al., 2020].

    Parameters
    ----------
    topology_type : str, one of "HEALTHY", "DIFFUSE", "PATCHY"
    N : int, grid dimension
    seed : int or None, random seed for reproducibility

    Returns
    -------
    SIG_L  : ndarray (N²,) — longitudinal conductivity
    SIG_T  : ndarray (N²,) — transverse conductivity
    K_VISC : ndarray (N²,) — viscous damping coefficient
    """
    n_nodes = N * N
    np.random.seed(seed)

    # Default healthy values
    SIG_L  = np.full(n_nodes, 1.0)    # longitudinal conductivity
    SIG_T  = np.full(n_nodes, 0.2)    # transverse (5:1 anisotropy ratio)
    K_VISC = np.full(n_nodes, 0.05)   # viscous damping

    if topology_type != "HEALTHY":
        # Generate noise field
        if topology_type == "DIFFUSE":
            # Uncorrelated: independent Bernoulli at each node
            noise = np.random.rand(N, N).flatten()
        else:  # PATCHY
            # Correlated: Gaussian filter (σ = 2.0) → clustered fibrosis
            noise = gaussian_filter(np.random.rand(N, N), sigma=2.0).flatten()

        # Normalise to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min())

        # Binary mask: nodes below 40th percentile → fibrotic
        fibro_mask = noise < 0.4

        # Apply fibrotic modifications
        SIG_L[fibro_mask]  = 0.1   # 10% conductivity
        SIG_T[fibro_mask]  = 0.1
        K_VISC[fibro_mask] = 0.25  # 5× stiffness

    return SIG_L, SIG_T, K_VISC


# =========================================================================
# 2. STIMULATION PROTOCOLS
# =========================================================================

def build_stimulus_masks(N, protocol):
    """
    Build S1 and S2 stimulus masks for a given protocol.

    Protocols
    ---------
    PLANAR  : S1 only — stimulates the leftmost 3 columns, generating a
              planar wave propagating rightward.
    SPIRAL  : S1 (planar, left edge) + S2 (cross-field, top half).  The
              S2 is delivered at t = 42 ms into the wake of the S1 wave
              to induce spiral-wave re-entry.
    """
    m1 = np.zeros(N * N)   # S1 mask
    m2 = np.zeros(N * N)   # S2 mask

    # S1: Leftmost 3 columns (planar wave initiation)
    for r in range(N):
        m1[r * N : r * N + 3] = 5.0

    if protocol == "SPIRAL":
        # S2: Top half of the domain (cross-field stimulus)
        for r in range(N // 2):
            m2[r * N : r * N + (N // 2)] = 5.0

    return m1, m2


# =========================================================================
# 3. SINGLE SIMULATION WORKER
# =========================================================================

def run_single_simulation(task_config):
    """
    Run one simulation for the ensemble.

    Parameters
    ----------
    task_config : tuple
        (run_id, topology_type, protocol, seed)

    Returns
    -------
    str : completion message with run_id
    """
    run_id, topo_type, protocol, seed = task_config

    # Create output directory for this run
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Generate tissue maps
    SL, ST, KV = generate_maps(topo_type, N, seed)

    # Stimulus timing
    if protocol == "PLANAR":
        timings = (1.0, 2.0, 9999.0, 0.0)   # S1 only
    else:  # SPIRAL
        timings = (1.0, 2.0, 42.0, 2.0)     # S1 then S2 at 42 ms

    # Build stimulus masks
    m1, m2 = build_stimulus_masks(N, protocol)

    # --- Run solver ---
    (U, HE, HM,
     ED_hist, ER_hist, EM_hist, ES_hist,
     MAP_D, MAP_M, MAP_K) = phs_hetero_core.run_solver(
        np.zeros(4 * N * N),  # initial condition: all zeros
        N, PARAMS_MATCHED,
        SL, ST, KV, m1, m2, timings, T_EVAL
    )

    # Determine if spiral was sustained (used for filtering)
    is_sustained = np.max(U[-3]) > 0.5

    # Save results
    np.savez(os.path.join(run_dir, "results.npz"),
             U=U, HE=HE, HM=HM,
             E_diff=ED_hist, E_reac=ER_hist,
             E_mech=EM_hist, E_stim=ES_hist,
             Map_Diff=MAP_D.reshape(N, N),
             Map_Mech=MAP_M.reshape(N, N),
             Map_K_Effort=MAP_K.reshape(N, N),
             SL=SL, ST=ST, KV=KV,
             is_sustained=is_sustained)

    return f"{run_id}: success"


# =========================================================================
# 4. MAIN — Ensemble Orchestration
# =========================================================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Build task list
    tasks = []
    for topo in ["HEALTHY", "DIFFUSE", "PATCHY"]:
        for proto in ["PLANAR", "SPIRAL"]:
            # 1 seed for healthy (deterministic), 50 for fibrotic (statistical)
            n_seeds = 1 if topo == "HEALTHY" else 50
            for s in range(n_seeds):
                tasks.append((
                    f"{topo}_{proto}_s{s:02d}",
                    topo, proto,
                    100 + s   # deterministic offset for reproducibility
                ))

    print(f"Total tasks: {len(tasks)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Starting ensemble...")

    # Multiprocessing: use half the available cores to avoid oversubscription
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(max(1, os.cpu_count() // 2)) as pool:
        results = pool.map(run_single_simulation, tasks)
        for r in results:
            print(r)

    print("\nEnsemble complete.")


if __name__ == "__main__":
    main()