"""
phs_3d_core.py — 3D Port-Hamiltonian Hypergraph Solver (Numba-JIT)
===================================================================

Paper reference
---------------
  Used for Figure 2 (algorithmic scalability in 3D anisotropic ventricular
  wedge).  Implements the condensed hypergraph Laplacian in 3D with full
  orthotropic conductivity.

Purpose
-------
  High-performance Numba-JIT kernel for 3D electromechanical simulations
  using the port-Hamiltonian hypergraph framework.  This kernel is called
  by run_3d_experiment_spiral.py for the ventricular-wedge scaling studies.

Key differences from the 2D kernel (phs_hetero_core.py)
-------------------------------------------------------
  1.  Operator splitting (Lie-Trotter): diffusion and reaction steps are
      decoupled for 3D efficiency.
  2.  Orthotropic conductivity: distinct D11 (fibre), D22 (sheet),
      D33 (normal), and D12 (cross-term) diffusivity tensors.
  3.  Different parameter values (see PARAMS_3D_SCALING in phs_models.py):
      a = 0.05 (lower excitation threshold for 3D scroll-wave stability),
      k_recruit = 2.0, k_max = 30.0.
  4.  Harmonic-mean interfacial conductance (same series-resistance
      derivation as the 2D kernel).

Physics implemented (see phs_models.py for full derivation)
-----------------------------------------------------------
  HL.1—HL.2  : Harmonic-mean interfacial conductance (hypergraph Laplacian)
  AP.1—AP.2  : Aliev-Panfilov electrophysiology
  NM.1—NM.3  : Nash-Panfilov / Silva active mechanics
  GYR.1—GYR.2: Port-Hamiltonian gyrator coupling
  MA.1       : Metabolic autoregulation

Dependencies
------------
  numpy, numba

Author
------
  Jagir R. Hussan
"""

import numpy as np
from numba import njit, prange


# =========================================================================
# 1. CONDENSED HYPERGRAPH LAPLACIAN — 3D Anisotropic
# =========================================================================
@njit(fastmath=True, parallel=True)
def compute_laplacian_3d_aniso(U, D11, D22, D33, D12, dx, dy, dz,
                                Nx, Ny, Nz):
    """
    Compute the condensed hypergraph Laplacian for a 3D orthotropic grid.

    This implements the harmonic-mean (series-resistance) formulation
    derived in the paper Appendix for the principal axes (fibre, sheet,
    normal), plus standard central differences for cross-terms.

    The harmonic mean g_eff = 2*g_i*g_j / (g_i + g_j + ε) correctly
    models the series-resistance topology of the static condensation,
    ensuring flux continuity across discontinuous interfaces (e.g.,
    fibrosis boundaries) and preventing artificial energy creation.

    Parameters
    ----------
    U   : ndarray (Nx, Ny, Nz) — voltage field
    D11, D22, D33, D12 : ndarray — diffusivity tensor components
    dx, dy, dz : float — grid spacings
    Nx, Ny, Nz : int   — grid dimensions

    Returns
    -------
    lap : ndarray (Nx, Ny, Nz) — Laplacian of U
    """
    lap = np.zeros_like(U)
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    idz2 = 1.0 / (dz * dz)
    idxy = 1.0 / (4.0 * dx * dy)

    # Regularisation constant to prevent division by zero
    eps = 1e-16

    for k in prange(1, Nz - 1):
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):

                # ========================================================
                # X-direction: fibre axis (Harmonic Mean / Series Conductance)
                # ========================================================
                g_x_curr = D11[i, j, k]
                g_x_next = D11[i + 1, j, k]
                g_x_prev = D11[i - 1, j, k]

                # Factor 2.0 accounts for half-cell length (dx/2) to the
                # interface in the harmonic-mean derivation.
                d11_p = (2.0 * g_x_curr * g_x_next) / (
                    g_x_curr + g_x_next + eps)
                d11_m = (2.0 * g_x_curr * g_x_prev) / (
                    g_x_curr + g_x_prev + eps)

                term_xx = (d11_p * (U[i+1, j, k] - U[i, j, k])
                           - d11_m * (U[i, j, k] - U[i-1, j, k])) * idx2

                # ========================================================
                # Y-direction: sheet axis
                # ========================================================
                g_y_curr = D22[i, j, k]
                g_y_next = D22[i, j+1, k]
                g_y_prev = D22[i, j-1, k]

                d22_p = (2.0 * g_y_curr * g_y_next) / (
                    g_y_curr + g_y_next + eps)
                d22_m = (2.0 * g_y_curr * g_y_prev) / (
                    g_y_curr + g_y_prev + eps)

                term_yy = (d22_p * (U[i, j+1, k] - U[i, j, k])
                           - d22_m * (U[i, j, k] - U[i, j-1, k])) * idy2

                # ========================================================
                # Z-direction: normal (cross-sheet) axis
                # ========================================================
                g_z_curr = D33[i, j, k]
                g_z_next = D33[i, j, k+1]
                g_z_prev = D33[i, j, k-1]

                d33_p = (2.0 * g_z_curr * g_z_next) / (
                    g_z_curr + g_z_next + eps)
                d33_m = (2.0 * g_z_curr * g_z_prev) / (
                    g_z_curr + g_z_prev + eps)

                term_zz = (d33_p * (U[i, j, k+1] - U[i, j, k])
                           - d33_m * (U[i, j, k] - U[i, j, k-1])) * idz2

                # ========================================================
                # Cross-terms (standard central differences)
                # Mixed derivatives use nodal values, not harmonic means.
                # ========================================================
                dvdy_ip1 = (U[i+1, j+1, k] - U[i+1, j-1, k])
                dvdy_im1 = (U[i-1, j+1, k] - U[i-1, j-1, k])
                term_xy1 = (D12[i+1, j, k] * dvdy_ip1
                            - D12[i-1, j, k] * dvdy_im1) * idxy

                dvdx_jp1 = (U[i+1, j+1, k] - U[i-1, j+1, k])
                dvdx_jm1 = (U[i+1, j-1, k] - U[i-1, j-1, k])
                term_xy2 = (D12[i, j+1, k] * dvdx_jp1
                            - D12[i, j-1, k] * dvdx_jm1) * idxy

                lap[i, j, k] = (term_xx + term_yy + term_zz
                                + term_xy1 + term_xy2)

    return lap


# =========================================================================
# 2. FULL ELECTROMECHANICAL REACTION STEP
# =========================================================================
@njit(fastmath=True, parallel=True)
def physics_reaction_step(U, V, Ta, Tai, lap, dt):
    """
    Advance the reaction (ODE) components by one time step dt using
    forward Euler integration (operator-split from diffusion).

    This implements:
      - Aliev-Panfilov electrophysiology (AP.1—AP.2)
      - Nash-Panfilov active mechanics (NM.1—NM.3)
      - Port-Hamiltonian gyrator coupling (GYR.1—GYR.2)
      - Metabolic autoregulation (MA.1)

    Parameters (see PARAMS_3D_SCALING in phs_models.py)
    ---------------------------------------------------
    All states U, V, Ta, Tai and the Laplacian lap are updated
    in-place (flattened, then reshaped).

    Parameter values differ from the 2D fibrosis kernel:
      a         = 0.05   (lower threshold for 3D scroll-wave stability)
      k_recruit = 2.0    (lower metabolic recruitment for 3D domain)
      k_max     = 30.0   (higher metabolic ceiling for 3D domain)
    """

    # --- Parameter values (PARAMS_3D_SCALING) ---
    k_base    = 8.0     # Baseline reaction rate                   [1/ms]
    a         = 0.05    # Excitation threshold (3D-specific)       [-]
    k_recruit = 2.0     # Metabolic recruitment gain               [1/ms]
    k_max     = 30.0    # Metabolic ceiling                        [1/ms]

    # Restitution parameters
    eps  = 0.01
    mu1  = 0.2
    mu2  = 0.3

    # Mechanical parameters
    c0       = 0.05    # Cross-bridge cycling rate                 [1/ms]
    alpha_fb = 0.05    # Gyrator coupling strength                 [-]
    k_visc  = 0.05    # Passive tension damping rate              [1/ms]

    N = U.size
    U_f   = U.ravel()
    V_f   = V.ravel()
    Ta_f  = Ta.ravel()
    Tai_f = Tai.ravel()
    lap_f = lap.ravel()

    for i in prange(N):
        u_val   = U_f[i]
        v_val   = V_f[i]
        ta_val  = Ta_f[i]
        tai_val = Tai_f[i]

        # --- 1. Metabolic Autoregulation (MA.1) ---
        diff_current = lap_f[i]
        load = 0.0
        if diff_current < 0:
            load = -diff_current

        k_local = k_base + k_recruit * load
        if k_local > k_max:
            k_local = k_max

        # --- 2. Electromechanical Gyrator (GYR.1—GYR.3) ---
        sigmoid = 1.0 / (1.0 + np.exp(-50.0 * (u_val - 0.05)))
        alpha_transducer = alpha_fb * sigmoid

        flow_act   = alpha_transducer * u_val      # → dTai (GYR.1)
        i_feedback = alpha_transducer * tai_val    # → du   (GYR.2)

        # --- 3. Reaction Dynamics ---

        # du/dt: Aliev-Panfilov fast reaction + gyrator feedback (AP.1)
        du_reac = (k_local * u_val * (u_val - a) * (1.0 - u_val)
                   - u_val * v_val)
        du_total = du_reac - i_feedback

        # dv/dt: Aliev-Panfilov slow recovery (AP.2)
        term_recov = (eps + (mu1 * v_val) / (mu2 + u_val))
        dv_total = term_recov * (-v_val
                                  - k_local * u_val * (u_val - a - 1.0))

        # dTai/dt: Calcium activation driven by gyrator flow (GYR.1)
        dtai_total = flow_act - (c0 * tai_val)

        # dTa/dt: Active tension with viscous damping (NM.2)
        dta_total = k_visc * (tai_val - ta_val)

        # Euler update (in-place)
        U_f[i]   += dt * du_total
        V_f[i]   += dt * dv_total
        Ta_f[i]  += dt * dta_total
        Tai_f[i] += dt * dtai_total

    return U, V, Ta, Tai


# =========================================================================
# 3. BOUNDARY CONDITIONS
# =========================================================================
@njit(parallel=True)
def apply_no_flux_bc(U):
    """Apply homogeneous Neumann (zero-flux) boundary conditions."""
    U[0, :, :]   = U[1, :, :]
    U[-1, :, :]  = U[-2, :, :]
    U[:, 0, :]   = U[:, 1, :]
    U[:, -1, :]  = U[:, -2, :]
    U[:, :, 0]   = U[:, :, 1]
    U[:, :, -1]  = U[:, :, -2]
    return U


# =========================================================================
# 4. TIME STEP — Operator Splitting (Lie-Trotter)
# =========================================================================
@njit
def rk4_step_3d(U, V, Ta, Tai, D11, D22, D33, D12,
                dx, dy, dz, dt):
    """
    Advance one full time step using Lie-Trotter operator splitting:

      1. Diffusion step:  U*   = U^n + dt * div(D grad U^n)
                          Apply Neumann BCs on U*
      2. Reaction step:   (U^{n+1}, V^{n+1}, Ta^{n+1}, Tai^{n+1})
                          = Reaction(U*, V^n, Ta^n, Tai^n, lap, dt)

    The Laplacian computed in step 1 is reused in step 2 for the
    metabolic autoregulation calculation.

    Note: This is strictly first-order in time.  Higher-order splitting
    schemes (Strang) are not used here because the fixed dt = 0.005 ms
    makes the temporal error negligible relative to the spatial
    discretisation.
    """
    # --- Step 1: Diffusion ---
    U = apply_no_flux_bc(U)

    lap = compute_laplacian_3d_aniso(
        U, D11, D22, D33, D12, dx, dy, dz,
        U.shape[0], U.shape[1], U.shape[2])

    U_star = U + dt * lap
    U_star = apply_no_flux_bc(U_star)

    # --- Step 2: Reaction ---
    U_new, V_new, Ta_new, Tai_new = physics_reaction_step(
        U_star, V, Ta, Tai, lap, dt)

    return U_new, V_new, Ta_new, Tai_new


# =========================================================================
# 5. ENERGY AUDIT
# =========================================================================
@njit(fastmath=True, parallel=True)
def compute_energy_flux_3d(U, V, Ta, Tai, lap, dx, dy, dz):
    """
    Compute global energy metrics for the thermodynamic audit.

    Returns
    -------
    H_elec_sum  : float — total stored energy Σ (0.5*u² + 0.5*Ta²)*dV
    P_diff_sum  : float — total diffusive dissipation Σ (-u * lap)*dV
    P_mech_sum  : float — total mechanical viscous dissipation Σ (c0*Tai²)*dV
    """
    dV = dx * dy * dz
    c0 = 0.05   # Must match physics_reaction_step

    H_elec_sum = 0.0
    P_diff_sum = 0.0
    P_mech_sum = 0.0

    U_flat   = U.ravel()
    Ta_flat  = Ta.ravel()
    Tai_flat = Tai.ravel()
    lap_flat = lap.ravel()

    for i in prange(U.size):
        u_val   = U_flat[i]
        tai_val = Tai_flat[i]

        # Stored energy: electrical (0.5*u²) + mechanical (0.5*Ta²)
        H_elec_sum += (0.5 * u_val**2 + 0.5 * Ta_flat[i]**2) * dV

        # Diffusive dissipation: -u * (divergence of diffusive flux)
        P_diff_sum += -(u_val * lap_flat[i]) * dV

        # Mechanical viscous dissipation: c0 * Tai²
        P_mech_sum += (c0 * tai_val**2) * dV

    return H_elec_sum, P_diff_sum, P_mech_sum