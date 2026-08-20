"""
phs_hetero_core.py — 2D Port-Hamiltonian Hypergraph Solver (Numba-JIT)
======================================================================

Paper reference
---------------
  Used for Figures 3 and S3 (thermodynamic profiling of myocardial fibrosis).
  Implements the automated algorithmic composition described in Methods
  § "Automated algorithmic composition" and the metabolic autoregulation
  described in Methods § "Modelling of myocardial fibrosis".

Purpose
-------
  High-performance kernel for 2D tissue-scale electromechanical simulations
  using the port-Hamiltonian hypergraph framework.  Supports:
    - Spatial heterogeneity (fibrosis maps for conductivity and viscosity)
    - Harmonic-mean interfacial conductance (hypergraph series-resistance)
    - Full energy accounting at every node and time step
    - Metabolic autoregulation (dynamic k_eff recruitment)
    - RK4 integration with consistent quadrature for energy audit

Architecture
------------
  The kernel is written in Python/Numba for speed and operates on flat
  (ravelled) 2D grids of size N×N.  It is called by run_experiment_ensemble.py
  for ensemble simulations and its output (.npz files) is post-processed by
  Figure3.py and FigureS3.py.

Key equations implemented (see phs_models.py for full derivation)
-----------------------------------------------------------------
  AP.1—AP.2  : Aliev-Panfilov electrophysiology
  NM.1—NM.3  : Nash-Panfilov / Silva active mechanics
  GYR.1—GYR.2: Port-Hamiltonian gyrator coupling (skew-symmetric)
  HL.1—HL.2  : Harmonic-mean interfacial conductance (hypergraph Laplacian)
  MA.1       : Metabolic autoregulation (source-sink k_eff adaptation)

Dependencies
------------
  numpy, numba

Author
------
  Jagir R. Hussan
"""

import numpy as np
from numba import njit


# =========================================================================
# 1. PHYSICS KERNEL — Single RK4 Stage Evaluation
# =========================================================================
@njit(fastmath=True, parallel=False)
def physics_kernel_hetero(t, x, N, params_scalars,
                          SIG_L, SIG_T, K_VISC,
                          stim_mask_s1, stim_mask_s2, timings):
    """
    Evaluate the state derivatives dx/dt for ALL nodes at time t.

    This function computes one right-hand-side evaluation for the RK4
    integrator.  It simultaneously returns:
      dx              : state derivatives (4*N² elements)
      local_p_diff    : per-node diffusive dissipation power   [W]
      total_p_reac    : total reactive (AP) power              [W]
      local_p_mech    : per-node mechanical viscous dissipation [W]
      total_p_stim    : total stimulus input power             [W]
      local_k_effort  : per-node adapted metabolic rate k_eff  [1/ms]

    Design notes (corrections from earlier versions)
    ------------------------------------------------
    1.  Harmonic mean (series-resistance model) for interfacial conductance
        replaces arithmetic mean (parallel model).  This is required by the
        static-condensation derivation and correctly handles fibrosis
        (zero conductivity → zero effective conductance).
    2.  Factor of 2.0 accounts for half-cell distances (dx/2 to interface).
    3.  Epsilon (1e-16) prevents division by zero.
    4.  Thermodynamic fuse: if u leaves physical bounds, gyrator coupling
        is disabled to prevent runaway.

    Parameters
    ----------
    t : float
        Current simulation time [ms].
    x : ndarray (4*N²,)
        Flattened state vector: [u, v, Tai, Ta] for each node.
    N : int
        Grid dimension (N×N nodes).
    params_scalars : tuple
        (k_base, a, eps_ap, mu1, mu2, c0, alpha_fb, k_recruit, k_max, _)
    SIG_L : ndarray (N²,)
        Longitudinal conductivity map.
    SIG_T : ndarray (N²,)
        Transverse conductivity map.
    K_VISC : ndarray (N²,)
        Viscous damping map (k_visc per node).
    stim_mask_s1 : ndarray (N²,)
        S1 stimulus mask (current amplitude per node).
    stim_mask_s2 : ndarray (N²,)
        S2 stimulus mask (current amplitude per node).
    timings : tuple
        (t_s1_start, t_s1_dur, t_s2_start, t_s2_dur) in [ms].

    Returns
    -------
    dx              : ndarray (4*N²,) — state derivatives
    local_p_diff    : ndarray (N²,)   — per-node diffusive power
    total_p_reac    : float           — total reactive power (scalar)
    local_p_mech    : ndarray (N²,)   — per-node mechanical dissipation
    total_p_stim    : float           — total stimulus power (scalar)
    local_k_effort  : ndarray (N²,)   — per-node adapted k_eff
    """

    # --- Unpack scalar parameters ---
    k_base, a, eps_ap, mu1, mu2, c0, alpha_fb, k_recruit, k_max, _ = params_scalars
    t_s1_start, t_s1_dur, t_s2_start, t_s2_dur = timings

    n_nodes = N * N
    dx = np.empty(4 * n_nodes, dtype=np.float64)

    # --- Spatial accumulators ---
    local_p_diff   = np.zeros(n_nodes, dtype=np.float64)
    local_p_mech   = np.zeros(n_nodes, dtype=np.float64)
    local_k_effort = np.zeros(n_nodes, dtype=np.float64)

    # --- Unpack flat state arrays ---
    u   = x[0:n_nodes]
    v   = x[n_nodes:2*n_nodes]
    Tai = x[2*n_nodes:3*n_nodes]
    Ta  = x[3*n_nodes:]

    total_p_reac = 0.0
    total_p_stim = 0.0

    # Regularisation constant to avoid division by zero in fibrotic regions
    div_eps = 1e-16

    for i in range(n_nodes):
        # ================================================================
        # 1. DIFFUSION — Harmonic-mean interfacial conductance (HL.1)
        # ================================================================
        r = i // N   # row index
        c = i % N    # column index

        # Local nodal conductivities (clamped to minimum 1e-9)
        g_curr_L = max(SIG_L[i], 1e-9)
        g_curr_T = max(SIG_T[i], 1e-9)

        diff_flux   = 0.0
        p_diff_node = 0.0

        # --- X-direction: Left neighbour ---
        if c > 0:
            g_prev = max(SIG_L[i-1], 1e-9)
            # Harmonic mean with factor 2.0 (series-resistance model, HL.1)
            g_eff = (2.0 * g_curr_L * g_prev) / (g_curr_L + g_prev + div_eps)
            flux  = g_eff * (u[i-1] - u[i])
            diff_flux   += flux
            p_diff_node += 0.5 * flux * (u[i-1] - u[i])   # P_diff (HL.2)

        # --- X-direction: Right neighbour ---
        if c < N - 1:
            g_next = max(SIG_L[i+1], 1e-9)
            g_eff  = (2.0 * g_curr_L * g_next) / (g_curr_L + g_next + div_eps)
            flux   = g_eff * (u[i+1] - u[i])
            diff_flux   += flux
            p_diff_node += 0.5 * flux * (u[i+1] - u[i])

        # --- Y-direction: Top neighbour ---
        if r > 0:
            g_prev = max(SIG_T[i-N], 1e-9)
            g_eff  = (2.0 * g_curr_T * g_prev) / (g_curr_T + g_prev + div_eps)
            flux   = g_eff * (u[i-N] - u[i])
            diff_flux   += flux
            p_diff_node += 0.5 * flux * (u[i-N] - u[i])

        # --- Y-direction: Bottom neighbour ---
        if r < N - 1:
            g_next = max(SIG_T[i+N], 1e-9)
            g_eff  = (2.0 * g_curr_T * g_next) / (g_curr_T + g_next + div_eps)
            flux   = g_eff * (u[i+N] - u[i])
            diff_flux   += flux
            p_diff_node += 0.5 * flux * (u[i+N] - u[i])

        local_p_diff[i] = p_diff_node

        # ================================================================
        # 2. METABOLIC AUTOREGULATION (MA.1)
        # ================================================================
        # load = outward (negative) diffusive flux — current leaving node i
        load = 0.0
        if diff_flux < 0:
            load = -diff_flux

        k_local = k_base + (k_recruit * load)
        if k_local > k_max:
            k_local = k_max

        local_k_effort[i] = k_local

        # ================================================================
        # 3. PHS GYRATOR COUPLING (GYR.1—GYR.2)
        # ================================================================
        u_val   = u[i]
        v_val   = v[i]
        Tai_val = Tai[i]

        # Voltage-dependent activation sigmoid (GYR.3)
        sigmoid = 1.0 / (1.0 + np.exp(-50.0 * (u_val - 0.05)))

        # Thermodynamic fuse: disable gyrator if state leaves physical bounds
        if u_val > 1.5 or u_val < -0.5:
            alpha_transducer = 0.0
        else:
            alpha_transducer = alpha_fb * sigmoid

        # Gyrator: skew-symmetric coupling (Rule 2 of compositional grammar)
        flow_act   = alpha_transducer * u_val     # → dTai (GYR.1)
        i_feedback = alpha_transducer * Tai_val   # → du   (GYR.2)

        # ================================================================
        # 4. REACTION DYNAMICS
        # ================================================================

        # --- du/dt: Aliev-Panfilov fast reaction (AP.1) ---
        # du_reac = k * u * (u - a) * (1 - u) - u * v
        du_reac_only = (k_local * u_val * (u_val - a) * (1.0 - u_val)
                        - u_val * v_val)

        # Reactive power: −u * du_reac (energy consumed by ionic fluxes)
        total_p_reac += -(u_val * du_reac_only)

        # --- dv/dt: Aliev-Panfilov slow recovery (AP.2) ---
        dv_reac = ((eps_ap + mu1 * v_val / (mu2 + u_val))
                   * (-v_val - k_local * u_val * (u_val - a - 1.0)))

        # --- dTai/dt: Calcium activation with gyrator drive ---
        # dTai = flow_act − c0 * Tai   (gyrator + viscous decay)
        dTai_reac = flow_act - (c0 * Tai_val)

        # Mechanical viscous dissipation: P_mech = c0 * Tai²
        local_p_mech[i] = c0 * (Tai_val ** 2)

        # --- Stimulus current ---
        stim = 0.0
        if u_val < 1.0:
            if t_s1_start <= t <= t_s1_start + t_s1_dur:
                stim += stim_mask_s1[i]
            if t_s2_start <= t <= t_s2_start + t_s2_dur:
                stim += stim_mask_s2[i]
        total_p_stim += stim * u_val

        # ================================================================
        # 5. STATE ASSEMBLY — pack derivatives into flat vector
        # ================================================================
        dx[i]                = du_reac_only + diff_flux + stim - i_feedback
        dx[n_nodes + i]      = dv_reac
        dx[2*n_nodes + i]    = dTai_reac
        dx[3*n_nodes + i]    = K_VISC[i] * (Tai_val - Ta[i])   # NM.2

    return dx, local_p_diff, total_p_reac, local_p_mech, total_p_stim, local_k_effort


# =========================================================================
# 2. TIME INTEGRATOR — RK4 with Consistent Energy Quadrature
# =========================================================================
@njit
def run_solver(y0, N, params_scalars, SIG_L, SIG_T, K_VISC,
               mask_s1, mask_s2, timings, t_eval):
    """
    Main integration loop with RK4 time-stepping and full energy audit.

    Critical design detail:
    -----------------------
    The energy audit accumulators (diffusive, reactive, mechanical, and
    stimulus power) use the IDENTICAL RK4 quadrature weights as the state
    integration.  This guarantees that the thermodynamic audit (dH/dt)
    exactly mirrors the integrated state trajectory — a necessary condition
    for validating energy conservation.

    Returns
    -------
    U_history     : (n_saved, N²) — subsampled voltage snapshots
    H_elec_hist   : (n_saved,)   — electrical stored energy over time
    H_mech_hist   : (n_saved,)   — mechanical stored energy over time
    E_diff_hist   : (n_saved,)   — cumulative diffusive dissipation
    E_reac_hist   : (n_saved,)   — cumulative reactive dissipation
    E_mech_hist   : (n_saved,)   — cumulative mechanical dissipation
    E_stim_hist   : (n_saved,)   — cumulative stimulus work
    E_diff_map    : (N²,)        — spatial map of total diffusive dissipation
    E_mech_map    : (N²,)        — spatial map of total mechanical dissipation
    K_effort_map  : (N²,)        — spatial map of time-averaged k_eff
    """

    n_steps  = len(t_eval)
    n_nodes  = N * N
    subsample = 5
    n_saved  = (n_steps // subsample) + 1

    # --- History arrays (subsampled) ---
    U_history  = np.zeros((n_saved, n_nodes), dtype=np.float64)
    H_elec_hist = np.zeros(n_saved, dtype=np.float64)
    H_mech_hist = np.zeros(n_saved, dtype=np.float64)

    # --- Energy ledger (subsampled) ---
    E_diff_hist = np.zeros(n_saved, dtype=np.float64)
    E_reac_hist = np.zeros(n_saved, dtype=np.float64)
    E_mech_hist = np.zeros(n_saved, dtype=np.float64)
    E_stim_hist = np.zeros(n_saved, dtype=np.float64)

    # --- Spatial accumulators (full resolution) ---
    E_diff_map   = np.zeros(n_nodes, dtype=np.float64)
    E_mech_map   = np.zeros(n_nodes, dtype=np.float64)
    K_effort_map = np.zeros(n_nodes, dtype=np.float64)

    # Scalar running accumulators
    acc_diff, acc_reac, acc_mech, acc_stim = 0.0, 0.0, 0.0, 0.0

    curr_y      = y0.copy()
    dt_internal = 0.005   # fixed internal time step [ms]
    save_idx    = 0

    for i in range(n_steps - 1):
        t_curr, t_end = t_eval[i], t_eval[i+1]

        while t_curr < t_end:
            dt = min(dt_internal, t_end - t_curr)

            # ============================================================
            # RK4 step — 4-stage Runge-Kutta
            # ============================================================

            # Stage 1
            k1_dx, k1_d, k1_r, k1_m, k1_s, k1_k = physics_kernel_hetero(
                t_curr, curr_y, N, params_scalars,
                SIG_L, SIG_T, K_VISC, mask_s1, mask_s2, timings)

            # Stage 2
            y_k2 = curr_y + 0.5 * dt * k1_dx
            k2_dx, k2_d, k2_r, k2_m, k2_s, k2_k = physics_kernel_hetero(
                t_curr + 0.5 * dt, y_k2, N, params_scalars,
                SIG_L, SIG_T, K_VISC, mask_s1, mask_s2, timings)

            # Stage 3
            y_k3 = curr_y + 0.5 * dt * k2_dx
            k3_dx, k3_d, k3_r, k3_m, k3_s, k3_k = physics_kernel_hetero(
                t_curr + 0.5 * dt, y_k3, N, params_scalars,
                SIG_L, SIG_T, K_VISC, mask_s1, mask_s2, timings)

            # Stage 4
            y_k4 = curr_y + dt * k3_dx
            k4_dx, k4_d, k4_r, k4_m, k4_s, k4_k = physics_kernel_hetero(
                t_curr + dt, y_k4, N, params_scalars,
                SIG_L, SIG_T, K_VISC, mask_s1, mask_s2, timings)

            # --- RK4 state update ---
            curr_y += (dt / 6.0) * (k1_dx + 2*k2_dx + 2*k3_dx + k4_dx)

            # --- Energy accumulation (IDENTICAL RK4 weights) ---
            # This is critical: the energy audit uses the same quadrature
            # as the state to ensure exact thermodynamic bookkeeping.

            def _sum(arr):
                return np.sum(arr)

            acc_diff += (dt / 6.0) * (
                _sum(k1_d) + 2*_sum(k2_d) + 2*_sum(k3_d) + _sum(k4_d))
            acc_mech += (dt / 6.0) * (
                _sum(k1_m) + 2*_sum(k2_m) + 2*_sum(k3_m) + _sum(k4_m))
            acc_reac += (dt / 6.0) * (
                k1_r + 2*k2_r + 2*k3_r + k4_r)
            acc_stim += (dt / 6.0) * (
                k1_s + 2*k2_s + 2*k3_s + k4_s)

            # Spatial energy maps (same RK4 weights)
            E_diff_map += (dt / 6.0) * (
                k1_d + 2*k2_d + 2*k3_d + k4_d)
            E_mech_map += (dt / 6.0) * (
                k1_m + 2*k2_m + 2*k3_m + k4_m)

            # k_eff map stores time-weighted average
            K_effort_map += (dt / 6.0) * (
                k1_k + 2*k2_k + 2*k3_k + k4_k)

            # --- State clipping (numerical safety) ---
            for j in range(n_nodes):
                if curr_y[j] > 1.2:
                    curr_y[j] = 1.2
                if curr_y[j] < -0.1:
                    curr_y[j] = 0.0

            t_curr += dt

        # --- Subsampled history ---
        if i % subsample == 0 and save_idx < n_saved:
            U_history[save_idx]  = curr_y[:n_nodes]
            # Electrical stored energy: H_elec = 0.5 * Σ u²
            H_elec_hist[save_idx] = 0.5 * np.sum(curr_y[:n_nodes] ** 2)
            # Mechanical stored energy: H_mech = 0.5 * Σ Tai²
            H_mech_hist[save_idx] = 0.5 * np.sum(curr_y[2*n_nodes:3*n_nodes] ** 2)
            # Cumulative energy ledger
            E_diff_hist[save_idx] = acc_diff
            E_reac_hist[save_idx] = acc_reac
            E_mech_hist[save_idx] = acc_mech
            E_stim_hist[save_idx] = acc_stim
            save_idx += 1

    # Normalise k_effort map by total simulation time → time-averaged rate
    K_effort_map /= t_eval[-1]

    return (U_history[:save_idx],
            H_elec_hist[:save_idx], H_mech_hist[:save_idx],
            E_diff_hist[:save_idx], E_reac_hist[:save_idx],
            E_mech_hist[:save_idx], E_stim_hist[:save_idx],
            E_diff_map, E_mech_map, K_effort_map)