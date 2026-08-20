"""
phs_models.py — Canonical Port-Hamiltonian Electromechanical Models
====================================================================

This module serves as the authoritative reference for all model equations,
parameter definitions, and their physical interpretations used across the
port-Hamiltonian hypergraph framework described in:

  Hussan, Omholt & Hunter, "Guaranteeing thermodynamically consistent
  multiphysics modeling by use of port-Hamiltonian hypergraphs" (2025).

All figure-generation scripts and simulation kernels in this repository
should draw their parameter defaults and equation definitions from this
module, or document any deliberate deviations with explicit justification.

====================================================================
TABLE OF CONTENTS
====================================================================
  1.  Aliev-Panfilov (AP) Electrophysiology Model
  2.  Nash-Panfilov / Silva Active Mechanics Model
  3.  Port-Hamiltonian Gyrator Coupling
  4.  Hypergraph Laplacian (Dissipative & Conservative)
  5.  Metabolic Autoregulation
  6.  Autonomic Control (Control by Interconnection)
  7.  Model Variants and Their Parameter Sets
====================================================================

====================================================================
1. ALIEV-PANFILOV ELECTROPHYSIOLOGY MODEL
====================================================================

The Aliev-Panfilov model [Aliev & Panfilov, 1996] is a phenomenological
two-variable reduction of cardiac excitation dynamics.  It captures the
essential non-linear behaviour of action-potential initiation, plateau,
and repolarisation while remaining computationally tractable for
tissue-scale simulations.

State variables
---------------
  u : dimensionless fast transmembrane potential     [-]
  v : dimensionless slow recovery variable            [-]

Governing equations
-------------------

  du/dt = -k * u * (u - a) * (1 - u) - u * v                (AP.1)

  dv/dt = (eps + mu1 * v / (mu2 + u)) * (-v - k * u * (u - a - 1))
                                                             (AP.2)

Parameter glossary
------------------
  k       : Reaction-rate scaling factor.  Controls the overall speed of
            excitation and recovery.                          [1/ms]
  a       : Excitation threshold.  The value of u above which the cubic
            reaction term becomes self-excitatory.            [-]
  eps     : Base recovery rate.  Small value governing the slow dynamics
            of repolarisation.                                [1/ms]
  mu1, mu2: Restitution parameters controlling the shape of the
            action-potential duration (APD) restitution curve.  [-]

Default parameter set (2D simulations, Figures 1, 3, 4, S1--S3)
---------------------------------------------------------------
  k   = 8.0      a    = 0.1      eps  = 0.01
  mu1 = 0.2      mu2  = 0.3

Note for 3D simulations (Figure 2, scaling_studies/)
------------------------------------------------------
  a   = 0.05     (lower threshold to sustain 3D scroll-wave re-entry
                   in larger, anisotropic domain — see parameters.md)

====================================================================
2. NASH-PANFILOV / SILVA ACTIVE MECHANICS MODEL
====================================================================

The active mechanics framework extends the Aliev-Panfilov model by
introducing an intermediate calcium-activation variable T_ai and a
macroscopic active stress variable T_a [Nash & Panfilov, 2004;
Silva et al., 2020].

State variables
---------------
  T_ai : internal calcium activation (dimensionless)          [-]
  T_a  : active mechanical tension                            [kPa]

Governing equations
-------------------

  dT_ai/dt = c0 * (k_u(u) - T_ai)                            (NM.1)

  dT_a/dt  = k_visc * (T_ai - T_a)                           (NM.2)

  k_u(u)   = 1 / (1 + exp(-s * (u - u_thr)))                 (NM.3)

Parameter glossary
------------------
  c0      : Cross-bridge cycling rate / calcium-activation coupling
            rate.  Also serves as mechanical viscous dissipation
            coefficient in the PHS dissipation matrix R.      [1/ms]
  k_visc  : Passive viscous damping rate for active tension relaxation.
                                                                 [1/ms]
  s       : Sigmoid steepness for voltage-dependent activation.   [-]
  u_thr   : Activation threshold in the sigmoid.                  [-]

Default parameter set (Figures 1, 3, S1--S3)
---------------------------------------------
  c0      = 1.0     (Figure 1)
  c0      = 0.05    (phs_hetero_core, Figures 3, S3)
  k_visc  = 0.05    (phs_hetero_core, Figures 3, S3)
  k_visc  = 0.1     (Figure 1 — equivalent to the 0.1 factor in dTa/dt)
  s       = 50.0
  u_thr   = 0.1     (Figures 1, S1)
  u_thr   = 0.05    (phs_hetero_core, Figures 3, 4, S3)

====================================================================
3. PORT-HAMILTONIAN GYRATOR COUPLING
====================================================================

The coupling between the electrical domain (u) and the mechanical
domain (T_ai) is implemented as a skew-symmetric gyrator (Rule 2 of
the compositional grammar).  This guarantees that the net power
transfer across the interface is identically zero.

Gyrator equations
-----------------

  flow_act   = alpha_fb * sigma(u) * u                       (GYR.1)
  I_feedback = alpha_fb * sigma(u) * T_ai                    (GYR.2)

  sigma(u)   = 1 / (1 + exp(-s_sig * (u - u_sig)))           (GYR.3)

where
  flow_act   : Activation flow driving T_ai (mechanical domain)
  I_feedback : Back-EMF current opposing u (electrical domain)
  alpha_fb   : Gyrator coupling strength                     [-]
  sigma(u)   : Voltage-dependent activation sigmoid
  s_sig      : Sigmoid steepness (typically 50 or 100)       [-]
  u_sig      : Sigmoid midpoint (typically 0.05)             [-]

Power balance
-------------
  P_net = I_feedback * u + flow_act * T_ai
        = alpha_fb * sigma(u) * (T_ai * u - u * T_ai) = 0

The skew-symmetry guarantees P_net ≡ 0 at machine precision.

Parameter glossary
------------------
  alpha_fb : Gyrator coupling strength.  Scales the bidirectional
             energy transduction between electrical and mechanical
             domains.                                          [-]
  s_sig    : Steepness of the activation sigmoid.  Higher values
             produce sharper threshold behaviour.              [-]
  u_sig    : Sigmoid midpoint (voltage at half-activation).    [-]

Default parameter set
---------------------
  alpha_fb = 0.05    (Figures 1, 3, S3, 3D scaling)
  alpha_fb = 0.15    (Figure 4 — higher coupling for autonomic control)
  s_sig    = 50.0    (phs_hetero_core, Figures 1, 3)
  s_sig    = 100.0   (Figure 4, Figure S1)
  u_sig    = 0.05    (phs_hetero_core, Figure 4)
  u_sig    = 0.1     (Figure 1)

====================================================================
4. HYPERGRAPH LAPLACIAN
====================================================================

4.1 Dissipative Hypergraph Laplacian (Electrical Diffusion)
-----------------------------------------------------------

For N cells coupled through a shared extracellular cleft represented
by a trace variable λ(t), the dissipation potential is:

  R*(u, λ) = 0.5 * Σ_i G_mix * (u_i - λ)²

Minimising with respect to λ yields the conductance-weighted average:

  λ = (Σ_i G_mix * u_i) / (Σ_i G_mix)

The resulting flux vector φ = ∇_u R* defines the hypergraph Laplacian:

  φ = L_hyper @ u

where L_hyper = G_diag - (g @ g^T) / (1^T @ g)

In the discrete (stencil) implementation, the effective conductance
between adjacent nodes i and j uses the **harmonic mean** (series-
resistance model) to correctly handle heterogeneous conductivity:

  g_eff = 2 * σ_i * σ_j / (σ_i + σ_j + ε)                     (HL.1)

with ε = 1e-16 to prevent division by zero in fully fibrotic regions.

The diffusive power dissipated at the interface is:

  P_diff(i→j) = 0.5 * g_eff * (u_j - u_i)²                   (HL.2)

4.2 Conservative Hypergraph Laplacian (Mechanical)
--------------------------------------------------

The same topological structure governs the conservative mechanical
domain, where the Laplacian enters the Hamiltonian as a quadratic
elastic potential rather than the dissipation matrix.

====================================================================
5. METABOLIC AUTOREGULATION
====================================================================

To prevent action-potential collapse in highly resistive (fibrotic)
regions, individual cells dynamically upregulate their effective
reaction rate k_eff in proportion to their instantaneous electrotonic
load (the diffusive flux lost to downstream neighbours):

  k_eff(i) = min(k_base + k_recruit * |φ_diff(i)|_neg, k_max) (MA.1)

where |φ_diff(i)|_neg is the magnitude of the negative (outward)
diffusive flux at node i.  This models the physiological phenomenon
of source-sink metabolic autoregulation [Myles et al., 2012].

Parameter glossary
------------------
  k_base    : Baseline metabolic reaction rate (resting k).     [1/ms]
  k_recruit : Metabolic recruitment gain.  Determines how aggressively
              k_eff increases with electrotonic load.           [1/ms]
  k_max     : Maximum (ceiling) reaction rate.  Prevents unbounded
              upregulation.                                     [1/ms]

Default parameter set (2D fibrosis, Figures 3, S3)
--------------------------------------------------
  k_base    = 8.0
  k_recruit = 5.0
  k_max     = 20.0

Note for 3D simulations (Figure 2, scaling_studies/)
------------------------------------------------------
  k_recruit = 2.0
  k_max     = 30.0
  (different values tuned for 3D anisotropic domain — see parameters.md)

====================================================================
6. AUTONOMIC CONTROL (Control by Interconnection)
====================================================================

The dual-mode neuro-controller (Figure 4) implements sympathetic
regulation strictly through power-conserving parameter modulation
rather than external state overwriting.

The inotropy gain G(t) is governed by a proportional-decay controller:

  dG/dt = k_gain_ctrl * (T_demand - T_peak) - k_decay * (G - G_min)
                                                              (AC.1)

The metabolic reaction rate k(t) is slaved to G(t) for dynamic
impedance matching:

  k(t) = k_rest + k_gain_metab * (G(t) - G_min)              (AC.2)

Parameter glossary
------------------
  G_min, G_max  : Bounds on the inotropy gain.                 [-]
  k_gain_ctrl   : Sympathetic recruitment rate.                [1/ms]
  k_decay_ctrl  : Enzymatic degradation rate of secondary messengers
                   (e.g., cAMP).                                [1/ms]
  T_demand      : Target peak tension setpoint.                [kPa]
  T_peak        : Actual measured peak tension.                [kPa]
  k_rest        : Baseline (resting) metabolic rate.           [1/ms]
  k_gain_metab  : Metabolic recruitment scaling factor.        [1/ms]

Default parameter set (Figure 4)
--------------------------------
  G_min = 0.5,  G_max = 5.0
  k_gain_ctrl   = 1.8
  k_decay_ctrl  = 0.12
  k_rest        = 8.0
  k_gain_metab  = 2.5
  T_demand_rest = 0.35
  T_demand_ex   = 0.95

====================================================================
7. MODEL VARIANTS AND THEIR PARAMETER SETS
====================================================================

  +------------------+--------+--------+------+------------+----------+---------+
  | Parameter        | Fig 1  | Fig S1 | Figs | Fig 4      | 3D Core  | Unit    |
  |                  |        |        | 3,S3 |            | (Fig 2)  |         |
  +------------------+--------+--------+------+------------+----------+---------+
  | k (AP rate)      | 8.0    | 0.0*   | 8.0  | 8.0        | 8.0      | 1/ms    |
  | a (AP thresh.)   | 0.1    | ---    | 0.1  | 0.1        | 0.05     | -       |
  | eps              | 0.01   | 0.01   | 0.01 | 0.01       | 0.01     | 1/ms    |
  | mu1, mu2         | 0.2,0.3| 0.2,0.3| 0.2,0.3|0.2,0.3| 0.2,0.3  | -       |
  | c0 (mech rate)   | 1.0    | ---    | 0.05 | 0.05       | 0.05     | 1/ms    |
  | k_visc (Ta damp) | 0.1    | ---    | 0.05 | 0.05       | 0.05     | 1/ms    |
  | alpha_fb (gyrat) | ---†   | 1.0*   | 0.05 | 0.15       | 0.05     | -       |
  | s_sig            | 50.0   | ---    | 50.0 | 100.0      | 50.0     | -       |
  | u_sig            | 0.1    | ---    | 0.05 | 0.05       | 0.05     | -       |
  | k_base (metab)   | ---    | ---    | 8.0  | ---        | 8.0      | 1/ms    |
  | k_recruit        | ---    | ---    | 5.0  | ---        | 2.0      | 1/ms    |
  | k_max            | ---    | ---    | 20.0 | ---        | 30.0     | 1/ms    |
  +------------------+--------+--------+------+------------+----------+---------+
  *  Figure S1 uses a linearised canonical transducer: k=0 removes AP kinetics;
     G_couple=1.0 (equivalent to alpha_fb) for pure gyrator validation.
  †  Figure 1 does not use explicit gyrator feedback; it uses the
     original Nash-Panfilov formulation without back-EMF.

====================================================================
REFERENCES
====================================================================

  Aliev, R. R. & Panfilov, A. V. (1996). A simple two-variable model
    of cardiac excitation. Chaos, Solitons & Fractals, 7(3), 293-301.
  Nash, M. P. & Panfilov, A. V. (2004). Electromechanical model of
    excitable tissue to study reentrant cardiac arrhythmias. Progress
    in Biophysics and Molecular Biology, 85(2-3), 501-522.
  Silva, J. R. et al. (2020). Simplified electromechanical model of
    cardiac tissue. (see main.tex references)
  van der Schaft, A. & Jeltsema, D. (2014). Port-Hamiltonian Systems
    Theory: An Introductory Overview. Foundations and Trends in
    Systems and Control, 1(2-3), 173-378.
  Myles, R. C. et al. (2012). Local source-sink mismatch as a
    mechanism of arrhythmogenesis. (see main.tex references)
  Brugnoli, A. et al. (2025). Finite element port-Hamiltonian systems.
    (see main.tex references)
"""

# =========================================================================
# PARAMETER DICTIONARIES — canonical defaults for each model variant
# =========================================================================

# --- 2D Simulations: Figures 1, 3, S2, S3 ---
PARAMS_2D_FIBROSIS = {
    'k':          8.0,    # Aliev-Panfilov reaction rate          [1/ms]
    'a':          0.1,    # Excitation threshold                  [-]
    'eps':        0.01,   # Base recovery rate                    [1/ms]
    'mu1':        0.2,    # APD restitution parameter             [-]
    'mu2':        0.3,    # APD restitution parameter             [-]
    'c0':         0.05,   # Cross-bridge cycling / calcium rate   [1/ms]
    'alpha_fb':   0.05,   # Gyrator coupling strength             [-]
    'k_visc':     0.05,   # Passive tension damping rate          [1/ms]
    'k_base':     8.0,    # Baseline metabolic reaction rate      [1/ms]
    'k_recruit':  5.0,    # Metabolic recruitment gain            [1/ms]
    'k_max':      20.0,   # Metabolic ceiling                    [1/ms]
    's_sig':      50.0,   # Activation sigmoid steepness          [-]
    'u_sig':      0.05,   # Activation sigmoid midpoint           [-]
}

# --- 3D Scaling Simulations: Figure 2 ---
PARAMS_3D_SCALING = {
    'k':          8.0,
    'a':          0.05,   # Lower threshold for 3D scroll-wave stability
    'eps':        0.01,
    'mu1':        0.2,
    'mu2':        0.3,
    'c0':         0.05,
    'alpha_fb':   0.05,
    'k_visc':     0.05,
    'k_base':     8.0,
    'k_recruit':  2.0,    # Lower recruitment for 3D domain
    'k_max':      30.0,   # Higher ceiling for 3D domain
    's_sig':      50.0,
    'u_sig':      0.05,
}

# --- Autonomic Control: Figure 4 ---
PARAMS_AUTONOMIC = {
    'k':          8.0,
    'a':          0.1,
    'eps':        0.01,
    'mu1':        0.2,
    'mu2':        0.3,
    'c0':         0.05,
    'alpha_fb':   0.05,   
    'k_visc':     0.05,
    's_sig':      100.0,
    'u_sig':      0.05,
    # Autonomic controller parameters
    'G_min':      0.5,
    'G_max':      5.0,
    'k_gain_ctrl':1.8,
    'k_decay_ctrl': 0.12,
    'k_rest':     8.0,
    'k_gain_metab': 2.5,
    'T_demand_rest': 0.35,
    'T_demand_ex':   0.95,
}

# --- 3-Cell Cluster: Figure 1 ---
PARAMS_3CELL = {
    'k':          8.0,
    'a':          0.1,
    'eps':        0.01,
    'mu1':        0.2,
    'mu2':        0.3,
    'c0':         1.0,    # Faster mechanical coupling for small cluster
    'G_cleft':    1.0,    # Cleft conductance (mixing)
    's_sig':      50.0,
    'u_sig':      0.1,
    'k_visc':     0.1,    # Tension relaxation rate
}

# --- Linearised Canonical Transducer: Figure S1 ---
PARAMS_FCU_VALIDATION = {
    'C_m':        1.0,    # Membrane capacitance (normalised)      [-]
    'C_act':      2.0,    # Activation capacitance (normalised)    [-]
    'R_leak':     0.05,   # Leak resistance                       [-]
    'R_mech':     0.05,   # Mechanical damping resistance          [-]
    'G_couple':   1.0,    # Gyrator coupling constant              [-]
    'k':          0.0,    # AP kinetics disabled for pure gyrator audit
    'eps':        0.01,
    'mu1':        0.2,
    'mu2':        0.3,
}