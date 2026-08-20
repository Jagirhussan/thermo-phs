"""
Figure1.py — Validation of the Hypergraph Laplacian in a 3-Cell Cluster
========================================================================

Paper reference
---------------
  Figure 1: "Validation of the hypergraph Laplacian based electromechanical
  coupling in a 3-cell cluster."
  Sections: Results § "The hypergraph representation does not impact dynamics"
            Methods § "Modelling of a 3-cell cluster"

Purpose
-------
  Compares two coupling topologies for a 3-cell electromechanical cluster:
    (a) Standard solver — dense pairwise graph Laplacian, O(N²)
    (b) hypLap solver  — hypergraph Laplacian via trace variable λ(t), O(N)

  Demonstrates that static condensation preserves dynamics to within
  integrator tolerance while reducing algorithmic complexity.

Output
------
  Six individual PDF panels saved to figures/:
    panel_a.pdf — Voltage u(t) with zoom inset (follower plateau)
    panel_b.pdf — Absolute error |u_standard − u_hypLap|
    panel_c.pdf — Lagrange multiplier λ(t) vs. arithmetic mean of u
    panel_d.pdf — Cleft potential (mixing zone) visualisation
    panel_e.pdf — Active tension T_a(t) transients
    panel_f.pdf — Electromechanical phase space u(t) vs T_a(t)

Models used
-----------
  - Aliev-Panfilov (1996) 2-variable cardiac electrophysiology
  - Nash-Panfilov / Silva et al. active mechanics (k_u sigmoid activation)
  - No explicit gyrator back-EMF (this is the pre-PHS Nash-Panfilov model)
  - Hypergraph Laplacian derived via trace-variable static condensation
    (see Methods § "Derivation of the hypergraph Laplacian via trace variables")

  All parameter values are drawn from PARAMS_3CELL in phs_models.py.

Dependencies
------------
  numpy, scipy, matplotlib, phs_models (local module)
  No external data files; all results are computed at runtime.

Author
------
  Jagir R. Hussan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from phs_models import PARAMS_3CELL

# =========================================================================
# 0. PUBLICATION STYLE CONFIGURATION
# =========================================================================
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.autolayout': True
})

# Colour palette — matches the TikZ visual legend in main.tex Figure 1
COLORS = {
    'cell1': '#E69F00',   # Orange — stimulated cell
    'cell2': '#0072B2',   # Blue   — follower 1
    'cell3': '#009E73',   # Green  — follower 2 (symmetrical to cell 2)
    'trace': '#D55E00',   # Vermilion — Lagrange multiplier λ(t)
    'gray':  '#999999',   # Grey — standard-solver overlay
    'overlay': '#222222',
}

# Output directory for figure panels
OUTPUT_DIR = 'figures'

# =========================================================================
# 1. PHYSICS — 3-Cell Electromechanical Model
# =========================================================================
class CardiacPhysics:
    """
    Encapsulates the Aliev-Panfilov electrophysiology and Nash-Panfilov
    active mechanics for a single cell.  Parameters are drawn from
    PARAMS_3CELL in phs_models.py.

    Parameters (see phs_models.py §1–§2 for derivations)
    ----------------------------------------------------
    k, a       : Aliev-Panfilov reaction rate and excitation threshold
    eps, mu1,
    mu2        : APD restitution parameters
    c0         : Cross-bridge cycling rate (calcium-activation coupling)
    G_cleft    : Cleft conductance for extracellular mixing
    s_sig      : Sigmoid steepness for activation
    u_sig      : Sigmoid midpoint for activation
    k_visc     : Passive tension damping rate
    """

    # Drawn from PARAMS_3CELL (single source of truth)
    k       = PARAMS_3CELL['k']
    a        = PARAMS_3CELL['a']
    eps      = PARAMS_3CELL['eps']
    mu1      = PARAMS_3CELL['mu1']
    mu2      = PARAMS_3CELL['mu2']
    c0       = PARAMS_3CELL['c0']
    G_cleft  = PARAMS_3CELL['G_cleft']
    s_sig    = PARAMS_3CELL['s_sig']
    u_sig    = PARAMS_3CELL['u_sig']
    k_visc   = PARAMS_3CELL['k_visc']

    @staticmethod
    def reaction_ode(u, v):
        """
        Aliev-Panfilov fast reaction term (AP.1).

        du_reac = k * u * (u - a) * (1 - u) - u * v
        """
        k, a = CardiacPhysics.k, CardiacPhysics.a
        return k * u * (u - a) * (1.0 - u) - u * v

    @staticmethod
    def recovery_ode(u, v):
        """
        Aliev-Panfilov slow recovery dynamics (AP.2).

        dv/dt = (eps + mu1*v/(mu2+u)) * (-v - k*u*(u - a - 1))
        """
        k, a    = CardiacPhysics.k, CardiacPhysics.a
        eps, mu1, mu2 = CardiacPhysics.eps, CardiacPhysics.mu1, CardiacPhysics.mu2
        factor = eps + (mu1 * v) / (mu2 + u)
        return factor * (-v - k * u * (u - a - 1.0))

    @staticmethod
    def mechanics_ode(u, Ta, Tai):
        """
        Nash-Panfilov active mechanics (NM.1–NM.3).

        Uses the original formulation WITHOUT explicit gyrator back-EMF.
        The mechanical domain is driven by the electrical sigmoid but does
        not feed back onto u (unlike Figures 3, 4, and S3).
        """
        c0, u_sig, s_sig = CardiacPhysics.c0, CardiacPhysics.u_sig, CardiacPhysics.s_sig
        k_u = 1.0 / (1.0 + np.exp(-s_sig * (u - u_sig)))
        dTai_dt = c0 * (k_u - Tai)
        dTa_dt  = CardiacPhysics.k_visc * (Tai - Ta)
        return dTa_dt, dTai_dt


# =========================================================================
# 2. SOLVER IMPLEMENTATIONS
# =========================================================================

def standard_dynamics(t, x):
    """
    Standard dense-matrix solver using all-to-all pairwise coupling.
    State vector (12 elements): [u(3), v(3), Ta(3), Tai(3)]
    """
    N = 3
    u, v    = x[0:N], x[N:2*N]
    Ta, Tai = x[2*N:3*N], x[3*N:4*N]

    I_stim = np.zeros(N)
    if 10.0 <= t <= 12.0:
        I_stim[0] = 5.0

    # Dense pairwise graph Laplacian (all-to-all, O(N²))
    L_full = np.array([[2, -1, -1],
                       [-1,  2, -1],
                       [-1, -1,  2]], dtype=float) * (1.0 / 3.0)
    I_coupling = CardiacPhysics.G_cleft * (L_full @ u)

    du        = CardiacPhysics.reaction_ode(u, v) - I_coupling + I_stim
    dv        = CardiacPhysics.recovery_ode(u, v)
    dTa, dTai = CardiacPhysics.mechanics_ode(u, Ta, Tai)

    return np.concatenate([du, dv, dTa, dTai])


def hybrid_dynamics(t, x):
    """
    Hypergraph Laplacian (hypLap) solver using trace-variable coupling.

    Each cell couples only to the shared extracellular cleft λ(t) = mean(u),
    collapsing O(N²) into O(N) via static condensation.
    """
    N = 3
    u, v    = x[0:N], x[N:2*N]
    Ta, Tai = x[2*N:3*N], x[3*N:4*N]

    I_stim = np.zeros(N)
    if 10.0 <= t <= 12.0:
        I_stim[0] = 5.0

    # Hypergraph trace variable — uniform conductance → arithmetic mean
    lambda_trace = np.mean(u)
    I_coupling = CardiacPhysics.G_cleft * (u - lambda_trace)

    du        = CardiacPhysics.reaction_ode(u, v) - I_coupling + I_stim
    dv        = CardiacPhysics.recovery_ode(u, v)
    dTa, dTai = CardiacPhysics.mechanics_ode(u, Ta, Tai)

    return np.concatenate([du, dv, dTa, dTai])


# =========================================================================
# 3. SIMULATION EXECUTION
# =========================================================================

T_SPAN = (0.0, 150.0)
T_EVAL = np.linspace(0.0, 150.0, 1000)
X0 = np.zeros(12)

print("Integrating standard (pairwise) solver...")
sol_standard = solve_ivp(
    standard_dynamics, T_SPAN, X0,
    t_eval=T_EVAL, method='Radau', rtol=1e-6
)

print("Integrating hybrid (hypergraph) solver...")
sol_hybrid = solve_ivp(
    hybrid_dynamics, T_SPAN, X0,
    t_eval=T_EVAL, method='Radau', rtol=1e-6
)

# Post-process
trace_standard = np.mean(sol_standard.y[0:3, :], axis=0)
trace_hybrid   = np.mean(sol_hybrid.y[0:3, :], axis=0)
u_hybrid       = sol_hybrid.y[0:3, :]
error_u        = np.abs(sol_standard.y[0:3, :] - sol_hybrid.y[0:3, :])


# =========================================================================
# 4. FIGURE PANEL GENERATION
# =========================================================================

FIG_SIZE = (3.2, 2.4)

# --- Panel (a): Voltage u(t) with zoom inset ---
fig, ax = plt.subplots(figsize=FIG_SIZE)
for i in range(3):
    ax.plot(sol_standard.t, sol_standard.y[i],
            color=COLORS['gray'], lw=4, alpha=0.3, zorder=1)
ax.plot(sol_hybrid.t, sol_hybrid.y[0], color=COLORS['cell1'], lw=2, zorder=3)
ax.plot(sol_hybrid.t, sol_hybrid.y[1], color=COLORS['cell2'], lw=3.5, zorder=4)
ax.plot(sol_hybrid.t, sol_hybrid.y[2], color=COLORS['cell3'], lw=2, ls=':', zorder=5)
ax.set_ylabel(r'Voltage $u(t)$')
ax.set_xlabel('Time (ms)')

axins = ax.inset_axes([0.55, 0.45, 0.42, 0.5])
for i in range(3):
    axins.plot(sol_standard.t, sol_standard.y[i],
               color=COLORS['gray'], lw=4, alpha=0.3, zorder=1)
axins.plot(sol_hybrid.t, sol_hybrid.y[0], color=COLORS['cell1'], lw=2, zorder=3)
axins.plot(sol_hybrid.t, sol_hybrid.y[1], color=COLORS['cell2'], lw=3.5, zorder=4)
axins.plot(sol_hybrid.t, sol_hybrid.y[2], color=COLORS['cell3'], lw=2, ls=':', zorder=5)
axins.set_xlim(18.0, 22.0)
axins.set_ylim(0.95, 1.01)
axins.tick_params(labelsize=8)
axins.set_xticks([18, 20, 22])
ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.7)
fig.savefig(f'{OUTPUT_DIR}/panel_a.pdf', bbox_inches='tight', transparent=True)
plt.close(fig)
print("  Saved figures/panel_a.pdf")

# --- Panel (b): Absolute error (log scale) ---
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.semilogy(sol_standard.t, error_u[0] + 1e-16, color=COLORS['cell1'], lw=1.5)
ax.semilogy(sol_standard.t, error_u[1] + 1e-16, color=COLORS['cell2'], lw=2.5)
ax.semilogy(sol_standard.t, error_u[2] + 1e-16, color=COLORS['cell3'], lw=2.5, ls=':')
ax.set_ylabel(r'Error(t) $|u_{dense} - u_{hyb}|$')
ax.set_xlabel('Time (ms)')
ax.set_ylim(1e-9, 1e-1)
fig.savefig(f'{OUTPUT_DIR}/panel_b.pdf', bbox_inches='tight', transparent=True)
plt.close(fig)
print("  Saved figures/panel_b.pdf")

# --- Panel (c): λ(t) vs arithmetic mean ---
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.plot(sol_standard.t, trace_standard, color=COLORS['cell2'], lw=4, alpha=0.3)
ax.plot(sol_hybrid.t, trace_hybrid, color=COLORS['trace'], lw=2, ls='--')
ax.set_ylabel('Mean potential(t)')
ax.set_xlabel('Time (ms)')
fig.savefig(f'{OUTPUT_DIR}/panel_c.pdf', bbox_inches='tight', transparent=True)
plt.close(fig)
print("  Saved figures/panel_c.pdf")

# --- Panel (d): Cleft potential / mixing zone ---
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.plot(sol_hybrid.t, u_hybrid[0], color=COLORS['cell1'], lw=1.5, alpha=0.5, zorder=2)
ax.plot(sol_hybrid.t, u_hybrid[1], color=COLORS['cell2'], lw=2.5, alpha=0.5, zorder=2)
ax.plot(sol_hybrid.t, u_hybrid[2], color=COLORS['cell3'], lw=1.5, ls=':', alpha=0.8, zorder=3)
ax.plot(sol_hybrid.t, trace_hybrid, color=COLORS['trace'], lw=3, ls='--', zorder=10)
ax.fill_between(sol_hybrid.t, 0, trace_hybrid, color=COLORS['trace'], alpha=0.15, zorder=1)
ax.set_ylabel(r'Cleft potential $\lambda(t)$')
ax.set_xlabel('Time (ms)')
fig.savefig(f'{OUTPUT_DIR}/panel_d.pdf', bbox_inches='tight', transparent=True)
plt.close(fig)
print("  Saved figures/panel_d.pdf")

# --- Panel (e): Active tension T_a(t) ---
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.plot(sol_hybrid.t, sol_hybrid.y[6], color=COLORS['cell1'], lw=2)
ax.plot(sol_hybrid.t, sol_hybrid.y[7], color=COLORS['cell2'], lw=2)
ax.plot(sol_hybrid.t, sol_hybrid.y[8], color=COLORS['cell3'], lw=2)
ax.set_ylabel(r'Tension $T_a(t)$')
ax.set_xlabel('Time (ms)')
fig.savefig(f'{OUTPUT_DIR}/panel_e.pdf', bbox_inches='tight', transparent=True)
plt.close(fig)
print("  Saved figures/panel_e.pdf")

# --- Panel (f): Phase space u(t) vs T_a(t) ---
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.plot(sol_hybrid.y[0], sol_hybrid.y[6], color=COLORS['cell1'], lw=2, alpha=0.9)
ax.plot(sol_hybrid.y[1], sol_hybrid.y[7], color=COLORS['cell2'], lw=2, alpha=0.9)
ax.plot(sol_hybrid.y[1], sol_hybrid.y[8], color=COLORS['cell3'], lw=2, alpha=0.9)
ax.set_xlabel(r'Voltage $u(t)$')
ax.set_ylabel(r'Tension $T_a(t)$')
fig.savefig(f'{OUTPUT_DIR}/panel_f.pdf', bbox_inches='tight', transparent=True)
plt.close(fig)
print("  Saved figures/panel_f.pdf")

print("\nDone. Exported panel_a.pdf through panel_f.pdf to figures/")