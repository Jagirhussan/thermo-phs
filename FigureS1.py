"""
FigureS1.py — Internal Consistency of the Functional Cell Unit (FCU)
=====================================================================

Paper reference
---------------
  Supplementary Figure S1: Validation of the port-Hamiltonian gyrator
  coupling and energy conservation for a single Functional Cell Unit.

Purpose
-------
  Validates that the electromechanical coupling implemented via a
  skew-symmetric gyrator (Rule 2 of the compositional grammar) is
  perfectly lossless — the energy residual dH/dt − P_in + P_diss
  remains at machine precision throughout a transient.

  Uses the *linearised canonical transducer*: a stripped-down FCU
  where all non-linear AP kinetics are disabled (k = 0) so that the
  core Dirac structure (J matrix) can be audited in isolation.

Output
------
  figures/S1.pdf — two-panel figure:
    (a) State trajectories: electrical potential u(t) and mechanical
        activation T_ai(t) driven by a 0.2 ms impulse.
    (b) Thermodynamic audit: total energy H(t), external work input
        P_in, dissipation rate P_diss, and the coupling residual.

Models used
-----------
  Linearised canonical transducer (PHS with k = 0, constant R_leak,
  R_mech, and G_couple).  See PARAMS_FCU_VALIDATION in phs_models.py.

Dependencies
------------
  numpy, scipy, matplotlib

Author
------
  Jagir R. Hussan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp

from phs_models import PARAMS_FCU_VALIDATION

# =========================================================================
# 0. PUBLICATION STYLE
# =========================================================================
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 14,
    'axes.linewidth': 1.2,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'legend.frameon': False,
})

# Colour palette
COLORS = {
    'u':      '#000000',   # Black  — electrical potential
    'Tai':    '#D55E00',   # Vermilion — mechanical activation
    'Energy': '#0072B2',   # Blue   — total energy H
    'Work':   '#009E73',   # Green  — external work input
    'Diss':   '#E69F00',   # Orange — dissipation rate
    'Resid':  '#CC79A7',   # Purple — energy residual
}


def style_axis(ax, panel_label=None):
    """Apply consistent panel styling: no top/right spines, optional label."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if panel_label:
        ax.text(-0.05, 1.05, panel_label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='bottom', ha='right')


# =========================================================================
# 1. PHYSICS — 4-Variable Linearised Canonical Transducer
# =========================================================================
def run_fcu_validation():
    """
    Simulate the linearised canonical transducer and perform a
    full First-Law audit.

    State vector:  x = [u, v, T_ai, T_a]
      u    : electrical potential (fast)
      v    : recovery variable (disabled with k = 0)
      T_ai : calcium activation
      T_a  : active tension

    The PHS structure (see Methods § "The port-Hamiltonian integration"):

      dx/dt = [J - R] @ ∇H + B @ u_stim

      J = [[0,   0, -G_couple/C_act,  0],
           [0,   0,              0,   0],
           [G_couple/C_m,       0,   0,  0],
           [0,            0,       0,   0]]

      R = diag(R_leak, 0, R_mech, 0)

      H(x) = 0.5 * u² / C_m  +  0.5 * T_ai² / C_act

    Because J is skew-symmetric and R is positive semi-definite,
    the First Law dictates:  dH/dt = P_in − P_diss
    with residual ≡ 0 to machine precision.
    """

    # --- Parameters drawn from PARAMS_FCU_VALIDATION in phs_models.py ---
    C_m      = PARAMS_FCU_VALIDATION['C_m']
    C_act    = PARAMS_FCU_VALIDATION['C_act']
    R_leak   = PARAMS_FCU_VALIDATION['R_leak']
    R_mech   = PARAMS_FCU_VALIDATION['R_mech']
    G_couple = PARAMS_FCU_VALIDATION['G_couple']

    # AP kinetics DISABLED (k = 0) for pure Dirac-structure audit
    k   = PARAMS_FCU_VALIDATION['k']
    eps = PARAMS_FCU_VALIDATION['eps']
    mu1 = PARAMS_FCU_VALIDATION['mu1']
    mu2 = PARAMS_FCU_VALIDATION['mu2']

    x0 = [0.0, 0.0, 0.0, 0.0]   # All states at rest

    # --- Stimulus: brief 0.2 ms current pulse ---
    def get_stim(t):
        return 5.0 if 1.0 <= t <= 1.2 else 0.0

    # --- FCU dynamics (PHS form) ---
    def fcu_dynamics(t, x):
        u, v, Tai, Ta = x

        # Effort variables = gradients of the Hamiltonian
        e_elec = u / C_m     # ∂H/∂u  — electrical effort (voltage)
        e_mech = Tai / C_act  # ∂H/∂Tai — mechanical effort (stress)

        # Electrical domain: J_{13} = −G_couple → du gets −G_couple * e_mech
        # This is the back-EMF: mechanical load opposes electrical potential.
        du = (-R_leak * u) - (G_couple * e_mech) + get_stim(t)

        # Recovery variable (dormant: k = 0 → −v term only)
        dv = (eps + (mu1 * v) / (mu2 + u)) * (-v)

        # Mechanical domain: J_{31} = +G_couple → dTai gets +G_couple * e_elec
        # This is the gyrator actuation: electrical potential drives mechanics.
        dTai = (G_couple * e_elec) - (R_mech * Tai)

        # Passive force transmission (viscous delay)
        dTa = 0.1 * (Tai - Ta)

        return [du, dv, dTai, dTa]

    # --- Integration ---
    t_eval = np.linspace(0, 30, 3000)
    sol = solve_ivp(fcu_dynamics, (0, 30), x0,
                    t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-9)
    u, v, Tai, Ta = sol.y

    # =====================================================================
    # FIRST-LAW AUDIT
    # =====================================================================
    # Reconstruct the instantaneous rates to compute the energy residual:
    #   Residual(t) = P_in(t) − dH/dt(t) − P_diss(t)
    # which must be identically zero for a valid PHS.

    du_dt   = np.zeros_like(u)
    dTai_dt = np.zeros_like(Tai)
    I_stim_trace = np.zeros_like(u)

    for i, ti in enumerate(sol.t):
        rates = fcu_dynamics(ti, [u[i], v[i], Tai[i], Ta[i]])
        du_dt[i]    = rates[0]
        dTai_dt[i]  = rates[2]
        I_stim_trace[i] = get_stim(ti)

    # Total stored energy (Hamiltonian)
    H_total = 0.5 * (u**2) / C_m + 0.5 * (Tai**2) / C_act

    # Input power: P_in = e_elec * I_stim  (effort × flow)
    e_elec = u / C_m
    P_in   = e_elec * I_stim_trace

    # Dissipated power: P_diss = ∇H^T @ R @ ∇H
    #  = R_leak * (∂H/∂u)²  +  R_mech * (∂H/∂Tai)²
    P_diss = (R_leak / C_m) * u**2 + (R_mech / C_act) * Tai**2

    # Analytic dH/dt = (∂H/∂u) * du/dt  +  (∂H/∂Tai) * dTai/dt
    dH_dt_analytic = (u / C_m) * du_dt + (Tai / C_act) * dTai_dt

    # Residual: should be ~0 to machine precision
    residual = P_in - dH_dt_analytic - P_diss

    # =====================================================================
    # 2. PLOTTING
    # =====================================================================
    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

    # --- Panel (a): Transduction dynamics ---
    ax1 = fig.add_subplot(gs[0])

    # Stimulus band (grey fill)
    ax1.axvspan(1.0, 1.2, color='gray', alpha=0.2, label='Impulse')

    ax1.plot(sol.t, u,   color=COLORS['u'],   lw=2.5,
             label=r'Potential $u$ (elec)')
    ax1.plot(sol.t, Tai, color=COLORS['Tai'], ls='--', lw=2.5,
             label=r'Activation $T_{ai}$ (mech)')

    ax1.set_ylabel('State magnitude (a.u.)')
    ax1.legend(loc='upper right', frameon=False)
    ax1.set_xlim(0, 30)
    style_axis(ax1, 'a')

    # --- Panel (b): Thermodynamic audit ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Twin axis for input work (filled area)
    ax2_twin = ax2.twinx()

    # Total energy H(t) — dashed, primary axis
    ax2.plot(sol.t, H_total, color=COLORS['Energy'], lw=3, ls='--',
             label=r'Total energy $\mathcal{H}$', alpha=0.5)

    # External work input — filled area, secondary axis
    ax2_twin.fill_between(sol.t, 0, P_in, color=COLORS['Work'], alpha=0.9)
    ax2_twin.set_ylabel('External work input (W)', color=COLORS['Work'])
    ax2_twin.tick_params(axis='y', labelcolor=COLORS['Work'])
    ax2_twin.spines['top'].set_visible(False)
    ax2_twin.spines['left'].set_visible(False)
    ax2_twin.spines['right'].set_visible(True)
    ax2_twin.spines['right'].set_color(COLORS['Work'])

    # Dissipation rate — dotted, primary axis
    ax2.plot(sol.t, P_diss, color=COLORS['Diss'], ls=':', lw=2,
             label=r'Dissipation rate')

    # Energy residual — solid thin, primary axis
    ax2.plot(sol.t, residual, color=COLORS['Resid'], lw=1.5,
             label=r'Energy residual $\approx 0$')

    # Annotation: First Law equation
    ax2.text(15, 0.5,
             r'$\frac{d\mathcal{H}}{dt} = P_{in} - P_{diss}$ ',
             ha='center', va='center', fontsize=12, fontweight='bold',
             color=COLORS['Energy'],
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=COLORS['Energy'], alpha=0.8))

    ax2.set_ylabel('Energy / Dissipation')
    ax2.set_xlabel('Time (s)')
    style_axis(ax2, 'b')

    # Ensure primary axis lines render on top of filled area
    ax2.set_zorder(ax2_twin.get_zorder() + 1)
    ax2.patch.set_visible(False)

    # Combined legend
    handles, labels = ax2.get_legend_handles_labels()
    patch_work = mpatches.Patch(color=COLORS['Work'], alpha=0.2,
                                label='External work input')
    ax2.legend(handles + [patch_work],
               labels + ['External work input'],
               loc='upper right')

    plt.tight_layout()
    plt.savefig('figures/S1.pdf')
    print("Saved figures/S1.pdf")


if __name__ == "__main__":
    run_fcu_validation()