"""
Figure3.py — Thermodynamic Audit of Myocardial Fibrosis (Post-Processing)
=========================================================================

Paper reference
---------------
  Figure 3: "Thermodynamic audit and metabolic consequences of tissue fibrosis."
  Results § "Energetic consequences of myocardial fibrosis"

Purpose
-------
  Reads the ensemble simulation results produced by run_experiment_ensemble.py
  and generates the publication figures:
    - Panel 1 (a–d): Diffusive cost, efficiency gap, metabolic stress maps
    - Panel 2 (e):   Combined thermodynamic audit (energy conservation check)

  The script computes ensemble means and standard deviations over the
  stochastic fibrotic realisations and compares them against the
  deterministic healthy baseline.

Input
-----
  results_thermo_audit/HEALTHY_SPIRAL_*/results.npz
  results_thermo_audit/PATCHY_SPIRAL_*/results.npz

Output
------
  figures/Figure3_panel_1.pdf  — panels a, b, c, d
  figures/Figure3_panel_2.pdf  — panel e (combined audit)

Dependencies
------------
  numpy, matplotlib

Author
------
  Jagir R. Hussan
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.gridspec import GridSpec


# =========================================================================
# 0. CONFIGURATION
# =========================================================================
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'sans-serif',
    'axes.labelsize': 20,
    'axes.linewidth': 1.5,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Colour palette
COLORS = {
    'primary':    '#0072B2',   # Blue
    'secondary':  '#E69F00',   # Orange (fibrotic)
    'tertiary':   '#009E73',   # Green  (healthy)
    'quaternary': '#D55E00',   # Vermilion
    'gray':       '#555555',
    'error':      'red',
}

# Paths (relative to repository root)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(REPO_ROOT, "results_thermo_audit")
PLOTS_DIR  = os.path.join(REPO_ROOT, "figures")


# =========================================================================
# 1. UTILITY FUNCTIONS
# =========================================================================

def style_axis(ax, panel_label=None):
    """Apply consistent panel styling."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if panel_label:
        ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='bottom', ha='right')


def load_data_robust(pattern):
    """
    Load all results.npz files matching a glob pattern.

    Returns a list of dicts, each containing the data from one
    simulation run.  Runs missing the Map_K_Effort field (legacy
    format) are silently skipped.
    """
    search_path = os.path.join(OUTPUT_DIR, pattern, "results.npz")
    files = glob.glob(search_path)
    files.sort()
    results = []
    for f in files:
        try:
            with np.load(f) as data:
                if 'Map_K_Effort' not in data:
                    continue
                results.append(dict(data))
        except Exception:
            pass
    return results


def safe_aggregate(data_list, func):
    """
    Aggregate time-series data across ensemble members.

    Computes the mean and standard deviation of func(d) across all
    ensemble members, truncating to the shortest time series length.

    Parameters
    ----------
    data_list : list of dicts
    func      : callable, maps a data dict to a 1D array

    Returns
    -------
    t     : ndarray — time vector [ms]
    mean  : ndarray — ensemble mean
    std   : ndarray — ensemble standard deviation
    """
    arrays = [func(d) for d in data_list if len(d['HE']) > 10]
    if not arrays:
        return np.array([]), np.array([]), np.array([])
    min_len = min(len(arr) for arr in arrays)
    truncated = np.array([arr[:min_len] for arr in arrays])
    t = np.linspace(0, 120, min_len)
    return t, np.mean(truncated, axis=0), np.std(truncated, axis=0)


# =========================================================================
# 2. MAIN
# =========================================================================

def main():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    print("Loading simulation data...")
    healthy_data = load_data_robust("HEALTHY_SPIRAL_*")
    patchy_data  = load_data_robust("PATCHY_SPIRAL_*")

    if not healthy_data or not patchy_data:
        print("Error: No simulation data found.")
        print(f"Expected location: {OUTPUT_DIR}/HEALTHY_SPIRAL_*/results.npz")
        print("Run 'run_experiment_ensemble.py' first.")
        return

    # Representative run for spatial maps (first ensemble member)
    h_rep = healthy_data[0]
    p_rep = patchy_data[0]
    t_rep = np.linspace(0, 120, len(h_rep['HE']))

    # =====================================================================
    # FIGURE 3 — PANELS a, b, c, d  (Panel 1)
    # =====================================================================
    fig1 = plt.figure(figsize=(12, 11), constrained_layout=True)
    gs1  = GridSpec(2, 2, figure=fig1, height_ratios=[1, 1.2])

    # --- Panel (a): Relative Diffusive Cost ---
    ax_a = fig1.add_subplot(gs1[0, 0])

    def get_diff_cost(d):
        """
        Relative diffusive cost (%) defined as:
          cost(t) = E_diff(t) / (E_diff(t) + E_mech(t)) × 100

        where E_diff is cumulative diffusive (Joule) dissipation and
        E_mech is cumulative mechanical viscous dissipation.
        """
        passive = d['E_diff'] + d['E_mech']
        ratio = np.divide(d['E_diff'], passive,
                          out=np.zeros_like(d['E_diff']),
                          where=passive > 1e-6)
        return ratio * 100.0

    t_h, h_mean, _       = safe_aggregate(healthy_data, get_diff_cost)
    t_p, p_mean, p_std   = safe_aggregate(patchy_data, get_diff_cost)

    ax_a.plot(t_p, p_mean, color=COLORS['secondary'], lw=2,
              label='Fibrotic')
    ax_a.fill_between(t_p, p_mean - p_std, p_mean + p_std,
                      color=COLORS['secondary'], alpha=0.2)
    ax_a.plot(t_h, h_mean, color=COLORS['tertiary'], lw=2,
              label='Healthy')

    # --- Inset: magnified view of the divergence window (42–95 ms) ---
    axins = ax_a.inset_axes([0.45, 0.65, 0.5, 0.5])

    axins.plot(t_p, p_mean, color=COLORS['secondary'], lw=2)
    axins.fill_between(t_p, p_mean - p_std, p_mean + p_std,
                       color=COLORS['secondary'], alpha=0.2)
    axins.plot(t_h, h_mean, color=COLORS['tertiary'], lw=2)

    # Shade the excess structural burden gap (42–80 ms)
    mask = (t_p >= 42) & (t_p <= 80)
    axins.fill_between(t_p[mask], h_mean[mask], p_mean[mask],
                       color=COLORS['secondary'], alpha=0.3, hatch='///')

    axins.text(75, 49.5, "Excess structural\nburden",
               color=COLORS['secondary'],
               fontsize=10, fontweight='bold', ha='center', va='center')

    axins.set_xlim(38, 95)
    axins.set_ylim(40, 56)
    axins.tick_params(labelsize=10)
    axins.grid(True, alpha=0.3)

    ax_a.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5)

    # Annotation: metabolic masking (the algebraic dilution artifact)
    ax_a.annotate('Metabolic\ndilution', xy=(95, 42), xytext=(80, 25),
                  arrowprops=dict(facecolor='black', shrink=0.05,
                                  width=1.5, headwidth=8),
                  fontsize=12, ha='center')

    ax_a.set_ylabel("Diffusive cost (%)")
    ax_a.set_ylim(0, 100)
    ax_a.set_xlabel("Time (ms)")
    ax_a.legend(loc="lower right", ncol=2)
    style_axis(ax_a, 'a')

    # --- Panel (b): Efficiency Gap (work invested vs. stored energy) ---
    ax_b = fig1.add_subplot(gs1[0, 1])

    # Individual ensemble traces (faint, α = 0.05)
    for d in healthy_data:
        x_trace = d['HM'] + d['E_mech']
        y_trace = d['HM']
        ax_b.plot(x_trace, y_trace, color=COLORS['tertiary'],
                  alpha=0.05, lw=1)

    for d in patchy_data:
        x_trace = d['HM'] + d['E_mech']
        y_trace = d['HM']
        ax_b.plot(x_trace, y_trace, color=COLORS['secondary'],
                  alpha=0.05, lw=1)

    # Ensemble mean loops
    _, h_HM_mean, _   = safe_aggregate(healthy_data, lambda d: d['HM'])
    _, h_Mech_mean, _ = safe_aggregate(healthy_data, lambda d: d['E_mech'])
    _, p_HM_mean, _   = safe_aggregate(patchy_data,  lambda d: d['HM'])
    _, p_Mech_mean, _ = safe_aggregate(patchy_data,  lambda d: d['E_mech'])

    ax_b.plot(p_HM_mean + p_Mech_mean, p_HM_mean,
              color=COLORS['secondary'], lw=3, label='Fibrotic\nmean')
    ax_b.plot(h_HM_mean + h_Mech_mean, h_HM_mean,
              color=COLORS['tertiary'], lw=3, label='Healthy\nmean')

    # Annotate viscous dissipation gap (Δη)
    h_work_peak   = (h_HM_mean + h_Mech_mean).max()
    h_stored_peak = h_HM_mean.max()
    p_stored_peak = p_HM_mean.max()

    x_loc = h_work_peak

    ax_b.annotate('', xy=(x_loc / 2, p_stored_peak - 10),
                  xytext=(x_loc / 2, h_stored_peak - 10),
                  arrowprops=dict(arrowstyle='<->', color=COLORS['error'],
                                  lw=2.5))

    ax_b.text(150, (h_stored_peak + p_stored_peak) / 2 - 15,
              r"Viscous dissipation",
              color=COLORS['error'], va='center',
              fontsize=11, fontweight='bold')
    ax_b.text(700, (h_stored_peak + p_stored_peak) / 2 - 45,
              r"($\Delta \eta$)",
              color=COLORS['error'], va='center',
              fontsize=11, fontweight='bold')

    ax_b.set_xlabel("Work invested (J)")
    ax_b.set_ylabel(r"Stored energy $\mathcal{H}$ (J)")
    ax_b.legend(loc='lower right', ncol=2, framealpha=0.5)
    style_axis(ax_b, 'b')

    # --- Panels (c) & (d): Metabolic Stress Maps ---
    def plot_map(ax, data):
        """Plot time-averaged k_eff spatial map."""
        k_map = data['Map_K_Effort'].copy()
        im = ax.imshow(k_map, cmap='afmhot', vmin=8.0, vmax=8.3,
                       origin='lower')
        ax.axis('off')
        return im

    ax_c = fig1.add_subplot(gs1[1, 0])
    plot_map(ax_c, h_rep)
    style_axis(ax_c, 'c')

    ax_d = fig1.add_subplot(gs1[1, 1])
    im = plot_map(ax_d, p_rep)
    style_axis(ax_d, 'd')

    cbar = fig1.colorbar(im, ax=[ax_c, ax_d], orientation='horizontal',
                         fraction=0.05, pad=0.05)
    cbar.set_label("$k_{eff}$ recruitment")

    fig1.savefig(os.path.join(PLOTS_DIR, "Figure3_panel_1.pdf"))
    print("Saved figures/Figure3_panel_1.pdf")

    # =====================================================================
    # FIGURE 3 — PANEL e: Combined Thermodynamic Audit (Panel 2)
    # =====================================================================
    fig2, ax_combined = plt.subplots(figsize=(10, 8), constrained_layout=True)

    def plot_combined_audit(ax, data_rep, color, label_prefix, y_text_offset):
        """
        Plot observed vs. predicted total stored energy and report
        the maximum thermodynamic drift (should be < 2.2%).

        The predicted energy is computed from the First Law:
          H_pred(t) = H(0) + E_stim(t) − Σ E_diss(t)
        """
        # Total stored energy: H_state = H_elec + H_mech
        H_state = data_rep['HE'] + data_rep['HM']

        # Total dissipated energy (all channels)
        Total_Diss = (data_rep['E_diff'] + data_rep['E_mech']
                      + data_rep['E_reac'])

        # Predicted energy from First-Law bookkeeping
        H_pred = (H_state[0] + data_rep['E_stim']) - Total_Diss

        # Observed (solid) and predicted (dashed)
        ax.plot(t_rep, H_state, color=color, lw=3, alpha=0.6,
                label=f'{label_prefix} observed')
        ax.plot(t_rep, H_pred, color=color, ls='--', lw=2,
                label=f'{label_prefix} predicted')

        # Maximum relative drift
        denom = np.where(H_state > 1e-3, H_state, 1.0)
        error_pct = 100 * np.abs(H_state - H_pred) / denom
        max_err = np.max(error_pct)

        ax.text(0.5, y_text_offset,
                f"{label_prefix} max drift: {max_err:.3f}%",
                transform=ax.transAxes, fontsize=18, fontweight='bold',
                color=color,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax.tick_params(axis='both', labelsize=30)

    # Plot both healthy and fibrotic on the same axes
    plot_combined_audit(ax_combined, p_rep, COLORS['secondary'],
                        "Fibrotic", 0.85)
    plot_combined_audit(ax_combined, h_rep, COLORS['tertiary'],
                        "Healthy", 0.92)

    ax_combined.set_ylabel(r"Total stored energy $\mathcal{H}$ (J)",
                           fontsize=36)
    ax_combined.set_xlabel("Time (ms)", fontsize=36)
    ax_combined.legend(loc='lower right', fontsize=19, ncol=2,
                       framealpha=0.5)

    fig2.savefig(os.path.join(PLOTS_DIR, "Figure3_panel_2.pdf"))
    print("Saved figures/Figure3_panel_2.pdf")

    print("\nAll Figure 3 panels generated.")


if __name__ == "__main__":
    main()