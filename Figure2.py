"""
Figure2.py — Algorithmic Scalability and Thermodynamic Gain (Post-Processing)
=============================================================================

Paper reference
---------------
  Figure 2: "Algorithmic scalability, thermodynamic gain, and topological
  complexity in 3D anisotropic media."
  Results § "Improved scalability in 3D anisotropic geometries"

Purpose
-------
  Generates three publication panels from 3D ventricular wedge simulation
  outputs:
    Panel A — Computational scalability: wall-clock time/step and throughput
              vs. domain size, with theoretical O(N) and O(N²) bounds.
    Panel B — Volumetric rendering of the steady-state spiral wave
              (t = 500 ms) from the 2.0M-node simulation.
    Panel D — Thermodynamic gain: temporal evolution of global stored
              energy H(t) across grid resolutions.

Input
-----
  Log files from scaling studies:
    scaling_studies/scaling_study_*/log_*.txt
  VTK output from the largest simulation:
    scaling_studies/scaling_study_250/threeD_spiral_audit/audit_3d_100000.vtk

Output
------
  figures/Fig2_Panel_A.pdf — scalability (wall-clock + throughput)
  figures/Fig2_Panel_B.pdf — spiral-wave volumetric slices (XY, XZ, YZ)
  figures/Fig2_Panel_D.pdf — energy gain H(t) across grid resolutions

Dependencies
------------
  numpy, matplotlib, pandas

Author
------
  Jagir R. Hussan
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.transforms as transforms


# =========================================================================
# 0. CONFIGURATION
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

COLORS = {
    'primary':   '#0072B2',
    'secondary': '#E69F00',
    'tertiary':  '#009E73',
    'gray':      '#7f7f7f',
}

# Output directory for figures
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# =========================================================================
# 1. DATA LOADING — Scaling Logs (Panels A & D)
# =========================================================================

# Log files from scaling studies (relative to repository root)
# Each entry: (path, label)
# The label is used for legend entries (e.g., "0.7M" = 0.7 million nodes)
SCALING_LOG_FILES = [
    # Paths match output of: python scaling_studies/run_scaling_study.py --preset <name>
    ('scaling_studies/scaling_study_140x2/log_140x2.txt',       '0.7M'),
    ('scaling_studies/scaling_study_160x2/log_160x2.txt',       '1.0M'),
    ('scaling_studies/scaling_study_150_cons/log_150_cons.txt', '1.2M'),
    ('scaling_studies/scaling_study_163/log_163.txt',           '1.3M'),
    ('scaling_studies/scaling_study_175/log_175.txt',           '1.4M'),
    ('scaling_studies/scaling_study_200/log_200.txt',           '1.6M'),
    ('scaling_studies/scaling_study_250/log_250.txt',           '2.0M'),
]

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# Resolve paths relative to repository root
SCALING_LOG_FILES = [
    (os.path.join(REPO_ROOT, f), l) for f, l in SCALING_LOG_FILES
]


def parse_log(filename, label):
    """
    Parse a scaling-study log file to extract:
      - Energy time series: (Time_ms, Energy_J) pairs
      - Metadata: domain size (M nodes), time/step, max stored energy

    The log format is produced by the EnergyAudit.report() method
    in scaling_studies/run_3d_experiment_spiral.py.
    """
    data = []
    meta = {'label': label}
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"  Warning: {filename} not found — skipping.")
        return None, None

    for line in lines:
        if "[Scalability] Domain Size:" in line:
            meta['nodes_m'] = float(
                re.search(r"Domain Size:\s+([\d\.]+)", line).group(1))
        if "[Scalability] Time/Step:" in line:
            meta['time_step'] = float(
                re.search(r"Time/Step:\s+([\d\.]+)", line).group(1))
        if "[Energy]      Max Stored:" in line:
            meta['max_energy'] = float(
                re.search(r"Max Stored:\s+([\d\.e\+\-]+)", line).group(1))

    # Parse per-timestep energy entries
    pattern = r"T=\s*([\d\.]+)ms\s*\|\s*Energy\(J\):\s*([\d\.e\+\-]+)"
    for line in lines:
        match = re.search(pattern, line)
        if match:
            data.append({
                'Time_ms': float(match.group(1)),
                'Energy_J': float(match.group(2)),
            })
    return pd.DataFrame(data), meta


# Load all logs
results = []
for filepath, label in SCALING_LOG_FILES:
    df, meta = parse_log(filepath, label)
    if df is not None:
        results.append((df, meta))


# =========================================================================
# 2. DATA LOADING — VTK Spiral Snapshot (Panel B)
# =========================================================================

# Largest simulation VTK output (2.0M nodes, t = 500 ms)
VTK_FILE = os.path.join(
    REPO_ROOT,
    "scaling_studies/scaling_study_250/threeD_spiral_audit/audit_3d_100000.vtk"
)

# Slice positions through the 3D volume (fractional: 0.5 = middle)
SLICE_X_RATIO = 0.5
SLICE_Y_RATIO = 0.5
SLICE_Z_RATIO = 0.5

# Voltage visualisation settings
V_MIN, V_MAX = 0.0, 1.0
CMAP_V = "inferno"


def parse_vtk_dimensions(filepath):
    """Read the DIMENSIONS line from an ASCII VTK structured-points file."""
    nx, ny, nz = 0, 0, 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("DIMENSIONS"):
                parts = line.split()
                nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
                break
    return nx, ny, nz


def load_vtk_multifield(filepath, nx, ny, nz):
    """
    Fast parser for multi-scalar ASCII VTK STRUCTURED_POINTS files.

    Returns a dict mapping field names to 3D numpy arrays (Fortran order).
    """
    data_dict = {}
    n_points = nx * ny * nz

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    n_lines = len(lines)

    while i < n_lines:
        line = lines[i].strip()

        if line.startswith("SCALARS"):
            parts = line.split()
            field_name = parts[1]

            # Skip "LOOKUP_TABLE default"
            i += 2

            # Read numeric block (may span multiple lines)
            values = np.fromstring(
                ''.join(lines[i:i + (n_points // 9 + 2)]),
                sep=' ', count=n_points
            )

            arr = values.reshape((nx, ny, nz), order='F')
            data_dict[field_name] = arr

            i += (n_points // 9 + 2)
        else:
            i += 1

    return data_dict


# Load VTK data for spiral visualisation
nx, ny, nz = parse_vtk_dimensions(VTK_FILE)
data_vtk = load_vtk_multifield(VTK_FILE, nx, ny, nz)
voltage_field = data_vtk.get("Voltage_u",
                              np.zeros((nx, ny, nz)))


# =========================================================================
# 3. PLOTTING
# =========================================================================

# --- Panel D: Energy Gain H(t) across resolutions ---
fig_d, ax_d = plt.subplots(figsize=(7.2, 6))
cmap = cm.get_cmap('viridis', len(results) + 1)

for i, (df, meta) in enumerate(results):
    lw = 3.0 if i == len(results) - 1 else 1.5
    alpha = 1.0 if i == len(results) - 1 else 0.6
    ax_d.plot(df['Time_ms'], df['Energy_J'],
              color=cmap(i), lw=lw, alpha=alpha,
              label=(f"{meta['label']} "
                     f"($H_{{max}}={meta['max_energy']:.0f}$J)"))

# Annotate super-linear energy gain
e_min = results[0][1]['max_energy']
e_max = results[-1][1]['max_energy']
gain  = ((e_max - e_min) / e_min) * 100
ax_d.annotate(
    f'+{gain:.0f}% Energy gain\n(super-linear)',
    xy=(100, e_max), xytext=(150, e_max - 1),
    arrowprops=dict(facecolor='black', arrowstyle='->'),
    fontsize=10, fontweight='bold')

ax_d.set_xlabel('Time (ms)', fontsize=11)
ax_d.set_ylabel(r'Total stored energy $\mathcal{H}$ (J)', fontsize=12)
ax_d.legend(loc='lower right', ncol=2)
fig_d.tight_layout()
fig_d.savefig(os.path.join(PLOTS_DIR, "Fig2_Panel_D.pdf"),
              bbox_inches='tight')
print("Saved figures/Fig2_Panel_D.pdf")
plt.close(fig_d)

# --- Panel A: Computational Scalability ---
fig_a, ax_a = plt.subplots(figsize=(7.2, 6))

regex_time       = re.compile(r"Time/Step:\s*([\d\.]+)")
regex_throughput = re.compile(r"Throughput:\s*([\d\.]+)")

domain_sizes = []
time_per_step = []
throughput = []

print("\nParsing scalability metrics...")
for filepath, label in SCALING_LOG_FILES:
    size_val = float(label.replace('M', ''))
    t_step = None
    t_put  = None

    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "[Scalability] Time/Step:" in line:
                    match = regex_time.search(line)
                    if match:
                        t_step = float(match.group(1))
                elif "[Scalability] Throughput:" in line:
                    match = regex_throughput.search(line)
                    if match:
                        t_put = float(match.group(1))

        if t_step is not None and t_put is not None:
            domain_sizes.append(size_val)
            time_per_step.append(t_step)
            throughput.append(t_put)
            print(f"  {label:>4}: Time/Step = {t_step:>6.2f} ms | "
                  f"Throughput = {t_put:>5.1f} M Updates/s")
        else:
            print(f"  Warning: Missing metrics in "
                  f"{os.path.basename(filepath)}")
    except FileNotFoundError:
        print(f"  Error: {filepath} not found")

# Compute theoretical scaling curves (anchored to smallest domain)
domain_sizes = np.array(domain_sizes)
N_base = domain_sizes[0]
T_base = time_per_step[0]

# O(N) — linear scaling
theoretical_O_N  = T_base * (domain_sizes / N_base)

# O(N²) — quadratic scaling (dense Laplacian)
theoretical_O_N2 = T_base * (domain_sizes / N_base) ** 2

# Left axis: Time/Step + theoretical bounds
line_obs  = ax_a.plot(domain_sizes, time_per_step,
                       marker='o', color='black', linewidth=2.5,
                       markersize=8, label='Observed')
line_ON   = ax_a.plot(domain_sizes, theoretical_O_N,
                       linestyle='--', color='#1f77b4', linewidth=2,
                       label=r'Theoretical $O(N)$')
line_ON2  = ax_a.plot(domain_sizes, theoretical_O_N2,
                       linestyle=':', color='#d62728', linewidth=2,
                       label=r'Theoretical $O(N^2)$')

ax_a.set_xlabel('Domain size (million nodes)', fontsize=12)
ax_a.set_ylabel('Steady-state time per step (ms)', fontsize=12)
ax_a.grid(True, linestyle='--', alpha=0.6)

# Right axis: Hardware throughput
ax_a2 = ax_a.twinx()
line_tput = ax_a2.plot(domain_sizes, throughput,
                        marker='s', linestyle='-.', color='#2ca02c',
                        linewidth=2, markersize=8, alpha=0.85,
                        label='Throughput')
ax_a2.set_ylabel('Throughput (million updates/sec)',
                  color='#2ca02c', fontsize=12)
ax_a2.tick_params(axis='y', labelcolor='#2ca02c')

# Combined legend
lines  = line_obs + line_ON + line_ON2 + line_tput
labels = [l.get_label() for l in lines]
ax_a2.legend(lines, labels, loc='upper right', framealpha=0.9,
             fontsize=10)

fig_a.tight_layout()
fig_a.savefig(os.path.join(PLOTS_DIR, "Fig2_Panel_A.pdf"),
              bbox_inches='tight')
print("Saved figures/Fig2_Panel_A.pdf")
plt.close(fig_a)

# --- Panel B: Volumetric Spiral Rendering ---
fig_b, ax_b = plt.subplots(figsize=(6, 6))
ax_b.axis('off')

# Extract orthogonal slices
idx_x = int(nx * SLICE_X_RATIO)
idx_y = int(ny * SLICE_Y_RATIO)
idx_z = int(nz * SLICE_Z_RATIO)

xy = voltage_field[:, :, idx_z]     # XY plane at fixed Z
xz = voltage_field[:, idx_y, :]     # XZ plane at fixed Y
yz = voltage_field[idx_x, :, :]     # YZ plane at fixed X

# Pad slices to equal height for horizontal stacking
h_xy, _ = xy.shape
h_xz, _ = xz.shape
h_yz, _ = yz.shape
max_h = max(h_xy, h_xz, h_yz)


def pad_with_nan(arr, target_h):
    """Pad a 2D array vertically with NaN to reach target_h rows."""
    pad_h = target_h - arr.shape[0]
    if pad_h <= 0:
        return arr
    nan_block = np.full((pad_h, arr.shape[1]), np.nan)
    return np.vstack([arr, nan_block])


spacer_width = 20
spacer = np.full((max_h, spacer_width), np.nan)

xy_p = pad_with_nan(xy, max_h)
xz_p = pad_with_nan(xz, max_h)
yz_p = pad_with_nan(yz, max_h)

combined = np.hstack([xy_p, spacer, xz_p, spacer, yz_p])

im = ax_b.imshow(combined, origin='lower',
                  cmap=CMAP_V, vmin=V_MIN, vmax=V_MAX)
ax_b.set_facecolor('white')

# Plane labels centred above each slice
trans = transforms.blended_transform_factory(
    ax_b.transData, ax_b.transAxes)

center_xy = xy.shape[1] / 2.0
center_xz = xy.shape[1] + spacer_width + (xz.shape[1] / 2.0)
center_yz = (xy.shape[1] + spacer_width + xz.shape[1]
             + spacer_width + (yz.shape[1] / 2.0))

ax_b.text(center_xy, 1.02, "XY", ha='center', va='bottom',
           transform=trans, fontsize=16, fontweight='bold')
ax_b.text(center_xz, 1.02, "XZ", ha='center', va='bottom',
           transform=trans, fontsize=16, fontweight='bold')
ax_b.text(center_yz, 1.02, "YZ", ha='center', va='bottom',
           transform=trans, fontsize=16, fontweight='bold')

fig_b.tight_layout()
fig_b.savefig(os.path.join(PLOTS_DIR, "Fig2_Panel_B.pdf"),
              bbox_inches='tight')
print("Saved figures/Fig2_Panel_B.pdf")
plt.close(fig_b)

print("\nAll Figure 2 panels generated.")