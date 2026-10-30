"""
run_scaling_study.py — Parameterized 3D Ventricular Wedge Scaling Runner
========================================================================

Paper reference
---------------
  Figure 2: Algorithmic scalability and thermodynamic gain.
  Methods § "High-resolution 3D ventricular wedge simulations"

Purpose
-------
  Accepts grid dimensions and output
  paths from the command line, or uses named presets for the exact
  configurations reported in the paper.

Usage
-----
  # Run a paper preset:
  python scaling_studies/run_scaling_study.py --preset 250

  # Run with explicit grid dimensions:
  python scaling_studies/run_scaling_study.py --nx 200 --ny 175 --nz 40

  # List all available presets:
  python scaling_studies/run_scaling_study.py --list-presets

  Warning: These simulations are EXTREMELY computationally expensive.
  The largest preset (2.0M nodes) takes ~11 hours on a modern CPU.
  See the table below for approximate run times.

Paper Presets (reproduce Figure 2 exactly)
-------------------------------------------
  +----------+-------------+-----------+--------------+----------------+
  | Preset   | Grid (NX×   | Nodes     | Label in     | Approx.        |
  |          | NY×NZ)      | (million) | Figure 2     | wall time      |
  +----------+-------------+-----------+--------------+----------------+
  | 140x2    | 140×140×40  | 0.78 M    | 0.7M         | ~1.5 h         |
  | 160x2    | 160×160×40  | 1.02 M    | 1.0M         | ~2.5 h         |
  | 150_cons | 200×150×40  | 1.20 M    | 1.2M         | ~3.5 h         |
  | 163      | 200×163×40  | 1.30 M    | 1.3M         | ~4.0 h         |
  | 175      | 200×175×40  | 1.40 M    | 1.4M         | ~4.5 h         |
  | 200      | 200×200×40  | 1.60 M    | 1.6M         | ~6.0 h         |
  | 250      | 200×250×40  | 2.00 M    | 2.0M         | ~11.0 h        |
  +----------+-------------+-----------+--------------+----------------+

Output directory structure
--------------------------
  scaling_studies/scaling_study_{preset}/
    log_{preset}.txt               — Energy/log output (parsed by Figure2.py)
    threeD_spiral_audit/
      energy_audit.csv             — Time series of H, P_diff, P_mech
      audit_3d_*.vtk              — Volumetric snapshots (if VTK enabled)

Common parameters (shared across all presets)
---------------------------------------------
  DX = 0.03 cm          DY = 0.03 * NY/NX cm    DZ = 0.03 cm
  DT = 0.005 ms         T_MAX = 500 ms
  SIGMA_L = 0.3e-3 S/cm  SIGMA_T = 0.1e-3 S/cm
  ANGLE_ENDO = −60°      ANGLE_EPI = +60°

Dependencies
------------
  numpy, phs_3d_core (local module)

Author
------
  Jagir R. Hussan
"""

import os
import sys
import time
import argparse
import numpy as np

# Lazy import — heavy Numba modules loaded only when running a simulation.
# This allows --list-presets and --help to work without a full HPC environment.
_LAZY_IMPORTS = None

def _get_phs3d():
    global _LAZY_IMPORTS
    if _LAZY_IMPORTS is None:
        from phs_3d_core import (rk4_step_3d, compute_laplacian_3d_aniso,
                                   compute_energy_flux_3d)
        _LAZY_IMPORTS = (rk4_step_3d, compute_laplacian_3d_aniso,
                          compute_energy_flux_3d)
    return _LAZY_IMPORTS


# =========================================================================
# 0. PAPER PRESETS — Exact configurations from Figure 2
# =========================================================================

PRESETS = {
    # preset  → (NX, NY, NZ, label_for_log)
    '140x2':    (140, 140, 40, '0.7M'),
    '160x2':    (160, 160, 40, '1.0M'),
    '150_cons': (200, 150, 40, '1.2M'),
    '163':      (200, 163, 40, '1.3M'),
    '175':      (200, 175, 40, '1.4M'),
    '200':      (200, 200, 40, '1.6M'),
    '250':      (200, 250, 40, '2.0M'),
}

# Shared physics parameters (identical across all presets)
DX         = 0.03              # Grid spacing X [cm]
DZ         = 0.03              # Grid spacing Z [cm]
DT         = 0.005             # Internal time step [ms]
T_MAX      = 500.0             # Total simulation time [ms]
SIGMA_L    = 0.3e-3            # Longitudinal conductivity [S/cm]
SIGMA_T    = 0.1e-3            # Transverse conductivity  [S/cm]
ANGLE_ENDO = -60.0             # Endocardial fibre angle [deg]
ANGLE_EPI  = +60.0             # Epicardial fibre angle   [deg]

# S2 trigger probe (fractional position)
PROBE_FRAC_X = 0.25
PROBE_FRAC_Y = 0.50
PROBE_FRAC_Z = 0.50

# S2 trigger voltage criteria
V_TRIGGER_MAX = 2.642
V_TRIGGER_MIN = 0.30

# I/O
VTK_EVERY_N_STEPS = 0   # Set to e.g. 200 to enable VTK output (expensive)
PROGRESS_EVERY_N   = 200
AUDIT_EVERY_N      = 50


# =========================================================================
# 1. SPATIAL METRICS (for optional VTK output)
# =========================================================================

def compute_spatial_metrics(U, Ta, lap,
                            k_base=8.0, k_recruit=2.0, k_max=30.0):
    """
    Compute diagnostic spatial maps from instantaneous state.
    See phs_3d_core.py §5 for field descriptions.
    """
    H_density = 0.5 * (U ** 2) + 0.5 * (Ta ** 2)
    P_diff_density = -1.0 * (U * lap)
    load = np.abs(lap) * (lap < 0).astype(np.float64)
    K_map = k_base + (k_recruit * load)
    K_map = np.clip(K_map, k_base, k_max)
    return H_density, P_diff_density, K_map


def save_vtk_advanced(U, V, Ta, lap, step, dx, dy, dz, output_dir):
    """Save state + thermodynamics to ASCII VTK.  See phs_3d_core.py."""
    nx, ny, nz = U.shape
    filename = os.path.join(output_dir, f"audit_3d_{step:04d}.vtk")
    H_density, P_diss, K_eff = compute_spatial_metrics(U, Ta, lap)

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"PHS-3D Spiral Thermodynamics Step {step}\n")
        f.write("ASCII\nDATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write("ORIGIN 0 0 0\n")
        f.write(f"SPACING {dx} {dy} {dz}\n")
        f.write(f"POINT_DATA {nx * ny * nz}\n")

        def write_scalar(name, data):
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            for val in data.ravel(order='F'):
                f.write(f"{val:.5f} ")
            f.write("\n")

        write_scalar("Voltage_u", U)
        write_scalar("Recovery_v", V)
        write_scalar("ActiveTension_Ta", Ta)
        write_scalar("Energy_Density_H", H_density)
        write_scalar("Dissipation_Power_Pdiff", P_diss)
        write_scalar("Metabolic_Stiffness_k", K_eff)


# =========================================================================
# 2. ENERGY AUDIT LOGGER
# =========================================================================

class EnergyAudit:
    """Accumulates and reports thermodynamic + scalability metrics."""

    def __init__(self, total_nodes):
        self.history = []
        self.total_nodes = total_nodes

    def log(self, t, H_elec, P_diff, P_mech, compute_time_per_step):
        self.history.append(
            [t, H_elec, P_diff, P_mech, compute_time_per_step])

    def report(self):
        data = np.array(self.history)
        avg_step_time = np.mean(data[:, 4])
        nups = self.total_nodes / avg_step_time

        report_lines = [
            "",
            "=== THERMODYNAMIC AUDIT & SCALABILITY REPORT ===",
            f"[Scalability] Domain Size: {self.total_nodes / 1e6:.2f} "
            f"Million Nodes",
            f"[Scalability] Throughput:  {nups / 1e6:.2f} "
            f"Million Updates/Sec",
            f"[Scalability] Time/Step:   {avg_step_time * 1000:.2f} ms",
            f"[Energy]      Max Stored:  {np.max(data[:, 1]):.4e} J",
            f"[Energy]      Max Diffusive Cost: "
            f"{np.max(data[:, 2]):.4e} W",
            "================================================",
        ]
        return "\n".join(report_lines), data


# =========================================================================
# 3. WEDGE SETUP
# =========================================================================

def setup_wedge(nx, ny, nz, dy):
    """
    Initialise the 3D transmural wedge with fibre rotation.

    Parameters
    ----------
    nx, ny, nz : int  — grid dimensions
    dy         : float — Y grid spacing [cm]
    """
    print(f"--- Setting up 3D Wedge ({nx}×{ny}×{nz}) ---")

    U   = np.zeros((nx, ny, nz), dtype=np.float64)
    V   = np.zeros((nx, ny, nz), dtype=np.float64)
    Ta  = np.zeros((nx, ny, nz), dtype=np.float64)
    Tai = np.zeros((nx, ny, nz), dtype=np.float64)

    D11 = np.zeros((nx, ny, nz))
    D22 = np.zeros((nx, ny, nz))
    D33 = np.zeros((nx, ny, nz))
    D12 = np.zeros((nx, ny, nz))

    print("--- Generating Fibre Rotation ---")
    for k in range(nz):
        frac  = k / (nz - 1)
        deg   = ANGLE_ENDO + frac * (ANGLE_EPI - ANGLE_ENDO)
        theta = np.radians(deg)
        c, s  = np.cos(theta), np.sin(theta)

        D11[:, :, k] = SIGMA_L * c**2 + SIGMA_T * s**2
        D22[:, :, k] = SIGMA_L * s**2 + SIGMA_T * c**2
        D12[:, :, k] = (SIGMA_L - SIGMA_T) * c * s
        D33[:, :, k] = SIGMA_T

    return U, V, Ta, Tai, D11, D22, D33, D12


# =========================================================================
# 4. MAIN SIMULATION LOOP
# =========================================================================

def run_simulation(nx, ny, nz, output_dir, log_filename):
    """
    Run a full S1-S2 arrhythmia-induction simulation.

    Parameters
    ----------
    nx, ny, nz : int   — grid dimensions
    output_dir : str   — directory for VTK and CSV output
    log_filename : str — path to the log file
    """
    dy = 0.03 * ny / nx      # DY scales with aspect ratio

    U, V, Ta, Tai, D11, D22, D33, D12 = setup_wedge(nx, ny, nz, dy)

    steps         = int(T_MAX / DT)
    trigger_state = 0
    s2_timer      = 0.0
    total_nodes   = nx * ny * nz

    # Probe location
    probe_x = int(nx * PROBE_FRAC_X)
    probe_y = int(ny * PROBE_FRAC_Y)
    probe_z = int(nz * PROBE_FRAC_Z)

    audit = EnergyAudit(total_nodes)

    print(f"--- Sentinel Probe at ({probe_x}, {probe_y}, {probe_z}) ---",
          flush=True)

    # Open log file for streaming output
    log_file = open(log_filename, 'w')

    t_sim_start = time.time()

    # Resolve lazy imports once before the simulation loop
    rk4_step_3d, compute_laplacian_3d_aniso, compute_energy_flux_3d = _get_phs3d()

    for step in range(1, steps + 1):
        step_start = time.time()
        curr_time  = step * DT

        # --- S1 Stimulus: planar wave from left boundary (t < 5 ms) ---
        if curr_time < 5.0:
            U[0:5, :, :] = 1.0

        # --- Physics step ---
        U, V, Ta, Tai = rk4_step_3d(
            U, V, Ta, Tai, D11, D22, D33, D12,
            DX, dy, DZ, DT)

        # --- Energy audit ---
        if step % AUDIT_EVERY_N == 0:
            lap_audit = compute_laplacian_3d_aniso(
                U, D11, D22, D33, D12, DX, dy, DZ, nx, ny, nz)
            H_elec, P_diff, P_mech = compute_energy_flux_3d(
                U, V, Ta, Tai, lap_audit, DX, dy, DZ)
            step_duration = time.time() - step_start
            audit.log(curr_time, H_elec, P_diff, P_mech, step_duration)

        # --- S2 Trigger Logic ---
        probe_u = U[probe_x, probe_y, probe_z]
        probe_v = V[probe_x, probe_y, probe_z]

        if trigger_state == 0:
            if probe_u > 0.8:
                print(f"[T={curr_time:.1f}ms] S1 Arrived.", flush=True)
                log_file.write(
                    f"[T={curr_time:.1f}ms] S1 Arrived.\n")
                trigger_state = 1

        elif trigger_state == 1:
            if probe_v < V_TRIGGER_MAX and probe_u < 0.2:
                if probe_v > V_TRIGGER_MIN:
                    print(f"[T={curr_time:.1f}ms] Target Acquired "
                          f"(V={probe_v:.3f}). FIRING S2!", flush=True)
                    log_file.write(
                        f"[T={curr_time:.1f}ms] Target Acquired "
                        f"(V={probe_v:.3f}). FIRING S2!\n")
                    trigger_state = 2
                    s2_timer = 0.0

        elif trigger_state == 2:
            if s2_timer < 5.0:
                U[0:nx, 0:ny // 2, :] = 1.0
                s2_timer += DT
            else:
                print(f"[T={curr_time:.1f}ms] S2 Complete.", flush=True)
                log_file.write(
                    f"[T={curr_time:.1f}ms] S2 Complete.\n")
                trigger_state = 3

        # --- Progress ---
        if step % PROGRESS_EVERY_N == 0:
            elapsed = time.time() - t_sim_start
            h_now = H_elec if 'H_elec' in dir() else 0.0
            msg = (f"T={curr_time:6.1f}ms | "
                   f"Energy(J): {h_now:.2e} | "
                   f"Wall: {elapsed:.1f}s | "
                   f"State={trigger_state} | "
                   f"V={probe_v:.3f}")
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()

        # --- VTK output (optional, expensive) ---
        if VTK_EVERY_N_STEPS > 0 and step % VTK_EVERY_N_STEPS == 0:
            vtk_dir = os.path.join(output_dir, "threeD_spiral_audit")
            os.makedirs(vtk_dir, exist_ok=True)
            lap_current = compute_laplacian_3d_aniso(
                U, D11, D22, D33, D12, DX, dy, DZ, nx, ny, nz)
            save_vtk_advanced(U, V, Ta, lap_current, step,
                              DX, dy, DZ, vtk_dir)

    # --- Final report ---
    report_str, audit_data = audit.report()
    print(report_str)
    log_file.write(report_str + "\n")

    # Save CSV
    csv_path = os.path.join(output_dir, "threeD_spiral_audit",
                            "energy_audit.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    np.savetxt(csv_path, audit_data, delimiter=",",
               header="Time_ms,H_elec_J,P_diff_W,P_mech_W,Compute_Sec",
               comments='')

    log_file.close()
    print(f"Log saved to {log_filename}")
    print(f"CSV saved to {csv_path}")
    print("Simulation complete.")


# =========================================================================
# 5. COMMAND-LINE INTERFACE
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="3D Ventricular Wedge S1-S2 Scaling Study Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scaling_studies/run_scaling_study.py --preset 250
  python scaling_studies/run_scaling_study.py --nx 140 --ny 140 --nz 40
  python scaling_studies/run_scaling_study.py --list-presets
        """)

    parser.add_argument('--preset', type=str, default=None,
                        help='Named preset from the paper (see --list-presets)')
    parser.add_argument('--nx', type=int, default=None,
                        help='Grid dimension X (fibre direction)')
    parser.add_argument('--ny', type=int, default=None,
                        help='Grid dimension Y (sheet direction)')
    parser.add_argument('--nz', type=int, default=None,
                        help='Grid dimension Z (transmural)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory '
                             '(default: scaling_studies/scaling_study_{preset}/)')
    parser.add_argument('--list-presets', action='store_true',
                        help='List all paper presets and exit')

    args = parser.parse_args()

    # --- List presets ---
    if args.list_presets:
        print("\nPaper Presets for Figure 2:\n")
        print(f"{'Preset':<12} {'Grid (NX×NY×NZ)':<20} {'Nodes':<12} "
              f"{'Label':<8}")
        print("-" * 52)
        for name, (nx, ny, nz, label) in PRESETS.items():
            nodes_m = nx * ny * nz / 1e6
            print(f"{name:<12} {nx}×{ny}×{nz:<18} {nodes_m:.2f} M{'':<6} "
                  f"{label:<8}")
        print()
        return

    # --- Resolve grid dimensions ---
    if args.preset:
        if args.preset not in PRESETS:
            print(f"Error: Unknown preset '{args.preset}'.")
            print(f"Available: {', '.join(PRESETS.keys())}")
            print("Use --list-presets to see all options.")
            sys.exit(1)
        nx, ny, nz, label = PRESETS[args.preset]
        preset_name = args.preset
    elif args.nx and args.ny and args.nz:
        nx, ny, nz = args.nx, args.ny, args.nz
        label = f"{nx}x{ny}x{nz}"
        preset_name = f"{ny}"  # Match naming convention
    else:
        parser.print_help()
        print("\nError: Specify either --preset or all of --nx, --ny, --nz.")
        sys.exit(1)

    # --- Output directory ---
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Place alongside this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir,
                                  f"scaling_study_{preset_name}")

    os.makedirs(output_dir, exist_ok=True)
    log_filename = os.path.join(output_dir, f"log_{preset_name}.txt")

    print(f"\n{'='*60}")
    print(f"  Scaling Study: {nx}×{ny}×{nz}  "
          f"({nx * ny * nz / 1e6:.2f}M nodes)")
    print(f"  Label: {label}")
    print(f"  Output: {output_dir}")
    print(f"  Log:    {log_filename}")
    print(f"{'='*60}\n")

    run_simulation(nx, ny, nz, output_dir, log_filename)


if __name__ == "__main__":
    main()