"""
run_3d_experiment_spiral.py — 3D Ventricular Wedge S1-S2 Protocol (Hero Run)
=============================================================================

Paper reference
---------------
  Figure 2: Algorithmic scalability and thermodynamic gain.
  Methods § "High-resolution 3D ventricular wedge simulations"

Purpose
-------
  Reference implementation of the 3D transmural ventricular wedge
  simulation.  This script is a thin wrapper around the parameterized
  runner (run_scaling_study.py) that uses the default "hero run"
  configuration: 140×140×40 nodes (0.78M).

  For running other grid sizes, use run_scaling_study.py directly:
    python scaling_studies/run_scaling_study.py --preset 250

Usage
-----
  python scaling_studies/run_3d_experiment_spiral.py

  ⚠ Warning: ~1.5 hours on a modern workstation for the default grid.

Output
------
  scaling_studies/scaling_study_140x2/
    log_140x2.txt
    threeD_spiral_audit/energy_audit.csv

Dependencies
------------
  numpy, scaling_studies.phs_3d_core, scaling_studies.run_scaling_study

Author
------
  Jagir R. Hussan
"""

import os
import sys

# Ensure the scaling_studies directory is on the path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from run_scaling_study import run_simulation, PRESETS

# =========================================================================
# Default hero-run configuration (140×140×40 — the smallest grid)
# =========================================================================
NX, NY, NZ, LABEL = PRESETS['140x2']

OUTPUT_DIR = os.path.join(_script_dir, "scaling_study_140x2")
LOG_FILE   = os.path.join(OUTPUT_DIR, "log_140x2.txt")


def main():
    print("Running hero 3D spiral simulation (140×140×40)...")
    print("For other grid sizes, use: python run_scaling_study.py --preset <name>")
    run_simulation(NX, NY, NZ, OUTPUT_DIR, LOG_FILE)


if __name__ == "__main__":
    main()