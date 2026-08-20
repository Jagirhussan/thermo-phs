# Model Parameters — Port-Hamiltonian Hypergraph Framework

Comprehensive parameter reference for all models and simulations in:

> Hussan, Omholt & Hunter, *"Guaranteeing thermodynamically consistent multiphysics modeling by use of port-Hamiltonian hypergraphs"* (2026)

---

## Table of Contents

1. [Aliev-Panfilov Electrophysiology Parameters](#1-aliev-panfilov-electrophysiology-parameters)
2. [Nash-Panfilov / Silva Active Mechanics Parameters](#2-nash-panfilov--silva-active-mechanics-parameters)
3. [Port-Hamiltonian Gyrator Coupling Parameters](#3-port-hamiltonian-gyrator-coupling-parameters)
4. [Hypergraph Laplacian & Diffusion Parameters](#4-hypergraph-laplacian--diffusion-parameters)
5. [Metabolic Autoregulation Parameters](#5-metabolic-autoregulation-parameters)
6. [Autonomic Control Parameters](#6-autonomic-control-parameters)
7. [Tissue Topology & Fibrosis Parameters](#7-tissue-topology--fibrosis-parameters)
8. [Numerical & Simulation Parameters](#8-numerical--simulation-parameters)
9. [Linearised Canonical Transducer Parameters (Figure S1)](#9-linearised-canonical-transducer-parameters-figure-s1)
10. [Per-Figure Parameter Summary](#10-per-figure-parameter-summary)

---

## 1. Aliev-Panfilov Electrophysiology Parameters

Govern the fast (u) and slow (v) dynamics of cardiac excitation.

| Symbol | Name | SI Unit | Normalised | Fig 1 | Fig S1 | Figs 3,S3 | Fig 4 | 3D (Fig 2) | Defined in |
|--------|------|---------|------------|-------|--------|-----------|-------|------------|------------|
| `k` | Reaction-rate scaling factor | ms⁻¹ | — | 8.0 | 0.0* | 8.0 | 8.0 | 8.0 | `phs_models.py`, `phs_hetero_core.py`, `phs_3d_core.py` |
| `a` | Excitation threshold | — | dimensionless | 0.1 | — | 0.1 | 0.1 | **0.05**† | `phs_hetero_core.py`, `phs_3d_core.py` |
| `ε` (eps) | Base recovery rate | ms⁻¹ | — | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | all physics modules |
| `μ₁` (mu1) | APD restitution parameter 1 | — | dimensionless | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | all physics modules |
| `μ₂` (mu2) | APD restitution parameter 2 | — | dimensionless | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | all physics modules |

\* `k = 0` in Figure S1 disables AP kinetics to isolate the pure Dirac structure.  
† `a = 0.05` in the 3D kernel is a deliberate choice: the lower excitation threshold sustains stable scroll-wave re-entry in the larger, anisotropic 3D domain. This does not affect the validity of the 2D results where `a = 0.1`.

**Equations** (see `phs_models.py` §1 for full derivation):

```
du/dt = −k·u·(u − a)·(1 − u) − u·v                                    (AP.1)
dv/dt = (ε + μ₁·v/(μ₂ + u)) · (−v − k·u·(u − a − 1))                  (AP.2)
```

---

## 2. Nash-Panfilov / Silva Active Mechanics Parameters

Govern the calcium activation (T_ai) and active tension (T_a) dynamics.

| Symbol | Name | SI Unit | Normalised | Fig 1 | Figs 3,S3 | Fig 4 | 3D (Fig 2) | Defined in |
|--------|------|---------|------------|-------|-----------|-------|------------|------------|
| `c₀` (c0) | Cross-bridge cycling / calcium coupling rate | ms⁻¹ | — | **1.0**† | 0.05 | 0.05 | 0.05 | `phs_hetero_core.py`, `phs_3d_core.py` |
| `k_visc` | Passive viscous damping rate | ms⁻¹ | — | **0.1**† | 0.05 | 0.05 | 0.05 | `phs_hetero_core.py`, `phs_3d_core.py` |
| `s` (s_sig) | Activation sigmoid steepness | — | dimensionless | 50.0 | 50.0 | 100.0 | 50.0 | `phs_hetero_core.py`, Figure scripts |
| `u_thr` (u_sig) | Activation sigmoid midpoint | — | dimensionless | 0.1 | 0.05 | 0.05 | 0.05 | `phs_hetero_core.py`, Figure scripts |

† Figure 1 uses different mechanical parameters (`c0 = 1.0`, `k_visc = 0.1`) because it simulates the original Nash-Panfilov model without explicit gyrator back-EMF, and a faster mechanical coupling rate is needed for the small 3-cell cluster to produce visible tension transients within the 150 ms simulation window.

**Equations**:

```
k_u(u)    = 1 / (1 + exp(−s·(u − u_thr)))                             (NM.3)
dT_ai/dt  = c₀ · (k_u(u) − T_ai)                                      (NM.1)
dT_a/dt   = k_visc · (T_ai − T_a)                                     (NM.2)
```

---

## 3. Port-Hamiltonian Gyrator Coupling Parameters

Govern the skew-symmetric energy transduction between electrical (u) and mechanical (T_ai) domains. The gyrator guarantees `P_net ≡ 0` at the interface (Rule 2 of the compositional grammar).

| Symbol | Name | SI Unit | Fig 1 | Fig S1 | Figs 3,S3 | Fig 4 | 3D (Fig 2) | Defined in |
|--------|------|---------|-------|--------|-----------|-------|------------|------------|
| `α_fb` (alpha_fb) | Gyrator coupling strength | dimensionless | —* | 1.0† | 0.05 | 0.05 | 0.05 | `phs_hetero_core.py`, `phs_3d_core.py` |
| `G_cpl` / `G_cleft` | Cleft conductance (Fig 1) / equivalent coupling | dimensionless | 1.0 | 1.0 | — | — | — | `Figure1.py` |

\* Figure 1 uses the original Nash-Panfilov formulation without explicit gyrator feedback (no back-EMF).  
† Figure S1 uses `G_couple = 1.0` as the gyrator constant in the linearised canonical transducer.  

**Equations**:

```
flow_act   = α_fb · σ(u) · u                                          (GYR.1)
I_feedback = α_fb · σ(u) · T_ai                                       (GYR.2)
σ(u)       = 1 / (1 + exp(−s_sig · (u − u_sig)))                      (GYR.3)
```

**Power balance**: `P_net = I_feedback · u + flow_act · T_ai = α_fb · σ(u) · (T_ai·u − u·T_ai) = 0`

---

## 4. Hypergraph Laplacian & Diffusion Parameters

Govern the spatial coupling between nodes via the condensed hypergraph Laplacian.

| Symbol | Name | SI Unit | Fig 1 | Figs 3,S3 | 3D (Fig 2) | Defined in |
|--------|------|---------|-------|-----------|------------|------------|
| `G_cleft` | Cleft (extracellular) conductance | dimensionless | 1.0 | — | — | `Figure1.py` |
| `σ_L` (SIG_L) | Longitudinal conductivity | dimensionless | — | 1.0 (healthy) / 0.1 (fibrotic) | 0.3×10⁻³ S/cm | `phs_hetero_core.py`, `phs_3d_core.py` |
| `σ_T` (SIG_T) | Transverse conductivity | dimensionless | — | 0.2 (healthy) / 0.1 (fibrotic) | 0.1×10⁻³ S/cm | `phs_hetero_core.py`, `phs_3d_core.py` |
| `σ_N` | Normal (cross-sheet) conductivity | S/cm | — | — | 0.1×10⁻³ | `phs_3d_core.py` |
| `ε` (div_eps) | Regularisation constant | — | — | 10⁻¹⁶ | 10⁻¹⁶ | `phs_hetero_core.py`, `phs_3d_core.py` |

**Harmonic mean interfacial conductance** (series-resistance model):

```
g_eff = 2 · σ_i · σ_j / (σ_i + σ_j + ε)                               (HL.1)
```

This correctly models the series topology of the static condensation and prevents non-physical current leakage at fibrosis boundaries.

---

## 5. Metabolic Autoregulation Parameters

Govern the dynamic upregulation of the effective reaction rate `k_eff` in response to electrotonic load (source-sink mismatch).

| Symbol | Name | SI Unit | Figs 3,S3 | 3D (Fig 2) | Defined in |
|--------|------|---------|-----------|------------|------------|
| `k_base` | Baseline metabolic rate | ms⁻¹ | 8.0 | 8.0 | `phs_hetero_core.py`, `phs_3d_core.py` |
| `k_recruit` | Metabolic recruitment gain | ms⁻¹ | **5.0**† | **2.0**† | `phs_hetero_core.py`, `phs_3d_core.py` |
| `k_max` | Metabolic ceiling | ms⁻¹ | **20.0**† | **30.0**† | `phs_hetero_core.py`, `phs_3d_core.py` |

† The 3D kernel uses lower recruitment gain (`k_recruit = 2.0` vs `5.0`) but a higher ceiling (`k_max = 30.0` vs `20.0`) than the 2D fibrosis kernel. These values were tuned for the larger spatial domain and different anisotropy of the ventricular wedge. They affect metabolic stress map magnitudes but do not change the qualitative behaviour.

**Equation**:

```
k_eff(i) = min(k_base + k_recruit · |φ_diff(i)|_neg,  k_max)         (MA.1)
```

where `|φ_diff(i)|_neg` is the magnitude of the outward (negative) diffusive flux at node i.

---

## 6. Autonomic Control Parameters

Govern the dual-mode sympathetic neuro-controller (Figure 4 only).

| Symbol | Name | SI Unit | Value | Defined in |
|--------|------|---------|-------|------------|
| `G_min` | Minimum inotropy gain | dimensionless | 0.5 | `Figure4.py`, `phs_models.py` |
| `G_max` | Maximum inotropy gain | dimensionless | 5.0 | `Figure4.py`, `phs_models.py` |
| `k_gain_ctrl` | Sympathetic recruitment rate | ms⁻¹ | 1.8 | `Figure4.py` |
| `k_decay_ctrl` | cAMP enzymatic degradation rate | ms⁻¹ | 0.12 | `Figure4.py` |
| `k_rest` | Baseline metabolic rate at rest | ms⁻¹ | 8.0 | `Figure4.py` |
| `k_gain_metab` | Metabolic recruitment scaling factor | ms⁻¹ | 2.5 | `Figure4.py` |
| `T_demand_rest` | Target tension at rest | kPa | 0.35 | `Figure4.py` |
| `T_demand_ex` | Target tension during exercise | kPa | 0.95 | `Figure4.py` |

**Equations**:

```
dG/dt  = k_gain_ctrl · (T_demand − T_peak) − k_decay_ctrl · (G − G_min)  (AC.1)
k(t)   = k_rest + k_gain_metab · (G(t) − G_min)                           (AC.2)
```

---

## 7. Tissue Topology & Fibrosis Parameters

Govern the spatial heterogeneity of the 2D tissue models (Figure 3, S2, S3).

| Symbol | Name | Value | Defined in |
|--------|------|-------|------------|
| `N` | Grid dimension (2D) | 80 × 80 nodes | `run_experiment_ensemble.py` |
| `burden` | Fibrosis burden (fraction fibrotic) | 40% | `run_experiment_ensemble.py` |
| `σ_filter` | Gaussian filter sigma (patchy fibrosis) | 2.0 grid units | `run_experiment_ensemble.py` |
| `σ_fibrotic` | Conductivity in fibrotic regions | 0.1 (10% of healthy) | `run_experiment_ensemble.py` |
| `k_visc_fibrotic` | Viscosity in fibrotic regions | 0.25 (5× healthy) | `run_experiment_ensemble.py` |
| `anisotropy_ratio` | σ_L / σ_T in healthy tissue | 5:1 (1.0 / 0.2) | `run_experiment_ensemble.py` |

**3D Ventricular Wedge Parameters** (Figure 2):

| Symbol | Name | Value | Defined in |
|--------|------|-------|------------|
| `NX, NY, NZ` | 3D grid dimensions | 140×140×40 (scalable) | `run_3d_experiment_spiral.py` |
| `DX, DY, DZ` | Grid spacings | 0.03 cm (X,Z); 0.03×175/200 cm (Y) | `run_3d_experiment_spiral.py` |
| `ANGLE_ENDO` | Fibre angle at endocardium | −60° | `run_3d_experiment_spiral.py` |
| `ANGLE_EPI` | Fibre angle at epicardium | +60° | `run_3d_experiment_spiral.py` |

---

## 8. Numerical & Simulation Parameters

| Symbol | Name | Value | Used in |
|--------|------|-------|---------|
| `dt_internal` | Internal RK4 time step (2D) | 0.005 ms | `phs_hetero_core.py` |
| `DT` | Time step (3D operator splitting) | 0.005 ms | `run_3d_experiment_spiral.py` |
| `T_MAX` / `T_END` | Simulation duration (2D) | 120 ms | `run_experiment_ensemble.py` |
| `T_MAX` | Simulation duration (3D) | 500 ms | `run_3d_experiment_spiral.py` |
| `subsample` | History subsampling factor | 5 | `phs_hetero_core.py` |
| `rtol` | Integrator relative tolerance (Figure 1) | 10⁻⁶ | `Figure1.py` |
| `rtol, atol` | Integrator tolerance (Figure S1) | 10⁻⁹ | `FigureS1.py` |
| `S1_start, S1_dur` | S1 planar stimulus timing | 1.0 ms, 2.0 ms | `run_experiment_ensemble.py` |
| `S2_start, S2_dur` | S2 cross-field stimulus timing | 42.0 ms, 2.0 ms | `run_experiment_ensemble.py` |
| `I_stim` | Stimulus current amplitude | 5.0 (dimensionless) | `Figure1.py`, `run_experiment_ensemble.py` |
| `n_ensemble` | Number of stochastic realisations (fibrotic) | 50 | `run_experiment_ensemble.py` |

---

## 9. Linearised Canonical Transducer Parameters (Figure S1)

Used exclusively in Figure S1 to validate the pure Dirac structure (all AP kinetics disabled).

| Symbol | Name | Value | Unit |
|--------|------|-------|------|
| `C_m` | Membrane capacitance | 1.0 | normalised |
| `C_act` | Activation capacitance | 2.0 | normalised |
| `R_leak` | Electrical leak resistance | 0.05 | normalised |
| `R_mech` | Mechanical damping resistance | 0.05 | normalised |
| `G_couple` | Gyrator coupling constant | 1.0 | normalised |

---

## 10. Per-Figure Parameter Summary

| Figure | Model | Key Differences from Baseline |
|--------|-------|-------------------------------|
| **Fig 1** | 3-cell cluster | `c0=1.0`, `k_visc=0.1`, `u_sig=0.1`, no gyrator back-EMF, `G_cleft=1.0` |
| **Fig S1** | Linearised transducer | `k=0` (AP off), `G_couple=1.0`, `C_act=2.0`, `R_leak=R_mech=0.05` |
| **Figs 3, S3** | 2D fibrosis | `PARAMS_2D_FIBROSIS`: `a=0.1`, `c0=0.05`, `α_fb=0.05`, `k_recruit=5.0`, `k_max=20.0` |
| **Fig S2** | Tissue maps only | Same topology generation as Figs 3,S3; no dynamics |
| **Fig 4** | Autonomic control | `PARAMS_AUTONOMIC`: `α_fb=0.15`, `s_sig=100.0`, plus controller parameters (§6) |
| **Fig 2** | 3D scaling | `PARAMS_3D_SCALING`: `a=0.05`, `k_recruit=2.0`, `k_max=30.0`, `σ_L=0.3e-3`, `σ_T=0.1e-3` |

---

## Notes on Units

- All state variables (u, v, T_ai) are **dimensionless** normalised quantities.
- Active tension T_a is reported in **kPa** in Figure 4 (autonomic control) and in normalised units elsewhere.
- Time is in **milliseconds (ms)** throughout.
- Conductivities in the 3D kernel use **S/cm** (siemens per centimetre); in the 2D kernel they are dimensionless normalised values.
- Stored energy H and dissipated energy are reported in **joules (J)** in Figures 2 and 3, representing the normalised Hamiltonian integral over the tissue volume.
- Power quantities (P_diff, P_mech, P_stim) are reported in **watts (W)**.

---

 Last updated 2025-08-20.