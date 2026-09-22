# Photonics: Silicon Photonic Tensor Accelerator Digital Twin

A second-order, experimentally-validated silicon photonic integrated circuit (PIC) simulation, autograd, and compilation engine built with PyTorch. Calibrated against 220 nm Silicon-on-Insulator (SOI) commercial foundry process design kits (PDKs) such as AIM Photonics, IMEC, and AMF.

> 📖 **Deep-Dive Architecture & Developer Guide:** For a comprehensive mathematical breakdown, tensor execution diagrams, and developer onboarding instructions, see [`ARCHITECTURE_AND_DEVELOPER_GUIDE.md`](ARCHITECTURE_AND_DEVELOPER_GUIDE.md).

---

## 🌟 Overview

The **Photonics** engine bridges the gap between deep learning software (PyTorch) and analog silicon photonic tensor computing hardware. It models physical non-idealities with second-order accuracy, enables noise-aware training (NAT), and provides hardware-level diagnostic evaluations (ENOB, SINAD, optical energy per MAC).

### Key Highlights
- **100% PyTorch & Autograd Native:** Every physical layer is differentiable via PyTorch autograd and Straight-Through Estimators (STE).
- **GPU Accelerated:** Scalable batch execution on CUDA with $>130,000$ vector propagations/sec on consumer GPUs.
- **Physical Realism:** 14 verified physical phenomena including non-local heat conduction, modal birefringence, optical nonlinearities, and backreflections.

---

## 🏗️ Architecture & Module Organization

```
photonics/
├── src/
│   ├── config.py             # Foundry PDK constants, physical tolerances & config
│   ├── calibration/          # Empirical wafer parameter identification & Bayesian fitting
│   │   └── parameter_fitting.py
│   ├── compiler/             # Amortized Neural Inverse Compiler & FFT compression
│   │   └── inverse_compiler.py
│   ├── models/               # Differentiable Photonic Mesh Digital Twin
│   │   └── digital_twin.py
│   ├── nn/                   # Learnable Step-Size Quantizer (LSQ) & DAC resolution
│   │   └── quantizer.py
│   ├── physics/              # Physics modeling engine
│   │   ├── mzi.py            # 2x2 MZI transfer matrix & directional coupler dispersion
│   │   ├── thermal_2d.py     # 2D Finite-Difference screened Poisson heat equation solver
│   │   ├── thermal.py        # Thermal crosstalk Green's function & BNNLS predistortion
│   │   ├── polarization.py   # Full Jones vector modal birefringence (TE/TM) & PDL
│   │   ├── nonlinear_optics.py # TPA, FCA, SPM, and FCD dispersion (RK4 CW solver)
│   │   ├── backreflection.py # Coherent Fabry-Pérot multi-cavity standing waves
│   │   ├── bend_loss.py      # Conformal mapping bend radiation loss & mismatch
│   │   ├── temporal_noise.py # 1/f flicker noise, micro-heater aging & dark current drift
│   │   ├── photodiode.py     # Multi-mode readout (direct, balanced, homodyne I/Q)
│   │   ├── spatial_wafer.py  # Gaussian Random Field (GRF) PVT wafer perturbations
│   │   └── routing_loss.py   # Waveguide crossings & progressive attenuation
│   ├── runtime/              # Closed-loop Thermal Eigenmode Decomposition (TED) daemon
│   │   └── ted_daemon.py
│   ├── training/             # Physics-aware loss functions
│   │   └── losses.py         # In-Situ Adjoint, Fidelity, Unitary Drift & Thermal losses
│   └── utils/                # Hardware evaluation & decomposition tools
│       ├── evaluations.py    # ENOB, SINAD, fJ/MAC & matrix fidelity audits
│       └── decomposition.py  # Universal Clements decomposition with diagonal screen D
│
├── tests/                    # 19 automated test & benchmark suites
│   ├── test_unitarity.py
│   ├── test_gradient_flow.py
│   ├── test_physics_validation.py
│   ├── test_matrix_field_equivalence.py
│   ├── test_calibration_fitting.py
│   ├── test_thermal_2d.py
│   ├── test_polarization.py
│   ├── test_nonlinear_optics.py
│   ├── test_backreflection.py
│   ├── test_bend_loss.py
│   ├── test_temporal_noise.py
│   ├── test_phase2_integration.py
│   ├── test_advanced_evaluations_and_losses.py
│   └── benchmark_performance.py
│
├── run_all_tests.py          # Master test & benchmark execution script
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Package metadata & build configuration
└── .gitignore                # Git ignore rules for clean repository
```

---

## 🔬 Core Physics Capabilities

1. **2D Finite-Difference Thermal Solver (`src/physics/thermal_2d.py`):**
   - Solves the steady-state screened Poisson/Helmholtz heat equation on the SOI device layer with heat dissipation through the BOX layer.
   - Computes multi-heater Green's function matrix $G = G^T$ and inverts thermal crosstalk: $\mathbf{P}_{\text{drive}} = K^{-1} \mathbf{\Delta T}$.

2. **Dual-Polarization Jones Vectors & PDL (`src/physics/polarization.py`):**
   - Full $2 \times 2$ state tracking: $\mathbf{E}_p = [E_{p, \text{TE}}, E_{p, \text{TM}}]^T$.
   - Models large silicon structural birefringence ($\Delta n_{\text{eff}} = 2.445 - 1.785 = 0.660$), bend rotation cross-coupling, and polarization-dependent loss (PDL).

3. **Silicon Optical Nonlinearities (`src/physics/nonlinear_optics.py`):**
   - High power density effects: Two-Photon Absorption (TPA, $\beta_{\text{TPA}}$), Free-Carrier Absorption (FCA, $\sigma_{\text{FCA}}$), Self-Phase Modulation (SPM, $n_2$), and Free-Carrier Dispersion (FCD).

4. **Coherent Fabry-Pérot Backreflections (`src/physics/backreflection.py`):**
   - Boundary discontinuities at grating couplers ($-25\text{ dB}$), waveguide crossings ($-35\text{ dB}$), and couplers ($-40\text{ dB}$) forming multi-cavity coherent standing-wave ripples.

7. **Universal $U(N)$ Clements Synthesis & Field Equivalence (`src/utils/decomposition.py`):**
   - Canonical Clements mesh decomposition providing $N(N-1)$ MZI internal/external phase degrees of freedom plus an $N$-element output diagonal phase screen $D$ ($N^2$ total real degrees of freedom).
   - Exact analytical commutation through diagonal phase screens and topological bubble sorting matching the physical column schedule.
   - Guaranteed machine-precision reconstruction ($\|U_{\text{target}} - U_{\text{recon}}\|_F < 10^{-14}$, fidelity $1.00000000000000$) on Haar random unitaries.
   - Rigorous mathematical equivalence between $O(N)$ field propagation `propagate_field()` and $O(N^2)$ transfer matrix multiplication `compute_transfer_matrix()` ($< 10^{-14}$ error across all modes and loss settings).

8. **Bounded Non-Negative Thermal Predistortion (`src/physics/thermal.py`):**
   - Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) solving Bounded Non-Negative Least Squares (BNNLS) for thermo-optic crosstalk inversion.
   - Strictly enforces box physical constraints $0 \le \theta_{\text{drive}} \le \theta_{\max}$ (no unphysical negative Joule heating).
   - Incorporates temperature coefficient of resistance (TCR) electro-thermal feedback.

9. **Empirical Parameter Identification & Calibration (`src/calibration/parameter_fitting.py`):**
   - Differentiable Bayesian parameter estimator distinguishing the prior nominal foundry hypothesis from the empirical hardware model.
   - Estimates directional coupler split imbalances $(\epsilon_1, \epsilon_2 \in [-0.25, 0.25])$ and static lithographic phase biases $(\phi_{\text{int}})$ from diagnostic optical transmission sweeps.
   - Achieves $>80\%$ error reduction without invasive physical characterization.

10. **Unified Optical Readout Hierarchy (`src/physics/photodiode.py`):**
    - Multi-mode photodetection supporting:
      - `direct`: Single-ended square-law detection with physical dark current ($I_{\text{dark}}$) DC baseline and shot noise.
      - `dual_rail`: Balanced differential photodiode pair ($I_1 - I_2$) eliminating common-mode DC drift.
      - `homodyne_i` & `homodyne_q`: Coherent local oscillator (LO) mixing recovering full in-phase ($I$) and quadrature ($Q$) complex optical field amplitudes.

---

## 🎯 Physics Losses & Hardware Evaluation Metrics

### Physics-Aware Losses (`src/training/losses.py`)
- **`InSituAdjointLoss`:** Simulates on-chip backpropagation via forward-backward optical field overlap integrals ($O(1)$ scaling, Hughes et al., *Optica* 2018; Pai et al., *Science* 2023).
- **`PhotonicFidelityLoss`:** Trace fidelity: $\mathcal{L}_{\text{fid}} = 1 - |\text{Tr}(U^\dagger V)|^2 / N^2$.
- **`UnitaryDriftPenalty`:** Penalizes non-unitary transmission from insertion loss and asymmetric coupling: $\frac{1}{N} \|U^\dagger U - I\|_F^2$.
- **`SpatialThermalGradientPenalty`:** Penalizes rapid temperature jumps between neighboring heaters: $\sum_{\langle i, j \rangle} (P_i - P_j)^2$.

### Hardware Evaluation Suite (`src/utils/evaluations.py`)
- **Effective Number of Bits (ENOB):** Signal-to-Noise-and-Distortion (SINAD) conversion: $\text{ENOB} = (\text{SINAD}_{\text{dB}} - 1.76) / 6.02$.
- **Optical Energy per MAC:** Quantifies optical joules ($\text{fJ/MAC}$) and mean photon counts using Poisson quantum detection limits (Wang & McMahon, *Nature Comm.* 2022).
- **Full Hardware Report:** Automated multi-point audit verifying fidelity, ENOB, energy, and thermal compliance.

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/photonics.git
cd photonics

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install in editable mode
pip install -e .
```

### Running the Test Suite

Execute all 17 verification and benchmark suites:

```bash
# Run all tests (headless text report)
python run_all_tests.py

# Run all tests and generate publication-grade visual diagnostics & plots
python run_all_tests.py --plot

# Or generate the test plots directly
python tests/generate_test_plots.py
```

All generated visual diagnostics are saved in high resolution (300 DPI) under `reports/plots/`:
1. `01_unitarity_and_reconstruction.png`: Haar unitary decomposition & exact field-matrix equivalence
2. `02_field_matrix_equivalence.png`: Output field profiles & error residuals across all modes
3. `03_silicon_nonlinear_optics.png`: Optical transmission vs input power, TPA + FCA, and SPM nonlinear phase shift
4. `04_thermal_and_bnnls_predistortion.png`: Thermal Green's function coupling matrix and BNNLS crosstalk cancellation
5. `05_hardware_parameter_calibration.png`: Convergence of coupler split error $(\epsilon_1, \epsilon_2)$ and lithographic phase offsets
6. `06_readout_and_noise_breakdown.png`: Photodiode direct, balanced dual-rail, and coherent homodyne I/Q constellation
7. `07_dispersion_and_backreflection.png`: C-band chromatic dispersion and Fabry-Pérot multi-cavity backreflection ripples
8. `08_gpu_benchmarks.png`: Forward latency and vector throughput scaling across mesh sizes ($N=4 \dots 64$)
9. `09_executive_test_dashboard.png`: Unified 6-panel executive test and health dashboard

Run individual test suites via `pytest`:

```bash
pytest tests/test_unitarity.py
pytest tests/test_thermal_2d.py
pytest tests/test_advanced_evaluations_and_losses.py
```

---

## 📊 Benchmark Throughput (NVIDIA RTX 3050 Ti Laptop GPU / CUDA)

| Mesh Size | Active MZIs | Optical Depth | Batch 1024 Latency | Throughput (vectors/sec) |
| :---: | :---: | :---: | :---: | :---: |
| **$N = 4$** | 6 | 4 | $7.45\text{ ms}$ | **$137,483\text{ vec/s}$** |
| **$N = 8$** | 28 | 8 | $15.09\text{ ms}$ | **$67,875\text{ vec/s}$** |
| **$N = 16$** | 120 | 16 | $35.44\text{ ms}$ | **$28,896\text{ vec/s}$** |
| **$N = 32$** | 496 | 32 | $133.44\text{ ms}$ | **$7,674\text{ vec/s}$** |

---

## 📄 License

MIT License.
