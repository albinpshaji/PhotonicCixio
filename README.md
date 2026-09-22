# Photonics: Silicon Photonic Tensor Accelerator Digital Twin

A second-order, experimentally-validated silicon photonic integrated circuit (PIC) simulation, autograd, and compilation engine built with PyTorch. Calibrated against 220 nm Silicon-on-Insulator (SOI) commercial foundry process design kits (PDKs) such as AIM Photonics, IMEC, and AMF.

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
│   ├── compiler/             # Amortized Neural Inverse Compiler & FFT compression
│   │   └── inverse_compiler.py
│   ├── models/               # Differentiable Photonic Mesh Digital Twin
│   │   └── digital_twin.py
│   ├── nn/                   # Learnable Step-Size Quantizer (LSQ) & DAC resolution
│   │   └── quantizer.py
│   ├── physics/              # Physics modeling engine
│   │   ├── mzi.py            # 2x2 MZI transfer matrix & directional coupler dispersion
│   │   ├── thermal_2d.py     # 2D Finite-Difference screened Poisson heat equation solver
│   │   ├── thermal.py        # Thermal crosstalk Green's function & predistortion
│   │   ├── polarization.py   # Full Jones vector modal birefringence (TE/TM) & PDL
│   │   ├── nonlinear_optics.py # TPA, FCA, SPM, and FCD dispersion
│   │   ├── backreflection.py # Coherent Fabry-Pérot multi-cavity standing waves
│   │   ├── bend_loss.py      # Conformal mapping bend radiation loss & mismatch
│   │   ├── temporal_noise.py # 1/f flicker noise, micro-heater aging & dark current drift
│   │   ├── photodiode.py     # Square-law detection, shot, thermal & RIN noise
│   │   ├── spatial_wafer.py  # Gaussian Random Field (GRF) PVT wafer perturbations
│   │   └── routing_loss.py   # Waveguide crossings & progressive attenuation
│   ├── runtime/              # Closed-loop Thermal Eigenmode Decomposition (TED) daemon
│   │   └── ted_daemon.py
│   ├── training/             # Physics-aware loss functions
│   │   └── losses.py         # In-Situ Adjoint, Fidelity, Unitary Drift & Thermal losses
│   └── utils/                # Hardware evaluation & decomposition tools
│       ├── evaluations.py    # ENOB, SINAD, fJ/MAC & matrix fidelity audits
│       └── decomposition.py  # Clements & Reck matrix factorizations
│
├── tests/                    # 17 automated test & benchmark suites
│   ├── test_unitarity.py
│   ├── test_gradient_flow.py
│   ├── test_physics_validation.py
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

5. **Waveguide Bend Radiation Loss (`src/physics/bend_loss.py`):**
   - Conformal mapping Marcuse radiation loss $\alpha_{\text{bend}}(R) = C_1 e^{-C_2 R}$ with S-bend minimum radius routing constraints.

6. **1/f Low-Frequency Flicker Noise & Aging (`src/physics/temporal_noise.py`):**
   - Hooge's low-frequency flicker noise in photodiode/TIA readout, Arrhenius micro-heater resistance drift, and trap-state dark current aging.

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
python run_all_tests.py
```

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
