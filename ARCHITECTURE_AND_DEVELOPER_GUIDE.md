# Silicon Photonic Tensor Accelerator Digital Twin
## Complete System Architecture, Physics Foundations & Developer Onboarding Guide

**Target Audience:** AI systems engineers, photonic hardware designers, research scientists, and autonomous AI agents working on or extending this codebase.  
**Platform Target:** Silicon-on-Insulator (SOI, 220 nm Si device layer / 2 µm $\text{SiO}_2$ BOX substrate) multi-channel coherent optical matrix accelerators.  
**Framework:** PyTorch (Autograd Differentiable, CUDA GPU Accelerated).

---

## Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [End-to-End System Architecture & Data Flow](#2-end-to-end-system-architecture--data-flow)
3. [Governing Mathematical Models & Physical Effects](#3-governing-mathematical-models--physical-effects)
4. [Codebase Map & Module Reference](#4-codebase-map--module-reference)
5. [Step-by-Step Forward & Backward Execution Pipeline](#5-step-by-step-forward--backward-execution-pipeline)
6. [Loss Functions & Hardware Evaluation Metrics](#6-loss-functions--hardware-evaluation-metrics)
7. [Developer & AI Agent Extension Guide](#7-developer--ai-agent-extension-guide)
8. [Testing & Benchmarking Cheat Sheet](#8-testing--benchmarking-cheat-sheet)

---

## 1. Executive Summary & Problem Statement

### 1.1 The Promise of Optical Computing
Coherent photonic integrated circuits (PICs) perform matrix-vector multiplication (MVM) at the speed of light by transmitting coherent laser signals through an interconnected mesh of tunable Mach-Zehnder Interferometers (MZIs) and directional couplers:
$$\mathbf{E}_{\text{out}} = U \mathbf{E}_{\text{in}}$$
Because light propagation through silicon waveguides takes only $\sim 100\text{ ps}$ across a full chip with near-zero runtime energy for matrix multiplication, photonic accelerators promise orders-of-magnitude higher throughput (tera-MAC/s) and energy efficiency (sub-fJ/MAC) compared to digital electronic processors (GPUs/TPUs).

### 1.2 The Physical Reality (The Analog Usability Gap)
Mathematical linear algebra assumes ideal, lossless, infinite-precision unitary operators ($U^\dagger U = I$). However, physical silicon chips fabricated in standard 220 nm SOI processes experience severe analog non-idealities:

1. **Thermal Crosstalk & Drift:** Silicon's high thermo-optic coefficient ($dn/dT \approx 1.86 \times 10^{-4}\text{ K}^{-1}$) means micro-heaters dissipate Joulean heat that diffuses non-locally into neighboring interferometers ($\lambda_{\text{th}} \approx 55\,\mu\text{m}$), perturbing phase settings by several radians.
2. **Coupler Imprecisions:** Lithographic roughness causes directional coupler splitting ratios to deviate ($\kappa = 0.5 \pm \epsilon$), imposing an extinction ratio floor ($T_{\min} \approx 4\epsilon^2$) and preventing perfect zero transmission.
3. **Broken Unitarity:** Waveguide propagation loss ($1.85\text{ dB/cm}$), bend radiation loss, and coupler insertion loss ($0.15\text{ dB/stage}$) attenuate optical fields asymmetrically, destroying mathematical unitarity ($U^\dagger U \ne I$).
4. **Birefringence & Polarization-Dependent Loss (PDL):** Sub-micron waveguides exhibit huge structural birefringence ($n_{\text{eff, TE}} \approx 2.45$ vs $n_{\text{eff, TM}} \approx 1.78$). Mode mixing at bends and crossings degrades signal integrity.
5. **Optical Nonlinearities:** At optical powers above $\sim 10\text{ mW}$, high spatial confinement ($A_{\text{eff}} \approx 0.1\,\mu\text{m}^2$) triggers Two-Photon Absorption (TPA) and Free-Carrier Absorption (FCA), causing power-dependent saturation.
6. **Cavity Backreflections:** Grating couplers and crossings reflect $-25\text{ dB}$ to $-35\text{ dB}$ of coherent light backward, forming multi-cavity Fabry-Pérot resonant ripples across the C-band.
7. **Readout Noise & Low Resolution:** Square-law photodiode detection is corrupted by Poisson shot noise, TIA thermal noise, laser RIN, and low-frequency 1/f flicker noise. Combined with 4- to 8-bit DAC quantization, the **effective resolution drops to 2 to 4 bits (ENOB)**.

### 1.3 How This Framework Solves the Problem
Directly deploying standard deep learning models (from PyTorch/ONNX) onto uncompensated hardware causes benchmark accuracy to collapse from $>95\%$ to $<40\%$ (random guessing).

This framework provides:
- A **Differentiable Digital Twin** that simulates all 14 physical phenomena from first principles.
- An **Amortized Neural Inverse Compiler** that predicts pre-distorted phase configurations in $<1\text{ ms}$.
- **Noise-Aware Training (NAT)** and **Learnable Step-Size Quantization (LSQ)** to make neural models robust to hardware noise.
- An **In-Situ Adjoint Backpropagation Engine** simulating on-chip optical gradient calculation.

---

## 2. End-to-End System Architecture & Data Flow

```
                      Target PyTorch Tensor / Weight Matrix W
                                         │
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │ Amortized Neural Inverse Compiler (G_Omega)  │
                 │ - Analytical Clements Decomposition          │
                 │ - Neural Phase Parameter Inversion           │
                 │ - Thermal Crosstalk Pre-distortion (K^-1)    │
                 └───────────────────────┬──────────────────────┘
                                         │ Target Phases (θ, ϕ)
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │ DAC Quantization Layer (quantizer.py)        │
                 │ - 4 to 8 bit Phase Discretization            │
                 │ - INL / DNL Non-linearity Modeling           │
                 │ - Straight-Through Estimators (STE) Gradients│
                 └───────────────────────┬──────────────────────┘
                                         │ Discrete Drive Voltages / Powers
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PHOTONIC MESH DIGITAL TWIN (src/models/digital_twin.py)                                │
│                                                                                        │
│   1. 2D Finite-Difference Thermal Solver (src/physics/thermal_2d.py)                   │
│      - Solves steady-state screened Poisson heat equation: -∇²T + T/λ² = Q/(κ t)       │
│      - Computes die-wide temperature field T(x, y) & inter-heater coupling matrix K   │
│                                                                                        │
│   2. Directional Coupler & MZI Core (src/physics/mzi.py)                              │
│      - Dual-drive symmetric MZI transfer matrix with common/differential phases       │
│      - Foundry coupler imprecisions (κ = 0.50 ± 0.04) and extinction limit T_min       │
│      - C-band wavelength dispersion: n_eff(λ) = n_0 + (dn/dλ)(λ - λ_0)                │
│                                                                                        │
│   3. Full Jones Vector Polarization Engine (src/physics/polarization.py)               │
│      - 2x2 state tracking: [E_TE, E_TM]^T                                              │
│      - Modal birefringence (Δn_eff = 0.66) and differential crossing PDL (dB)          │
│                                                                                        │
│   4. Optical Nonlinearity Pipeline (src/physics/nonlinear_optics.py)                   │
│      - Power-dependent Two-Photon Absorption (TPA) & Free-Carrier Absorption (FCA)     │
│      - Self-Phase Modulation (SPM) & Free-Carrier Dispersion (FCD)                     │
│                                                                                        │
│   5. Progressive Loss & Geometry (src/physics/routing_loss.py, bend_loss.py)          │
│      - Waveguide propagation loss (1.85 dB/cm) & crossing insertion loss (0.04 dB/ea)  │
│      - Conformal mapping bend radiation loss α_bend(R) = C1 * exp(-C2 * R)             │
│                                                                                        │
│   6. Multi-Cavity Fabry-Pérot Backreflection (src/physics/backreflection.py)           │
│      - Grating coupler (-25 dB) & crossing (-35 dB) boundary reflections               │
│      - Wavelength transmission ripples across C-band (1530 - 1565 nm)                  │
│                                                                                        │
│   7. Photodetector & Receiver Readout Stack (src/physics/photodiode.py)                │
│      - Square-law optical power detection: P = |E_TE|² + |E_TM|²                       │
│      - Poisson shot noise + TIA thermal Johnson noise + Laser RIN                      │
│      - Hooge 1/f low-frequency flicker noise & temporal aging drift                    │
│      - Balanced Photodiode (BPD) CMRR & TIA voltage saturation clipping                │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │ Hardware Metrics & Loss Diagnostics          │
                 │ - Effective Number of Bits (ENOB) & SINAD    │
                 │ - Optical Energy per MAC (fJ/MAC, photons)   │
                 │ - Matrix Trace Fidelity: F(U, V)             │
                 │ - In-Situ Adjoint Backprop Gradients         │
                 └──────────────────────────────────────────────┘
```

---

## 3. Governing Mathematical Models & Physical Effects

### 3.1 The Elementary 2×2 Mach-Zehnder Interferometer (MZI)
Each elementary spatial unit is composed of two 50:50 directional couplers enclosing internal phase modulators ($\theta$) and external phase delays ($\phi$):
$$T_{\text{MZI}}(\theta, \phi) = \begin{bmatrix} e^{i\phi} & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} \sqrt{1-\kappa_2} & i\sqrt{\kappa_2} \\ i\sqrt{\kappa_2} & \sqrt{1-\kappa_2} \end{bmatrix} \begin{bmatrix} e^{i\theta} & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} \sqrt{1-\kappa_1} & i\sqrt{\kappa_1} \\ i\sqrt{\kappa_1} & \sqrt{1-\kappa_1} \end{bmatrix}$$
Under ideal 50:50 splitting ($\kappa_1 = \kappa_2 = 0.5$):
$$T_{\text{ideal}}(\theta, \phi) = i e^{i\theta/2} \begin{bmatrix} e^{i\phi} \sin(\theta/2) & e^{i\phi} \cos(\theta/2) \\ \cos(\theta/2) & -\sin(\theta/2) \end{bmatrix}$$

### 3.2 Coupler Imprecision & Extinction Leakage Floor
In commercial foundries, directional coupler power splitting deviates:
$$\kappa = 0.5 \pm \epsilon$$
When $\kappa \ne 0.5$, destructive interference cannot completely cancel light. The minimum transmission floor in the cross state is:
$$T_{\min} \approx 4\epsilon^2$$
For $\epsilon = 0.04$, $T_{\min} \approx 0.0064$ (extinction ratio limited to $\sim 21.9\text{ dB}$).

### 3.3 2D Finite-Difference Screened Poisson Thermal Solver
The steady-state temperature profile $T(x, y)$ in the thin silicon device layer satisfies the 2D screened Poisson equation:
$$-\nabla^2 T(x, y) + \frac{1}{\lambda_{\text{th}}^2} T(x, y) = \frac{Q(x, y)}{\kappa_{\text{Si}} t_{\text{Si}}}$$
where:
- $\lambda_{\text{th}} = \sqrt{\frac{\kappa_{\text{Si}} t_{\text{Si}} t_{\text{BOX}}}{\kappa_{\text{BOX}}}} \approx 55\,\mu\text{m}$ is the lateral thermal decay length.
- $Q(x, y)$ is the dissipated Joulean electrical heating power density ($\text{W/m}^2$).
- $\kappa_{\text{Si}} = 148\text{ W/(m}\cdot\text{K)}$, $\kappa_{\text{BOX}} = 1.4\text{ W/(m}\cdot\text{K)}$.
- $t_{\text{Si}} = 220\text{ nm}$, $t_{\text{BOX}} = 2.0\,\mu\text{m}$.

The framework solves this over the die grid via 5-point finite-difference stencils, computing the exact Green's function coupling matrix $G_{ij}$ and inverting multi-heater crosstalk:
$$\mathbf{P}_{\text{drive}} = K^{-1} \mathbf{\Delta T}_{\text{target}}$$

### 3.4 Full Jones Vector Polarization & Modal Birefringence
Waveguide fields are tracked as dual-polarization 2-vectors:
$$\mathbf{E}_p = \begin{bmatrix} E_{p, \text{TE}} \\ E_{p, \text{TM}} \end{bmatrix}$$
Silicon strip waveguides ($450 \times 220\text{ nm}$) possess massive structural birefringence:
$$\Delta n_{\text{eff}} = n_{\text{eff, TE}} - n_{\text{eff, TM}} = 2.445 - 1.785 = 0.660$$
Waveguide bends induce cross-polarization coupling modeled via rotation matrices:
$$J_{\text{rot}}(\theta_{\text{pol}}) = \begin{bmatrix} \cos\theta_{\text{pol}} & -\sin\theta_{\text{pol}} \\ \sin\theta_{\text{pol}} & \cos\theta_{\text{pol}} \end{bmatrix}$$
Waveguide crossings introduce differential Polarization-Dependent Loss (PDL):
$$\text{Loss}_{\text{TE}} \approx 0.025\text{ dB/crossing}, \quad \text{Loss}_{\text{TM}} \approx 0.080\text{ dB/crossing}$$

### 3.5 Silicon Optical Nonlinearities (TPA, FCA, SPM, FCD)
At high optical powers in sub-micron waveguides ($A_{\text{eff}} \approx 0.055\,\mu\text{m}^2$):
$$\frac{dP}{dz} = -\alpha_{\text{lin}} P - k_{\text{tpa}} P^2 - k_{\text{fca}} P^3$$
where calibrated 220 nm SOI constants are:
- $\beta_{\text{TPA}} = 6.5 \times 10^{-12}\text{ m/W}$ (Dinu et al., *Appl. Phys. Lett.* 2003)
- $k_{\text{tpa}} = \frac{\beta_{\text{TPA}}}{A_{\text{eff}}} \approx 118.2\text{ W}^{-1}\text{m}^{-1}$
- $k_{\text{fca}} = \frac{\sigma_{\text{FCA}} \tau_c \beta_{\text{TPA}}}{2 h \nu A_{\text{eff}}^2} \approx 3.04 \times 10^4\text{ W}^{-2}\text{m}^{-1}$ ($\tau_c = 2.5\text{ ns}$, $\sigma_{\text{FCA}} = 1.45 \times 10^{-21}\text{ m}^2$)
- $n_2 = 4.5 \times 10^{-18}\text{ m}^2/\text{W}$ (Kerr self-phase modulation)
- Critical runaway threshold: $P_{\text{crit}} = 50\text{ mW}$. Power above $50\text{ mW}$ triggers high-confinement thermal runaway warnings.
Integrated via a 4th-order Runge-Kutta (RK4) solver with autograd-stable non-negative power bounds.

### 3.6 Coherent Fabry-Pérot Multi-Cavity Backreflections
Boundary discontinuities at grating couplers ($R_1 = -25\text{ dB}$) and crossings ($R_2 = -35\text{ dB}$) create internal optical cavities. The multi-path transfer function is:
$$H_{\text{FP}}(\lambda) = \frac{\sqrt{(1 - R_1)(1 - R_2)}}{1 - \sqrt{R_1 R_2} \exp\left( i \frac{4\pi n_{\text{eff}} L_{\text{cav}}}{\lambda} \right)}$$
This generates periodic transmission ripples with Free Spectral Range:
$$\text{FSR} = \frac{\lambda_0^2}{2 n_g L_{\text{cav}}}$$

### 3.7 Readout Noise & Transimpedance Amplifier (TIA) Physics
The total electrical noise variance in the detected photocurrent is the sum of four uncorrelated stochastic processes:
$$\sigma_{\text{total}}^2 = \sigma_{\text{shot}}^2 + \sigma_{\text{thermal}}^2 + \sigma_{\text{RIN}}^2 + \sigma_{1/f}^2$$
where:
- **Poisson Shot Noise:** $\sigma_{\text{shot}}^2 = 2 q (\mathcal{R} P_{\text{opt}} + I_{\text{dark}}) B$
- **Johnson-Nyquist Thermal Noise:** $\sigma_{\text{thermal}}^2 = \frac{4 k_B T B}{R_L}$
- **Laser Relative Intensity Noise:** $\sigma_{\text{RIN}}^2 = \text{RIN} \cdot (\mathcal{R} P_{\text{opt}})^2 B$
- **Hooge 1/f Flicker Noise:** $\sigma_{1/f}^2 = \alpha_H I^2 \ln(f_c / f_{\min})$

Readout modes include single-ended direct detection (with physical dark current DC offset $I_{\text{dark}}$), dual-rail balanced detection ($I_1 - I_2$), and coherent homodyne ($I$ and $Q$ quadrature mixing with a local oscillator).

### 3.8 Universal $U(N)$ Clements Mesh Decomposition & Output Phase Screen
An arbitrary unitary matrix $U \in U(N)$ requires $N^2$ real degrees of freedom. A Clements triangular/rectangular mesh provides $M = N(N-1)/2$ MZIs, each with 2 phase shifters ($\theta_m, \phi_m$), totaling $N(N-1)$ parameters. Universal representation requires an additional $N$-element output diagonal phase screen:
$$U = D(\vec{\gamma}) \prod_{l=1}^L U_{\text{col}, l}(\vec{\theta}_l, \vec{\phi}_l), \quad D(\vec{\gamma}) = \text{diag}\left( e^{i \gamma_1}, \dots, e^{i \gamma_N} \right)$$
Our canonical decomposition (`src/utils/decomposition.py`) commutes all left and right nulling transformations through the diagonal matrix analytically, bubble-sorts disjoint operations into physical digital twin columns, and guarantees double-precision reconstruction ($\|U_{\text{target}} - U_{\text{recon}}\|_F < 10^{-14}$) on Haar unitaries.

### 3.9 Bounded Non-Negative Least Squares (BNNLS) Thermal Predistortion
Because micro-heaters operate strictly on Joulean dissipation ($P_j = I_j^2 R_j \ge 0$), drive powers cannot be negative. Unconstrained linear inversion ($P = K^{-1} \theta$) yields unphysical negative powers on large arrays.
We implement the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) solving:
$$\min_{\mathbf{P} \ge 0} \frac{1}{2} \|K \mathbf{P} - \boldsymbol{\theta}_{\text{target}}\|_2^2 + \frac{\lambda_{\text{tikh}}}{2} \|\mathbf{P}\|_2^2 \quad \text{s.t.} \quad 0 \le P_i \le P_{\max}$$
with $O(1/k^2)$ accelerated convergence and temperature-dependent resistance feedback:
$$R_i(T) = R_0 \left[ 1 + \alpha_{\text{TCR}} (T_i - T_0) \right]$$

### 3.10 Empirical Foundry-to-Hardware Parameter Calibration
Manufacturing variations induce directional coupler split imbalances ($\kappa = 0.5 \pm \epsilon$) and static lithographic phase offsets ($\phi_{\text{int}}$).
Our differentiable estimator (`src/calibration/parameter_fitting.py`) fits on-chip non-idealities from transmission matrices:
$$\min_{\boldsymbol{\epsilon}_1, \boldsymbol{\epsilon}_2, \boldsymbol{\phi}_{\text{int}}} \frac{1}{K} \sum_{k=1}^K \|T_{\text{PIC}}(\theta_k, \phi_k; \boldsymbol{\epsilon}, \boldsymbol{\phi}_{\text{int}}) - T_{\text{meas}, k}\|_F^2 + \lambda_{\text{prior}} \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\text{prior}}\|_2^2$$
reducing transfer matrix RMSE by $>80\%$ without opening the physical package.

---

## 4. Codebase Map & Module Reference

All simulation source code resides under [`photonics/src/`](src/):

| File Path | Primary Class / Functions | Description & Responsibilities |
| :--- | :--- | :--- |
| **[`src/config.py`](src/config.py)** | `PhotonicConfig`, `FoundryPDK` | Central configuration dataclass. Stores nominal waveguide dimensions, refractive indices, loss coefficients, temperature coefficients, noise parameters, and device precision. |
| **[`src/models/digital_twin.py`](src/models/digital_twin.py)** | `PhotonicMeshDigitalTwin` | Master PyTorch `nn.Module`. Manages mesh coordinate layout, cascades physical effect modules, executes forward optical propagation (field & matrix), and applies detector readout. |
| **[`src/calibration/parameter_fitting.py`](src/calibration/parameter_fitting.py)** | `MeshParameterEstimator`, `PhotonicCalibrationDataset` | Differentiable Bayesian optimizer identifying coupler split errors ($\epsilon_1, \epsilon_2$) and intrinsic phase biases ($\phi_{\text{int}}$) from diagnostic transmission sweeps. |
| **[`src/physics/mzi.py`](src/physics/mzi.py)** | `mzi_transfer_matrix`, `directional_coupler_matrix` | Evaluates analytic $2 \times 2$ transfer matrices for symmetric dual-drive MZIs with coupler splitting variations and wavelength dispersion. |
| **[`src/physics/thermal_2d.py`](src/physics/thermal_2d.py)** | `compute_fd_thermal_greens_function` | Solves the 2D screened Poisson heat equation on the SOI die using finite-difference stencils to produce the exact multi-heater Green's matrix. |
| **[`src/physics/thermal.py`](src/physics/thermal.py)** | `ThermalCrosstalkModel`, `predistort_bnnls` | Orchestrates steady-state thermal diffusion, self-heating, heater resistance thermal feedback, and BNNLS FISTA predistortion ($0 \le P \le P_{\max}$). |
| **[`src/physics/polarization.py`](src/physics/polarization.py)** | `JonesPolarizationModel` | Tracks full $2 \times 2$ Jones vectors ($E_{\text{TE}}, E_{\text{TM}}$), modal birefringence phase delays, bend rotation, and crossing PDL. |
| **[`src/physics/nonlinear_optics.py`](src/physics/nonlinear_optics.py)** | `SiliconNonlinearOptics` | Continuous-wave (CW) power-basis Runge-Kutta 4th order solver for TPA, FCA, Kerr SPM, and FCD dispersion, with 50 mW critical threshold monitoring. |
| **[`src/physics/backreflection.py`](src/physics/backreflection.py)** | `FabryPerotBackreflection` | Implements coherent multi-cavity standing waves, reflectance at grating couplers/crossings, and C-band spectral ripple filtering. |
| **[`src/physics/bend_loss.py`](src/physics/bend_loss.py)** | `WaveguideBendLossModel` | Calculates bend radiation loss via conformal mapping (Marcatili/Marcuse asymptotic) and S-bend geometry constraints. |
| **[`src/physics/temporal_noise.py`](src/physics/temporal_noise.py)** | `TemporalNoiseAgingModel` | Generates 1/f low-frequency flicker noise in photodetectors and computes long-term Arrhenius micro-heater resistance aging. |
| **[`src/physics/photodiode.py`](src/physics/photodiode.py)** | `PhotodetectorArray`, `detect_homodyne` | Converts complex optical fields to electrical signals supporting direct detection (with dark current $I_{\text{dark}}$), dual-rail balanced detection, and coherent homodyne ($I/Q$). |
| **[`src/physics/routing_loss.py`](src/physics/routing_loss.py)** | `ClementsPhysicalRouting` | Implements physical waveguide routing layouts, tracks crossing counts per channel, and applies progressive insertion loss. |
| **[`src/physics/spatial_wafer.py`](src/physics/spatial_wafer.py)** | `SpatialWaferMap` | Synthesizes correlated wafer-level process variations (width, thickness) via 2D Gaussian Random Fields (GRFs). |
| **[`src/nn/quantizer.py`](src/nn/quantizer.py)** | `DACQuantizer`, `STEQuantizeFunction` | Quantizes continuous phase shifts into $b$-bit discrete levels ($b \in [4, 8]$) with Straight-Through Estimators (STE) for autograd. |
| **[`src/compiler/inverse_compiler.py`](src/compiler/inverse_compiler.py)** | `GenerativePhotonicCompiler` | Deep neural inverse compiler predicting optimal phase configurations from target weight matrices in $<1\text{ ms}$. |
| **[`src/training/losses.py`](src/training/losses.py)** | `InSituAdjointLoss`, `PhotonicFidelityLoss`, `UnitaryDriftPenalty`, `SpatialThermalGradientPenalty`, `CompositePhotonicLoss` | Suite of physics-aware loss functions for noise-aware training, unitary drift prevention, and optical in-situ backpropagation. |
| **[`src/utils/evaluations.py`](src/utils/evaluations.py)** | `evaluate_effective_resolution_enob`, `evaluate_matrix_fidelity`, `evaluate_optical_energy_per_mac`, `evaluate_full_hardware_system` | Rigorous hardware evaluation tools computing SINAD, ENOB, optical energy (fJ/MAC, photons/MAC), and trace fidelity. |
| **[`src/utils/decomposition.py`](src/utils/decomposition.py)** | `clements_decompose_np`, `clements_reconstruct_torch` | Canonical Clements matrix decomposition with output diagonal phase screen $D$, exact commutation, and column bubble sorting for universal $U(N)$ synthesis. |

---

## 5. Step-by-Step Forward & Backward Execution Pipeline

### 5.1 The Forward Pass (`twin(E_in, theta, phi)`)

When an optical field vector $\mathbf{E}_{\text{in}} \in \mathbb{C}^{B \times N}$ is passed to `PhotonicMeshDigitalTwin`, the following pipeline executes:

```python
# 1. Quantize continuous phase controls to discrete DAC levels
theta_q = self.dac_quantizer(theta)
phi_q = self.dac_quantizer(phi)

# 2. Compute Joulean heater powers and 2D thermal crosstalk diffusion
# Solves: Delta_T = K_thermal @ P_heater
delta_T = self.thermal_model(theta_q, phi_q)

# 3. Add wafer-level spatial correlated variations (PVT drift)
effective_theta = theta_q + delta_T_theta + delta_wafer_theta
effective_phi = phi_q + delta_T_phi + delta_wafer_phi

# 4. Construct the N x N optical transfer matrix U_pic across Clements layers
# Sequentially multiplies block-diagonal 2x2 MZI transfer matrices
U_pic = self.compute_transfer_matrix(effective_theta, effective_phi)

# 5. Apply Fabry-Pérot cavity backreflection spectral ripple
U_pic = self.backreflection_model(U_pic, wavelength=self.cfg.wavelength)

# 6. Apply full Jones vector modal birefringence & polarization loss (if enabled)
E_jones = self.polarization_model(E_in, U_pic)

# 7. Apply intensity-dependent nonlinear absorption (TPA & FCA)
E_opt = self.nonlinear_model(E_jones)

# 8. Detect optical power via square-law receiver with realistic noise
# P_out = |E_opt|^2 + Noise(shot, thermal, RIN, flicker) -> TIA saturation -> ADC
readout_power = self.photodetectors(E_opt)
```

### 5.2 The Backward Pass (Autograd & Straight-Through Estimators)

Because physical DAC discretization is a step function (whose derivative is zero almost everywhere), standard backpropagation would cause vanishing gradients ($\nabla_\theta \mathcal{L} = 0$).

The quantizer uses **Straight-Through Estimators (STE)**:
$$\text{Forward:} \quad \hat{\theta} = \text{round}\left(\frac{\theta}{\Delta}\right) \cdot \Delta$$
$$\text{Backward:} \quad \frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial \hat{\theta}} \cdot \mathbf{1}_{\theta_{\min} \le \theta \le \theta_{\max}}$$
This allows gradients to flow backwards through the entire chain:
$$\frac{\partial \mathcal{L}}{\partial \theta} \longleftarrow \text{Photodetector} \longleftarrow \text{MZI Mesh} \longleftarrow \text{Thermal Kernel} \longleftarrow \text{DAC Quantizer}$$
enabling gradient-based optimization directly on the physical hardware model.

---

## 6. Loss Functions & Hardware Evaluation Metrics

### 6.1 Physics-Aware Losses (`src/training/losses.py`)

1. **Matrix Trace Fidelity Loss (`PhotonicFidelityLoss`):**
   Measures normalized overlap between target unitary $U$ and physical matrix $V$:
   $$\mathcal{L}_{\text{fid}}(U, V) = 1 - \frac{1}{N^2} |\text{Tr}(U^\dagger V)|^2$$

2. **Unitary Drift Penalty (`UnitaryDriftPenalty`):**
   Penalizes loss of optical energy and asymmetric loss imbalance:
   $$\mathcal{L}_{\text{drift}}(V) = \frac{1}{N} \|V^\dagger V - I\|_F^2$$

3. **Spatial Thermal Gradient Penalty (`SpatialThermalGradientPenalty`):**
   Penalizes steep temperature differences between neighboring heaters to prevent thermal mechanical stresses:
   $$\mathcal{L}_{\text{thermal}}(\mathbf{P}) = \sum_{\langle i, j \rangle} (P_i - P_j)^2$$

4. **In-Situ Adjoint Backpropagation Loss (`InSituAdjointLoss`):**
   Directly computes the forward-backward optical field overlap integral at each phase shifter (Hughes et al., *Optica* 2018; Pai et al., *Science* 2023):
   $$\nabla_{\theta_k} \mathcal{L} = 2 \, \text{Im} \left\{ E_{\text{fwd}, k}^* E_{\text{adj}, k} \right\}$$

5. **Composite Photonic Loss (`CompositePhotonicLoss`):**
   Multi-objective combination:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{fid}} \mathcal{L}_{\text{fid}} + \lambda_{\text{drift}} \mathcal{L}_{\text{drift}} + \lambda_{\text{th}} \mathcal{L}_{\text{thermal}}$$

### 6.2 Hardware Evaluation Metrics (`src/utils/evaluations.py`)

- **Effective Number of Bits (ENOB):**
  Evaluates Signal-to-Noise-and-Distortion ratio ($\text{SINAD}$) and converts to effective hardware resolution:
  $$\text{ENOB} = \frac{\text{SINAD}_{\text{dB}} - 1.76}{6.02}$$
- **Optical Energy per MAC:**
  Computes optical power per multiply-accumulate operation ($\text{fJ/MAC}$) and mean photon count $\langle N_{\text{photons}} \rangle$ using quantum Poisson statistics ($E_{\text{photon}} = h c / \lambda_0 \approx 0.1282\text{ aJ}$).
- **Automated Diagnostic Suite (`evaluate_full_hardware_system`):**
  Audits trace fidelity, broken unitarity, thermal gradients, and ENOB in a unified report.

---

## 7. Developer & AI Agent Extension Guide

When contributing new features or modifying this repository, adhere to the following architectural guidelines:

### Rule 1: Preserve Autograd Differentiability
- Always use native PyTorch tensor operations (`torch.matmul`, `torch.sin`, `torch.complex`).
- Do **NOT** convert tensors to NumPy arrays inside forward passes.
- When discrete steps are required, use or subclass `STEQuantizeFunction`.

### Rule 2: Respect Tensor Shape & Dtype Standards
- Field vectors: Complex tensors of shape `(batch_size, n_modes)` or `(batch_size, n_modes, 2)` for Jones polarization vectors.
- Transfer matrices: Complex tensors of shape `(n_modes, n_modes)`.
- Phase parameters: Real float tensors of shape `(total_mzis,)`.
- Default complex dtype: `torch.complex64`; default float dtype: `torch.float32`.

### Rule 3: Always Respect Foundry Parameter Bounds
- Directional coupler errors: $\kappa = 0.50 \pm 0.04$.
- Silicon thermo-optic coefficient: $dn/dT = 1.86 \times 10^{-4}\text{ K}^{-1}$.
- Thermal decay length in $\text{SiO}_2$: $\lambda_{\text{thermal}} = 55.0\,\mu\text{m}$.
- Waveguide propagation loss: $1.85\text{ dB/cm}$.
- Cumulative MZI stage loss: $0.15\text{ dB/stage}$.

### Rule 4: How to Add a New Physical Effect
1. Create your physics module under `src/physics/<new_effect>.py` inheriting from `torch.nn.Module`.
2. Expose the effect's parameters in `PhotonicConfig` (`src/config.py`).
3. Instantiate and invoke the effect within `PhotonicMeshDigitalTwin.forward()` or `compute_transfer_matrix()`.
4. Create a dedicated test file under `tests/test_<new_effect>.py`.
5. Register the test script in `TEST_SCRIPTS` inside `run_all_tests.py`.

---

## 8. Testing & Benchmarking Cheat Sheet

### Run the Master Test Suite (17 Comprehensive Test Suites)
```bash
# From inside the photonics/ directory:
.venv/bin/python run_all_tests.py
```

### Run Specific Test Suites via Pytest
```bash
# 1. Unitarity & Mathematical Conservation
.venv/bin/python -m pytest tests/test_unitarity.py

# 2. Autograd & Backward Gradient Flow
.venv/bin/python -m pytest tests/test_gradient_flow.py

# 3. 2D Finite-Difference Thermal Heat Equation Solver
.venv/bin/python -m pytest tests/test_thermal_2d.py

# 4. Full Jones Vector Polarization & PDL
.venv/bin/python -m pytest tests/test_polarization.py

# 5. Silicon Optical Nonlinearities (TPA/FCA/SPM)
.venv/bin/python -m pytest tests/test_nonlinear_optics.py

# 6. Coherent Multi-Cavity Fabry-Pérot Backreflection
.venv/bin/python -m pytest tests/test_backreflection.py

# 7. Waveguide Bend Radiation Loss & Mode Mismatch
.venv/bin/python -m pytest tests/test_bend_loss.py

# 8. 1/f Low-Frequency Flicker Noise & Aging
.venv/bin/python -m pytest tests/test_temporal_noise.py

# 9. Advanced Loss Functions & Hardware Evaluations
.venv/bin/python -m pytest tests/test_advanced_evaluations_and_losses.py

# 10. Phase 2 Full Pipeline Integration
.venv/bin/python -m pytest tests/test_phase2_integration.py

# 11. GPU Performance & Scalability Benchmarks
.venv/bin/python tests/benchmark_performance.py
```

### Verified Benchmark Throughput (NVIDIA GeForce RTX 3050 Ti Laptop GPU / CUDA)
| Mesh Modes ($N$) | Total MZIs ($M$) | Optical Depth | Batch Size | Latency | Throughput |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 4$** | 6 | 4 | 1024 | $7.17\text{ ms}$ | **$142,716\text{ vec/s}$** |
| **$N = 8$** | 28 | 8 | 1024 | $16.63\text{ ms}$ | **$61,562\text{ vec/s}$** |
| **$N = 16$** | 120 | 16 | 1024 | $35.64\text{ ms}$ | **$28,730\text{ vec/s}$** |
| **$N = 32$** | 496 | 32 | 1024 | $141.73\text{ ms}$ | **$7,225\text{ vec/s}$** |

---

## 📚 Foundational Literature & Theoretical References
For full mathematical derivations, Green's function proofs, and 220 nm SOI PDK parameter references, consult the companion document in the parent repository:
- `../papers/PAPERS_SUMMARY_AND_EQUATIONS.md` (Compendium of 23 premier peer-reviewed papers from *Nature*, *Science*, *Optica*, and *Light: Science & Applications*).
