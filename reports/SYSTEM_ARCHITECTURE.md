# Silicon Photonic Digital Twin: System Architecture & Workflow

This document provides the complete system architecture and signal flow of the **Silicon Photonic Coherent Matrix Multiplier Digital Twin**. 

When viewed on GitHub, the diagram below is automatically rendered as an interactive flowchart.

---

## 1. System Architecture Flowchart

```mermaid
flowchart TD
    %% =========================================================================
    %% TOP LEVEL INPUTS
    %% =========================================================================
    subgraph Inputs["System Ingestion & Dual Inputs"]
        direction LR
        AI_W["AI Weight Matrix W ∈ ℝ^(N×N)<br/>(PyTorch Neural Network Layer)"]
        OPT_IN["Input Optical Feature Vector E_in ∈ ℂ^N<br/>(CW Laser Carrier λ = 1550 nm)"]
    end

    %% =========================================================================
    %% PHASE 1: STATIC OPTICAL COMPILATION & PRE-COMPENSATION
    %% =========================================================================
    subgraph Phase1["Phase 1: Optical Compilation & Pre-Compensation (Static Matrix Engine)"]
        direction TB
        CLEMENTS["1. Clements Nulling Scan<br/>• Optica 2016 SVD / QR Decomposition<br/>• Generates M = N(N-1)/2 Phase Pairs (θ, φ)<br/>• Generates N Diagonal Phases γ"]
        CALIB["2. Calibration Parameter Fitting<br/>• Inverts Foundry Wafer Variances<br/>• Recovers Coupler Errors (ε₁, ε₂)<br/>• Offsets Intrinsic Arm Bias φ_int"]
        BNNLS["3. BNNLS Thermal Inversion<br/>• Solves P_drive = K⁻¹ · P_target<br/>• 2D Screened Poisson Heat PDE<br/>• Cancels Lateral Heat Bleed (~55 µm)"]
        DAC["4. DAC Discretization & Jitter<br/>• 6-bit / 8-bit Voltage Discretization<br/>• Injects DNL (0.3 LSB) & INL (0.5 LSB)<br/>• Analog Phase Jitter (σ = 0.008 rad)<br/>• Straight-Through Estimators (STE)"]

        CLEMENTS --> CALIB --> BNNLS --> DAC
    end

    %% =========================================================================
    %% PHASE 2: DYNAMIC WAVE PROPAGATION ENGINE
    %% =========================================================================
    subgraph Phase2["Phase 2: Dynamic Wave Propagation Engine (Mesh SOI Physics)"]
        direction TB
        GC_IN["Input Fiber Grating Couplers<br/>• -3.0 dB Insertion Loss<br/>• C-Band Spectral Dispersion"]
        
        subgraph MeshCascade["Layered Clements Planar MZI Matrix (N Columns, M MZIs)"]
            direction TB
            MZI_CELLS["MZI Unit Cells<br/>• 2×2 Unitary Rotations<br/>• Directional Couplers C(ε₁, ε₂)<br/>• Intrinsic Roughness Phase Bias φ_int"]
            ROUTING["Planar Waveguide Routing<br/>• 0.15 dB/col Progressive Stage Loss<br/>• Waveguide Crossings (-40 dB Crosstalk, 0.025 dB Loss)<br/>• Waveguide Bend Radiation Loss"]
            NONLINEAR["Silicon Nonlinear Optics<br/>• Intensity-Dependent Step: I = |E|²<br/>• Two-Photon Absorption (TPA)<br/>• Free-Carrier Absorption (FCA)<br/>• Kerr Self-Phase Modulation (SPM)"]
            DIAG_SCREEN["Output Diagonal Phase Screen<br/>• N Single-Mode Phase Shifters D_diag(γ)<br/>• Completes Universal U(N) Synthesis"]
            
            MZI_CELLS --> ROUTING --> NONLINEAR --> DIAG_SCREEN
        end
        
        GC_OUT["Output Fiber Grating Couplers<br/>• -3.0 dB Insertion Loss<br/>• Outputs Complex Fields E_out"]

        GC_IN --> MeshCascade --> GC_OUT
    end

    %% =========================================================================
    %% PHASE 3: RECEIVER TRANSDUCTION & READOUT NOISE
    %% =========================================================================
    subgraph Phase3["Phase 3: Receiver Transduction & Noise Readout (Hardware Interface)"]
        direction TB
        BPD["1. Balanced Photodiodes (BPD)<br/>• Germanium-on-Si PIN Detectors<br/>• Power to Current: I_ph = R · |E_out|²<br/>• 30 dB CMRR (Signed Arithmetic)"]
        NOISE["2. Stochastic Noise Injection<br/>• Quantum Poisson Shot Noise<br/>• Johnson-Nyquist Thermal Hiss<br/>• Laser Relative Intensity Noise (RIN = -145 dB/Hz)"]
        TIA["3. Transimpedance Amplification (TIA)<br/>• Current-to-Voltage: V = I_ph · R_tia<br/>• Rail-to-Rail Clamping at V_sat = 1.2 V"]
        ADC["4. Output ADC Quantization<br/>• 8-bit Output Digitization<br/>• ENOB Degradation ≈ 5.8 - 6.2 bits<br/>• Delivers Digital Word y_pred ∈ ℝ^N"]

        BPD --> NOISE --> TIA --> ADC
    end

    %% =========================================================================
    %% PHASE 4: SYSTEM BENCHMARKING & RETRAINING
    %% =========================================================================
    subgraph Phase4["Phase 4: System Benchmarking & Hardware-Aware Retraining"]
        direction LR
        BENCH["Verification & Diagnostics<br/>• Unitarity Error: < 10⁻¹⁵ (Ideal)<br/>• Total On-Chip Loss: 1.104 dB<br/>• Flight Latency: ~100 ps<br/>• GPU Batching: > 10,000 vectors/ms"]
        TRAIN["Differentiable AI Retraining<br/>• Evaluates Task Loss L(y_pred, y_true)<br/>• Autograd Backward Propagation<br/>• Hardware-Aware Weight Updates"]
    end

    %% =========================================================================
    %% SYSTEM SIGNAL FLOW CONNECTIONS
    %% =========================================================================
    AI_W -->|Target Matrix W| CLEMENTS
    DAC -->|Calibrated Voltages V_DAC| MeshCascade

    OPT_IN -->|Optical Carrier| GC_IN
    GC_OUT -->|Optical Fields E_out| BPD

    ADC -->|Electronic Predictions y_pred| Phase4

    %% =========================================================================
    %% DIFFERENTIABLE FEEDBACK LOOP (HARDWARE-IN-THE-LOOP RETRAINING)
    %% =========================================================================
    TRAIN -.->|Differentiable Backprop via STE Gradients ∂L/∂W| AI_W

    %% =========================================================================
    %% STYLING AND COLOR ACCENTS
    %% =========================================================================
    classDef inputStyle fill:#111827,stroke:#38bdf8,stroke-width:1.5px,color:#f3f4f6;
    classDef p1Style fill:#172033,stroke:#00e5ff,stroke-width:1.5px,color:#f3f4f6;
    classDef p2Style fill:#1a192e,stroke:#c084fc,stroke-width:1.5px,color:#f3f4f6;
    classDef p3Style fill:#241d19,stroke:#f59e0b,stroke-width:1.5px,color:#f3f4f6;
    classDef p4Style fill:#13231e,stroke:#10b981,stroke-width:1.5px,color:#f3f4f6;

    class AI_W,OPT_IN inputStyle;
    class CLEMENTS,CALIB,BNNLS,DAC p1Style;
    class GC_IN,MZI_CELLS,ROUTING,NONLINEAR,DIAG_SCREEN,GC_OUT p2Style;
    class BPD,NOISE,TIA,ADC p3Style;
    class BENCH,TRAIN p4Style;
```

---

## 2. High-Resolution Publication Graphic

For technical papers, reports, or slides, a rendered 300 DPI dark scientific graphic is available:

![Silicon Photonic Digital Twin Architecture](plots/12_system_architecture_diagram.png)

*(File location: [`reports/plots/12_system_architecture_diagram.png`](plots/12_system_architecture_diagram.png))*

---

## 3. Operational Phases Breakdown

### Phase 1: Optical Compilation & Hardware Pre-Compensation (Static Engine)
* **Goal:** Translates abstract AI weight tensors into calibrated physical control voltages.
* **Key Code Modules:**
  * [`src/utils/decomposition.py`](../src/utils/decomposition.py): Executes the canonical Clements Givens-rotation nulling scan to produce $M = \frac{N(N-1)}{2}$ phase pairs $(\theta_m, \phi_m)$ and $N$ diagonal phases $\gamma_k$.
  * [`src/calibration/parameter_fitting.py`](../src/calibration/parameter_fitting.py): Gradient-based parameter estimation that offsets static fabrication deviations ($\epsilon_1, \epsilon_2, \phi_{\mathrm{int}}$).
  * [`src/physics/thermal.py`](../src/physics/thermal.py): Inverts the 2D Poisson thermal kernel $\mathbf{K}^{-1}$ via Bounded Non-Negative Least Squares (BNNLS) to pre-cancel lateral thermal bleed across micro-heaters ($\approx 55\text{ µm}$ decay).
  * [`src/nn/quantizer.py`](../src/nn/quantizer.py): Discretizes continuous phase angles to 6-bit or 8-bit DAC levels with DNL/INL non-linearities and thermal phase jitter.

### Phase 2: Dynamic Wave Propagation Engine (Mesh SOI Physics)
* **Goal:** High-throughput optical matrix multiplication directly in the complex electromagnetic wave domain.
* **Key Code Modules:**
  * [`src/models/digital_twin.py`](../src/models/digital_twin.py): Orchestrates column-by-column sparse field updates across $N$ physical layers.
  * [`src/physics/mzi.py`](../src/physics/mzi.py): Physical $2 \times 2$ MZI transfer matrices with non-ideal directional coupling ($C(\epsilon)$).
  * [`src/physics/routing_loss.py`](../src/physics/routing_loss.py): Inter-column planar waveguide crossings ($-40\text{ dB}$ crosstalk, $0.025\text{ dB}$ loss).
  * [`src/physics/loss_model.py`](../src/physics/loss_model.py): Cumulative waveguide propagation attenuation ($1.5\text{ dB/cm}$).
  * [`src/physics/nonlinear_optics.py`](../src/physics/nonlinear_optics.py): Power-dependent continuous-wave Two-Photon Absorption (TPA), Free-Carrier Absorption (FCA), and Kerr Self-Phase Modulation (SPM).

### Phase 3: Receiver Transduction & Noise Readout (Hardware Interface)
* **Goal:** Transduces optical power back into digital words for the electronic host.
* **Key Code Modules:**
  * [`src/physics/photodiode.py`](../src/physics/photodiode.py):
    * Balanced Homodyne Photodetectors (BPD) yielding photocurrent $I_{\mathrm{photo}} = \mathcal{R} \cdot |E_{\mathrm{out}}|^2$.
    * Injects Poisson quantum shot noise, Johnson thermal hiss, and laser Relative Intensity Noise (RIN = $-145\text{ dB/Hz}$).
    * Transimpedance Amplifier (TIA) conversion with rail-to-rail saturation clamping at $V_{\mathrm{sat}} = 1.2\text{ V}$.
    * Output ADC quantization delivering the final digital prediction vector $y_{\mathrm{pred}} \in \mathbb{R}^N$ with an Effective Number of Bits ($\text{ENOB} \approx 5.8 - 6.2\text{ bits}$).

### Phase 4: System Benchmarking & Hardware-Aware Retraining
* **Goal:** Closes the loop between physical hardware and AI model accuracy.
* **Key Code Modules:**
  * [`tests/run_all_tests.py`](../tests/run_all_tests.py): Validates exact double-precision unitarity ($< 10^{-15}$ error in ideal mode) across all 19 test suites.
  * [`src/training/losses.py`](../src/training/losses.py): Backpropagates task loss gradients $\frac{\partial \mathcal{L}}{\partial W}$ through the PyTorch Straight-Through Estimators (STE). The AI autonomously learns to compensate for chip loss, thermal bleed, and DAC quantization during training.

---

## 4. Verification & Testing

To execute the test suite and verify all modules in this architecture:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all 19 physics test suites
python tests/run_all_tests.py

# Regenerate all 12 publication diagnostic plots
python tests/generate_test_plots.py
```
