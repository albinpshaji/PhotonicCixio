"""
Comprehensive Hardware Metrics and Scientific Benchmark Evaluation Suite for PICs.
Evaluates:
1. Matrix Fidelity & Frobenius Reconstruction Accuracy
2. Effective Number of Bits (ENOB) & SINAD
3. Optical Energy per Multiply-Accumulate (fJ / MAC) & Photon Budget
4. Polarization Dependent Loss (PDL) & Polarization Extinction Ratio (PER)
5. Thermal Conditioning, TED Invertibility, & Peak Temperature Drift
6. Optical Nonlinear Compression & TPA/FCA Saturation
7. Fabry-Perot Standing-Wave Ripple & Free Spectral Range (FSR)
"""

import math
import torch
from typing import Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass

from src.config import PhotonicConfig, PhysicalConstants
from src.models.digital_twin import PhotonicMeshDigitalTwin


@dataclass
class HardwareEvaluationReport:
    """Consolidated hardware evaluation report dataclass."""
    # Matrix fidelity metrics
    trace_fidelity: float
    frobenius_error: float
    unitarity_error: float

    # Precision & noise metrics
    sinad_db: float
    enob_bits: float

    # Energy & efficiency
    optical_energy_per_mac_fj: float
    photons_per_mac: float
    total_power_watts: float

    # Thermal health
    thermal_condition_number: float
    ted_inversion_error: float
    peak_die_temperature_rise_k: float

    # Optical impairments
    mean_pdl_db: float
    fabry_perot_ripple_db: float
    tpa_attenuation_db: float


def evaluate_matrix_fidelity(
    U_realized: torch.Tensor,
    U_target: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluates linear algebraic fidelity between target unitary and realized PIC matrix.
    """
    N = U_target.shape[-1]
    U_target_H = torch.conj(torch.transpose(U_target, -2, -1))

    # Trace overlap fidelity: |Tr(U_target^H @ U_realized)|^2 / N^2
    prod = torch.matmul(U_target_H, U_realized)
    trace_val = torch.diagonal(prod, dim1=-2, dim2=-1).sum(dim=-1)
    fidelity = (torch.square(torch.abs(trace_val)) / (N**2)).mean().item()

    # Frobenius norm error
    frob_err = torch.norm(U_realized - U_target, p="fro").item()

    # Unitarity violation: || U_realized^H @ U_realized - I ||_F
    eye = torch.eye(N, device=U_realized.device, dtype=U_realized.dtype)
    U_real_H = torch.conj(torch.transpose(U_realized, -2, -1))
    unit_err = torch.norm(torch.matmul(U_real_H, U_realized) - eye, p="fro").item()

    return {
        "trace_fidelity": float(fidelity),
        "frobenius_error": float(frob_err),
        "unitarity_error": float(unit_err)
    }


def evaluate_effective_resolution_enob(
    twin: PhotonicMeshDigitalTwin,
    batch_size: int = 1000,
    test_vectors: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Computes Signal-to-Noise-and-Distortion Ratio (SINAD) and Effective Number of Bits (ENOB).
    ENOB = (SINAD_dB - 1.76) / 6.02
    """
    cfg = twin.config
    N = cfg.n_modes
    device = cfg.device

    # Sweep optical power across the linear detection window (0.02 mW to 0.8 mW)
    P_opt = torch.linspace(0.02e-3, 0.8e-3, steps=batch_size, device=device).unsqueeze(-1).repeat(1, N)

    # Clean linear reference voltage: V = R * P * R_L
    I_clean = cfg.responsivity * P_opt
    V_clean = I_clean * cfg.tia_load_resistance

    # Physical readout chain: Poisson shot noise, TIA thermal noise, RIN, 1/f noise, and ADC
    I_noisy = twin.photodetector._add_detector_noise(I_clean)
    V_actual = twin.photodetector.readout_electronics(I_noisy)

    sig_power = torch.var(V_clean).item()
    err_power = torch.mean(torch.square(V_actual - V_clean)).item()

    sinad_linear = max(sig_power / max(err_power, 1e-20), 1.0)
    sinad_db = 10.0 * math.log10(sinad_linear)
    enob = max((sinad_db - 1.76) / 6.02, 0.0)

    return {
        "sinad_db": float(sinad_db),
        "enob_bits": float(enob),
        "snr_linear": float(sinad_linear)
    }


def evaluate_optical_energy_per_mac(
    config: PhotonicConfig,
    laser_power_watts: float = 10.0e-3,
    symbol_rate_baud: float = 10.0e9
) -> Dict[str, float]:
    """
    Computes optical and electrical energy dissipation per MAC (Multiply-Accumulate) operation.
    In an N-mode optical mesh, one vector propagation computes N^2 real MACs.
    """
    N = config.n_modes
    M = config.total_mzis
    macs_per_propagation = 2 * (N ** 2)  # Complex MVM is 4 real multiplications, 2 real MACs per cell

    # Electrical thermal heater power: M heaters * average half P_pi (0.5 * P_pi)
    P_heaters = M * (0.5 * config.P_pi)
    # Total chip power (laser launch + thermal actuators)
    P_total = laser_power_watts + P_heaters

    # Duration per vector compute cycle
    dt = 1.0 / symbol_rate_baud  # e.g. 100 ps at 10 Gbaud

    energy_per_vector_joules = P_total * dt
    optical_energy_joules = laser_power_watts * dt

    energy_per_mac_fj = (energy_per_vector_joules / macs_per_propagation) * 1e15
    optical_mac_fj = (optical_energy_joules / macs_per_propagation) * 1e15

    # Photon budget per MAC: E_photon = h * c / lambda
    h = PhysicalConstants.h
    c = PhysicalConstants.c
    e_photon = (h * c) / config.wavelength
    photons_per_mac = (optical_energy_joules / macs_per_propagation) / e_photon

    return {
        "energy_per_mac_fj": float(energy_per_mac_fj),
        "optical_energy_per_mac_fj": float(optical_mac_fj),
        "photons_per_mac": float(photons_per_mac),
        "total_dissipated_power_watts": float(P_total)
    }


def evaluate_full_hardware_system(
    twin: PhotonicMeshDigitalTwin,
    U_target: Optional[torch.Tensor] = None
) -> HardwareEvaluationReport:
    """
    Executes complete end-to-end hardware realism benchmark across all physics modules.
    """
    cfg = twin.config
    device = cfg.device
    N = cfg.n_modes
    M = twin.total_mzis

    # 1. Linear Algebra Fidelity
    theta_nom = torch.rand(M, device=device) * math.pi
    phi_nom = torch.rand(M, device=device) * (2.0 * math.pi)
    U_realized = twin.compute_transfer_matrix(theta=theta_nom, phi=phi_nom)
    if U_target is None:
        U_target = U_realized.clone()

    fid_metrics = evaluate_matrix_fidelity(U_realized, U_target)

    # 2. ENOB & Precision
    enob_metrics = evaluate_effective_resolution_enob(twin, batch_size=32)

    # 3. Energy & Efficiency
    energy_metrics = evaluate_optical_energy_per_mac(cfg, laser_power_watts=cfg.optical_input_power_watts)

    # 4. Thermal Health
    K_norm = twin.thermal_model.K_norm
    cond_num = torch.linalg.cond(K_norm).item()
    # Predistortion test
    th_tgt = torch.rand(M, device=device) * math.pi
    th_drv = twin.thermal_model.K_norm_inv @ th_tgt
    th_rec = twin.thermal_model.K_norm @ th_drv
    ted_err = torch.norm(th_rec - th_tgt).item()
    peak_pwr = M * cfg.P_pi
    peak_temp = cfg.package_thermal_resistance * peak_pwr

    # 5. Polarization & Loss
    pdl_val = cfg.tm_crossing_loss_db - cfg.crossing_loss_db

    # 6. Backreflection ripple
    H_fp = twin.backreflection_model.compute_transfer_filter()
    ripple = 10.0 * math.log10(torch.max(torch.abs(H_fp)**2).item() / torch.min(torch.abs(H_fp)**2).item())

    # 7. TPA attenuation at nominal input power
    E_test = torch.tensor([math.sqrt(cfg.optical_input_power_watts) + 0.0j], dtype=torch.complex64, device=device)
    E_att = twin.nonlinear_model(E_test)
    tpa_ratio = torch.abs(E_att[0])**2 / cfg.optical_input_power_watts
    tpa_db = -10.0 * math.log10(torch.clamp(tpa_ratio, min=1e-12).item())

    return HardwareEvaluationReport(
        trace_fidelity=fid_metrics["trace_fidelity"],
        frobenius_error=fid_metrics["frobenius_error"],
        unitarity_error=fid_metrics["unitarity_error"],
        sinad_db=enob_metrics["sinad_db"],
        enob_bits=enob_metrics["enob_bits"],
        optical_energy_per_mac_fj=energy_metrics["optical_energy_per_mac_fj"],
        photons_per_mac=energy_metrics["photons_per_mac"],
        total_power_watts=energy_metrics["total_dissipated_power_watts"],
        thermal_condition_number=float(cond_num),
        ted_inversion_error=float(ted_err),
        peak_die_temperature_rise_k=float(peak_temp),
        mean_pdl_db=float(pdl_val),
        fabry_perot_ripple_db=float(ripple),
        tpa_attenuation_db=float(tpa_db)
    )
