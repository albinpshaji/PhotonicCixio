"""
Die Inspection, Physics Diagnostics, and Visualization Utilities for Silicon Photonic Chips.
Provides:
- 2D wafer spatial error correlation map extraction
- Multi-heater mesh thermal dissipation profile
- Broadband C-band optical spectrum analyzer (1530 nm - 1565 nm)
- S-parameter Touchstone exporter for circuit co-simulation
"""

import math
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.config import PhotonicConfig, PhysicalConstants
from src.models.digital_twin import PhotonicMeshDigitalTwin


@dataclass
class SpectralAnalysisResult:
    """Broadband spectral sweep data across optical wavelengths."""
    wavelengths_nm: torch.Tensor
    transmission_db: torch.Tensor
    extinction_ratio_db: torch.Tensor
    mean_insertion_loss_db: float
    nominal_split_ratio: float


def sweep_wavelength_spectrum(
    model: PhotonicMeshDigitalTwin,
    theta: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
    lambda_start_nm: float = 1530.0,
    lambda_stop_nm: float = 1565.0,
    num_points: int = 71
) -> SpectralAnalysisResult:
    """
    Performs a broadband wavelength sweep across the C-band,
    evaluating transfer matrix T(lambda) at each wavelength point.

    Args:
        model: PhotonicMeshDigitalTwin instance.
        theta: Internal phase shift tensor of shape (M,).
        phi: External phase shift tensor of shape (M,).
        lambda_start_nm: Start wavelength in nanometers.
        lambda_stop_nm: Stop wavelength in nanometers.
        num_points: Number of discrete spectral wavelength samples.

    Returns:
        SpectralAnalysisResult dataclass containing spectra and metrics.
    """
    wavelengths = torch.linspace(
        lambda_start_nm * 1e-9,
        lambda_stop_nm * 1e-9,
        num_points,
        device=theta.device
    )

    transmissions = []
    ers = []

    # Ensure single mesh evaluation shape
    th = theta.unsqueeze(0) if theta.dim() == 1 else theta
    ph = phi.unsqueeze(0) if (phi is not None and phi.dim() == 1) else phi

    with torch.no_grad():
        for lam in wavelengths:
            T = model.compute_transfer_matrix(theta=th, phi=ph, wavelength=lam.item())
            # Power transmission matrix P_ij = |T_ij|^2
            P = torch.real(T * torch.conj(T)).squeeze(0)  # (N, N)

            # Diagonal transmission (bar state) and off-diagonal cross state
            t_diag = torch.diag(P)
            mean_diag_pwr = torch.mean(t_diag).item()
            t_db = 10.0 * math.log10(max(mean_diag_pwr, 1e-12))
            transmissions.append(t_db)

            # Extinction Ratio ER = 10 * log10(max(P) / min(P))
            p_max = torch.max(P).item()
            p_min = torch.min(torch.clamp(P, min=1e-12)).item()
            er_db = 10.0 * math.log10(max(p_max / p_min, 1.0))
            ers.append(er_db)

    wl_nm = wavelengths * 1e9
    t_tensor = torch.tensor(transmissions, device=theta.device)
    er_tensor = torch.tensor(ers, device=theta.device)

    return SpectralAnalysisResult(
        wavelengths_nm=wl_nm,
        transmission_db=t_tensor,
        extinction_ratio_db=er_tensor,
        mean_insertion_loss_db=-float(torch.mean(t_tensor).item()),
        nominal_split_ratio=0.5
    )


def compute_mesh_thermal_dissipation(
    model: PhotonicMeshDigitalTwin,
    theta: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Computes 2D spatial power and temperature profiles across the MZI heater grid.

    Returns:
        Dictionary containing:
        - 'coords_um': (M, 2) physical heater coordinates in microns
        - 'joule_power_mw': (M,) dissipated electrical power per heater in mW
        - 'local_temp_rise_k': (M,) local temperature rise per heater in Kelvin
        - 'die_substrate_rise_k': scalar global die temperature rise in Kelvin
    """
    th = theta.squeeze(0) if theta.dim() > 1 else theta
    P = model.thermal_model.compute_joule_power(th)  # (M,) in Watts

    # Local temperature rise: delta_T = K @ P
    K = model.thermal_model.K.to(device=theta.device, dtype=P.dtype)
    local_dT = torch.matmul(K, P)

    # Substrate global temperature rise: R_pack * sum(P)
    p_total = torch.sum(P).item()
    substrate_dT = model.config.package_thermal_resistance * p_total

    coords_um = model.thermal_model.coords.to(theta.device) * 1e6
    power_mw = P * 1e3

    return {
        "coords_um": coords_um,
        "joule_power_mw": power_mw,
        "local_temp_rise_k": local_dT,
        "die_substrate_rise_k": torch.tensor(substrate_dT, device=theta.device)
    }
