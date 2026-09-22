"""
Unit Test: Broadband Wavelength Dispersion & Directional Coupler Spectrum.
Validates:
1. Directional coupler coupling coefficient kappa(lambda) dispersion across C-band.
2. Coupler excess insertion loss attenuation.
3. Extinction ratio degradation at detuned wavelengths.
"""

import sys
import os
import math
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PhotonicConfig
from src.physics.mzi import directional_coupler_matrix, mzi_transfer_matrix


def test_directional_coupler_dispersion():
    """Verifies that directional coupler split ratio shifts with optical wavelength."""
    eps0 = torch.tensor([0.0], dtype=torch.float32)
    lambda_0 = 1550e-9

    # At nominal wavelength, bar and cross powers should both be 0.5
    C_nom = directional_coupler_matrix(eps0, wavelength=lambda_0, lambda_0=lambda_0)
    bar_nom = torch.abs(C_nom[0, 0, 0]) ** 2
    cross_nom = torch.abs(C_nom[0, 0, 1]) ** 2
    assert abs(bar_nom.item() - 0.5) < 1e-4
    assert abs(cross_nom.item() - 0.5) < 1e-4

    # At blue-shifted wavelength 1530 nm (delta_lambda = -20 nm)
    C_blue = directional_coupler_matrix(eps0, wavelength=1530e-9, lambda_0=lambda_0)
    bar_blue = (torch.abs(C_blue[0, 0, 0]) ** 2).item()
    cross_blue = (torch.abs(C_blue[0, 0, 1]) ** 2).item()

    # Total power must still be conserved (lossless coupler)
    assert abs(bar_blue + cross_blue - 1.0) < 1e-4
    # Cross coupling must shift away from 0.5 due to dispersion
    assert abs(cross_blue - 0.5) > 0.001, f"Coupling did not shift with wavelength: {cross_blue}"


def test_coupler_excess_loss():
    """Verifies that coupler excess loss attenuates field amplitude."""
    eps0 = torch.tensor([0.0], dtype=torch.float32)
    loss_db = 0.10  # 0.10 dB excess loss

    C_loss = directional_coupler_matrix(eps0, excess_loss_db=loss_db)
    # Total power transmission = sum(|C_ij|^2) / 1.0 = 10^(-loss_dB / 10)
    p_out = (torch.abs(C_loss[0, 0, 0]) ** 2 + torch.abs(C_loss[0, 0, 1]) ** 2).item()
    expected_p = 10.0 ** (-loss_db / 10.0)
    assert abs(p_out - expected_p) < 1e-4, f"Loss mismatch: {p_out} vs {expected_p}"


def test_mzi_dispersion_extinction_ratio():
    """Verifies that MZI destructive interference is degraded at detuned wavelengths."""
    theta = torch.tensor([0.0], dtype=torch.float32)  # Bar state (destructive interference in cross port)
    phi = torch.tensor([0.0], dtype=torch.float32)
    eps = torch.tensor([0.0], dtype=torch.float32)
    lambda_0 = 1550e-9

    # At 1550 nm: ideal cancellation
    T_0 = mzi_transfer_matrix(theta=theta, phi=phi, epsilon1=eps, epsilon2=eps, wavelength=lambda_0)
    p_cross_0 = (torch.abs(T_0[0, 0, 0]) ** 2).item()

    # At 1565 nm (+15 nm detuned): dispersion creates split mismatch -> non-zero leakage
    T_detuned = mzi_transfer_matrix(theta=theta, phi=phi, epsilon1=eps, epsilon2=eps, wavelength=1565e-9)
    p_cross_detuned = (torch.abs(T_detuned[0, 0, 0]) ** 2).item()

    assert p_cross_detuned > p_cross_0, (
        f"Off-wavelength leakage should increase: {p_cross_detuned:.4e} vs {p_cross_0:.4e}"
    )


if __name__ == "__main__":
    test_directional_coupler_dispersion()
    test_coupler_excess_loss()
    test_mzi_dispersion_extinction_ratio()
    print("ALL DISPERSION TESTS PASSED.")
