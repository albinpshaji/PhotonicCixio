"""
Tests for 2D Finite-Difference Poisson/Helmholtz Thermal Solver on SOI Platform.
Validates:
1. Green's function matrix symmetry (reciprocity G_ij = G_ji).
2. Diagonal normalization (diag(K_norm) == 1.0).
3. Distance decay behavior across multi-heater 2D Clements layout.
4. Matrix conditioning and invertibility for TED predistortion.
5. PyTorch autograd gradient flow through FD thermal solver.
"""

import sys
import os
import math
import torch
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import PhotonicConfig, FoundryPDK
from src.physics.thermal import compute_clements_layout_coordinates, ThermalCrosstalkModel
from src.physics.thermal_2d import compute_fd_thermal_greens_function


def test_fd_thermal_matrix_properties():
    """Verify symmetry, normalization, and shape of FD thermal Green's function."""
    n_modes = 8
    config = PhotonicConfig(n_modes=n_modes, thermal_solver="finite_difference")
    coords, _ = compute_clements_layout_coordinates(
        n_modes=n_modes,
        pitch_x=config.heater_pitch_x,
        pitch_y=config.heater_pitch_y
    )

    G, K_norm = compute_fd_thermal_greens_function(
        coords=coords,
        heater_pitch_x=config.heater_pitch_x,
        heater_pitch_y=config.heater_pitch_y,
        grid_res=64,
        thermal_impedance=config.thermal_impedance,
        device="cpu"
    )

    M = config.total_mzis
    assert G.shape == (M, M)
    assert K_norm.shape == (M, M)

    # 1. Reciprocity (Symmetry)
    asymmetry = torch.max(torch.abs(G - G.T)).item()
    assert asymmetry < 1e-5, f"Thermal Green's function must be symmetric (asymmetry={asymmetry})"

    # 2. Diagonal normalization
    diag = torch.diagonal(K_norm)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5), "diag(K_norm) must be exactly 1.0"

    # 3. Off-diagonal elements non-negative and strictly less than 1.0
    off_diag_mask = ~torch.eye(M, dtype=torch.bool)
    off_diag = K_norm[off_diag_mask]
    assert (off_diag >= 0.0).all(), "Thermal coupling must be non-negative"
    assert (off_diag < 1.0).all(), "Off-diagonal thermal coupling must be strictly less than self-heating"
    assert off_diag.max() > 0.001, "Nearest neighbor thermal coupling should be measurable (> 0.1%)"


def test_fd_thermal_distance_decay():
    """Verify that thermal coupling monotonically decreases with physical Euclidean distance."""
    n_modes = 8
    config = PhotonicConfig(n_modes=n_modes, thermal_solver="finite_difference")
    coords, _ = compute_clements_layout_coordinates(
        n_modes=n_modes,
        pitch_x=config.heater_pitch_x,
        pitch_y=config.heater_pitch_y
    )

    _, K_norm = compute_fd_thermal_greens_function(
        coords=coords,
        heater_pitch_x=config.heater_pitch_x,
        heater_pitch_y=config.heater_pitch_y,
        grid_res=64,
        thermal_impedance=config.thermal_impedance,
        device="cpu"
    )

    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    dist = torch.norm(diff, dim=-1)

    # For heater 0, find a near neighbor vs a far neighbor
    dists_from_0 = dist[0]
    coupling_from_0 = K_norm[0]

    # Sort by distance
    sorted_indices = torch.argsort(dists_from_0)
    # Exclude self at index 0
    near_idx = sorted_indices[1].item()
    far_idx = sorted_indices[-1].item()

    assert dists_from_0[near_idx] < dists_from_0[far_idx]
    assert coupling_from_0[near_idx] > coupling_from_0[far_idx], (
        f"Near heater coupling ({coupling_from_0[near_idx]:.4f}) must exceed far coupling ({coupling_from_0[far_idx]:.4f})"
    )


def test_fd_thermal_crosstalk_model_forward_and_backward():
    """Verify that ThermalCrosstalkModel runs forward and autograd backward with FD solver."""
    config = PhotonicConfig(n_modes=6, thermal_solver="finite_difference")
    model = ThermalCrosstalkModel(config)

    M = config.total_mzis
    theta_in = torch.randn(4, M, requires_grad=True)

    theta_out = model(theta_in)
    assert theta_out.shape == (4, M)

    loss = theta_out.sum()
    loss.backward()

    assert theta_in.grad is not None
    assert not torch.isnan(theta_in.grad).any()
    assert (theta_in.grad.abs() > 0.0).any()


def test_fd_thermal_predistortion_inversion():
    """Verify that K_norm is invertible and inverts thermal crosstalk accurately."""
    config = PhotonicConfig(n_modes=6, thermal_solver="finite_difference")
    model = ThermalCrosstalkModel(config)

    M = config.total_mzis
    target_theta = torch.rand(M) * math.pi

    # Predistorted command
    theta_drive = model.K_norm_inv @ target_theta
    # Thermal response: K_norm @ theta_drive
    realized_theta = model.K_norm @ theta_drive

    inv_err = torch.max(torch.abs(realized_theta - target_theta)).item()
    assert inv_err < 1e-4, f"Thermal predistortion inversion error too high: {inv_err}"


if __name__ == "__main__":
    test_fd_thermal_matrix_properties()
    test_fd_thermal_distance_decay()
    test_fd_thermal_crosstalk_model_forward_and_backward()
    test_fd_thermal_predistortion_inversion()
    print("ALL 2D FD THERMAL TESTS PASSED.")
