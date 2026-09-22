"""
Comprehensive End-to-End Integration Tests for Phase 2 Physics Modules:
1. 2D Finite-Difference Poisson/Helmholtz Thermal Solver
2. Silicon Optical Nonlinearities (TPA, FCA, SPM, FCD)
3. Full Jones Vector Polarization Tracking (TE/TM cross-coupling & PDL)
4. Coherent Multi-Cavity Fabry-Perot Backreflection Ripple
5. Waveguide Bend Radiation Loss and Mode-Mismatch Scattering
6. 1/f Low-Frequency Flicker Noise and Temporal Aging Drift
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
from src.models.digital_twin import PhotonicMeshDigitalTwin


def test_phase2_full_pipeline_forward():
    """Verify that all 6 physics modules execute harmoniously in forward propagation."""
    config = PhotonicConfig(
        n_modes=8,
        thermal_solver="finite_difference",
        enable_nonlinear_optics=True,
        enable_backreflection=True,
        enable_bend_loss=True,
        enable_flicker_noise=True,
        enable_aging=True,
        operating_hours=2500.0,
        enable_polarization=True,
        enable_noise=True,
        enable_adc=True
    )
    twin = PhotonicMeshDigitalTwin(config)

    batch_size = 4
    # Input optical field: (Batch, N)
    E_in = torch.randn(batch_size, config.n_modes, dtype=torch.complex64)
    theta = torch.rand(batch_size, twin.total_mzis) * math.pi
    phi = torch.rand(batch_size, twin.total_mzis) * (2.0 * math.pi)

    # 1. Single-ended detection with full Jones vectors
    E_out, measured = twin(E_in=E_in, theta=theta, phi=phi, balanced=False)
    assert E_out.shape == (batch_size, config.n_modes, 2), f"Expected Jones output, got {E_out.shape}"
    assert measured.shape == (batch_size, config.n_modes), f"Expected measured readout, got {measured.shape}"
    assert not torch.isnan(measured).any()

    # 2. Balanced detection with full Jones vectors
    E_out_bal, measured_bal = twin(E_in=E_in, theta=theta, phi=phi, balanced=True)
    assert measured_bal.shape == (batch_size, config.n_modes // 2)
    assert not torch.isnan(measured_bal).any()


def test_phase2_full_pipeline_autograd():
    """Verify end-to-end differentiability through all 6 combined physics layers."""
    config = PhotonicConfig(
        n_modes=6,
        thermal_solver="finite_difference",
        enable_nonlinear_optics=True,
        enable_backreflection=True,
        enable_bend_loss=True,
        enable_flicker_noise=True,
        enable_aging=True,
        operating_hours=1000.0,
        enable_polarization=True,
        enable_noise=False,     # Disable stochastic sampling for deterministic gradient check
        enable_adc=False        # Disable discrete quantization for smooth autograd check
    )
    twin = PhotonicMeshDigitalTwin(config)

    theta = torch.rand(twin.total_mzis, requires_grad=True)
    phi = torch.rand(twin.total_mzis, requires_grad=True)
    E_in = torch.ones(config.n_modes, dtype=torch.complex64)

    E_out, measured = twin(E_in=E_in, theta=theta, phi=phi)

    # Objective: maximize detected optical power in channel 0
    loss = torch.sum(measured)
    loss.backward()

    assert theta.grad is not None
    assert phi.grad is not None
    assert not torch.isnan(theta.grad).any()
    assert not torch.isnan(phi.grad).any()
    assert (theta.grad.abs() > 0.0).any()
    assert (phi.grad.abs() > 0.0).any()


def test_phase2_ideal_mode_bypass():
    """Verify that ideal_mode=True completely bypasses all Phase 2 physics."""
    config = PhotonicConfig(
        n_modes=4,
        ideal_mode=True,
        enable_nonlinear_optics=True,
        enable_backreflection=True,
        enable_bend_loss=True,
        enable_polarization=True
    )
    twin = PhotonicMeshDigitalTwin(config)

    theta = torch.rand(twin.total_mzis) * math.pi
    phi = torch.rand(twin.total_mzis) * (2.0 * math.pi)
    T = twin.compute_transfer_matrix(theta=theta, phi=phi)

    # In ideal mode, Clements mesh is strictly unitary: T^H @ T == I
    eye = torch.eye(4, dtype=torch.complex64)
    unitarity_err = torch.norm(torch.matmul(T.mH, T) - eye).item()
    assert unitarity_err < 1e-5, f"Ideal mode unitarity violated: error={unitarity_err}"


if __name__ == "__main__":
    test_phase2_full_pipeline_forward()
    test_phase2_full_pipeline_autograd()
    test_phase2_ideal_mode_bypass()
    print("ALL PHASE 2 INTEGRATION TESTS PASSED.")
