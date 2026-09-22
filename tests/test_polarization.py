"""
Tests for Full Jones Vector Polarization Dynamics and PDL.
Validates:
1. Pure TE input coupling into TM mode via waveguide bend mixing.
2. Power conservation of 2x2 rotation coupling operator.
3. Modal birefringence phase differential between TE and TM modes.
4. Polarization-dependent loss (PDL) across waveguide crossings.
5. Square-law total power detection across both polarizations.
6. Smooth PyTorch autograd gradient backpropagation.
"""

import sys
import os
import math
import torch
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import PhotonicConfig
from src.physics.polarization import JonesPolarizationModel


def test_bend_polarization_mixing():
    """Verify that pure TE mode couples into TM mode after traversing bends."""
    config = PhotonicConfig(n_modes=4, enable_polarization=True, polarization_coupling_rad=0.03)
    pol_model = JonesPolarizationModel(config)

    # Pure TE state: [1.0 + 0j, 0.0 + 0j]
    E_te_pure = torch.zeros(4, 2, dtype=torch.complex64)
    E_te_pure[:, 0] = 1.0 + 0.0j

    # Apply bend coupling
    E_rotated = pol_model.apply_bend_coupling(E_te_pure)

    # TM amplitude should be non-zero: sin(0.03) approx 0.03
    tm_power = torch.abs(E_rotated[:, 1])**2
    assert (tm_power > 0.0005).all(), f"Bend coupling must transfer power to TM mode, got {tm_power}"

    # Total power must be strictly conserved by pure rotation: |E_TE|^2 + |E_TM|^2 == 1.0
    total_power = pol_model.total_optical_power(E_rotated)
    assert torch.allclose(total_power, torch.ones_like(total_power), atol=1e-5)


def test_modal_birefringence_phase_drift():
    """Verify that TE and TM components accumulate different phase shifts over propagation."""
    config = PhotonicConfig(n_modes=4, enable_polarization=True)
    pol_model = JonesPolarizationModel(config)

    # Equal superposition of TE and TM: [1.0, 1.0]
    E_in = torch.ones(4, 2, dtype=torch.complex64)
    L = 100.0e-6  # 100 um propagation distance

    E_out = pol_model.propagate_polarization(E_in, distance=L)

    phase_te = torch.angle(E_out[0, 0]).item()
    phase_tm = torch.angle(E_out[0, 1]).item()

    phase_diff = abs(phase_te - phase_tm)
    assert phase_diff > 0.01, f"Modal birefringence must produce phase differential: {phase_diff}"


def test_crossing_polarization_dependent_loss():
    """Verify that TM mode experiences higher crossing attenuation than TE mode."""
    config = PhotonicConfig(
        n_modes=4,
        enable_polarization=True,
        crossing_loss_db=0.025,
        tm_crossing_loss_db=0.080
    )
    pol_model = JonesPolarizationModel(config)

    # Equal power in TE and TM: [1.0, 1.0]
    E_in = torch.ones(4, 2, dtype=torch.complex64)
    crossings = torch.tensor([5.0, 10.0, 15.0, 20.0])

    E_out = pol_model.apply_crossing_pdl(E_in, num_crossings=crossings)

    te_power = torch.abs(E_out[:, 0])**2
    tm_power = torch.abs(E_out[:, 1])**2

    # TM power must be strictly less than TE power due to higher PDL
    assert (tm_power < te_power).all(), f"TM power ({tm_power}) must be lower than TE ({te_power}) due to PDL"


def test_polarization_autograd():
    """Verify smooth backward gradient flow through Jones matrix operations."""
    config = PhotonicConfig(n_modes=4, enable_polarization=True)
    pol_model = JonesPolarizationModel(config)

    E_in = torch.randn(4, 2, dtype=torch.complex64, requires_grad=True)
    E_rot = pol_model.apply_bend_coupling(E_in)
    power = pol_model.total_optical_power(E_rot)

    loss = power.sum()
    loss.backward()

    assert E_in.grad is not None
    assert not torch.isnan(E_in.grad).any()
    assert (E_in.grad.abs() > 0.0).any()


def test_polarization_ideal_mode():
    """Ideal mode must bypass polarization effects."""
    config = PhotonicConfig(n_modes=4, ideal_mode=True)
    pol_model = JonesPolarizationModel(config)

    E_in = torch.randn(4, 2, dtype=torch.complex64)
    E_rot = pol_model.apply_bend_coupling(E_in)

    assert torch.equal(E_in, E_rot)


if __name__ == "__main__":
    test_bend_polarization_mixing()
    test_modal_birefringence_phase_drift()
    test_crossing_polarization_dependent_loss()
    test_polarization_autograd()
    test_polarization_ideal_mode()
    print("ALL POLARIZATION TESTS PASSED.")
