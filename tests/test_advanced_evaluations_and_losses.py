"""
Tests for Physics-Aware Loss Functions and Hardware Realism Evaluation Metrics.
Validates:
1. Matrix fidelity loss, unitary drift penalty, and thermal gradient smoothing.
2. In-situ adjoint gradient computation (Hughes Optica 2018 / Pai Science 2023).
3. Composite photonic loss autograd differentiability.
4. Hardware ENOB calculation and Signal-to-Noise-and-Distortion Ratio (SINAD).
5. Energy per MAC (fJ/MAC) and sub-photon quantum scaling (Wang Nature Comm 2022).
6. Comprehensive full-system hardware evaluation report.
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
from src.models.digital_twin import PhotonicMeshDigitalTwin
from src.training.losses import (
    PhotonicFidelityLoss,
    UnitaryDriftPenalty,
    SpatialThermalGradientPenalty,
    InSituAdjointLoss,
    CompositePhotonicLoss,
)
from src.utils.evaluations import (
    evaluate_matrix_fidelity,
    evaluate_effective_resolution_enob,
    evaluate_optical_energy_per_mac,
    evaluate_full_hardware_system,
)


def test_photonic_fidelity_and_unitary_loss():
    """Verify matrix fidelity and unitarity penalties."""
    N = 4
    U_ideal = torch.eye(N, dtype=torch.complex64)
    loss_fid = PhotonicFidelityLoss()
    loss_unit = UnitaryDriftPenalty()

    # Identical matrix has exactly 0.0 fidelity loss
    assert loss_fid(U_ideal, U_ideal).item() < 1e-6
    # Unitary matrix has exactly 0.0 unitary drift penalty
    assert loss_unit(U_ideal).item() < 1e-6

    # Perturbed non-unitary matrix
    U_perturbed = U_ideal * 0.7
    assert loss_fid(U_ideal, U_perturbed).item() > 0.05
    assert loss_unit(U_perturbed).item() > 0.05


def test_thermal_gradient_penalty():
    """Verify that steep spatial power gradients between adjacent heaters are penalized."""
    cfg = PhotonicConfig(n_modes=4)
    twin = PhotonicMeshDigitalTwin(cfg)
    loss_therm = SpatialThermalGradientPenalty(twin.thermal_model.coords)

    # Uniform heater powers -> 0 penalty
    P_uniform = torch.full((twin.total_mzis,), 10.0e-3)
    assert loss_therm(P_uniform).item() < 1e-7

    # Non-uniform heater powers (alternating 0 and 20 mW)
    P_nonuniform = torch.tensor([0.0, 20e-3] * (twin.total_mzis // 2))
    assert loss_therm(P_nonuniform).item() > 1e-6


def test_in_situ_adjoint_loss():
    """Verify in-situ adjoint interference gradient calculation."""
    adjoint_module = InSituAdjointLoss()

    # Forward optical fields: e^(i * 0)
    E_fwd1 = torch.tensor([1.0 + 0.0j])
    E_fwd2 = torch.tensor([0.0 + 1.0j])
    # Backward adjoint error fields: e^(i * pi/4)
    E_adj1 = torch.tensor([0.7071 + 0.7071j])
    E_adj2 = torch.tensor([0.7071 + 0.7071j])

    grad = adjoint_module(E_fwd1, E_fwd2, E_adj1, E_adj2)
    assert grad.shape == (1,)
    assert not torch.isnan(grad).any()
    assert abs(grad.item()) > 0.01


def test_composite_loss_autograd():
    """Verify smooth backward gradient flow through CompositePhotonicLoss."""
    cfg = PhotonicConfig(n_modes=4, ideal_mode=False)
    twin = PhotonicMeshDigitalTwin(cfg)
    loss_fn = CompositePhotonicLoss(twin.thermal_model.coords)

    theta = torch.rand(twin.total_mzis, requires_grad=True)
    U_realized = twin.compute_transfer_matrix(theta=theta)
    U_target = torch.eye(cfg.n_modes, dtype=torch.complex64)

    losses = loss_fn(
        U_realized=U_realized,
        U_target=U_target,
        heater_powers=twin.thermal_model.compute_joule_power(theta)
    )

    total_loss = losses["total_loss"]
    total_loss.backward()

    assert theta.grad is not None
    assert not torch.isnan(theta.grad).any()
    assert (theta.grad.abs() > 0.0).any()


def test_hardware_evaluations_enob_and_energy():
    """Verify ENOB, energy per MAC, and full hardware system report."""
    cfg = PhotonicConfig(n_modes=8, dac_bits=6, adc_bits=8)
    twin = PhotonicMeshDigitalTwin(cfg)

    # ENOB evaluation
    enob_res = evaluate_effective_resolution_enob(twin, batch_size=16)
    assert 2.0 <= enob_res["enob_bits"] <= 8.5
    assert enob_res["sinad_db"] > 10.0

    # Energy per MAC evaluation
    energy_res = evaluate_optical_energy_per_mac(cfg)
    assert energy_res["energy_per_mac_fj"] > 0.1
    assert energy_res["photons_per_mac"] > 10.0

    # Full hardware system report
    report = evaluate_full_hardware_system(twin)
    assert report.trace_fidelity > 0.0
    assert report.thermal_condition_number >= 1.0
    assert report.peak_die_temperature_rise_k > 0.0
    assert report.enob_bits > 0.0


if __name__ == "__main__":
    test_photonic_fidelity_and_unitary_loss()
    test_thermal_gradient_penalty()
    test_in_situ_adjoint_loss()
    test_composite_loss_autograd()
    test_hardware_evaluations_enob_and_energy()
    print("ALL ADVANCED EVALUATIONS AND LOSSES TESTS PASSED.")
