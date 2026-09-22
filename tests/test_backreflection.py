"""
Tests for Coherent Backreflection and Fabry-Perot Ripple.
Validates:
1. Wavelength-dependent transmission ripple (FSR and oscillation amplitude).
2. Mode-dependent cavity path variations.
3. Smooth PyTorch autograd gradient backpropagation.
4. Ideal mode bypass (strictly 1.0 transmission).
"""

import sys
import os
import math
import torch
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import PhotonicConfig, PhysicalConstants
from src.physics.backreflection import FabryPerotBackreflection


def test_spectral_ripple_and_fsr():
    """Verify that transmission exhibits periodic Fabry-Perot interference oscillations."""
    config = PhotonicConfig(n_modes=4, enable_backreflection=True)
    fp_model = FabryPerotBackreflection(config)

    # Sweep wavelength across 10 nm around 1550 nm to cover multiple FSR periods (FSR ~ 2.8 nm)
    wavelengths = torch.linspace(1545e-9, 1555e-9, steps=300)
    transmissions = []

    for lam in wavelengths:
        H = fp_model.compute_transfer_filter(wavelength=lam.item())
        T = torch.abs(H[0])**2
        transmissions.append(T.item())

    t_tensor = torch.tensor(transmissions)
    t_min = t_tensor.min().item()
    t_max = t_tensor.max().item()

    ripple_db = 10.0 * math.log10(t_max / t_min)
    assert 0.01 < ripple_db < 1.0, f"Expected ripple between 0.01 and 1.0 dB, got {ripple_db:.4f} dB"

    # With FSR ~ 2.8 nm over 10 nm sweep, we expect multiple extrema
    diffs = torch.diff(t_tensor)
    sign_changes = torch.sum(diffs[:-1] * diffs[1:] < 0).item()
    assert sign_changes >= 2, f"Expected at least 2 extrema across 10 nm sweep, got {sign_changes}"


def test_mode_dependent_ripple():
    """Different modes must experience different cavity lengths and distinct spectral phases."""
    config = PhotonicConfig(n_modes=4, enable_backreflection=True)
    fp_model = FabryPerotBackreflection(config)

    H = fp_model.compute_transfer_filter(wavelength=1550e-9)
    # Modes 0 and 3 have different cavity lengths
    assert abs(H[0].item() - H[3].item()) > 1e-4, "Modes must have different cavity responses"


def test_backreflection_autograd():
    """Verify smooth backward gradient flow through backreflection operator."""
    config = PhotonicConfig(n_modes=4, enable_backreflection=True)
    fp_model = FabryPerotBackreflection(config)

    E_in = torch.randn(4, dtype=torch.complex64, requires_grad=True)
    E_out = fp_model(E_in)

    loss = torch.sum(torch.abs(E_out)**2)
    loss.backward()

    assert E_in.grad is not None
    assert not torch.isnan(E_in.grad).any()
    assert (E_in.grad.abs() > 0.0).any()


def test_backreflection_ideal_mode():
    """Ideal mode must completely disable backreflection ripple."""
    config = PhotonicConfig(n_modes=4, ideal_mode=True)
    fp_model = FabryPerotBackreflection(config)

    E_in = torch.randn(4, dtype=torch.complex64)
    E_out = fp_model(E_in)

    assert torch.equal(E_in, E_out), "Ideal mode must leave field completely unmodified"


if __name__ == "__main__":
    test_spectral_ripple_and_fsr()
    test_mode_dependent_ripple()
    test_backreflection_autograd()
    test_backreflection_ideal_mode()
    print("ALL BACKREFLECTION TESTS PASSED.")
