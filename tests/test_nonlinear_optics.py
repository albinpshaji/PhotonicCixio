"""
Tests for Silicon Optical Nonlinearities (TPA, FCA, SPM, FCD).
Validates:
1. Convergence to linear propagation at microwatt power levels.
2. Sub-linear transmission and power saturation at high optical intensity (>10 mW).
3. Intensity-dependent SPM phase shift.
4. Smooth PyTorch autograd gradient backpropagation.
5. Bypass under ideal_mode=True.
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
from src.physics.nonlinear_optics import SiliconNonlinearOptics


def test_low_power_linearity():
    """At low power (1 uW), nonlinear effects must be virtually zero."""
    config = PhotonicConfig(enable_nonlinear_optics=True)
    nl_model = SiliconNonlinearOptics(config)

    # 1 microwatt optical field amplitude: sqrt(1e-6 W)
    E_low = torch.tensor([math.sqrt(1e-6) + 0.0j], dtype=torch.complex64)
    E_out = nl_model(E_low)

    power_in = torch.abs(E_low)**2
    power_out = torch.abs(E_out)**2

    rel_diff = torch.abs(power_out - power_in) / power_in
    assert rel_diff.item() < 1e-4, f"Low-power linearity violated: rel_diff={rel_diff.item()}"


def test_high_power_saturation():
    """At high power (50 mW), TPA and FCA must introduce measurable power attenuation."""
    config = PhotonicConfig(enable_nonlinear_optics=True)
    nl_model = SiliconNonlinearOptics(config)

    # 50 mW power
    p_high = 50e-3
    E_high = torch.tensor([math.sqrt(p_high) + 0.0j], dtype=torch.complex64)
    # Propagate across a 1 mm test waveguide
    L_test = 1.0e-3
    E_out = nl_model(E_high, segment_length=L_test)

    power_out = torch.abs(E_out)**2
    attenuation_ratio = (power_out / p_high).item()

    assert attenuation_ratio < 0.98, f"High power should exhibit TPA/FCA absorption: ratio={attenuation_ratio}"
    assert attenuation_ratio > 0.15, f"Transmission should remain physically bounded: ratio={attenuation_ratio}"


def test_spm_phase_shift():
    """High optical intensity must induce an intensity-dependent self-phase modulation."""
    config = PhotonicConfig(enable_nonlinear_optics=True)
    nl_model = SiliconNonlinearOptics(config)

    E_10mw = torch.tensor([math.sqrt(10e-3) + 0.0j], dtype=torch.complex64)
    E_out = nl_model(E_10mw, segment_length=2.0e-3)

    phase_shift = torch.angle(E_out).item()
    assert abs(phase_shift) > 1e-4, f"SPM phase shift should be non-zero at 10 mW: phase={phase_shift}"


def test_nonlinear_autograd():
    """Verify smooth backward gradient flow through nonlinear operators."""
    config = PhotonicConfig(enable_nonlinear_optics=True)
    nl_model = SiliconNonlinearOptics(config)

    E_in = torch.randn(8, dtype=torch.complex64, requires_grad=True)
    E_out = nl_model(E_in)

    loss = torch.sum(torch.abs(E_out)**2)
    loss.backward()

    assert E_in.grad is not None
    assert not torch.isnan(E_in.grad).any()
    assert (E_in.grad.abs() > 0.0).any()


def test_ideal_mode_bypass():
    """Ideal mode must completely disable nonlinear optics."""
    config = PhotonicConfig(ideal_mode=True)
    nl_model = SiliconNonlinearOptics(config)

    E_high = torch.tensor([math.sqrt(100e-3) + 0.0j], dtype=torch.complex64)
    E_out = nl_model(E_high)

    assert torch.equal(E_high, E_out), "Ideal mode must leave field completely unmodified"


if __name__ == "__main__":
    test_low_power_linearity()
    test_high_power_saturation()
    test_spm_phase_shift()
    test_nonlinear_autograd()
    test_ideal_mode_bypass()
    print("ALL NONLINEAR OPTICS TESTS PASSED.")
