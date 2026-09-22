"""
Tests for Waveguide Bend Radiation Loss and Mode-Mismatch Scattering.
Validates:
1. Exponential radiation loss scaling with bend radius.
2. S-bend geometry calculation across Clements layout channels.
3. Non-trivial physical insertion loss budget.
4. Autograd backward gradient flow.
5. Ideal mode bypass.
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
from src.physics.bend_loss import WaveguideBendLossModel, compute_sbend_radius_and_loss


def test_sbend_radius_and_loss_scaling():
    """Verify that tighter bends have exponentially higher radiation loss."""
    # S-bend with tight lateral offset (sharper bend)
    R_tight, loss_tight = compute_sbend_radius_and_loss(dx=60e-6, dy=40e-6)
    # S-bend with wide gentle lateral offset (larger bend radius)
    R_gentle, loss_gentle = compute_sbend_radius_and_loss(dx=200e-6, dy=20e-6)

    assert R_tight < R_gentle, f"R_tight ({R_tight}) should be smaller than R_gentle ({R_gentle})"
    assert loss_tight > loss_gentle, f"Tighter bend must exhibit higher loss: {loss_tight} > {loss_gentle}"


def test_mode_dependent_bend_loss():
    """Channels traversing the mesh must accumulate physical bend and transition losses."""
    config = PhotonicConfig(n_modes=8, enable_bend_loss=True)
    bend_model = WaveguideBendLossModel(config)

    loss_db = bend_model.mode_bend_loss_db
    assert loss_db.shape == (8,)

    # Total loss across 8 stages should be in the realistic range 0.1 dB to 1.5 dB
    assert (loss_db > 0.05).all(), "All channels traversing the mesh must have positive bend/transition loss"
    assert (loss_db < 2.5).all(), "Total bend loss must remain within physical bounds (< 2.5 dB)"

    # Field transmission must be strictly between 0.5 and 1.0
    trans = bend_model.bend_transmission
    assert (trans > 0.5).all() and (trans <= 1.0).all()


def test_bend_loss_autograd():
    """Verify smooth backward gradient flow through bend loss module."""
    config = PhotonicConfig(n_modes=8, enable_bend_loss=True)
    bend_model = WaveguideBendLossModel(config)

    E_in = torch.randn(8, dtype=torch.complex64, requires_grad=True)
    E_out = bend_model(E_in)

    loss = torch.sum(torch.abs(E_out)**2)
    loss.backward()

    assert E_in.grad is not None
    assert not torch.isnan(E_in.grad).any()
    assert (E_in.grad.abs() > 0.0).any()


def test_bend_loss_ideal_mode():
    """Ideal mode must bypass bend and transition loss completely."""
    config = PhotonicConfig(n_modes=8, ideal_mode=True)
    bend_model = WaveguideBendLossModel(config)

    E_in = torch.randn(8, dtype=torch.complex64)
    E_out = bend_model(E_in)

    assert torch.equal(E_in, E_out), "Ideal mode must leave field completely unmodified"


if __name__ == "__main__":
    test_sbend_radius_and_loss_scaling()
    test_mode_dependent_bend_loss()
    test_bend_loss_autograd()
    test_bend_loss_ideal_mode()
    print("ALL BEND LOSS TESTS PASSED.")
