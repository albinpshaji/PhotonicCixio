"""
Tests for 1/f Flicker Noise and Physical Temporal Aging Drift.
Validates:
1. Signal-dependent 1/f flicker noise contribution in photodiode readout.
2. Photodiode dark current aging degradation over operating hours.
3. Micro-heater resistance logarithmic aging in thermal crosstalk module.
4. Ideal mode bypass.
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
from src.physics.photodiode import PhotodetectorArray
from src.physics.thermal import ThermalCrosstalkModel
from src.physics.temporal_noise import TemporalNoiseAndAgingModel


def test_flicker_noise_scaling():
    """Verify that enabling 1/f flicker noise increases photocurrent variance."""
    cfg_no_flicker = PhotonicConfig(enable_noise=True, enable_flicker_noise=False, enable_adc=False)
    cfg_with_flicker = PhotonicConfig(enable_noise=True, enable_flicker_noise=True, enable_adc=False)

    pd_no_flicker = PhotodetectorArray(cfg_no_flicker)
    pd_with_flicker = PhotodetectorArray(cfg_with_flicker)

    # High signal current: 5 mA
    I_test = torch.full((5000,), 5.0e-3)

    noise_samples_base = pd_no_flicker._add_detector_noise(I_test) - I_test
    noise_samples_flicker = pd_with_flicker._add_detector_noise(I_test) - I_test

    std_base = torch.std(noise_samples_base).item()
    std_flicker = torch.std(noise_samples_flicker).item()

    assert std_flicker > std_base, f"1/f flicker noise must increase noise std: {std_flicker} > {std_base}"


def test_photodiode_dark_current_aging():
    """Verify that operating hours increase photodetector dark current linearly."""
    cfg_fresh = PhotonicConfig(enable_aging=False, operating_hours=0.0)
    cfg_aged = PhotonicConfig(enable_aging=True, operating_hours=5000.0, dark_current_aging_rate=0.02)

    pd_fresh = PhotodetectorArray(cfg_fresh)
    pd_aged = PhotodetectorArray(cfg_aged)

    # 5000 hours at 2% / 1000 hrs = 10% increase
    expected_ratio = 1.0 + 0.02 * (5000.0 / 1000.0)  # 1.10
    actual_ratio = pd_aged.I_dark / pd_fresh.I_dark

    assert abs(actual_ratio - expected_ratio) < 1e-4, f"Aged dark current mismatch: {actual_ratio} vs {expected_ratio}"


def test_heater_resistance_aging():
    """Verify that micro-heater resistance increases logarithmically with lifetime."""
    cfg_fresh = PhotonicConfig(enable_aging=False, operating_hours=0.0)
    cfg_aged = PhotonicConfig(enable_aging=True, operating_hours=1000.0, heater_aging_rate=0.005)

    th_fresh = ThermalCrosstalkModel(cfg_fresh)
    th_aged = ThermalCrosstalkModel(cfg_aged)

    expected_factor = 1.0 + 0.005 * math.log10(1.0 + 1000.0)
    actual_factor = th_aged.R0 / th_fresh.R0

    assert abs(actual_factor - expected_factor) < 1e-4, f"Heater aging factor mismatch: {actual_factor} vs {expected_factor}"
    assert th_aged.R0 > th_fresh.R0, "Aged heater resistance must be strictly greater than fresh resistance"


def test_aging_ideal_mode_bypass():
    """Ideal mode must disable all noise and aging drift."""
    cfg_ideal = PhotonicConfig(ideal_mode=True, enable_aging=True, operating_hours=10000.0)
    pd = PhotodetectorArray(cfg_ideal)
    th = ThermalCrosstalkModel(cfg_ideal)

    assert pd.I_dark == cfg_ideal.dark_current
    assert th.R0 == cfg_ideal.heater_resistance


if __name__ == "__main__":
    test_flicker_noise_scaling()
    test_photodiode_dark_current_aging()
    test_heater_resistance_aging()
    test_aging_ideal_mode_bypass()
    print("ALL TEMPORAL NOISE & AGING TESTS PASSED.")
