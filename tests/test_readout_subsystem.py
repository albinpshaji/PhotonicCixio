"""
Unit Test: Balanced Photodetection, TIA Saturation, and ADC Readout.
Validates:
1. Balanced Photodiode (BPD) common-mode rejection ratio (CMRR).
2. Transimpedance amplifier (TIA) soft saturation non-linearity.
3. Output ADC discretization with Straight-Through Estimator.
4. Grating coupler spectral transmission bandpass envelope.
"""

import sys
import os
import math
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PhotonicConfig
from src.physics.photodiode import PhotodetectorArray


def test_bpd_cmrr_suppression():
    """Verifies that balanced photodetection suppresses common-mode optical signals."""
    cfg = PhotonicConfig(
        n_modes=8,
        enable_balanced_detection=True,
        bpd_cmrr_db=30.0,
        enable_noise=False
    )
    detector = PhotodetectorArray(cfg)

    # Identical common-mode input power: 1 mW on both arms
    p_in = 1.0e-3
    E_pos = torch.tensor([math.sqrt(p_in)], dtype=torch.complex64)
    E_neg = torch.tensor([math.sqrt(p_in)], dtype=torch.complex64)

    # BPD differential output
    I_diff = detector.detect_balanced(E_pos, E_neg, add_noise=False).item()
    I_single = (detector.R * p_in)

    # Relative suppression ratio
    suppression_ratio = abs(I_diff) / I_single
    # Expected: 10^(-CMRR / 20) = 10^(-30 / 20) approx 0.0316
    expected_ratio = 10.0 ** (-30.0 / 20.0)
    assert abs(suppression_ratio - expected_ratio) < 1e-4, (
        f"CMRR suppression mismatch: {suppression_ratio:.4e} vs {expected_ratio:.4e}"
    )


def test_tia_saturation():
    """Verifies that TIA voltage saturates at v_sat under high input currents."""
    cfg = PhotonicConfig(n_modes=8, enable_adc=False, tia_saturation_voltage=1.2)
    detector = PhotodetectorArray(cfg)

    # Very large input current (100 mA into 1 kOhm = 100 V linear)
    I_huge = torch.tensor([0.10], dtype=torch.float32)
    V_out = detector.readout_electronics(I_huge).item()

    # Must be asymptotically clamped to v_sat = 1.2 V
    assert abs(V_out - 1.2) < 0.01, f"TIA did not saturate at 1.2V: {V_out}"


def test_adc_quantization():
    """Verifies that ADC output consists of discrete levels."""
    cfg = PhotonicConfig(
        n_modes=8,
        enable_adc=True,
        adc_bits=8,
        tia_saturation_voltage=1.2
    )
    detector = PhotodetectorArray(cfg)

    # Sweep input voltages across dynamic range
    I_sweep = torch.linspace(-0.002, 0.002, 1000)
    V_adc = detector.readout_electronics(I_sweep)

    # Number of unique quantized output values must not exceed 2^8 = 256
    unique_levels = torch.unique(V_adc)
    assert len(unique_levels) <= 256, f"Too many ADC levels: {len(unique_levels)}"
    assert len(unique_levels) > 10, f"ADC did not quantize into multiple levels: {len(unique_levels)}"


def test_grating_coupler_bandpass():
    """Verifies that grating coupler attenuates off-center wavelengths."""
    cfg = PhotonicConfig(
        n_modes=8,
        wavelength=1550e-9,
        grating_coupler_loss_db=3.0,
        grating_bandwidth=35.0e-9
    )
    detector = PhotodetectorArray(cfg)

    E_0 = torch.tensor([1.0], dtype=torch.complex64)

    # Peak transmission at 1550 nm
    E_center = detector.apply_grating_coupler(E_0, wavelength=1550e-9)
    p_center = (torch.abs(E_center) ** 2).item()

    # Off-peak transmission at 1550 nm + 35 nm = 1585 nm
    E_edge = detector.apply_grating_coupler(E_0, wavelength=1585e-9)
    p_edge = (torch.abs(E_edge) ** 2).item()

    # 1-dB bandwidth means transmission drops by ~ 1 dB at the edge
    drop_db = 10.0 * math.log10(p_center / p_edge)
    assert drop_db > 0.8, f"Grating did not attenuate off-center wavelength: drop = {drop_db:.2f} dB"


def test_homodyne_detection_quadratures():
    """Verifies that balanced optical homodyne detection linearly recovers In-Phase (I) and Quadrature (Q) fields."""
    cfg = PhotonicConfig(n_modes=4, enable_noise=False, ideal_mode=False, bpd_cmrr_db=60.0)
    detector = PhotodetectorArray(cfg)

    # Test field: E = A * exp(i * phi) with known amplitude and phase
    A = math.sqrt(2.0e-3)  # sqrt(2 mW)
    phi = math.pi / 4.0    # 45 deg -> Re(E) = Im(E) = 1 mW^0.5
    E_test = torch.tensor([A * math.cos(phi) + 1.0j * A * math.sin(phi)], dtype=torch.complex64)

    P_lo = 1.0e-3  # 1 mW LO
    # Expected I-channel: 2 * R * sqrt(P_lo) * Re(E)
    expected_I = 2.0 * detector.R * math.sqrt(P_lo) * (A * math.cos(phi))
    # Expected Q-channel: 2 * R * sqrt(P_lo) * Im(E)
    expected_Q = 2.0 * detector.R * math.sqrt(P_lo) * (A * math.sin(phi))

    I_meas = detector.detect_homodyne(E_test, mode="homodyne_i", P_lo_watts=P_lo, add_noise=False).item()
    Q_meas = detector.detect_homodyne(E_test, mode="homodyne_q", P_lo_watts=P_lo, add_noise=False).item()

    assert abs(I_meas - expected_I) / expected_I < 1e-2, f"Homodyne I mismatch: {I_meas:.4e} vs {expected_I:.4e}"
    assert abs(Q_meas - expected_Q) / expected_Q < 1e-2, f"Homodyne Q mismatch: {Q_meas:.4e} vs {expected_Q:.4e}"


def test_dark_current_mean_photocurrent():
    """Verifies that dark current appears as a physical DC baseline in single-ended readout."""
    cfg = PhotonicConfig(n_modes=4, dark_current=10.0e-9, enable_noise=False, ideal_mode=False)
    detector = PhotodetectorArray(cfg)

    # Zero optical power: pure dark current
    E_zero = torch.zeros(4, dtype=torch.complex64)
    I_dark_out = detector(E_zero, add_noise=False, readout_mode="direct")

    expected_dark = torch.full((4,), 10.0e-9, dtype=torch.float32)
    assert torch.allclose(I_dark_out, expected_dark, rtol=1e-4), (
        f"Dark current DC baseline mismatch: {I_dark_out} vs {expected_dark}"
    )


if __name__ == "__main__":
    test_bpd_cmrr_suppression()
    test_tia_saturation()
    test_adc_quantization()
    test_grating_coupler_bandpass()
    test_homodyne_detection_quadratures()
    test_dark_current_mean_photocurrent()
    print("ALL READOUT SUBSYSTEM TESTS PASSED.")
