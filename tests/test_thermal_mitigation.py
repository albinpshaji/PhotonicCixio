import math
import torch
import pytest
from src.config import PhotonicConfig
from src.physics.thermal import ThermalCrosstalkModel


@pytest.fixture
def crosstalk_model():
    cfg = PhotonicConfig(
        n_modes=4,
        enable_thermal_crosstalk=True,
        enable_tcr=False,
        package_thermal_resistance=0.0,
        ideal_mode=False
    )
    return ThermalCrosstalkModel(cfg)


def test_baseline_raw_bnnls_exhibits_bleed_floor(crosstalk_model):
    """Verifies that un-wrapped raw BNNLS clamps at 0 mW and exhibits ~0.32 rad residual error."""
    theta_target = torch.tensor([0.05, 3.10, 2.95, 3.05, 0.40, 2.90], dtype=torch.float32)
    theta_bnnls = crosstalk_model.predistort(
        theta_target,
        method="bnnls",
        enable_phase_wrapping=False,
        bias_power_watts=0.0,
        max_power_watts=0.050,
        max_iter=100
    )
    # Heater 0 clamped to 0 mW
    p_bnnls = theta_bnnls / math.pi * crosstalk_model.P_pi
    assert p_bnnls[0].item() == pytest.approx(0.0, abs=1e-5)

    # Actual phase shows residual error ~0.32 rad
    theta_real = crosstalk_model(theta_bnnls)
    err = torch.abs(theta_real - theta_target)
    assert err[0].item() > 0.25  # Unavoidable physical bleed floor
    assert err[1].item() < 0.05  # Active channels track well


def test_2pi_phase_wrapping_mitigation(crosstalk_model):
    """Verifies that 2pi phase wrapping eliminates negative power requirement and drops error to < 0.006 rad."""
    theta_target = torch.tensor([0.05, 3.10, 2.95, 3.05, 0.40, 2.90], dtype=torch.float32)
    theta_bnnls_wrapped = crosstalk_model.predistort(
        theta_target,
        method="bnnls",
        enable_phase_wrapping=True,
        bias_power_watts=0.0,
        max_power_watts=0.050,
        max_iter=150
    )
    # Powers must be strictly positive and within bounds
    p_wrapped = (theta_bnnls_wrapped / math.pi * crosstalk_model.P_pi) * 1e3
    assert (p_wrapped >= 0.0).all()
    assert (p_wrapped <= 50.0).all()
    assert p_wrapped[0].item() > 30.0  # Heater 0 is now driven positively

    # Optical error evaluated modulo 2pi
    theta_real = crosstalk_model(theta_bnnls_wrapped)
    err_mod = torch.abs(torch.remainder(theta_real - theta_target + math.pi, 2 * math.pi) - math.pi)
    assert err_mod.max().item() < 0.010
    assert err_mod[0].item() < 0.006


def test_thermal_prebiasing_mitigation(crosstalk_model):
    """Verifies that thermal pre-biasing enables virtual cooling and drops error to < 0.005 rad."""
    theta_target = torch.tensor([0.05, 3.10, 2.95, 3.05, 0.40, 2.90], dtype=torch.float32)
    bias_power = 0.010  # 10 mW
    theta_bias_drive = crosstalk_model.predistort(
        theta_target,
        method="bnnls",
        enable_phase_wrapping=False,
        bias_power_watts=bias_power,
        max_power_watts=0.050,
        max_iter=150
    )
    # Check powers: all in [0, 50 mW]
    p_mw = (theta_bias_drive / math.pi * crosstalk_model.P_pi) * 1e3
    assert (p_mw >= 0.0).all()
    assert (p_mw <= 50.0).all()
    # Heater 0 reduced power below 10 mW baseline to compensate neighbor crosstalk
    assert p_mw[0].item() < 10.0
    assert p_mw[0].item() > 5.0

    # Compute realized phase relative to pre-bias baseline
    baseline_drift = crosstalk_model.compute_thermal_bias_baseline(bias_power)
    theta_real = crosstalk_model(theta_bias_drive)
    logical_phase = theta_real - baseline_drift
    err = torch.abs(logical_phase - theta_target)
    assert err.max().item() < 0.005
    assert err[0].item() < 0.001
