"""
Unit Test: Advanced Electro-Thermal Dynamics and Substrate Packaging.
Validates:
1. Joule heating power under Temperature Coefficient of Resistance (TCR).
2. Package substrate thermal impedance and global baseline drift.
3. Transient dynamic thermal ODE step-response convergence.
"""

import sys
import os
import math
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PhotonicConfig
from src.physics.thermal import ThermalCrosstalkModel


def test_tcr_power_compression():
    """Verifies that TCR compresses dissipated power under constant voltage drive."""
    cfg = PhotonicConfig(n_modes=8, enable_tcr=True, ideal_mode=False)
    thermal = ThermalCrosstalkModel(cfg)

    # Command pi phase shift (target power = P_pi = 21.5 mW)
    theta = torch.tensor([math.pi], dtype=torch.float32)
    P_joule = thermal.compute_joule_power(theta).item()

    # With TCR alpha > 0, resistance increases, so V^2 / R(T) < V^2 / R_0
    assert P_joule < cfg.P_pi, f"TCR should compress power: {P_joule*1e3:.2f} mW vs {cfg.P_pi*1e3:.2f} mW"
    # Compression should be around 5% - 15% for typical parameters
    ratio = P_joule / cfg.P_pi
    assert 0.80 < ratio < 0.98, f"Unexpected TCR compression ratio: {ratio:.3f}"


def test_package_substrate_drift():
    """Verifies that package thermal resistance produces die temperature rise proportional to total power."""
    cfg = PhotonicConfig(n_modes=8, package_thermal_resistance=15.0, ideal_mode=False)
    thermal = ThermalCrosstalkModel(cfg)

    # All 28 heaters driven to 21.5 mW
    P_heaters = torch.full((cfg.total_mzis,), cfg.P_pi, dtype=torch.float32)
    P_total = P_heaters.sum().item()

    drift = thermal.compute_package_heating(P_heaters).item()
    # Expected global phase drift: 0.05 * R_pack * P_total
    expected_drift = 0.05 * cfg.package_thermal_resistance * P_total
    assert abs(drift - expected_drift) < 1e-4, f"Substrate drift mismatch: {drift} vs {expected_drift}"


def test_transient_thermal_ode_convergence():
    """Verifies that transient thermal ODE step response converges to steady state."""
    cfg = PhotonicConfig(
        n_modes=8,
        thermal_mode="transient",
        thermal_time_constant=10.0e-6,
        package_thermal_resistance=0.0,
        enable_tcr=False,
        ideal_mode=False
    )
    thermal = ThermalCrosstalkModel(cfg)

    theta_cmd = torch.full((cfg.total_mzis,), 1.5, dtype=torch.float32)
    # Steady state
    cfg.thermal_mode = "steady_state"
    steady_state = thermal(theta_cmd)
    cfg.thermal_mode = "transient"

    # Step response with dt = 1 us over 60 us (6 * tau)
    dt = 1.0e-6
    state = torch.zeros_like(theta_cmd)
    for step in range(60):
        state = thermal(theta_cmd, state_T=state, dt=dt)

    # After 6 * tau, state must be within 1% of steady state (1 - exp(-6) = 0.9975)
    rel_diff = torch.norm(state - steady_state) / torch.norm(steady_state)
    assert rel_diff < 0.01, f"Transient did not converge to steady state: rel_diff = {rel_diff.item():.4e}"


if __name__ == "__main__":
    test_tcr_power_compression()
    test_package_substrate_drift()
    test_transient_thermal_ode_convergence()
    print("ALL ADVANCED THERMAL TESTS PASSED.")
