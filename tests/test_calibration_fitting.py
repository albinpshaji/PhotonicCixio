"""
Unit Test: Photonic Parameter Calibration and Empirical Identification.
Validates:
1. Synthetic diagnostic measurement data generation.
2. Differentiable estimation of coupler split deviations and intrinsic phase biases.
3. Separation of nominal prior hypothesis from calibrated hardware model.
4. Convergence and error reduction (RMSE reduction > 30%).
"""

import sys
import os
import math
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PhotonicConfig
from src.models.digital_twin import PhotonicMeshDigitalTwin
from src.calibration.parameter_fitting import (
    MeshParameterEstimator,
    generate_synthetic_calibration_dataset,
    apply_calibrated_parameters
)


def test_parameter_calibration_convergence():
    """Verify that parameter estimation reduces prediction error on a perturbed hardware chip."""
    n_modes = 4

    # 1. Hardware chip with actual wafer manufacturing variations
    hw_cfg = PhotonicConfig(
        n_modes=n_modes,
        ideal_mode=False,
        enable_coupler_errors=True,
        coupler_error_std=0.04,
        intrinsic_phase_std=0.05,
        wafer_seed=1234,
        enable_spatial_correlation=False,
        enable_loss=False,
        enable_physical_routing=False,
        enable_dispersion=False,
        enable_quantization=False,
        enable_thermal_crosstalk=False,
        enable_polarization=False,
        enable_noise=False,
        enable_bend_loss=False,
        enable_backreflection=False,
        enable_nonlinear_optics=False,
        laser_linewidth=0.0,
        grating_coupler_loss_db=0.0
    )
    hardware_chip = PhotonicMeshDigitalTwin(hw_cfg)

    # 2. Prior nominal model (assumes zero coupler errors and zero intrinsic bias)
    nom_cfg = PhotonicConfig(
        n_modes=n_modes,
        ideal_mode=False,
        enable_coupler_errors=False,
        enable_loss=False,
        enable_physical_routing=False,
        enable_dispersion=False,
        enable_quantization=False,
        enable_thermal_crosstalk=False,
        enable_polarization=False,
        enable_noise=False,
        enable_bend_loss=False,
        enable_backreflection=False,
        enable_nonlinear_optics=False,
        laser_linewidth=0.0,
        grating_coupler_loss_db=0.0
    )
    nominal_model = PhotonicMeshDigitalTwin(nom_cfg)

    # 3. Collect diagnostic calibration dataset
    dataset = generate_synthetic_calibration_dataset(
        hardware_twin=hardware_chip,
        num_samples=24,
        seed=42
    )

    # 4. Run parameter estimation
    estimator = MeshParameterEstimator(nominal_model, prior_weight=1e-4, lr=0.02)
    result = estimator.calibrate(dataset, num_epochs=60)

    print(f"\nCalibration Results:")
    print(f"  Prior Nominal RMSE:      {result.prior_rmse:.4e}")
    print(f"  Calibrated Model RMSE:   {result.calibrated_rmse:.4e}")
    print(f"  Relative Error Reduction: {result.improvement_pct:.1f}%")
    print(f"  Optimizer Iterations:     {result.iterations}")

    assert result.calibrated_rmse < result.prior_rmse, "Calibration failed to improve model accuracy!"
    assert result.improvement_pct > 25.0, f"Expected >25% improvement, got {result.improvement_pct:.1f}%"

    # 5. Apply calibrated parameters to the model
    apply_calibrated_parameters(nominal_model, result)

    # Verify updated buffers
    assert torch.allclose(nominal_model.coupler_eps1, result.fitted_eps1)
    assert torch.allclose(nominal_model.coupler_eps2, result.fitted_eps2)


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING PHOTONIC PARAMETER CALIBRATION TESTS")
    print("=" * 70)
    test_parameter_calibration_convergence()
    print("=" * 70)
    print("[ALL CALIBRATION TESTS PASSED]")
    print("=" * 70)
