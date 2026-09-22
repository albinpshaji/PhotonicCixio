"""
Foundry-to-Hardware Parameter Calibration and Empirical Identification Framework.
Mathematical references:
- Bandyopadhyay et al. (2021), arXiv:2103.04993: "Hardware Error Correction for Silicon Photonic Meshes"
- Pai et al. (2019), IEEE JSTQE 26(2), 1-19: "Matrix Optimization on Universal Unitary Photonic Circuits"
- Bogaerts et al. (2020), Nature 586, 207-216: "Programmable photonic circuits"

Distinguishes:
1. Prior Nominal Hypothesis: Uncalibrated foundry PDK parameters and spatial wafer priors.
2. Calibrated Hardware Model: Empirically fitted on-chip non-idealities:
   - Directional coupler splitting deviations: epsilon_1, epsilon_2 in [-0.25, 0.25]
   - Static lithographic phase biases: phi_intrinsic in [0, 2*pi)
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from src.models.digital_twin import PhotonicMeshDigitalTwin
from src.config import PhotonicConfig


@dataclass
class PhotonicCalibrationDataset:
    """
    Experimental diagnostic dataset collected from chip transmission measurements.
    """
    thetas: torch.Tensor          # (K, M) commanded internal phases
    phis: torch.Tensor            # (K, M) commanded external phases
    measured_matrices: torch.Tensor # (K, N, N) observed complex transmission matrices
    diag_phases: Optional[torch.Tensor] = None # (K, N) optional output diagonal phases
    wavelengths: Optional[torch.Tensor] = None # (K,) optional test wavelengths


@dataclass
class CalibrationResult:
    """
    Identified physical parameters and convergence metrics.
    """
    fitted_eps1: torch.Tensor
    fitted_eps2: torch.Tensor
    fitted_phi_intrinsic: torch.Tensor
    prior_rmse: float
    calibrated_rmse: float
    improvement_pct: float
    iterations: int
    converged: bool


def generate_synthetic_calibration_dataset(
    hardware_twin: PhotonicMeshDigitalTwin,
    num_samples: int = 32,
    seed: int = 42
) -> PhotonicCalibrationDataset:
    """
    Generates synthetic diagnostic measurements from an instantiated 'hardware' digital twin
    with unknown static wafer defects.
    """
    device = hardware_twin.device
    M = hardware_twin.total_mzis
    N = hardware_twin.n_modes

    gen = torch.Generator(device=device).manual_seed(seed)
    thetas = torch.rand(num_samples, M, generator=gen, device=device) * math.pi
    phis = torch.rand(num_samples, M, generator=gen, device=device) * (2.0 * math.pi)

    # Diagnostic measurement sweep (lossless transmission matrix baseline)
    with torch.no_grad():
        measured_mats = []
        for k in range(num_samples):
            T_k = hardware_twin.compute_transfer_matrix(thetas[k], phis[k])
            measured_mats.append(T_k)
        measured_matrices = torch.stack(measured_mats, dim=0)

    return PhotonicCalibrationDataset(
        thetas=thetas,
        phis=phis,
        measured_matrices=measured_matrices
    )


class MeshParameterEstimator:
    """
    Differentiable optimizer that identifies physical coupler split errors and phase biases
    by minimizing the Frobenius residual against experimental transmission measurements.
    """
    def __init__(
        self,
        nominal_twin: PhotonicMeshDigitalTwin,
        prior_weight: float = 1e-3,
        lr: float = 0.01
    ):
        self.nominal_twin = nominal_twin
        self.device = nominal_twin.device
        self.complex_dtype = nominal_twin.complex_dtype
        self.prior_weight = prior_weight
        self.lr = lr
        self.M = nominal_twin.total_mzis
        self.N = nominal_twin.n_modes

    def calibrate(
        self,
        dataset: PhotonicCalibrationDataset,
        num_epochs: int = 80,
        tol: float = 1e-6
    ) -> CalibrationResult:
        """
        Executes gradient-based parameter estimation.

        Loss = MSE(T_model, T_meas) + prior_weight * (||eps1 - eps1_0||^2 + ||eps2 - eps2_0||^2)
        """
        thetas = dataset.thetas.to(self.device)
        phis = dataset.phis.to(self.device)
        T_target = dataset.measured_matrices.to(device=self.device, dtype=self.complex_dtype)
        K = thetas.shape[0]

        # Prior nominal values (from nominal twin)
        eps1_prior = self.nominal_twin.coupler_eps1.clone().detach()
        eps2_prior = self.nominal_twin.coupler_eps2.clone().detach()
        phi_int_prior = self.nominal_twin.phi_intrinsic.clone().detach()

        # Learnable parameters (initialized to prior nominal hypothesis)
        eps1_param = nn.Parameter(eps1_prior.clone().requires_grad_(True))
        eps2_param = nn.Parameter(eps2_prior.clone().requires_grad_(True))
        phi_int_param = nn.Parameter(phi_int_prior.clone().requires_grad_(True))

        optimizer = torch.optim.Adam([eps1_param, eps2_param, phi_int_param], lr=self.lr)

        # Baseline error before calibration
        with torch.no_grad():
            T_prior_list = []
            for k in range(K):
                T_p = self.nominal_twin.compute_transfer_matrix(thetas[k], phis[k])
                T_prior_list.append(T_p)
            T_prior = torch.stack(T_prior_list, dim=0)
            prior_rmse = torch.sqrt(torch.mean(torch.abs(T_prior - T_target) ** 2)).item()

        # Training loop
        prev_loss = float("inf")
        converged = False

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            clamped_eps1 = torch.clamp(eps1_param, -0.25, 0.25)
            clamped_eps2 = torch.clamp(eps2_param, -0.25, 0.25)

            # Forward pass over diagnostic batch with overridden parameters
            T_pred_list = []
            for k in range(K):
                T_k = self.nominal_twin.compute_transfer_matrix(
                    thetas[k],
                    phis[k],
                    override_eps1=clamped_eps1,
                    override_eps2=clamped_eps2,
                    override_phi_intrinsic=phi_int_param
                )
                T_pred_list.append(T_k)
            T_pred = torch.stack(T_pred_list, dim=0)

            # Fit loss + Bayesian prior penalty
            mse_loss = torch.mean(torch.abs(T_pred - T_target) ** 2)
            prior_penalty = self.prior_weight * (
                torch.sum((eps1_param - eps1_prior) ** 2) +
                torch.sum((eps2_param - eps2_prior) ** 2) +
                torch.sum((phi_int_param - phi_int_prior) ** 2)
            )
            total_loss = mse_loss + prior_penalty

            total_loss.backward()
            optimizer.step()

            loss_val = total_loss.item()
            if abs(prev_loss - loss_val) < tol:
                converged = True
                break
            prev_loss = loss_val

        # Final evaluation
        with torch.no_grad():
            fitted_eps1 = torch.clamp(eps1_param.detach(), -0.25, 0.25)
            fitted_eps2 = torch.clamp(eps2_param.detach(), -0.25, 0.25)
            fitted_phi = phi_int_param.detach()

            T_cal_list = []
            for k in range(K):
                T_k = self.nominal_twin.compute_transfer_matrix(
                    thetas[k],
                    phis[k],
                    override_eps1=fitted_eps1,
                    override_eps2=fitted_eps2,
                    override_phi_intrinsic=fitted_phi
                )
                T_cal_list.append(T_k)
            T_cal = torch.stack(T_cal_list, dim=0)
            calibrated_rmse = torch.sqrt(torch.mean(torch.abs(T_cal - T_target) ** 2)).item()

        improvement_pct = max(0.0, (prior_rmse - calibrated_rmse) / max(prior_rmse, 1e-12)) * 100.0

        return CalibrationResult(
            fitted_eps1=fitted_eps1,
            fitted_eps2=fitted_eps2,
            fitted_phi_intrinsic=fitted_phi,
            prior_rmse=prior_rmse,
            calibrated_rmse=calibrated_rmse,
            improvement_pct=improvement_pct,
            iterations=epoch + 1,
            converged=converged
        )


def apply_calibrated_parameters(
    twin: PhotonicMeshDigitalTwin,
    result: CalibrationResult
):
    """
    Applies the identified physical parameters to a PhotonicMeshDigitalTwin instance,
    converting it from nominal prior to calibrated hardware twin.
    """
    twin.coupler_eps1.copy_(result.fitted_eps1)
    twin.coupler_eps2.copy_(result.fitted_eps2)
    twin.phi_intrinsic.copy_(result.fitted_phi_intrinsic)
