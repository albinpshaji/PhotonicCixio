"""
Physics-Aware Loss Functions for Optical Neural Networks and Photonic Tensor Compilers.
Mathematical references:
- Hughes et al. (Optica 2018), Vol. 5, pp. 864-871 (In-situ backpropagation)
- Pai et al. (Science 2023), Vol. 380, pp. 398-404 (Hardware-in-the-loop adjoint descent)
- Wang et al. (Nature Comm 2022), Vol. 13, 123 (Sub-photon noise-aware bounds)
- Zhu et al. (ICCAD 2020) & Sunny et al. (DAC 2021) (Thermal gradient penalty)
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class PhotonicFidelityLoss(nn.Module):
    """
    Measures matrix fidelity between target unitary matrix U_target and simulated U_twin:
        L_fid = 1.0 - (1 / N^2) * |Tr(U_target^H @ U_twin)|^2

    Yields L_fid in [0.0, 1.0], where L_fid = 0.0 indicates perfect phase/amplitude alignment
    up to a global unobservable phase factor e^(i * phi_global).
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, U_target: torch.Tensor, U_twin: torch.Tensor) -> torch.Tensor:
        """
        Args:
            U_target: Target unitary matrix of shape (..., N, N).
            U_twin: Realized physical transfer matrix of shape (..., N, N).

        Returns:
            Scalar fidelity loss.
        """
        N = U_target.shape[-1]
        # Inner product: Tr(U_target^H @ U_twin)
        U_target_H = torch.conj(torch.transpose(U_target, -2, -1))
        prod = torch.matmul(U_target_H, U_twin)
        trace_val = torch.diagonal(prod, dim1=-2, dim2=-1).sum(dim=-1)

        # Normalized overlap fidelity: |Tr(U_target^H @ U_twin)|^2 / N^2
        fidelity = torch.square(torch.abs(trace_val)) / (N**2)
        fidelity_loss = 1.0 - torch.clamp(fidelity, 0.0, 1.0)
        return fidelity_loss.mean()


class UnitaryDriftPenalty(nn.Module):
    """
    Penalizes deviations from mathematical unitarity caused by optical insertion losses,
    directional coupler deviations, and asymmetric channel attenuation:
        L_unit = (1 / N) * || U^H @ U - I ||_F^2
    """
    def __init__(self):
        super().__init__()

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        """
        Args:
            U: Complex transfer matrix of shape (..., N, N).

        Returns:
            Scalar unitary deviation penalty.
        """
        N = U.shape[-1]
        device = U.device
        dtype = U.dtype

        eye = torch.eye(N, device=device, dtype=dtype)
        U_H = torch.conj(torch.transpose(U, -2, -1))
        diff = torch.matmul(U_H, U) - eye

        # Frobenius norm squared
        frobenius_sq = torch.sum(torch.abs(diff)**2, dim=(-2, -1))
        return (frobenius_sq / N).mean()


class SpatialThermalGradientPenalty(nn.Module):
    """
    Penalizes sharp temperature differences between physically adjacent micro-heaters:
        L_thermal = sum_{<i, j>} (P_i - P_j)^2

    Minimizing this penalty suppresses non-local thermal bleed across the die,
    reduces mechanical silicon stress, and keeps heater powers within TED decoupling bounds.
    """
    def __init__(self, coords: torch.Tensor, neighbor_radius: float = 160.0e-6):
        super().__init__()
        M = coords.shape[0]
        # Precompute neighbor adjacency mask based on Euclidean layout distance
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        # Adjacent if distance <= neighbor_radius and i != j
        adj = (dist <= neighbor_radius) & (~torch.eye(M, dtype=torch.bool, device=coords.device))
        self.register_buffer("adjacency_mask", adj)

    def forward(self, heater_powers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heater_powers: Tensor of shape (..., M) containing dissipated heater powers (Watts).

        Returns:
            Scalar thermal smoothness penalty.
        """
        # Pairwise power difference: (..., M, 1) - (..., 1, M)
        p_diff = heater_powers.unsqueeze(-1) - heater_powers.unsqueeze(-2)
        p_diff_sq = torch.square(p_diff)

        # Mask only adjacent pairs
        masked_diffs = p_diff_sq * self.adjacency_mask.to(p_diff_sq.device).float()
        num_pairs = torch.clamp(self.adjacency_mask.sum(), min=1.0)
        return (masked_diffs.sum(dim=(-2, -1)) / num_pairs).mean()


class InSituAdjointLoss(nn.Module):
    """
    Implements the Hughes-Pai in-situ photonic adjoint variable gradient formulation:
        d(L) / d(theta_m) = - 2 * Im( E_forward_m * (E_adjoint_m)* )

    Computes the exact physical optical interference term between the forward
    propagating wavefield and the backward injected adjoint error field.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        E_forward_arm1: torch.Tensor,
        E_forward_arm2: torch.Tensor,
        E_adjoint_arm1: torch.Tensor,
        E_adjoint_arm2: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates physical in-situ gradient update vectors for MZI phase shifters.

        Args:
            E_forward_arm1, E_forward_arm2: Forward optical fields inside MZI arms.
            E_adjoint_arm1, E_adjoint_arm2: Backward adjoint fields injected at output.

        Returns:
            In-situ phase gradient tensor dL/d(theta) of shape (..., M).
        """
        # Interference between forward and adjoint fields inside the phase-modulated arm
        overlap1 = E_forward_arm1 * torch.conj(E_adjoint_arm1)
        overlap2 = E_forward_arm2 * torch.conj(E_adjoint_arm2)

        # Physical gradient is proportional to imaginary part of inter-arm interference
        grad1 = -2.0 * torch.imag(overlap1)
        grad2 = 2.0 * torch.imag(overlap2)
        return grad1 + grad2


class CompositePhotonicLoss(nn.Module):
    """
    Master composite loss function combining task objective, matrix fidelity,
    unitarity regularization, thermal gradient penalty, and optical power constraints:
        L_total = L_task + w_fid * L_fid + w_unit * L_unit + w_therm * L_therm + w_pwr * L_pwr
    """
    def __init__(
        self,
        coords: torch.Tensor,
        w_fidelity: float = 1.0,
        w_unitarity: float = 0.1,
        w_thermal: float = 0.05,
        w_power: float = 0.01,
        max_optical_power_watts: float = 10.0e-3
    ):
        super().__init__()
        self.w_fidelity = w_fidelity
        self.w_unitarity = w_unitarity
        self.w_thermal = w_thermal
        self.w_power = w_power
        self.max_optical_power = max_optical_power_watts

        self.fidelity_loss = PhotonicFidelityLoss()
        self.unitary_penalty = UnitaryDriftPenalty()
        self.thermal_penalty = SpatialThermalGradientPenalty(coords)

    def forward(
        self,
        U_realized: torch.Tensor,
        U_target: Optional[torch.Tensor] = None,
        heater_powers: Optional[torch.Tensor] = None,
        optical_fields: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes composite loss breakdown dictionary.
        """
        losses = {}
        total = torch.tensor(0.0, device=U_realized.device)

        if U_target is not None:
            l_fid = self.fidelity_loss(U_target, U_realized)
            losses["fidelity_loss"] = l_fid
            total = total + self.w_fidelity * l_fid

        l_unit = self.unitary_penalty(U_realized)
        losses["unitarity_loss"] = l_unit
        total = total + self.w_unitarity * l_unit

        if heater_powers is not None:
            l_therm = self.thermal_penalty(heater_powers)
            losses["thermal_loss"] = l_therm
            total = total + self.w_thermal * l_therm

        if optical_fields is not None:
            # Penalize optical power exceeding TPA threshold
            pwr = torch.abs(optical_fields)**2
            excess_pwr = torch.clamp(pwr - self.max_optical_power, min=0.0)
            l_pwr = torch.square(excess_pwr).mean()
            losses["power_penalty"] = l_pwr
            total = total + self.w_power * l_pwr

        losses["total_loss"] = total
        return losses
