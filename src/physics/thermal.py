"""
Non-Local Thermal Diffusion, Multi-Heater Crosstalk, and Bounded Regularized Predistortion (BNNLS).
Mathematical references:
- Zhu et al. (2020), IEEE/ACM ICCAD 2020, pp. 1-7
- Zhang et al. (2023), arXiv:2312.01403 (OplixNet)
- Sunny et al. (2021), ACM/IEEE DAC 2021, pp. 1069-1074 (CrossLight)
- Gürses & Hajimiri (2022), arXiv:2206.04525 / IEEE JSTQE
"""

import sys
import os
import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

# Ensure project root is in path for standalone execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import PhotonicConfig


def compute_clements_layout_coordinates(
    n_modes: int,
    pitch_x: float = 120.0e-6,
    pitch_y: float = 80.0e-6,
    include_diagonal_screen: bool = False
) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
    """
    Computes the 2D spatial coordinates (x, y) for all MZIs and optional output
    diagonal phase screen heaters in a planar Clements rectangular mesh.

    Args:
        n_modes: Total optical modes N.
        pitch_x: Horizontal column pitch (meters).
        pitch_y: Vertical waveguide track pitch (meters).
        include_diagonal_screen: If True, appends coordinates for the N output heaters.

    Returns:
        coords: Tensor of shape (M, 2) or (M + N, 2) with (x, y) coordinates in meters.
        mzi_indices: List of tuples (col_idx, mode_p, mode_q) for each actuator.
    """
    coords = []
    mzi_indices = []

    for col in range(n_modes):
        if col % 2 == 0:
            pairs = [(2 * k, 2 * k + 1) for k in range(n_modes // 2)]
        else:
            pairs = [(2 * k + 1, 2 * k + 2) for k in range((n_modes - 2) // 2)]

        for p, q in pairs:
            x = col * pitch_x
            y = 0.5 * (p + q) * pitch_y
            coords.append([x, y])
            mzi_indices.append((col, p, q))

    if include_diagonal_screen:
        # Output diagonal phase screen column at col = N
        diag_col = n_modes
        for k in range(n_modes):
            x = diag_col * pitch_x
            y = k * pitch_y
            coords.append([x, y])
            mzi_indices.append((diag_col, k, k))

    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    return coords_tensor, mzi_indices


def build_thermal_crosstalk_matrix(
    coords: torch.Tensor,
    thermal_decay_length: float = 55.0e-6,
    thermal_impedance: float = 18.5
) -> torch.Tensor:
    """
    Constructs the non-local thermal diffusion matrix K in R^(K x K).
    K_ij = kappa_0 * exp( - ||x_i - x_j||_2 / lambda_thermal )
    """
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (K, K, 2)
    distances = torch.norm(diff, dim=-1)             # (K, K)
    K = thermal_impedance * torch.exp(-distances / max(thermal_decay_length, 1e-9))
    return K


class ThermalCrosstalkModel(nn.Module):
    """
    PyTorch module modeling multi-heater lateral thermal diffusion and
    constrained Bounded Non-Negative Least Squares (BNNLS) predistortion.
    """
    def __init__(self, config: PhotonicConfig):
        super().__init__()
        self.config = config
        self.n_modes = config.n_modes
        self.total_mzis = config.total_mzis
        self.enabled = config.enable_thermal_crosstalk and not config.ideal_mode

        # 1. Base coordinates for M internal MZI heaters
        coords, self.mzi_indices = compute_clements_layout_coordinates(
            n_modes=config.n_modes,
            pitch_x=config.heater_pitch_x,
            pitch_y=config.heater_pitch_y,
            include_diagonal_screen=False
        )
        self.register_buffer("coords", coords)

        # 2. Extended coordinates for M + N heaters (including output diagonal screen)
        coords_full, self.actuator_indices_full = compute_clements_layout_coordinates(
            n_modes=config.n_modes,
            pitch_x=config.heater_pitch_x,
            pitch_y=config.heater_pitch_y,
            include_diagonal_screen=True
        )
        self.register_buffer("coords_full", coords_full)

        # Build coupling matrices
        K = build_thermal_crosstalk_matrix(
            coords=coords,
            thermal_decay_length=config.thermal_decay_length,
            thermal_impedance=config.thermal_impedance
        )
        K_norm = K / max(config.thermal_impedance, 1e-6)
        self.register_buffer("K", K)
        self.register_buffer("K_norm", K_norm)

        K_full = build_thermal_crosstalk_matrix(
            coords=coords_full,
            thermal_decay_length=config.thermal_decay_length,
            thermal_impedance=config.thermal_impedance
        )
        K_norm_full = K_full / max(config.thermal_impedance, 1e-6)
        self.register_buffer("K_full", K_full)
        self.register_buffer("K_norm_full", K_norm_full)

        # Unconstrained pseudoinverse for legacy fast inversion
        self.register_buffer("K_norm_inv", torch.linalg.pinv(K_norm))
        self.register_buffer("K_norm_full_inv", torch.linalg.pinv(K_norm_full))

        # Electro-thermal and packaging constants
        self.P_pi = config.P_pi
        self.max_heater_power = getattr(config, "max_heater_power_watts", 0.050)
        self.tikhonov_lambda = getattr(config, "thermal_tikhonov_lambda", 1e-3)

        if getattr(config, "enable_aging", False) and getattr(config, "operating_hours", 0.0) > 0:
            hours = config.operating_hours
            rate = getattr(config, "heater_aging_rate", 0.005)
            aging_factor = 1.0 + rate * math.log10(1.0 + hours)
            self.R0 = config.heater_resistance * aging_factor
        else:
            self.R0 = config.heater_resistance

        self.alpha_tcr = config.tcr_coefficient
        self.R_pack = config.package_thermal_resistance
        self.tau = config.thermal_time_constant
        self.enable_tcr = config.enable_tcr and not config.ideal_mode

    def compute_joule_power(self, theta_command: torch.Tensor) -> torch.Tensor:
        """
        Calculates electrical Joule heating power P (Watts) from commanded phase,
        accounting for voltage drive and Temperature Coefficient of Resistance (TCR).
        """
        P_ideal = (torch.clamp(theta_command, min=0.0) / math.pi) * self.P_pi

        if not self.enable_tcr:
            return P_ideal

        delta_T_pi = 40.0
        delta_T_local = (torch.clamp(theta_command, min=0.0) / math.pi) * delta_T_pi
        R_T = self.R0 * (1.0 + self.alpha_tcr * delta_T_local)
        V_sq = P_ideal * self.R0
        P_realized = V_sq / torch.clamp(R_T, min=1.0)
        return P_realized

    def compute_package_heating(self, P: torch.Tensor) -> torch.Tensor:
        """Calculates global die temperature rise from total chip power dissipation."""
        if not self.enabled or self.config.package_thermal_resistance <= 0:
            return torch.zeros_like(P[..., :1])

        P_total = torch.sum(P, dim=-1, keepdim=True)
        delta_T_die = self.R_pack * P_total
        return 0.05 * delta_T_die

    def forward(
        self,
        theta: torch.Tensor,
        state_T: Optional[torch.Tensor] = None,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        """
        Applies non-local thermal crosstalk and TCR electro-thermal feedback.
        Supports both M internal MZI phases or M + N (MZI + diagonal) actuator phases.
        """
        if not self.enabled:
            return theta

        is_full = (theta.shape[-1] == self.total_mzis + self.n_modes)
        K_norm = (self.K_norm_full if is_full else self.K_norm).to(device=theta.device, dtype=theta.dtype)

        if self.config.thermal_mode == "transient" and dt is not None:
            decay = math.exp(-dt / max(self.tau, 1e-9))
            theta_target_bleed = torch.matmul(theta, K_norm.T)
            if state_T is None:
                state_T = torch.zeros_like(theta)
            return state_T * decay + theta_target_bleed * (1.0 - decay)

        if self.enable_tcr:
            P = self.compute_joule_power(theta)
            theta_eff = (P / self.P_pi) * math.pi
        else:
            theta_eff = theta

        theta_actual = torch.matmul(theta_eff, K_norm.T)

        if self.config.package_thermal_resistance > 0:
            P_tot = (torch.clamp(theta, min=0.0) / math.pi) * self.P_pi
            global_drift = self.compute_package_heating(P_tot)
            theta_actual = theta_actual + global_drift

        return theta_actual

    def predistort(
        self,
        theta_target: torch.Tensor,
        method: str = "linear",
        max_power_watts: Optional[float] = None,
        regularization_lambda: Optional[float] = None,
        max_iter: int = 50,
        tol: float = 1e-5
    ) -> torch.Tensor:
        """
        Inverts thermal crosstalk to compute predistorted actuator drive settings.

        Args:
            theta_target: Target phase tensor of shape (..., K).
            method: 'linear' for theoretical unconstrained pseudoinverse (machine precision),
                    'bnnls' for bounded non-negative regularized physical drive (0 <= P <= P_max).
            max_power_watts: Maximum electrical power per actuator (for BNNLS).
            regularization_lambda: Tikhonov regularization weight (for BNNLS).
            max_iter: Maximum iterations (for BNNLS).
            tol: Convergence tolerance (for BNNLS).

        Returns:
            theta_drive: Predistorted drive phase tensor of shape (..., K).
        """
        if not self.enabled:
            return theta_target

        if method == "bnnls":
            return self.predistort_bnnls(
                theta_target=theta_target,
                max_power_watts=max_power_watts,
                regularization_lambda=regularization_lambda,
                max_iter=max_iter,
                tol=tol
            )

        # Theoretical Linear Unconstrained Inversion via K_norm^-1
        device = theta_target.device
        dtype = theta_target.dtype
        is_full = (theta_target.shape[-1] == self.total_mzis + self.n_modes)
        K_inv = (self.K_norm_full_inv if is_full else self.K_norm_inv).to(device=device, dtype=dtype)

        # 1. Remove global substrate heating offset if active
        if self.config.package_thermal_resistance > 0 and self.enabled:
            P_tot = (torch.clamp(theta_target, min=0.0) / math.pi) * self.P_pi
            global_drift = self.compute_package_heating(P_tot)
            theta_sub = torch.clamp(theta_target - global_drift, min=0.0)
        else:
            theta_sub = theta_target

        # 2. Linear spatial inversion via K_norm^-1
        theta_eff = torch.matmul(theta_sub, K_inv.T)

        # 3. Inverse TCR mapping
        if self.enable_tcr and self.enabled:
            P_target = (torch.clamp(theta_eff, min=0.0) / math.pi) * self.P_pi
            delta_T = self.config.thermal_impedance * P_target
            P_command = P_target * (1.0 + self.alpha_tcr * delta_T)
            theta_drive = (P_command / self.P_pi) * math.pi
        else:
            theta_drive = theta_eff

        return theta_drive

    def predistort_bnnls(
        self,
        theta_target: torch.Tensor,
        max_power_watts: Optional[float] = None,
        regularization_lambda: Optional[float] = None,
        max_iter: int = 50,
        tol: float = 1e-5
    ) -> torch.Tensor:
        """
        Inverts thermal crosstalk using Bounded Non-Negative Least Squares (BNNLS) with FISTA.
        Guarantees:
            0 <= theta_drive <= theta_max
        where theta_max = (P_max / P_pi) * pi.
        """
        if not self.enabled:
            return theta_target

        device = theta_target.device
        dtype = theta_target.dtype
        is_full = (theta_target.shape[-1] == self.total_mzis + self.n_modes)
        K_mat = (self.K_norm_full if is_full else self.K_norm).to(device=device, dtype=dtype)

        P_max = max_power_watts if max_power_watts is not None else self.max_heater_power
        theta_max = (P_max / max(self.P_pi, 1e-6)) * math.pi
        lam = regularization_lambda if regularization_lambda is not None else self.tikhonov_lambda

        # 1. Remove package substrate heating baseline
        if self.config.package_thermal_resistance > 0:
            P_tot = (torch.clamp(theta_target, min=0.0) / math.pi) * self.P_pi
            global_drift = self.compute_package_heating(P_tot)
            theta_sub = torch.clamp(theta_target - global_drift, min=0.0)
        else:
            theta_sub = theta_target

        # 2. Spectral radius calculation for Lipschitz step size
        with torch.no_grad():
            K_dim = K_mat.shape[-1]
            v = torch.ones(K_dim, 1, device=device, dtype=dtype) / math.sqrt(K_dim)
            for _ in range(8):
                v = torch.matmul(K_mat.T, torch.matmul(K_mat, v))
                norm_v = torch.norm(v)
                if norm_v > 0:
                    v = v / norm_v
            L_grad = torch.norm(torch.matmul(K_mat.T, torch.matmul(K_mat, v))).item() + lam
            step_size = 1.0 / max(L_grad, 1e-6)

        # 3. FISTA acceleration loop
        x = torch.clamp(theta_sub.clone(), 0.0, theta_max)
        y = x.clone()
        t = 1.0

        for _ in range(max_iter):
            Ky = torch.matmul(y, K_mat.T)
            grad = torch.matmul(Ky - theta_sub, K_mat) + lam * y
            x_next = torch.clamp(y - step_size * grad, 0.0, theta_max)

            rel_change = torch.norm(x_next - x) / (torch.norm(x) + 1e-12)
            if rel_change.item() < tol:
                x = x_next
                break

            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            y = x_next + ((t - 1.0) / t_next) * (x_next - x)
            x = x_next
            t = t_next

        # 4. Inverse TCR adjustment if enabled
        if self.enable_tcr:
            P_target = (x / math.pi) * self.P_pi
            delta_T = self.config.thermal_impedance * P_target
            P_command = P_target * (1.0 + self.alpha_tcr * delta_T)
            theta_drive = torch.clamp((P_command / self.P_pi) * math.pi, 0.0, theta_max)
        else:
            theta_drive = x

        return theta_drive

    def get_crosstalk_ratio(self, mzi_idx1: int, mzi_idx2: int) -> float:
        """Returns the mutual thermal coupling ratio between two MZIs."""
        return float(self.K_norm[mzi_idx1, mzi_idx2].item())
