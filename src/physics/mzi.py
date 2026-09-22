"""
Mach-Zehnder Interferometer (MZI) 2x2 Transfer Operator with Directional Coupler Imprecisions.
Mathematical references:
- Clements et al. (2016), Optica 3(12), 1460-1466
- Fang et al. (2019), Optics Express 27(10), 14009-14029
- Pai et al. (2019), IEEE JSTQE 26(2), 1-19
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


def directional_coupler_matrix(
    epsilon: torch.Tensor,
    wavelength: Optional[Union[float, torch.Tensor]] = None,
    lambda_0: float = 1550e-9,
    excess_loss_db: float = 0.0,
    complex_dtype: torch.dtype = torch.complex64
) -> torch.Tensor:
    """
    Constructs the 2x2 field transfer matrix of a non-ideal directional coupler
    with split-ratio deviation epsilon and optional wavelength dispersion.

    C(epsilon, lambda) = a_coupler * [ [ sqrt(0.5 - eps_eff),     i * sqrt(0.5 + eps_eff) ],
                                       [ i * sqrt(0.5 + eps_eff), sqrt(0.5 - eps_eff)     ] ]

    Args:
        epsilon: Tensor of shape (...) containing coupling ratio deviations.
        wavelength: Operating optical wavelength in meters (scalar or tensor).
        lambda_0: Nominal center wavelength (1550 nm).
        excess_loss_db: Directional coupler excess insertion loss (dB).
        complex_dtype: Complex dtype (torch.complex64 or torch.complex128).

    Returns:
        Tensor of shape (..., 2, 2) representing coupler transfer matrices.
    """
    eps_eff = epsilon

    # Model wavelength dispersion if wavelength deviates from lambda_0
    if wavelength is not None:
        if isinstance(wavelength, (int, float)):
            delta_lambda = wavelength - lambda_0
        else:
            delta_lambda = wavelength - lambda_0
        # Coupled-mode dispersion slope: d(kappa)/d(lambda) approx 8.5e4 m^-1 (0.085 nm^-1)
        dispersion_shift = 8.5e4 * delta_lambda
        eps_eff = eps_eff + dispersion_shift

    # Clamp effective epsilon to ensure physical bounds: kappa in [0.001, 0.999]
    eps_clamped = torch.clamp(eps_eff, -0.499, 0.499)
    bar_amp = torch.sqrt(0.5 - eps_clamped).to(complex_dtype)
    cross_amp = (1.0j * torch.sqrt(0.5 + eps_clamped)).to(complex_dtype)

    # Coupler excess insertion loss
    if excess_loss_db > 0:
        a_dc = 10.0 ** (-excess_loss_db / 20.0)
        bar_amp = bar_amp * a_dc
        cross_amp = cross_amp * a_dc

    # Stack into 2x2 matrix: [[bar, cross], [cross, bar]]
    row1 = torch.stack([bar_amp, cross_amp], dim=-1)
    row2 = torch.stack([cross_amp, bar_amp], dim=-1)
    return torch.stack([row1, row2], dim=-2)


def mzi_transfer_matrix(
    theta: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
    epsilon1: Optional[torch.Tensor] = None,
    epsilon2: Optional[torch.Tensor] = None,
    phi_intrinsic: Optional[torch.Tensor] = None,
    wavelength: Optional[Union[float, torch.Tensor]] = None,
    excess_loss_db: float = 0.0,
    ideal_mode: bool = False,
    complex_dtype: torch.dtype = torch.complex64
) -> torch.Tensor:
    """
    Computes the 2x2 MZI transfer matrix for arbitrary batch shapes (...).

    In ideal mode (epsilon1 = epsilon2 = 0, phi_intrinsic = 0):
        T(theta, phi) = i * exp(i*theta/2) * [ [ exp(i*phi) * sin(theta/2),  cos(theta/2)  ],
                                                [ exp(i*phi) * cos(theta/2), -sin(theta/2) ] ]

    In realistic physical mode:
        T(theta, phi; eps1, eps2) = C(eps2) @ diag(exp(i*(theta + phi_intrinsic)), 1) @ C(eps1) @ diag(exp(i*phi), 1)

    Args:
        theta: Internal phase shift tensor of shape (...) in radians.
        phi: External phase shift tensor of shape (...) in radians. Defaults to zero if None.
        epsilon1: Input coupler split error of shape (...).
        epsilon2: Output coupler split error of shape (...).
        phi_intrinsic: Static intrinsic lithographic phase bias of shape (...).
        wavelength: Operating optical wavelength in meters.
        excess_loss_db: Coupler excess insertion loss in dB.
        ideal_mode: If True, uses the exact analytic ideal formula.
        complex_dtype: Complex tensor dtype.

    Returns:
        Tensor of shape (..., 2, 2) in complex_dtype.
    """
    if phi is None:
        phi = torch.zeros_like(theta)

    device = theta.device
    one = torch.ones_like(theta, dtype=complex_dtype)

    if ideal_mode or (epsilon1 is None and epsilon2 is None and phi_intrinsic is None and wavelength is None):
        # Fast, exact closed-form ideal Clements MZI transfer operator
        half_theta = 0.5 * theta
        sin_half = torch.sin(half_theta).to(complex_dtype)
        cos_half = torch.cos(half_theta).to(complex_dtype)
        
        # exp(i * phi) and global phase factor i * exp(i * theta / 2)
        exp_i_phi = torch.exp(1.0j * phi.to(complex_dtype))
        global_phase = 1.0j * torch.exp(1.0j * half_theta.to(complex_dtype))

        t11 = global_phase * exp_i_phi * sin_half
        t12 = global_phase * cos_half
        t21 = global_phase * exp_i_phi * cos_half
        t22 = -global_phase * sin_half

        row1 = torch.stack([t11, t12], dim=-1)
        row2 = torch.stack([t21, t22], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    # Physical mode with directional coupler non-idealities
    if epsilon1 is None:
        epsilon1 = torch.zeros_like(theta)
    if epsilon2 is None:
        epsilon2 = torch.zeros_like(theta)

    C1 = directional_coupler_matrix(
        epsilon1,
        wavelength=wavelength,
        excess_loss_db=excess_loss_db,
        complex_dtype=complex_dtype
    )
    C2 = directional_coupler_matrix(
        epsilon2,
        wavelength=wavelength,
        excess_loss_db=excess_loss_db,
        complex_dtype=complex_dtype
    )

    # Internal phase with intrinsic phase error
    theta_total = theta
    if phi_intrinsic is not None:
        theta_total = theta_total + phi_intrinsic

    exp_i_theta = torch.exp(1.0j * theta_total.to(complex_dtype))
    zeros_c = torch.zeros_like(theta, dtype=complex_dtype)
    theta_mat = torch.stack([
        torch.stack([exp_i_theta, zeros_c], dim=-1),
        torch.stack([zeros_c, one], dim=-1)
    ], dim=-2)

    # External phase matrix: diag(exp(i*phi), 1)
    exp_i_phi = torch.exp(1.0j * phi.to(complex_dtype))
    phi_mat = torch.stack([
        torch.stack([exp_i_phi, zeros_c], dim=-1),
        torch.stack([zeros_c, one], dim=-1)
    ], dim=-2)

    # T = C2 @ Theta @ C1 @ Phi
    return C2 @ theta_mat @ C1 @ phi_mat


class MZIOperator(nn.Module):
    """
    Differentiable PyTorch module representing a physical MZI.
    Encapsulates internal/external phase parameters and fabrication errors.
    """
    def __init__(
        self,
        ideal_mode: bool = False,
        coupler_error_std: float = 0.04,
        complex_dtype: torch.dtype = torch.complex64
    ):
        super().__init__()
        self.ideal_mode = ideal_mode
        self.coupler_error_std = coupler_error_std
        self.complex_dtype = complex_dtype

        if not ideal_mode and coupler_error_std > 0:
            # Register fixed fabrication offsets (persistent state)
            self.register_buffer("epsilon1", torch.randn(1) * coupler_error_std)
            self.register_buffer("epsilon2", torch.randn(1) * coupler_error_std)
        else:
            self.register_buffer("epsilon1", torch.zeros(1))
            self.register_buffer("epsilon2", torch.zeros(1))

    def forward(
        self,
        theta: torch.Tensor,
        phi: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes 2x2 MZI transfer matrix for inputs theta and phi.
        """
        return mzi_transfer_matrix(
            theta=theta,
            phi=phi,
            epsilon1=None if self.ideal_mode else self.epsilon1,
            epsilon2=None if self.ideal_mode else self.epsilon2,
            ideal_mode=self.ideal_mode,
            complex_dtype=self.complex_dtype
        )
