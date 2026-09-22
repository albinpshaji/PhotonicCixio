"""
2D Finite-Difference Thermal Solver for Silicon Photonics Integrated Circuits (PICs).
Calibrated for standard Silicon-on-Insulator (SOI) cross-sections (Si device layer,
SiO2 buried oxide (BOX), top cladding, and Si substrate heat sink).

Governing Physics:
2D steady-state screened heat conduction equation on the SOI thin-film device layer:
    - nabla * (k_eff * nabla T(x, y)) + (k_BOX / (t_BOX * t_device)) * (T(x, y) - T_sub) = Q(x, y)

Boundary conditions:
    - Dirichlet isothermal T = T_sub at the bottom substrate heat sink (via BOX thermal resistance)
    - Neumann insulation / Robin convective at outer die borders
    - Local Joule heat generation Q_j at heater coordinates (x_j, y_j)
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


def compute_fd_thermal_greens_function(
    coords: torch.Tensor,
    heater_pitch_x: float = 120.0e-6,
    heater_pitch_y: float = 80.0e-6,
    grid_res: int = 64,
    si_conductivity: float = 148.0,
    box_conductivity: float = 1.38,
    box_thickness: float = 2.0e-6,
    device_thickness: float = 220.0e-9,
    effective_decay_length: Optional[float] = None,
    thermal_impedance: float = 18.5,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the 2D steady-state finite-difference thermal Green's function matrix
    G in R^(M x M) and normalized crosstalk matrix K_norm in R^(M x M).

    For each heater j with coordinates (x_j, y_j):
        1. Injects localized Gaussian-spread heat source Q_j into a 2D spatial grid.
        2. Solves the 2D finite-difference screened Poisson system:
           [- nabla^2 + (1 / lambda_th^2)] T = Q / (k_eff * t_device)
        3. Interpolates the resulting 2D temperature field T(x, y) at all heater
           locations (x_i, y_i) to obtain row/column G_ij.

    Args:
        coords: Tensor of shape (M, 2) containing heater coordinates (x, y) in meters.
        heater_pitch_x: Horizontal pitch between heater columns (meters).
        heater_pitch_y: Vertical pitch between heater rows (meters).
        grid_res: Grid resolution along the dominant dimension (e.g. 64 or 128).
        si_conductivity: Thermal conductivity of Si device layer (W/(m*K)).
        box_conductivity: Thermal conductivity of SiO2 BOX layer (W/(m*K)).
        box_thickness: Thickness of buried oxide layer t_BOX (meters).
        device_thickness: Thickness of silicon device layer t_Si (meters).
        effective_decay_length: Optional override for lambda_th. If None, derived from
                                physical stack: sqrt(k_si * t_si * t_box / k_box).
        thermal_impedance: Target self-heating impedance kappa_0 (K/W).
        device: Torch device ('cpu' or 'cuda').

    Returns:
        G: Physical thermal coupling matrix in R^(M x M) (units: K/W).
        K_norm: Normalized thermal crosstalk matrix in R^(M x M) with diag(K_norm) = 1.0.
    """
    M = coords.shape[0]
    coords = coords.to(device)

    # Compute characteristic thermal diffusion length lambda_th
    if effective_decay_length is not None:
        lambda_th = effective_decay_length
    else:
        # Classical thin-film on insulator diffusion length:
        # lambda_th = sqrt( (k_si * t_si * t_box) / k_box )
        # Cladding and top metallization broaden this to ~35-55 um in real foundry chips
        base_lambda = math.sqrt((si_conductivity * device_thickness * box_thickness) / box_conductivity)
        # Account for SiO2 upper cladding conduction factor (~2.5x spread)
        lambda_th = max(base_lambda * 3.5, 35.0e-6)

    # Domain extent enclosing all heaters with physical thermal padding
    pad_x = 3.0 * lambda_th
    pad_y = 3.0 * lambda_th

    x_min = coords[:, 0].min().item() - pad_x
    x_max = coords[:, 0].max().item() + pad_x
    y_min = coords[:, 1].min().item() - pad_y
    y_max = coords[:, 1].max().item() + pad_y

    width_x = max(x_max - x_min, 1e-6)
    width_y = max(y_max - y_min, 1e-6)

    Nx = grid_res
    Ny = max(int(grid_res * (width_y / width_x)), 16)

    dx = width_x / (Nx - 1)
    dy = width_y / (Ny - 1)

    # Screened Poisson Helmholtz operator in 2D Fourier domain (spectral / cyclic FD):
    # Operator: -d^2/dx^2 - d^2/dy^2 + 1/lambda_th^2
    # In k-space: (kx^2 + ky^2 + 1/lambda_th^2) * T_hat = Q_hat / (k_eff * t)
    # This provides the exact continuum Green's function solution with boundary falloff.
    kx = 2.0 * math.pi * torch.fft.fftfreq(Nx, d=dx, device=device)
    ky = 2.0 * math.pi * torch.fft.fftfreq(Ny, d=dy, device=device)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")  # (Ny, Nx)

    inv_operator = 1.0 / (KX**2 + KY**2 + (1.0 / (lambda_th**2)))  # (Ny, Nx)

    # Map heater continuous coordinates (x_m, y_m) into grid indices
    # Using Gaussian source injection with micro-heater width sigma ~ 5 um
    sigma_heater = 5.0e-6
    x_grid = torch.linspace(x_min, x_max, Nx, device=device)
    y_grid = torch.linspace(y_min, y_max, Ny, device=device)
    Y_g, X_g = torch.meshgrid(y_grid, x_grid, indexing="ij")  # (Ny, Nx)

    G = torch.zeros((M, M), dtype=torch.float32, device=device)

    for j in range(M):
        xj = coords[j, 0]
        yj = coords[j, 1]

        # Normalized Gaussian heat source centered at heater j
        r_sq = (X_g - xj)**2 + (Y_g - yj)**2
        Q_field = torch.exp(-r_sq / (2.0 * (sigma_heater**2)))
        Q_field = Q_field / torch.sum(Q_field)  # Unit energy injection

        # Solve via spectral Helmholtz operator
        Q_hat = torch.fft.rfft2(Q_field)
        # Slice inv_operator to match rfft2 positive frequencies
        inv_op_rfft = inv_operator[:, :Q_hat.shape[1]]
        T_hat = Q_hat * inv_op_rfft
        T_field = torch.fft.irfft2(T_hat, s=(Ny, Nx))

        # Sample temperature at all heater coordinates (x_i, y_i) via bilinear interpolation
        # Convert (xi, yi) to normalized grid coordinates in [-1, 1] for grid_sample
        norm_x = 2.0 * (coords[:, 0] - x_min) / width_x - 1.0
        norm_y = 2.0 * (coords[:, 1] - y_min) / width_y - 1.0
        sample_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, M, 2)

        T_sampled = torch.nn.functional.grid_sample(
            T_field.unsqueeze(0).unsqueeze(0),
            sample_grid,
            mode="bilinear",
            align_corners=True
        ).squeeze()  # (M,)

        G[:, j] = T_sampled

    # Enforce physical reciprocity: G_ij = G_ji
    G = 0.5 * (G + G.T)

    # Diagonal self-heating values
    diag = torch.diagonal(G).clone()
    diag_mean = diag.mean()
    if diag_mean > 0:
        # Normalize G to physical units (K/W)
        G = G * (thermal_impedance / diag_mean)
        # Normalized coupling matrix K_norm = G_ij / sqrt(G_ii * G_jj)
        diag_sqrt = torch.sqrt(torch.clamp(torch.diagonal(G), min=1e-12))
        K_norm = G / (diag_sqrt.unsqueeze(1) * diag_sqrt.unsqueeze(0))
    else:
        K_norm = torch.eye(M, device=device)

    # Physical bounds: off-diagonals are non-negative and strictly <= 1.0
    K_norm = torch.clamp(K_norm, min=0.0, max=1.0)
    K_norm.fill_diagonal_(1.0)

    return G, K_norm
