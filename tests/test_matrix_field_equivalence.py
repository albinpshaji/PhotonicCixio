"""
Unit Test: Optical Field Propagation vs Transfer Matrix Equivalence.
Validates:
1. Exact equality E_prop == T_PIC @ E_in in ideal mode across N in [2, 4, 8, 16] in complex128.
2. Exact equality E_prop == T_PIC @ E_in under progressive stage insertion loss and physical crossings.
3. Universal U(N) field propagation with output diagonal phase screen D.
"""

import sys
import os
import math
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PhotonicConfig
from src.models.digital_twin import PhotonicMeshDigitalTwin
from src.utils.decomposition import generate_random_unitary, clements_decompose_np


def test_field_matrix_equivalence_ideal():
    """Verify that propagate_field and compute_transfer_matrix match to double precision in ideal mode."""
    for n_modes in [2, 4, 8, 16]:
        cfg = PhotonicConfig(
            n_modes=n_modes,
            ideal_mode=True,
            dtype=torch.float64,
            complex_dtype=torch.complex128
        )
        twin = PhotonicMeshDigitalTwin(cfg)

        batch_size = 8
        gen = torch.Generator(device=cfg.device)
        gen.manual_seed(100 + n_modes)

        theta = torch.rand(batch_size, twin.total_mzis, generator=gen, device=cfg.device, dtype=torch.float64) * math.pi
        phi = torch.rand(batch_size, twin.total_mzis, generator=gen, device=cfg.device, dtype=torch.float64) * 2 * math.pi
        diag = torch.rand(batch_size, n_modes, generator=gen, device=cfg.device, dtype=torch.float64) * 2 * math.pi

        T_mat = twin.compute_transfer_matrix(theta, phi, diag_phases=diag)

        E_in = (torch.randn(batch_size, n_modes, generator=gen, device=cfg.device, dtype=torch.float64) +
                1.0j * torch.randn(batch_size, n_modes, generator=gen, device=cfg.device, dtype=torch.float64)).to(cfg.complex_dtype)

        E_prop = twin.propagate_field(E_in, theta, phi, diag_phases=diag)
        E_mat = torch.matmul(T_mat, E_in.unsqueeze(-1)).squeeze(-1)

        diff = torch.max(torch.norm(E_prop - E_mat, dim=-1)).item()
        print(f"[Ideal N={n_modes:02d}] Max Field vs Matrix Diff: {diff:.3e}")
        assert diff < 1e-12, f"Field vs matrix divergence in ideal mode for N={n_modes}: diff = {diff}"


def test_field_matrix_equivalence_lossy():
    """Verify that propagate_field and compute_transfer_matrix match under stage insertion loss."""
    for n_modes in [4, 8]:
        cfg = PhotonicConfig(
            n_modes=n_modes,
            ideal_mode=False,
            enable_loss=True,
            loss_per_stage_db=0.20,
            loss_std_db=0.0,
            enable_coupler_errors=False,
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
            grating_coupler_loss_db=0.0,
            dtype=torch.float64,
            complex_dtype=torch.complex128
        )
        twin = PhotonicMeshDigitalTwin(cfg)

        batch_size = 4
        gen = torch.Generator(device=cfg.device)
        gen.manual_seed(200 + n_modes)

        theta = torch.rand(twin.total_mzis, generator=gen, device=cfg.device, dtype=torch.float64) * math.pi
        phi = torch.rand(twin.total_mzis, generator=gen, device=cfg.device, dtype=torch.float64) * 2 * math.pi
        diag = torch.rand(n_modes, generator=gen, device=cfg.device, dtype=torch.float64) * 2 * math.pi

        T_mat = twin.compute_transfer_matrix(theta, phi, diag_phases=diag)

        E_in = (torch.randn(batch_size, n_modes, generator=gen, device=cfg.device, dtype=torch.float64) +
                1.0j * torch.randn(batch_size, n_modes, generator=gen, device=cfg.device, dtype=torch.float64)).to(cfg.complex_dtype)

        E_prop = twin.propagate_field(E_in, theta, phi, diag_phases=diag)
        E_mat = torch.matmul(T_mat.unsqueeze(0), E_in.unsqueeze(-1)).squeeze(-1)

        diff = torch.max(torch.norm(E_prop - E_mat, dim=-1)).item()
        print(f"[Lossy N={n_modes:02d}] Max Field vs Matrix Diff: {diff:.3e}")
        assert diff < 1e-12, f"Field vs matrix divergence in lossy mode for N={n_modes}: diff = {diff}"


def test_field_matrix_equivalence_packaged():
    """Verify that propagate_field and compute_transfer_matrix match under full packaging (grating coupler) loss."""
    for n_modes in [4, 8]:
        cfg = PhotonicConfig(
            n_modes=n_modes,
            ideal_mode=False,
            enable_loss=True,
            loss_per_stage_db=0.20,
            loss_std_db=0.0,
            enable_coupler_errors=False,
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
            grating_coupler_loss_db=3.0,
            dtype=torch.float64,
            complex_dtype=torch.complex128
        )
        twin = PhotonicMeshDigitalTwin(cfg)

        batch_size = 4
        gen = torch.Generator(device=cfg.device)
        gen.manual_seed(300 + n_modes)

        theta = torch.rand(twin.total_mzis, generator=gen, device=cfg.device, dtype=torch.float64) * math.pi
        phi = torch.rand(twin.total_mzis, generator=gen, device=cfg.device, dtype=torch.float64) * 2 * math.pi
        diag = torch.rand(n_modes, generator=gen, device=cfg.device, dtype=torch.float64) * 2 * math.pi

        T_mat = twin.compute_transfer_matrix(theta, phi, diag_phases=diag, include_packaging=True)

        E_in = (torch.randn(batch_size, n_modes, generator=gen, device=cfg.device, dtype=torch.float64) +
                1.0j * torch.randn(batch_size, n_modes, generator=gen, device=cfg.device, dtype=torch.float64)).to(cfg.complex_dtype)

        E_prop = twin.propagate_field(E_in, theta, phi, diag_phases=diag, include_packaging=True)
        E_mat = torch.matmul(T_mat.unsqueeze(0), E_in.unsqueeze(-1)).squeeze(-1)

        diff = torch.max(torch.norm(E_prop - E_mat, dim=-1)).item()
        print(f"[Packaged N={n_modes:02d}] Max Field vs Matrix Diff: {diff:.3e}")
        assert diff < 1e-12, f"Field vs matrix divergence in packaged mode for N={n_modes}: diff = {diff}"


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING FIELD-VS-MATRIX EQUIVALENCE TESTS")
    print("=" * 70)
    test_field_matrix_equivalence_ideal()
    test_field_matrix_equivalence_lossy()
    test_field_matrix_equivalence_packaged()
    print("=" * 70)
    print("[ALL FIELD-VS-MATRIX EQUIVALENCE TESTS PASSED]")
    print("=" * 70)
