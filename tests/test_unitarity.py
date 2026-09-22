"""
Mathematical Unitarity Test Suite.
Validates that in ideal mode (lossless, zero noise), the PhotonicMeshDigitalTwin
produces a mathematically perfect unitary matrix (T^dagger @ T = I) within machine precision
across multiple mode sizes N = 4, 8, 16, 32.
"""

import sys
import os
import math
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PhotonicConfig
from src.models.digital_twin import PhotonicMeshDigitalTwin
from src.utils.decomposition import generate_random_unitary


def test_unitarity_identity_phases():
    """Verify that zero phases produce an exact unitary matrix."""
    for n_modes in [4, 8, 16, 32]:
        cfg = PhotonicConfig(n_modes=n_modes, ideal_mode=True)
        twin = PhotonicMeshDigitalTwin(cfg)

        theta = torch.zeros(twin.total_mzis, device=cfg.device)
        phi = torch.zeros(twin.total_mzis, device=cfg.device)

        T = twin.compute_transfer_matrix(theta, phi)

        # Compute T^dagger @ T
        T_dag_T = torch.matmul(T.conj().T, T)
        eye = torch.eye(n_modes, device=cfg.device, dtype=cfg.complex_dtype)

        frobenius_err = torch.norm(T_dag_T - eye).item()
        print(f"[N={n_modes:02d}] Zero-Phase Frobenius Error ||T^H T - I||_F: {frobenius_err:.3e}")
        assert frobenius_err < 1e-5, f"Unitarity violated for N={n_modes}: error = {frobenius_err}"


def test_unitarity_random_phases():
    """Verify that arbitrary random continuous phase configurations are strictly unitary."""
    for n_modes in [4, 8, 16, 32]:
        cfg = PhotonicConfig(n_modes=n_modes, ideal_mode=True)
        twin = PhotonicMeshDigitalTwin(cfg)

        batch_size = 8
        torch.manual_seed(42 + n_modes)
        theta = torch.rand(batch_size, twin.total_mzis, device=cfg.device) * math.pi
        phi = torch.rand(batch_size, twin.total_mzis, device=cfg.device) * (2.0 * math.pi)

        T = twin.compute_transfer_matrix(theta, phi)  # (batch, N, N)

        # Batch conjugate transpose
        T_dag = T.conj().transpose(-2, -1)
        T_dag_T = torch.matmul(T_dag, T)
        eye = torch.eye(n_modes, device=cfg.device, dtype=cfg.complex_dtype).unsqueeze(0)

        max_err = torch.max(torch.norm(T_dag_T - eye, dim=(-2, -1))).item()
        print(f"[N={n_modes:02d}] Random Phase (Batch={batch_size}) Max Error: {max_err:.3e}")
        assert max_err < 1e-5, f"Unitarity violated under random phases for N={n_modes}: max_err = {max_err}"


def test_power_conservation_field_propagation():
    """Verify that field vector propagation preserves total optical energy sum(|E_out|^2) == sum(|E_in|^2)."""
    for n_modes in [4, 8, 16, 32]:
        cfg = PhotonicConfig(n_modes=n_modes, ideal_mode=True)
        twin = PhotonicMeshDigitalTwin(cfg)

        batch_size = 16
        torch.manual_seed(123 + n_modes)
        E_in = (torch.randn(batch_size, n_modes, device=cfg.device) + 
                1.0j * torch.randn(batch_size, n_modes, device=cfg.device)).to(cfg.complex_dtype)

        theta = torch.rand(batch_size, twin.total_mzis, device=cfg.device) * math.pi
        phi = torch.rand(batch_size, twin.total_mzis, device=cfg.device) * (2.0 * math.pi)

        E_out = twin.propagate_field(E_in, theta, phi)

        P_in = torch.sum(torch.real(E_in * torch.conj(E_in)), dim=-1)
        P_out = torch.sum(torch.real(E_out * torch.conj(E_out)), dim=-1)

        rel_diff = torch.max(torch.abs(P_out - P_in) / torch.clamp(P_in, min=1e-12)).item()
        print(f"[N={n_modes:02d}] Energy Conservation Relative Error |P_out - P_in| / P_in: {rel_diff:.3e}")
        assert rel_diff < 5e-5, f"Energy not conserved for N={n_modes}: rel_diff = {rel_diff}"


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING UNITARITY & MATHEMATICAL CONSERVATION TESTS")
    print("=" * 70)
    test_unitarity_identity_phases()
    test_unitarity_random_phases()
    test_power_conservation_field_propagation()
    print("=" * 70)
    print("[ALL UNITARITY TESTS PASSED]")
    print("=" * 70)
