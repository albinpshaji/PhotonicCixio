"""
Autograd and Gradient Flow Test Suite.
Validates that reverse-mode automatic differentiation propagates cleanly through
all physical layers and the Straight-Through Estimator (STE) DAC quantizer
without NaNs, Infs, or gradient vanishing.
"""

import sys
import os
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PhotonicConfig
from src.models.digital_twin import PhotonicMeshDigitalTwin
from src.nn.quantizer import DACQuantizer


def test_ste_quantizer_gradients():
    """Verify that STE passes unit gradients within bounds and zero outside."""
    quantizer = DACQuantizer(bits=4, val_min=0.0, val_max=math.pi, enabled=True)

    # In-bounds input
    x_in = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
    y_in = quantizer(x_in)
    loss_in = torch.sum(y_in * torch.tensor([1.0, 2.0, 3.0]))
    loss_in.backward()

    assert x_in.grad is not None, "STE gradient is None!"
    assert torch.allclose(x_in.grad, torch.tensor([1.0, 2.0, 3.0])), (
        f"Expected unit STE gradient, got {x_in.grad}"
    )
    print("[STE Quantizer] In-bounds gradient matches exactly: [1.0, 2.0, 3.0]")

    # Out-of-bounds input (below 0 and above pi)
    x_out = torch.tensor([-0.5, math.pi + 0.5], requires_grad=True)
    y_out = quantizer(x_out)
    loss_out = torch.sum(y_out)
    loss_out.backward()

    assert torch.allclose(x_out.grad, torch.tensor([0.0, 0.0])), (
        f"Expected zero gradient for out-of-bounds inputs, got {x_out.grad}"
    )
    print("[STE Quantizer] Out-of-bounds gradient is strictly zeroed.")


def test_full_pipeline_gradient_flow():
    """Verify end-to-end gradient flow through the complete physical simulation pipeline."""
    for n_modes in [4, 8, 16]:
        # Initialize realistic physical mode (all non-idealities active)
        cfg = PhotonicConfig(
            n_modes=n_modes,
            ideal_mode=False,
            enable_coupler_errors=True,
            enable_thermal_crosstalk=True,
            enable_loss=True,
            enable_quantization=True,
            enable_noise=False  # Deterministic for gradient check
        )
        twin = PhotonicMeshDigitalTwin(cfg)

        batch_size = 4
        theta = torch.nn.Parameter(torch.rand(batch_size, twin.total_mzis, device=cfg.device) * math.pi)
        phi = torch.nn.Parameter(torch.rand(batch_size, twin.total_mzis, device=cfg.device) * (2.0 * math.pi))

        # 1. Transfer matrix loss test
        T = twin.compute_transfer_matrix(theta, phi)
        target_T = torch.eye(n_modes, device=cfg.device, dtype=cfg.complex_dtype).unsqueeze(0).repeat(batch_size, 1, 1)

        loss_mat = torch.sum(torch.abs(T - target_T) ** 2)
        loss_mat.backward()

        assert theta.grad is not None, f"[N={n_modes}] theta.grad is None!"
        assert phi.grad is not None, f"[N={n_modes}] phi.grad is None!"
        assert not torch.isnan(theta.grad).any(), f"[N={n_modes}] NaNs in theta.grad!"
        assert not torch.isnan(phi.grad).any(), f"[N={n_modes}] NaNs in phi.grad!"
        assert not torch.isinf(theta.grad).any(), f"[N={n_modes}] Infs in theta.grad!"
        assert torch.norm(theta.grad) > 0.0, f"[N={n_modes}] theta.grad is zero!"
        assert torch.norm(phi.grad) > 0.0, f"[N={n_modes}] phi.grad is zero!"

        grad_norm = torch.norm(theta.grad).item()
        print(f"[N={n_modes:02d}] Physical Chain Backward Gradient Norm ||grad_theta||: {grad_norm:.3e}")


def test_optimization_step_convergence():
    """Verify that gradients can be used by an optimizer (Adam) to minimize a target loss."""
    cfg = PhotonicConfig(n_modes=4, ideal_mode=False, enable_noise=False)
    twin = PhotonicMeshDigitalTwin(cfg)

    # Trainable phase parameters
    theta = torch.nn.Parameter(torch.ones(twin.total_mzis, device=cfg.device) * 1.5)
    phi = torch.nn.Parameter(torch.ones(twin.total_mzis, device=cfg.device) * 1.5)

    optimizer = torch.optim.Adam([theta, phi], lr=0.05)
    target = torch.eye(4, device=cfg.device, dtype=cfg.complex_dtype)

    initial_loss = None
    final_loss = None

    for step in range(15):
        optimizer.zero_grad()
        T = twin.compute_transfer_matrix(theta, phi)
        loss = torch.sum(torch.abs(T - target) ** 2)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    print(f"[Optimization Step] Initial Loss: {initial_loss:.4f} -> Final Loss after 15 steps: {final_loss:.4f}")
    assert final_loss < initial_loss, "Optimizer failed to reduce loss!"


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING AUTOGRAD & GRADIENT FLOW TESTS")
    print("=" * 70)
    test_ste_quantizer_gradients()
    test_full_pipeline_gradient_flow()
    test_optimization_step_convergence()
    print("=" * 70)
    print("[ALL GRADIENT FLOW TESTS PASSED]")
    print("=" * 70)
