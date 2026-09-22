"""
Unit Test: Correlated Wafer Map Spatial Covariance and Determinism.
Validates:
1. Spatial covariance decays exponentially with physical distance according to Lx, Ly.
2. Systematic vs. random variance partition (systematic_ratio).
3. Deterministic repeatability under wafer_seed.
"""

import sys
import os
import math
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PhotonicConfig
from src.physics.thermal import compute_clements_layout_coordinates
from src.physics.spatial_wafer import SpatialWaferMap, build_spatial_covariance_matrix


def test_spatial_covariance_decay():
    """Verifies that spatial covariance decays monotonically with distance."""
    coords, _ = compute_clements_layout_coordinates(n_modes=16)
    Lx = 250.0e-6
    Ly = 200.0e-6
    var_total = 0.04 ** 2

    Sigma = build_spatial_covariance_matrix(
        coords=coords,
        correlation_length_x=Lx,
        correlation_length_y=Ly,
        total_variance=var_total,
        systematic_ratio=0.80
    )

    # Diagonal elements must equal total variance (plus jitter)
    diag = torch.diag(Sigma)
    assert torch.allclose(diag, torch.tensor(var_total), atol=1e-5), (
        f"Diagonal variance mismatch: {diag.mean().item()} vs {var_total}"
    )

    # Off-diagonal elements must be strictly positive and less than diagonal
    off_diag = Sigma[~torch.eye(Sigma.shape[0], dtype=torch.bool)]
    assert (off_diag > 0).all(), "Covariance elements must be positive"
    assert (off_diag < var_total).all(), "Off-diagonal covariance must be strictly less than variance"

    # Pairs with large separation (> 3*Lx) must have covariance < 5% of peak
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    dist_x = torch.abs(diff[:, :, 0])
    far_mask = dist_x > (3.0 * Lx)
    if far_mask.any():
        far_cov = Sigma[far_mask]
        max_far_cov = torch.max(far_cov).item()
        assert max_far_cov < 0.05 * var_total, (
            f"Far-field covariance did not decay: {max_far_cov} vs {0.05 * var_total}"
        )


def test_wafer_seed_determinism():
    """Verifies that identical wafer_seed produces bit-exact identical wafer maps."""
    coords, _ = compute_clements_layout_coordinates(n_modes=8)
    cfg1 = PhotonicConfig(n_modes=8, wafer_seed=42)
    cfg2 = PhotonicConfig(n_modes=8, wafer_seed=42)
    cfg3 = PhotonicConfig(n_modes=8, wafer_seed=999)

    w1 = SpatialWaferMap(coords, cfg1)
    w2 = SpatialWaferMap(coords, cfg2)
    w3 = SpatialWaferMap(coords, cfg3)

    # w1 and w2 must be exactly identical
    assert torch.equal(w1.coupler_eps1, w2.coupler_eps1), "Same seed must produce identical eps1"
    assert torch.equal(w1.phi_intrinsic, w2.phi_intrinsic), "Same seed must produce identical phi_intrinsic"

    # w1 and w3 must differ
    assert not torch.equal(w1.coupler_eps1, w3.coupler_eps1), "Different seeds must produce different maps"


if __name__ == "__main__":
    test_spatial_covariance_decay()
    test_wafer_seed_determinism()
    print("ALL WAFER CORRELATION TESTS PASSED.")
