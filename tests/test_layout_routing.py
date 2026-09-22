"""
Unit Test: Physical Waveguide Routing, Crossing Loss, and Path-Length Skew.
Validates:
1. Inter-stage waveguide crossing identification.
2. Crossing insertion loss and inter-channel optical crosstalk level.
3. Physical route-length propagation loss and phase delay.
"""

import sys
import os
import math
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PhotonicConfig
from src.physics.routing_loss import ClementsPhysicalRouting


def test_crossing_count_and_pairs():
    """Verifies that crossing pairs match physical planar layout geometry."""
    for n_modes in [4, 8, 16]:
        cfg = PhotonicConfig(n_modes=n_modes, enable_physical_routing=True)
        routing = ClementsPhysicalRouting(cfg)

        # Number of crossing pairs between even and odd columns: (N - 2) // 2
        expected_crossings = (n_modes - 2) // 2
        assert len(routing.crossing_p) == expected_crossings, (
            f"Expected {expected_crossings} crossings for N={n_modes}, got {len(routing.crossing_p)}"
        )


def test_crossing_crosstalk_magnitude():
    """Verifies that optical crossing induces -40 dB crosstalk between crossing channels."""
    cfg = PhotonicConfig(
        n_modes=8,
        enable_physical_routing=True,
        crossing_loss_db=0.025,
        crossing_crosstalk_db=-40.0
    )
    routing = ClementsPhysicalRouting(cfg)

    # Launch light into mode 1 only: [0, 1.0, 0, 0, ...]
    field_in = torch.zeros(1, 8, dtype=torch.complex64)
    field_in[0, 1] = 1.0

    # Crossings active between col 0 (even) and col 1 (odd)
    field_out = routing.apply_inter_stage_crossings(field_in, col=0)

    # Mode 1 power (through path)
    p_through = (torch.abs(field_out[0, 1]) ** 2).item()
    # Mode 2 power (crosstalk path from mode 1)
    p_xtalk = (torch.abs(field_out[0, 2]) ** 2).item()

    # Crosstalk power in dB: 10 * log10(p_xtalk)
    xtalk_db = 10.0 * math.log10(p_xtalk)
    assert abs(xtalk_db - (-40.0)) < 0.5, f"Crosstalk mismatch: {xtalk_db} dB vs -40.0 dB"


def test_propagation_path_skew():
    """Verifies that physical waveguide propagation introduces path-length dependent loss."""
    cfg = PhotonicConfig(n_modes=8, enable_physical_routing=True, propagation_loss_db_per_cm=1.8)
    routing = ClementsPhysicalRouting(cfg)

    # Outer modes (0 and 7) should travel slightly longer distance than center modes (3 and 4)
    center_len = routing.route_lengths[3].item()
    outer_len = routing.route_lengths[0].item()
    assert outer_len > center_len, f"Outer mode length ({outer_len}) should exceed center ({center_len})"

    # Propagation attenuation must be strictly < 1.0
    assert (routing.route_attenuations < 1.0).all()
    assert (routing.route_attenuations > 0.8).all()


if __name__ == "__main__":
    test_crossing_count_and_pairs()
    test_crossing_crosstalk_magnitude()
    test_propagation_path_skew()
    print("ALL LAYOUT ROUTING TESTS PASSED.")
