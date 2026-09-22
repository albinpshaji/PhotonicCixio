"""
Physical non-ideality simulation modules for silicon photonics.
"""

from src.physics.mzi import (
    directional_coupler_matrix,
    mzi_transfer_matrix,
    MZIOperator
)
from src.physics.thermal import (
    compute_clements_layout_coordinates,
    build_thermal_crosstalk_matrix,
    ThermalCrosstalkModel
)
from src.physics.loss_model import OpticalLossModel
from src.physics.photodiode import PhotodetectorArray

__all__ = [
    "directional_coupler_matrix",
    "mzi_transfer_matrix",
    "MZIOperator",
    "compute_clements_layout_coordinates",
    "build_thermal_crosstalk_matrix",
    "ThermalCrosstalkModel",
    "OpticalLossModel",
    "PhotodetectorArray"
]
