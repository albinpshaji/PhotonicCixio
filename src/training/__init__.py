"""Training utilities and physics-aware loss functions for photonic tensor processing."""
from src.training.losses import (
    PhotonicFidelityLoss,
    UnitaryDriftPenalty,
    SpatialThermalGradientPenalty,
    InSituAdjointLoss,
    CompositePhotonicLoss,
)

__all__ = [
    "PhotonicFidelityLoss",
    "UnitaryDriftPenalty",
    "SpatialThermalGradientPenalty",
    "InSituAdjointLoss",
    "CompositePhotonicLoss",
]
