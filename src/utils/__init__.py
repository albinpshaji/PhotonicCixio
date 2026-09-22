"""
Utility functions for photonic simulation and matrix decomposition.
"""

from src.utils.decomposition import clements_decompose_np, generate_random_unitary
from src.utils.diagnostics import sweep_wavelength_spectrum, compute_mesh_thermal_dissipation
from src.utils.evaluations import (
    evaluate_matrix_fidelity,
    evaluate_effective_resolution_enob,
    evaluate_optical_energy_per_mac,
    evaluate_full_hardware_system,
    HardwareEvaluationReport,
)

__all__ = [
    "clements_decompose_np",
    "generate_random_unitary",
    "sweep_wavelength_spectrum",
    "compute_mesh_thermal_dissipation",
    "evaluate_matrix_fidelity",
    "evaluate_effective_resolution_enob",
    "evaluate_optical_energy_per_mac",
    "evaluate_full_hardware_system",
    "HardwareEvaluationReport",
]
