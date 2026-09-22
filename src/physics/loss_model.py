"""
Progressive Optical Insertion Loss and Waveguide Attenuation Model.
Mathematical references:
- Bogaerts et al. (2020), Nature 586, 207-216
- Clements et al. (2016), Optica 3(12), 1460-1466
"""

import torch
import torch.nn as nn
from typing import Optional
from src.config import PhotonicConfig


class OpticalLossModel(nn.Module):
    """
    Models stage-by-stage progressive optical insertion loss across an N-layer mesh.
    Computes amplitude attenuation factors that break mathematical unitarity.
    """
    def __init__(self, config: PhotonicConfig):
        super().__init__()
        self.config = config
        self.n_modes = config.n_modes
        self.num_layers = config.num_layers
        self.enabled = config.enable_loss and not config.ideal_mode

        if self.enabled:
            # Generate static per-layer insertion loss (dB)
            # alpha_l = alpha_mean + delta_alpha_l
            torch.manual_seed(config.wafer_seed + 101)
            losses_db = config.loss_per_stage_db + torch.randn(self.num_layers) * config.loss_std_db
            # Ensure loss is strictly positive
            losses_db = torch.clamp(losses_db, min=0.01)
            self.register_buffer("stage_losses_db", losses_db)

            # Field transmission coefficient: a = 10^(-loss_dB / 20)
            stage_attenuations = torch.pow(10.0, -losses_db / 20.0)
            self.register_buffer("stage_attenuations", stage_attenuations)
        else:
            self.register_buffer("stage_losses_db", torch.zeros(self.num_layers))
            self.register_buffer("stage_attenuations", torch.ones(self.num_layers))

    def get_layer_attenuation(self, layer_idx: int) -> torch.Tensor:
        """Returns the scalar field attenuation factor for layer layer_idx."""
        if not self.enabled:
            return torch.tensor(1.0)
        return self.stage_attenuations[layer_idx]

    def total_loss_db(self) -> float:
        """Returns cumulative mean optical loss in dB through all N stages."""
        if not self.enabled:
            return 0.0
        return float(torch.sum(self.stage_losses_db).item())

    def apply_layer_loss(self, field: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Attenuates propagating complex optical field E across layer layer_idx.
        E_out = a_l * E_in
        """
        if not self.enabled:
            return field
        a = self.stage_attenuations[layer_idx].to(device=field.device, dtype=field.dtype)
        return field * a
