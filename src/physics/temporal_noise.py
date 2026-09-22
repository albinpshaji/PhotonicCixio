"""
Low-Frequency 1/f (Flicker) Noise and Device Temporal Aging Models.
Calibrated for Silicon Photodetector Readout Chains and Micro-Heaters.

Governing Physics:
1. Low-Frequency 1/f Flicker Noise (Hooge's Empirical Relation):
   S_1f(f) = (alpha_H / N_carriers) * (I^2 / f^gamma)
   where gamma approx 1.0, and alpha_H is Hooge's parameter.
   Flicker corner frequency f_c (typically 1 kHz - 100 kHz) marks where
   1/f noise power equals the white noise floor.

2. Photodiode Dark Current Temporal Drift:
   I_dark(t) = I_dark,0 * (1 + alpha_dark * (t / 1000 hours))
   Caused by trap state generation at the Si/SiO2 interface under reverse bias.

3. Micro-Heater Resistance Aging Drift:
   R(t) = R_0 * (1 + delta_R * log10(1 + t / t_0))
   Caused by electromigration and thermal grain annealing under continuous Joule heating.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.config import PhotonicConfig


def compute_flicker_noise_variance(
    I_mean: torch.Tensor,
    corner_freq: float = 10.0e3,
    f_min: float = 1.0,
    f_max: float = 10.0e9,
    alpha_H: float = 1e-4
) -> torch.Tensor:
    """
    Computes 1/f flicker noise variance sigma_1f^2 for a given photocurrent.

    Integrating S(f) = K * I^2 / f from f_min to corner_freq:
    sigma_1f^2 = K * I^2 * ln(corner_freq / f_min)

    Args:
        I_mean: Mean photocurrent tensor in Amperes.
        corner_freq: Flicker noise corner frequency (Hz).
        f_min: Minimum integration frequency (e.g. 1 Hz).
        f_max: Upper circuit bandwidth (Hz).
        alpha_H: Effective flicker scaling coefficient.

    Returns:
        Tensor of shape matching I_mean with 1/f noise variance (A^2).
    """
    if corner_freq <= f_min:
        return torch.zeros_like(I_mean)

    # Integrated 1/f noise power
    ln_ratio = math.log(max(corner_freq / f_min, 1.01))
    sigma_1f_sq = alpha_H * (I_mean ** 2) * ln_ratio
    return sigma_1f_sq


class TemporalNoiseAndAgingModel(nn.Module):
    """
    Module providing 1/f noise generation and lifetime aging adjustments
    for photodetectors and thermo-optic actuators.
    """
    def __init__(self, config: PhotonicConfig):
        super().__init__()
        self.config = config
        self.enable_flicker = config.enable_flicker_noise and not config.ideal_mode
        self.enable_aging = config.enable_aging and not config.ideal_mode
        self.operating_hours = config.operating_hours
        self.corner_freq = config.flicker_corner_freq

        # Dark current aging
        if self.enable_aging and self.operating_hours > 0:
            rate = config.dark_current_aging_rate
            self.dark_current_factor = 1.0 + rate * (self.operating_hours / 1000.0)
        else:
            self.dark_current_factor = 1.0

    def get_aged_dark_current(self, base_dark_current: float) -> float:
        """Returns dark current scaled by operating lifetime."""
        return base_dark_current * self.dark_current_factor

    def add_flicker_noise(self, I_current: torch.Tensor) -> torch.Tensor:
        """
        Adds 1/f flicker noise to current tensor.
        """
        if not self.enable_flicker:
            return I_current

        sigma_1f_sq = compute_flicker_noise_variance(
            I_mean=I_current,
            corner_freq=self.corner_freq,
            f_min=1.0,
            f_max=self.config.bandwidth
        )
        sigma_1f = torch.sqrt(torch.clamp(sigma_1f_sq, min=1e-26))
        noise = sigma_1f * torch.randn_like(I_current)
        return I_current + noise
