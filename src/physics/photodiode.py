"""
Photodiode Square-Law Optical Readout and Analog Noise Model.
Mathematical references:
- Hamerly et al. (2019), Phys. Rev. X 9, 021023
- Mourgias-Alexandris et al. (2022), Nature Communications 13, 5536
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from src.config import PhotonicConfig, PhysicalConstants
from src.nn.quantizer import STEQuantizeFunction


class PhotodetectorArray(nn.Module):
    """
    Simulates optical power absorption and analog electronic readout noise
    at the optical-to-electrical boundary.

    Models:
    - Square-law optical power detection: P_opt = |E_out|^2
    - Quantum Poisson shot noise, TIA thermal noise, and Laser RIN
    - Balanced Photodetection (BPD) architecture with finite CMRR
    - Transimpedance Amplifier (TIA) saturation non-linearity
    - High-speed output ADC quantization
    - Laser phase noise and grating coupler spectral transmission envelope
    """
    def __init__(self, config: PhotonicConfig):
        super().__init__()
        self.config = config
        self.enabled = config.enable_noise and not config.ideal_mode

        # Physical constants
        self.q = PhysicalConstants.q
        self.k_B = PhysicalConstants.k_B

        # Detector parameters
        self.R = config.responsivity
        # Dark current with optional lifetime aging drift
        if getattr(config, "enable_aging", False) and getattr(config, "operating_hours", 0.0) > 0:
            dark_rate = getattr(config, "dark_current_aging_rate", 0.02)
            self.I_dark = config.dark_current * (1.0 + dark_rate * (config.operating_hours / 1000.0))
        else:
            self.I_dark = config.dark_current
        self.R_L = config.tia_load_resistance
        self.B = config.bandwidth
        self.T = config.temperature
        self.lambda_0 = config.wavelength

        # TIA noise factor: F_n = 10^(NF_dB / 10)
        self.F_n = 10.0 ** (config.tia_noise_figure_db / 10.0)

        # Laser RIN linear factor (1/Hz): 10^(RIN_dB / 10)
        self.rin_linear = 10.0 ** (config.rin_db_per_hz / 10.0)

        # Constant thermal noise variance: sigma_th^2 = 4 * k_B * T * F_n * B / R_L
        self.sigma_th_sq = (4.0 * self.k_B * self.T * self.F_n * self.B) / self.R_L

        # Balanced Photodetection (BPD) responsivity mismatch via CMRR
        # Delta_R / R = 10^(-CMRR_dB / 20)
        delta_r_ratio = 10.0 ** (-config.bpd_cmrr_db / 20.0)
        self.R_pos = self.R * (1.0 + 0.5 * delta_r_ratio)
        self.R_neg = self.R * (1.0 - 0.5 * delta_r_ratio)

        # TIA saturation voltage
        self.v_sat = config.tia_saturation_voltage

        # ADC resolution
        self.adc_bits = config.adc_bits
        self.enable_adc = config.enable_adc and not config.ideal_mode

        # Grating coupler parameters
        self.gc_loss_linear = 10.0 ** (-config.grating_coupler_loss_db / 10.0)
        # 1-dB bandwidth to Gaussian sigma: sigma = BW / 2.355
        self.gc_sigma = max(config.grating_bandwidth / 2.355, 1e-12)

    def apply_grating_coupler(
        self,
        E: torch.Tensor,
        wavelength: Optional[float] = None
    ) -> torch.Tensor:
        """
        Applies wavelength-dependent grating coupler fiber-to-chip insertion loss.
        T_GC(lambda) = T_0 * exp( - (lambda - lambda_0)^2 / (2 * sigma_GC^2) )
        """
        if self.config.ideal_mode:
            return E

        lam = wavelength if wavelength is not None else self.lambda_0
        delta_lam = lam - self.lambda_0
        spectral_factor = math.exp(-0.5 * (delta_lam / self.gc_sigma) ** 2)
        power_trans = self.gc_loss_linear * spectral_factor
        amp_trans = math.sqrt(power_trans)
        return E * amp_trans

    def apply_laser_phase_noise(
        self,
        E_in: torch.Tensor,
        integration_time_sec: Optional[float] = None
    ) -> torch.Tensor:
        """
        Injects laser phase noise due to non-zero laser linewidth (Wiener phase drift).
        delta_phi ~ N(0, 2 * pi * Delta_nu * dt)
        """
        if self.config.ideal_mode or self.config.laser_linewidth <= 0:
            return E_in

        dt = integration_time_sec if integration_time_sec is not None else (1.0 / self.B)
        phase_variance = 2.0 * math.pi * self.config.laser_linewidth * dt
        phase_noise = math.sqrt(phase_variance) * torch.randn(
            E_in.shape[:-1],
            device=E_in.device,
            dtype=torch.float32
        ).unsqueeze(-1)
        phasor = torch.exp(1.0j * phase_noise).to(dtype=E_in.dtype)
        return E_in * phasor

    def _add_detector_noise(self, I_mean: torch.Tensor) -> torch.Tensor:
        """Adds signal-dependent shot noise, thermal noise, RIN, and 1/f flicker noise to photocurrent."""
        if not self.enabled:
            return I_mean

        sigma_shot_sq = 2.0 * self.q * (torch.abs(I_mean) + self.I_dark) * self.B
        sigma_rin_sq = self.rin_linear * torch.square(I_mean) * self.B

        # 1/f Flicker noise power below flicker corner frequency
        if getattr(self.config, "enable_flicker_noise", False) and not self.config.ideal_mode:
            fc = getattr(self.config, "flicker_corner_freq", 10.0e3)
            alpha_H = 1e-4
            ln_ratio = math.log(max(fc / 1.0, 1.01))
            sigma_1f_sq = alpha_H * torch.square(I_mean) * ln_ratio
        else:
            sigma_1f_sq = 0.0

        sigma_total_sq = sigma_shot_sq + self.sigma_th_sq + sigma_rin_sq + sigma_1f_sq
        sigma_total = torch.sqrt(torch.clamp(sigma_total_sq, min=1e-24))

        xi = torch.randn_like(I_mean)
        return I_mean + sigma_total * xi

    def detect_balanced(
        self,
        E_pos: torch.Tensor,
        E_neg: torch.Tensor,
        add_noise: bool = True
    ) -> torch.Tensor:
        """
        Simulates Balanced Photodetection (BPD) pair:
        I_diff = (R_pos * |E_pos|^2 + I_dark) - (R_neg * |E_neg|^2 + I_dark) + noise

        Returns differential photocurrent in Amperes (can be signed/negative).
        """
        P_pos = torch.real(E_pos * torch.conj(E_pos))
        P_neg = torch.real(E_neg * torch.conj(E_neg))

        i_dark_dc = self.I_dark if self.enabled else 0.0
        I_p = self.R_pos * P_pos + i_dark_dc
        I_n = self.R_neg * P_neg + i_dark_dc

        if self.enabled and add_noise:
            I_p = self._add_detector_noise(I_p)
            I_n = self._add_detector_noise(I_n)

        return I_p - I_n

    def detect_homodyne(
        self,
        E_sig: torch.Tensor,
        mode: str = "homodyne_i",
        P_lo_watts: float = 1e-3,
        add_noise: bool = True
    ) -> torch.Tensor:
        """
        Simulates balanced optical homodyne detection with a coherent local oscillator (LO).
        50:50 optical coupler mixes E_sig with E_LO, followed by balanced photodetection:
            E_pos = (E_sig + E_LO) / sqrt(2)
            E_neg = (E_sig - E_LO) / sqrt(2)
        - 'homodyne_i': In-phase quadrature (E_LO = sqrt(P_LO)) -> I ~ 2*R*sqrt(P_LO)*Re(E_sig)
        - 'homodyne_q': Quadrature component (E_LO = i*sqrt(P_LO)) -> I ~ 2*R*sqrt(P_LO)*Im(E_sig)

        Args:
            E_sig: Signal optical field tensor of shape (..., N).
            mode: 'homodyne_i' or 'homodyne_q'.
            P_lo_watts: Local oscillator power in Watts.
            add_noise: Whether to include quantum shot noise and electrical noise.

        Returns:
            Demodulated homodyne photocurrent tensor of shape (..., N).
        """
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        amp_lo = math.sqrt(max(P_lo_watts, 1e-12))
        if mode == "homodyne_q":
            E_lo = 1.0j * amp_lo
        else:
            E_lo = amp_lo

        E_pos = (E_sig + E_lo) * inv_sqrt2
        E_neg = (E_sig - E_lo) * inv_sqrt2

        return self.detect_balanced(E_pos, E_neg, add_noise=add_noise)

    def readout_electronics(self, I_photocurrent: torch.Tensor) -> torch.Tensor:
        """
        Passes photocurrent through TIA gain with saturation compression and ADC quantization.
        V_out = V_sat * tanh(I * R_TIA / V_sat)
        """
        if self.config.ideal_mode:
            return I_photocurrent

        # TIA amplification with soft saturation
        V_linear = I_photocurrent * self.R_L
        V_tia = self.v_sat * torch.tanh(V_linear / max(self.v_sat, 1e-3))

        # Output ADC quantization via STE
        if self.enable_adc:
            V_adc = STEQuantizeFunction.apply(V_tia, -self.v_sat, self.v_sat, self.adc_bits)
            return V_adc

        return V_tia

    def forward(
        self,
        E_out: torch.Tensor,
        add_noise: bool = True,
        readout_mode: Optional[str] = None
    ) -> torch.Tensor:
        """
        Detects optical field and returns measured photocurrent or digitized voltage.

        Args:
            E_out: Complex optical field tensor of shape (..., N).
            add_noise: Flag to conditionally enable/disable noise during forward pass.
            readout_mode: Optional override ('direct', 'dual_rail', 'homodyne_i', 'homodyne_q').

        Returns:
            Real-valued readout tensor of shape (..., N) or (..., N/2) for dual_rail.
        """
        mode = readout_mode or getattr(self.config, "readout_mode", "direct")

        if mode == "dual_rail":
            if E_out.shape[-1] % 2 != 0:
                raise ValueError(f"dual_rail readout requires even number of modes, got {E_out.shape[-1]}")
            return self.detect_balanced(E_out[..., 0::2], E_out[..., 1::2], add_noise=add_noise)

        if mode in ("homodyne_i", "homodyne_q"):
            return self.detect_homodyne(E_out, mode=mode, add_noise=add_noise)

        # Default: direct square-law optical power detection
        P_opt = torch.real(E_out * torch.conj(E_out))

        # Mean single-ended photocurrent including physical dark current baseline
        i_dark_dc = self.I_dark if self.enabled else 0.0
        I_sig = self.R * P_opt + i_dark_dc

        if not self.enabled or not add_noise:
            return I_sig

        # Add physical noise floor
        I_noisy = self._add_detector_noise(I_sig)
        I_clamped = torch.clamp(I_noisy, min=0.0)

        return I_clamped

    def compute_snr_db(self, optical_power_watts: float) -> float:
        """
        Calculates theoretical electrical Signal-to-Noise Ratio (SNR) in dB
        for a given optical power level.
        """
        I_sig = self.R * optical_power_watts
        sigma_shot_sq = 2.0 * self.q * (I_sig + self.I_dark) * self.B
        sigma_rin_sq = self.rin_linear * (I_sig ** 2) * self.B
        sigma_total_sq = sigma_shot_sq + self.sigma_th_sq + sigma_rin_sq

        snr_linear = (I_sig ** 2) / sigma_total_sq
        return 10.0 * math.log10(max(snr_linear, 1e-12))
