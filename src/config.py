"""
Simulation Configuration and Physical Benchmark Constants for Silicon Photonics.
Calibrated for standard 220 nm Silicon-on-Insulator (SOI) CMOS photonics foundry PDKs.
"""

from dataclasses import dataclass, field
from enum import Enum
import torch


class FoundryPDK(Enum):
    """Calibrated Silicon Photonics Process Design Kits (PDKs)."""
    AIM_PHOTONICS_220NM = "aim_photonics_220nm"
    IMEC_ISIPP50G = "imec_isipp50g"
    TSMC_SOI = "tsmc_soi"
    CUSTOM = "custom"


@dataclass(frozen=True)
class PhysicalConstants:
    """Foundry-standard physical constants (SOI platform, 1550 nm)."""
    # Thermo-optic coefficient of silicon (K^-1)
    dn_dT: float = 1.86e-4
    # Nominal vacuum optical wavelength (m)
    lambda_0: float = 1550e-9
    # Silicon core refractive index
    n_si: float = 3.45
    # Silicon dioxide cladding refractive index
    n_sio2: float = 1.444
    # Fundamental TE mode effective index in 450nm x 220nm waveguide
    n_eff: float = 2.445
    # Waveguide group index
    n_g: float = 4.18
    # Dispersion parameter D in ps / (nm * km)
    dispersion_D: float = -1000.0
    # Elementary charge (C)
    q: float = 1.602176634e-19
    # Boltzmann constant (J/K)
    k_B: float = 1.380649e-23
    # Planck constant (J*s)
    h: float = 6.62607015e-34
    # Speed of light in vacuum (m/s)
    c: float = 299792458.0


@dataclass
class PhotonicConfig:
    """
    Configuration parameters for the PhotonicMeshDigitalTwin.
    Allows granular toggling and calibration of foundry-level physical non-idealities.
    """
    # Active Foundry PDK preset
    pdk: FoundryPDK = FoundryPDK.AIM_PHOTONICS_220NM

    # Number of optical spatial modes (channels) - must be even
    n_modes: int = 8

    # Operating nominal optical wavelength (meters)
    wavelength: float = 1550e-9

    # Master switch: if True, disables all physical impairments for ideal math mode
    ideal_mode: bool = False

    # ---------------- 1. Directional Coupler & Wafer Spatial Correlation ----------------
    # Enable directional coupler splitting errors kappa = 0.5 +/- epsilon
    enable_coupler_errors: bool = True
    # Standard deviation of coupler splitting deviation epsilon
    coupler_error_std: float = 0.04
    # Optional fixed seed for static wafer-map generation
    wafer_seed: int = 42
    # Enable 2D correlated wafer map (Gaussian Random Field with spatial covariance)
    enable_spatial_correlation: bool = True
    # Spatial correlation length in X (horizontal die direction, meters)
    correlation_length_x: float = 250.0e-6
    # Spatial correlation length in Y (vertical die direction, meters)
    correlation_length_y: float = 200.0e-6
    # Fraction of systematic correlated variation vs random high-frequency variation (0.0 to 1.0)
    systematic_var_ratio: float = 0.75
    # Static intrinsic arm phase bias std (rad) from lithographic line-edge roughness
    intrinsic_phase_std: float = 0.03

    # ---------------- 2. Thermal Crosstalk, Electro-Thermal & Package Dynamics ----------------
    # Enable non-local lateral thermal bleed across micro-heaters
    enable_thermal_crosstalk: bool = True
    # Characteristic thermal decay length in SiO2 cladding (meters)
    thermal_decay_length: float = 55.0e-6
    # Self-heating thermal impedance kappa_0 (K/W)
    thermal_impedance: float = 18.5
    # Physical spacing between MZI heaters in the layout grid (meters)
    heater_pitch_x: float = 120.0e-6
    heater_pitch_y: float = 80.0e-6
    # Thermo-optic heater switching power P_pi (Watts)
    P_pi: float = 21.5e-3
    # Micro-heater electrical resistance (Ohms)
    heater_resistance: float = 1200.0
    # Enable Temperature Coefficient of Resistance (TCR) electro-thermal feedback
    enable_tcr: bool = True
    # TCR of heater material (K^-1, e.g. TiN or doped Si)
    tcr_coefficient: float = 0.0025
    # Package thermal resistance to ambient / heatsink (K/W)
    package_thermal_resistance: float = 15.0
    # Thermal time constant for transient dynamics (seconds)
    thermal_time_constant: float = 10.0e-6
    # Thermal simulation mode: 'steady_state' or 'transient'
    thermal_mode: str = "steady_state"
    # Thermal solver formulation: 'finite_difference' (2D Poisson/Helmholtz FD) or 'exponential' (lumped kernel)
    thermal_solver: str = "finite_difference"
    # Thermal conductivity of silicon device layer (W/(m*K))
    si_thermal_conductivity: float = 148.0
    # Thickness of buried oxide (BOX) layer (meters)
    box_thickness: float = 2.0e-6
    # Thermal conductivity of SiO2 BOX layer (W/(m*K))
    box_thermal_conductivity: float = 1.38
    # Grid resolution along dominant dimension for 2D FD solver
    thermal_grid_resolution: int = 64

    # ---------------- 3. Optical Insertion Loss & Physical Routing ----------------
    # Enable stage-by-stage optical attenuation
    enable_loss: bool = True
    # Mean optical loss per MZI stage (dB)
    loss_per_stage_db: float = 0.15
    # Stage-to-stage random loss variation std (dB)
    loss_std_db: float = 0.03
    # Enable physical waveguide routing effects (crossings & path skew)
    enable_physical_routing: bool = True
    # Waveguide propagation loss (dB/cm)
    propagation_loss_db_per_cm: float = 1.8
    # Insertion loss per waveguide crossing (dB)
    crossing_loss_db: float = 0.025
    # Inter-channel optical crosstalk per crossing (dB)
    crossing_crosstalk_db: float = -40.0

    # ---------------- 4. Broadband Dispersion & Laser Source ----------------
    # Enable wavelength-dependent coupling and phase dispersion
    enable_dispersion: bool = True
    # Nominal directional coupler interaction length (meters)
    coupler_coupling_length: float = 18.5e-6
    # Excess insertion loss per 3dB coupler (dB)
    coupler_excess_loss_db: float = 0.05
    # Input laser linewidth (Hz) causing phase noise
    laser_linewidth: float = 500.0e3
    # Grating coupler peak insertion loss per port (dB)
    grating_coupler_loss_db: float = 3.0
    # Grating coupler 1-dB optical bandwidth (meters)
    grating_bandwidth: float = 35.0e-9

    # ---------------- 5. Electrical Control & DAC Non-Idealities ----------------
    # Enable discrete DAC control with Straight-Through Estimator (STE)
    enable_quantization: bool = True
    # DAC resolution in bits (typically 4 to 8 bits)
    dac_bits: int = 6
    # Optional dedicated DAC resolution for output diagonal phase screen
    dac_phi_diag_bits: int = 6
    # Enable output diagonal phase screen (necessary for universal U(N) unitary synthesis)
    enable_diagonal_phase_screen: bool = True
    # Differential Non-Linearity (DNL) std in LSB
    dac_dnl_lsb: float = 0.3
    # Integral Non-Linearity (INL) peak in LSB
    dac_inl_lsb: float = 0.5
    # Enable analog voltage ripple / phase jitter
    enable_phase_jitter: bool = True
    # Electrical phase jitter std on heater control (radians)
    phase_jitter_std: float = 0.008

    # ---------------- 6. Balanced Photodetection & Readout ----------------
    # Optical readout mode: 'direct' (single-ended power), 'dual_rail' (differential pair 2k, 2k+1),
    # 'homodyne_i' (in-phase with local oscillator), 'homodyne_q' (quadrature with local oscillator)
    readout_mode: str = "direct"
    # Enable photodetector noise (shot, thermal, RIN)
    enable_noise: bool = True
    # Photodiode responsivity R (A/W)
    responsivity: float = 0.95
    # Photodiode dark current (Amperes)
    dark_current: float = 5.0e-9
    # Transimpedance amplifier (TIA) feedback resistance (Ohms)
    tia_load_resistance: float = 1000.0
    # TIA noise figure (dB)
    tia_noise_figure_db: float = 2.5
    # Photodetector / TIA electrical bandwidth (Hz)
    bandwidth: float = 10.0e9
    # Laser Relative Intensity Noise (dB/Hz)
    rin_db_per_hz: float = -145.0
    # Operating temperature (Kelvin)
    temperature: float = 300.0
    # Enable Balanced Photodetection (BPD) architecture for signed matrix math (legacy compatibility)
    enable_balanced_detection: bool = True
    # BPD Common-Mode Rejection Ratio (CMRR in dB)
    bpd_cmrr_db: float = 30.0
    # TIA saturation voltage (Volts)
    tia_saturation_voltage: float = 1.2
    # Enable output ADC quantization
    enable_adc: bool = True
    # Output ADC resolution (bits)
    adc_bits: int = 8

    # ---------------- 8. Second-Order Physics Extensions ----------------
    # Module 1: Coherent Backreflection & Multi-Cavity Fabry-Perot
    enable_backreflection: bool = True
    grating_reflectivity_db: float = -25.0
    crossing_reflectivity_db: float = -35.0
    coupler_reflectivity_db: float = -40.0
    cavity_length: float = 120.0e-6

    # Module 2: Nonlinear Optics (TPA, FCA, SPM, FCD)
    enable_nonlinear_optics: bool = True
    # Two-photon absorption coefficient (m/W) - calibrated for silicon at 1550 nm (~0.65-0.8 cm/GW = 6.5-8.0e-12 m/W)
    tpa_coefficient: float = 6.5e-12
    # Effective optical mode area (m^2) for 450 nm x 220 nm strip waveguide
    effective_mode_area: float = 0.055e-12
    # Free-carrier recombination lifetime (seconds)
    free_carrier_lifetime: float = 2.5e-9
    # Optical Kerr nonlinear index n2 (m^2/W)
    nonlinear_index_n2: float = 4.5e-18
    # Optical input power in Watts (CW)
    optical_input_power_watts: float = 5.0e-3  # 5 mW

    # Module 3: Polarization & Jones Vectors
    enable_polarization: bool = False
    n_eff_tm: float = 1.785
    polarization_coupling_rad: float = 0.02
    tm_crossing_loss_db: float = 0.08
    tm_bend_loss_factor: float = 2.5

    # Module 4: Advanced Thermal Predistortion (BNNLS)
    # Tikhonov regularization parameter for thermal inversion
    thermal_tikhonov_lambda: float = 1e-3
    # Maximum allowable microheater electrical power per actuator (Watts)
    max_heater_power_watts: float = 0.050

    # Module 5: Waveguide Bend Loss & Mode Mismatch
    enable_bend_loss: bool = True
    min_bend_radius: float = 5.0e-6  # meters
    waveguide_width: float = 450e-9  # meters
    coupler_gap_width: float = 350e-9  # meters
    bend_loss_coefficient_C1: float = 0.12  # dB
    bend_loss_coefficient_C2: float = 0.45e6  # 1/m
    transition_loss_db: float = 0.02  # dB per width transition

    # Module 6: 1/f Flicker Noise & Temporal Aging
    enable_flicker_noise: bool = True
    flicker_corner_freq: float = 10.0e3  # 10 kHz
    enable_aging: bool = False
    operating_hours: float = 0.0
    heater_aging_rate: float = 0.005  # 0.5% per decade hour
    dark_current_aging_rate: float = 0.02  # 2% per 1000 hours

    # ---------------- 7. Compute Device & Precision ----------------
    # Execution device ('cuda', 'cpu', or auto-detected)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    # Floating point dtype for real tensors
    dtype: torch.dtype = torch.float32
    # Complex dtype for optical wave fields
    complex_dtype: torch.dtype = torch.complex64

    def __post_init__(self):
        if self.n_modes < 2 or self.n_modes % 2 != 0:
            raise ValueError(f"n_modes must be an even positive integer (got {self.n_modes})")
        if self.ideal_mode:
            # Force disable all non-idealities when ideal mode is requested
            self.enable_coupler_errors = False
            self.enable_spatial_correlation = False
            self.enable_thermal_crosstalk = False
            self.enable_tcr = False
            self.enable_loss = False
            self.enable_physical_routing = False
            self.enable_dispersion = False
            self.enable_quantization = False
            self.enable_phase_jitter = False
            self.enable_noise = False
            self.enable_balanced_detection = False
            self.enable_adc = False
            self.enable_backreflection = False
            self.enable_nonlinear_optics = False
            self.enable_polarization = False
            self.enable_bend_loss = False
            self.enable_flicker_noise = False
            self.enable_aging = False

    @classmethod
    def from_pdk(cls, pdk: FoundryPDK, **kwargs) -> "PhotonicConfig":
        """
        Factory method to generate foundry-calibrated configurations.
        """
        if pdk == FoundryPDK.AIM_PHOTONICS_220NM:
            cfg = cls(
                pdk=pdk,
                P_pi=21.5e-3,
                heater_resistance=1200.0,
                heater_pitch_x=120.0e-6,
                heater_pitch_y=80.0e-6,
                propagation_loss_db_per_cm=1.8,
                crossing_loss_db=0.025,
                crossing_crosstalk_db=-40.0,
                coupler_error_std=0.035,
                bpd_cmrr_db=30.0,
                **kwargs
            )
        elif pdk == FoundryPDK.IMEC_ISIPP50G:
            cfg = cls(
                pdk=pdk,
                P_pi=18.0e-3,
                heater_resistance=1000.0,
                heater_pitch_x=100.0e-6,
                heater_pitch_y=75.0e-6,
                propagation_loss_db_per_cm=1.5,
                crossing_loss_db=0.020,
                crossing_crosstalk_db=-42.0,
                coupler_error_std=0.030,
                bpd_cmrr_db=32.0,
                **kwargs
            )
        elif pdk == FoundryPDK.TSMC_SOI:
            cfg = cls(
                pdk=pdk,
                P_pi=16.0e-3,
                heater_resistance=800.0,
                heater_pitch_x=90.0e-6,
                heater_pitch_y=60.0e-6,
                propagation_loss_db_per_cm=1.2,
                crossing_loss_db=0.015,
                crossing_crosstalk_db=-45.0,
                coupler_error_std=0.025,
                bpd_cmrr_db=35.0,
                **kwargs
            )
        else:
            cfg = cls(pdk=FoundryPDK.CUSTOM, **kwargs)
        return cfg

    @property
    def total_mzis(self) -> int:
        """Total number of MZIs in an N-mode Clements mesh: M = N*(N-1)/2."""
        return (self.n_modes * (self.n_modes - 1)) // 2

    @property
    def num_layers(self) -> int:
        """Total optical depth (number of columns) in Clements mesh: exactly N."""
        return self.n_modes
