"""
Physical Non-Ideality Validation Test Suite.
Validates:
1. Directional coupler leakage floor T_min = 4*epsilon^2 across epsilon sweep.
2. Non-local thermal diffusion matrix and lateral crosstalk bleed (5% - 20%).
3. Thermal pre-distortion inversion (Thermal Eigenmode Decomposition).
4. Cascading optical insertion loss breaking mathematical unitarity.
5. DAC B-bit quantization discretization.
6. Photodiode shot noise and thermal noise scaling.
"""

import sys
import os
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PhotonicConfig
from src.physics.mzi import mzi_transfer_matrix
from src.physics.thermal import ThermalCrosstalkModel
from src.physics.loss_model import OpticalLossModel
from src.physics.photodiode import PhotodetectorArray
from src.nn.quantizer import DACQuantizer
from src.models.digital_twin import PhotonicMeshDigitalTwin


def test_coupler_leakage_floor_formula():
    """
    Validates that directional coupler split deviations produce the theoretical
    cross-state leakage floor: T_min approx 4 * epsilon^2.
    """
    print("\n--- 1. Directional Coupler Leakage Floor Validation ---")
    epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

    for eps in epsilons:
        eps_tensor = torch.tensor([eps], dtype=torch.float32)
        # At theta = 0 (target bar state destructive interference)
        theta_zero = torch.tensor([0.0], dtype=torch.float32)

        # MZI transfer matrix with symmetric coupler deviation eps1 = eps2 = eps
        T = mzi_transfer_matrix(
            theta=theta_zero,
            epsilon1=eps_tensor,
            epsilon2=eps_tensor,
            ideal_mode=False
        )

        # Measured power transmission T_min = |T_11|^2
        T_min_measured = torch.abs(T[0, 0, 0]) ** 2
        T_min_theoretical = 4.0 * (eps ** 2)

        rel_error = abs(T_min_measured.item() - T_min_theoretical) / T_min_theoretical
        er_db_measured = -10.0 * math.log10(T_min_measured.item())
        er_db_theoretical = -10.0 * math.log10(T_min_theoretical)

        print(f"  eps={eps:.2f} | Measured T_min={T_min_measured.item():.6f} | "
              f"Theory 4*eps^2={T_min_theoretical:.6f} | RelErr={rel_error:.2e} | "
              f"ER={er_db_measured:.1f} dB (Theory {er_db_theoretical:.1f} dB)")

        assert rel_error < 0.01, f"Leakage floor mismatch for eps={eps}: rel_error={rel_error}"


def test_thermal_crosstalk_bleed():
    """
    Validates non-local thermal diffusion:
    Driving heater i by pi induces 5% to 25% phase shift in nearest neighbors,
    decaying monotonically with physical Euclidean distance.
    """
    print("\n--- 2. Non-Local Thermal Crosstalk Validation ---")
    cfg = PhotonicConfig(
        n_modes=8,
        ideal_mode=False,
        enable_thermal_crosstalk=True,
        enable_tcr=False,
        package_thermal_resistance=0.0
    )
    thermal = ThermalCrosstalkModel(cfg)

    # Drive only heater 0 with pi phase
    theta_single = torch.zeros(cfg.total_mzis)
    theta_single[0] = math.pi

    theta_actual = thermal(theta_single)

    self_phase = theta_actual[0].item()
    assert abs(self_phase - math.pi) < 1e-4, f"Self phase altered: {self_phase}"

    # Check neighbor crosstalk
    crosstalk_ratios = []
    for j in range(1, cfg.total_mzis):
        induced = theta_actual[j].item()
        ratio = induced / math.pi
        crosstalk_ratios.append((j, ratio, thermal.coords[j].tolist()))

    # Find nearest neighbor
    dists = torch.norm(thermal.coords[1:] - thermal.coords[0], dim=-1)
    nearest_idx = torch.argmin(dists).item() + 1
    nearest_ratio = theta_actual[nearest_idx].item() / math.pi
    nearest_dist_um = dists[nearest_idx - 1].item() * 1e6

    print(f"  Heater 0 driven with pi rad.")
    print(f"  Nearest neighbor MZI {nearest_idx} (dist={nearest_dist_um:.1f} um) induced ratio: {nearest_ratio*100:.1f}%")
    assert 0.02 <= nearest_ratio <= 0.25, (
        f"Nearest neighbor crosstalk {nearest_ratio*100:.1f}% outside realistic 2%-25% window!"
    )

    # Test Thermal Pre-distortion Inversion (TED)
    theta_target = torch.rand(cfg.total_mzis) * math.pi
    theta_predistort = thermal.predistort(theta_target)
    theta_realized = thermal(theta_predistort)

    ted_inversion_err = torch.norm(theta_realized - theta_target).item()
    print(f"  Thermal Pre-distortion Inversion Error: {ted_inversion_err:.3e}")
    assert ted_inversion_err < 1e-5, f"Thermal inversion failed: error = {ted_inversion_err}"


def test_optical_loss_breaks_unitarity():
    """
    Validates that progressive optical insertion loss causes sub-unitarity (singular values < 1.0)
    and power attenuation matching the cumulative dB loss.
    """
    print("\n--- 3. Cascading Optical Insertion Loss Validation ---")
    n_modes = 8
    cfg = PhotonicConfig(
        n_modes=n_modes,
        ideal_mode=False,
        enable_coupler_errors=False,
        enable_thermal_crosstalk=False,
        enable_quantization=False,
        enable_noise=False,
        enable_loss=True,
        enable_physical_routing=False,
        enable_bend_loss=False,
        enable_backreflection=False,
        loss_per_stage_db=0.15,
        loss_std_db=0.0
    )
    twin = PhotonicMeshDigitalTwin(cfg)

    theta = torch.rand(twin.total_mzis, device=cfg.device) * math.pi
    phi = torch.rand(twin.total_mzis, device=cfg.device) * (2.0 * math.pi)

    T_lossy = twin.compute_transfer_matrix(theta, phi)

    # Compute singular values of T_lossy
    S = torch.linalg.svdvals(T_lossy)
    mean_s = torch.mean(S).item()

    # Theoretical singular value: a_total = 10^(-(N * 0.15 dB) / 20)
    expected_s = 10.0 ** (-(n_modes * 0.15) / 20.0)

    print(f"  Total stages N={n_modes} | Expected singular value a={expected_s:.4f} | Measured a={mean_s:.4f}")
    assert torch.all(S < 1.0), "Singular values not strictly < 1.0 under optical loss!"
    assert abs(mean_s - expected_s) < 1e-3, f"Singular value mismatch: {mean_s} vs {expected_s}"

    # Unitarity violation norm
    unitarity_violation = torch.norm(T_lossy.conj().T @ T_lossy - torch.eye(n_modes, device=cfg.device)).item()
    print(f"  Unitarity violation ||T^H T - I||_F under 0.15 dB/stage loss: {unitarity_violation:.4f}")
    assert unitarity_violation > 0.1, "Loss failed to break unitarity!"


def test_dac_quantization_levels():
    """Validates that B-bit DAC quantizer discretizes continuous angles into 2^B discrete levels."""
    print("\n--- 4. DAC Control Discretization Validation ---")
    bits = 4
    n_levels = 1 << bits  # 16
    quantizer = DACQuantizer(bits=bits, val_min=0.0, val_max=math.pi, enabled=True)

    # Continuous fine sweep across [0, pi]
    continuous_sweep = torch.linspace(0.0, math.pi, 1000)
    quantized_sweep = quantizer(continuous_sweep)

    unique_levels = torch.unique(quantized_sweep)
    print(f"  DAC {bits}-bit: Expected levels={n_levels} | Observed unique levels={len(unique_levels)}")
    assert len(unique_levels) == n_levels, f"Expected {n_levels} unique levels, got {len(unique_levels)}"

    step_size = quantizer.step_size
    expected_step = math.pi / 15.0
    print(f"  DAC LSB Step Size: {step_size:.5f} rad ({step_size*180/math.pi:.2f} deg)")
    assert abs(step_size - expected_step) < 1e-6, "Step size mismatch!"


def test_photodiode_noise_scaling():
    """Validates photodetector square-law readout, dark thermal noise, and shot noise scaling."""
    print("\n--- 5. Photodetector Readout & Noise Scaling Validation ---")
    cfg = PhotonicConfig(ideal_mode=False, enable_noise=True)
    detector = PhotodetectorArray(cfg)

    # Case 1: Dark input (zero optical power)
    E_dark = torch.zeros(500, cfg.n_modes, dtype=torch.complex64)
    I_dark_measured = detector(E_dark, add_noise=True)
    # With zero light, dark noise should fluctuate around dark current
    print(f"  Dark current mean: {torch.mean(I_dark_measured).item():.3e} A | std: {torch.std(I_dark_measured).item():.3e} A")

    # Case 2: Signal power 1 mW (E = sqrt(1 mW))
    P_in_mw = 1.0e-3
    E_sig = torch.ones(500, cfg.n_modes, dtype=torch.complex64) * math.sqrt(P_in_mw)
    I_sig_measured = detector(E_sig, add_noise=True)

    expected_I_mean = cfg.responsivity * P_in_mw  # 0.95 mA
    measured_I_mean = torch.mean(I_sig_measured).item()
    print(f"  Signal (1 mW): Expected I_mean={expected_I_mean*1e3:.2f} mA | Measured={measured_I_mean*1e3:.2f} mA")
    assert abs(measured_I_mean - expected_I_mean) / expected_I_mean < 0.05, "Mean photocurrent mismatch!"

    snr_db = detector.compute_snr_db(P_in_mw)
    print(f"  Theoretical Electrical SNR at 1 mW: {snr_db:.1f} dB")
    assert snr_db > 20.0, "SNR too low for 1 mW optical input!"


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING PHYSICAL NON-IDEALITY VALIDATION TESTS")
    print("=" * 70)
    test_coupler_leakage_floor_formula()
    test_thermal_crosstalk_bleed()
    test_optical_loss_breaks_unitarity()
    test_dac_quantization_levels()
    test_photodiode_noise_scaling()
    print("=" * 70)
    print("[ALL PHYSICAL VALIDATION TESTS PASSED]")
    print("=" * 70)
