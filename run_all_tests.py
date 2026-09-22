#!/usr/bin/env python3
"""
Master Verification and Benchmark Runner for the Silicon Photonic Digital Twin.
Executes:
1. Mathematical Unitarity & Energy Conservation Tests
2. Autograd & Backward Gradient Flow Tests
3. Physical Non-Ideality Validation Tests
4. GPU Performance & Scalability Benchmarks
"""

import sys
import os
import subprocess
import time

DIR = os.path.dirname(os.path.abspath(__file__))

PYTHON_EXEC = None
for candidate in [
    os.path.join(DIR, ".venv", "bin", "python"),
    os.path.join(DIR, "..", ".venv", "bin", "python"),
]:
    if os.path.exists(candidate):
        PYTHON_EXEC = os.path.abspath(candidate)
        break

if not PYTHON_EXEC:
    PYTHON_EXEC = sys.executable

TEST_SCRIPTS = [
    ("Unitarity & Power Conservation", "tests/test_unitarity.py"),
    ("Autograd & Gradient Flow (STE)", "tests/test_gradient_flow.py"),
    ("Physical Non-Idealities Validation", "tests/test_physics_validation.py"),
    ("Wafer Spatial Correlation (GRF)", "tests/test_wafer_correlation.py"),
    ("Electro-Thermal Dynamics & Package", "tests/test_advanced_thermal.py"),
    ("Broadband Wavelength Dispersion", "tests/test_dispersion.py"),
    ("Waveguide Routing & Crossings", "tests/test_layout_routing.py"),
    ("Balanced Readout & ADC Subsystem", "tests/test_readout_subsystem.py"),
    ("2D Finite-Difference Thermal Solver", "tests/test_thermal_2d.py"),
    ("Silicon Optical Nonlinearities (TPA/FCA/SPM)", "tests/test_nonlinear_optics.py"),
    ("Coherent Backreflection & Fabry-Perot", "tests/test_backreflection.py"),
    ("Waveguide Bend Loss & Mode Mismatch", "tests/test_bend_loss.py"),
    ("1/f Flicker Noise & Temporal Aging", "tests/test_temporal_noise.py"),
    ("Full Jones Vector Polarization & PDL", "tests/test_polarization.py"),
    ("Phase 2 Full Physics Pipeline Integration", "tests/test_phase2_integration.py"),
    ("Physics Losses & Hardware Evaluations", "tests/test_advanced_evaluations_and_losses.py"),
    ("Field vs Matrix Equivalence", "tests/test_matrix_field_equivalence.py"),
    ("Photonic Parameter Calibration", "tests/test_calibration_fitting.py"),
    ("GPU Execution Performance Benchmark", "tests/benchmark_performance.py")
]


def run_suite():
    print("=" * 80)
    print("SILICON PHOTONIC DIGITAL TWIN - COMPLETE VERIFICATION SUITE")
    print(f"Target Environment: {PYTHON_EXEC}")
    print(f"Working Directory:  {DIR}")
    print("=" * 80)

    total_start = time.perf_counter()
    summary = []
    env = os.environ.copy()
    env["PYTHONPATH"] = DIR + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    for name, script in TEST_SCRIPTS:
        script_path = os.path.join(DIR, script)
        print(f"\n[RUNNING] {name} ({script})...")
        print("-" * 80)

        start_t = time.perf_counter()
        result = subprocess.run([PYTHON_EXEC, script_path], cwd=DIR, env=env, capture_output=True, text=True)
        elapsed = time.perf_counter() - start_t

        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        status = "PASSED" if result.returncode == 0 else "FAILED"
        print(f"[{status}] {name} in {elapsed:.2f} seconds")
        summary.append((name, status, elapsed))

    total_elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 80)
    print("EXECUTIVE VERIFICATION SUMMARY")
    print("=" * 80)
    all_passed = True
    for name, status, elapsed in summary:
        color = "PASS" if status == "PASSED" else "FAIL"
        print(f"  [{color}] {name:<45} | Elapsed: {elapsed:>6.2f}s")
        if status != "PASSED":
            all_passed = False

    print("-" * 80)
    print(f"Total Execution Time: {total_elapsed:.2f} seconds")
    if all_passed:
        print("[FINAL STATUS] ALL VERIFICATION TESTS AND BENCHMARKS PASSED PERFECTLY!")
        if "--plot" in sys.argv:
            print("\n" + "=" * 80)
            print("[GENERATING GRAPHICAL TEST DIAGNOSTICS & PLOTS]")
            print("=" * 80)
            plot_script = os.path.join(DIR, "tests", "generate_test_plots.py")
            subprocess.run([PYTHON_EXEC, plot_script], cwd=DIR, env=env)
        return 0
    else:
        print("[FINAL STATUS] ONE OR MORE VERIFICATION CHECKS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(run_suite())
