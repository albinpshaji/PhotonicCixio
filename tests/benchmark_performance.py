"""
GPU High-Throughput Benchmarking Suite.
Measures forward propagation latency, batch throughput, and execution scaling
across various mode counts (N = 4, 8, 16, 32) and batch sizes (B = 1 to 1024)
on the local NVIDIA GPU.
"""

import sys
import os
import time
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PhotonicConfig
from src.models.digital_twin import PhotonicMeshDigitalTwin


def benchmark_simulator(
    n_modes_list=[4, 8, 16, 32],
    batch_sizes=[1, 16, 64, 256, 1024],
    n_warmup=10,
    n_reps=50
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 85)
    print(f"PHOTONIC DIGITAL TWIN EXECUTION BENCHMARK ({device_name.upper()} - {device})")
    print("=" * 85)

    results = []

    for n_modes in n_modes_list:
        cfg = PhotonicConfig(n_modes=n_modes, ideal_mode=False, device=device)
        twin = PhotonicMeshDigitalTwin(cfg)
        print(f"\n[Benchmarking N={n_modes} Modes | M={twin.total_mzis} MZIs | Depth={twin.num_layers} Layers]")
        print("-" * 85)
        print(f"{'Batch Size':>12} | {'Field Latency':>15} | {'Field Throughput':>18} | {'Matrix Latency':>15} | {'Matrix Throughput':>18}")
        print("-" * 85)

        for B in batch_sizes:
            theta = torch.rand(B, twin.total_mzis, device=device) * math.pi
            phi = torch.rand(B, twin.total_mzis, device=device) * (2.0 * math.pi)
            E_in = (torch.randn(B, n_modes, device=device) + 
                    1.0j * torch.randn(B, n_modes, device=device)).to(cfg.complex_dtype)

            # --- Benchmark 1: Field Vector Propagation ---
            # Warmup
            for _ in range(n_warmup):
                _ = twin.propagate_field(E_in, theta, phi)
            if device == "cuda":
                torch.cuda.synchronize()

            start_t = time.perf_counter()
            for _ in range(n_reps):
                _ = twin.propagate_field(E_in, theta, phi)
            if device == "cuda":
                torch.cuda.synchronize()
            field_total_s = time.perf_counter() - start_t
            field_latency_ms = (field_total_s / n_reps) * 1000.0
            field_throughput = (B * n_reps) / field_total_s

            # --- Benchmark 2: Matrix Synthesis ---
            # Warmup
            for _ in range(n_warmup):
                _ = twin.compute_transfer_matrix(theta, phi)
            if device == "cuda":
                torch.cuda.synchronize()

            start_t = time.perf_counter()
            for _ in range(n_reps):
                _ = twin.compute_transfer_matrix(theta, phi)
            if device == "cuda":
                torch.cuda.synchronize()
            mat_total_s = time.perf_counter() - start_t
            mat_latency_ms = (mat_total_s / n_reps) * 1000.0
            mat_throughput = (B * n_reps) / mat_total_s

            print(f"{B:>12d} | {field_latency_ms:>12.3f} ms | {field_throughput:>14,.0f} vec/s | "
                  f"{mat_latency_ms:>12.3f} ms | {mat_throughput:>14,.0f} mat/s")

            results.append({
                "n_modes": n_modes,
                "batch_size": B,
                "field_latency_ms": field_latency_ms,
                "field_throughput": field_throughput,
                "mat_latency_ms": mat_latency_ms,
                "mat_throughput": mat_throughput
            })

    print("=" * 85)
    print("[BENCHMARK COMPLETED SUCCESSFULLY]")
    print("=" * 85)
    return results


if __name__ == "__main__":
    benchmark_simulator()
