"""
Exact Analytical Clements Planar Unitary Matrix Decomposition.
Mathematical references:
- Clements et al. (2016), Optica 3(12), 1460-1466: "Optimal design for universal multiport interferometers"
- Bandyopadhyay et al. (2021), arXiv:2103.04993: "Hardware Error Correction for Silicon Photonic Meshes"

Decomposes an arbitrary unitary matrix U in U(N) into:
1. M = N*(N-1)/2 MZI internal phases theta in [0, pi] and external phases phi in [0, 2*pi).
2. An N-element output diagonal phase screen diag_phases in [0, 2*pi).
The resulting sequence of MZIs strictly matches the layered column-by-column topology
of the PhotonicMeshDigitalTwin.
"""

import math
import numpy as np
import torch
from typing import Tuple, List, Optional


def _mzi_matrix_np(theta: float, phi: float) -> np.ndarray:
    """Computes standard 2x2 Clements MZI transfer matrix."""
    st = np.sin(theta / 2.0)
    ct = np.cos(theta / 2.0)
    ep = np.exp(1.0j * phi)
    gp = 1.0j * np.exp(1.0j * theta / 2.0)
    return gp * np.array([[ep * st, ct], [ep * ct, -st]], dtype=np.complex128)


def clements_decompose_np(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs the canonical Clements decomposition on an N x N unitary matrix U using NumPy (complex128).
    
    Decomposes U such that:
        U = diag(exp(i * diag_phases)) @ U_{layers-1} @ ... @ U_0
    where the order of MZIs in (thetas, phis) exactly matches the digital twin's layer_info.

    Args:
        U: Unitary matrix of shape (N, N), complex-valued.

    Returns:
        thetas: Array of length M = N*(N-1)/2 containing internal phases in [0, pi].
        phis: Array of length M containing external phases in [0, 2*pi).
        diag_phases: Array of length N containing output diagonal phase screen in [0, 2*pi).
    """
    U_curr = np.array(U, dtype=np.complex128, copy=True)
    N = U_curr.shape[0]
    M = (N * (N - 1)) // 2

    if N == 1:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), np.angle(np.diag(U_curr))

    right_ops = []  # list of (m, n, theta, phi)
    left_ops = []   # list of (m, n, theta, phi)

    # Clements nulling scan (alternating outside-in diagonals)
    for i in range(1, N):
        if i % 2 == 1:
            # Odd step: null bottom-left elements via right-multiplications
            for j in range(0, i):
                m = i - j - 1
                n = m + 1
                r = N - 1 - j
                u1 = U_curr[r, m]
                u2 = U_curr[r, n]
                if np.abs(u2) < 1e-15:
                    theta = 0.0
                    phi = 0.0
                elif np.abs(u1) < 1e-15:
                    theta = np.pi
                    phi = 0.0
                else:
                    theta = 2.0 * np.arctan(np.abs(u2) / np.abs(u1))
                    phi = float((np.angle(u1) - np.angle(u2) + np.pi) % (2.0 * np.pi))

                theta = float(np.mod(theta, 2.0 * np.pi))
                if theta > np.pi:
                    theta = 2.0 * np.pi - theta
                    phi = float((phi + np.pi) % (2.0 * np.pi))

                T = _mzi_matrix_np(theta, phi)
                T_inv = np.conjugate(T.T)
                U_curr[:, [m, n]] = U_curr[:, [m, n]] @ T_inv
                right_ops.append((m, n, theta, phi))
        else:
            # Even step: null top-right elements via left-multiplications
            for j in range(1, i + 1):
                m = N + j - i - 2
                n = m + 1
                r = n
                c = j - 1
                u1 = U_curr[m, c]
                u2 = U_curr[n, c]
                if np.abs(u1) < 1e-15:
                    theta = 0.0
                    phi = 0.0
                elif np.abs(u2) < 1e-15:
                    theta = np.pi
                    phi = 0.0
                else:
                    theta = 2.0 * np.arctan(np.abs(u1) / np.abs(u2))
                    phi = float((np.angle(u2) - np.angle(u1)) % (2.0 * np.pi))

                theta = float(np.mod(theta, 2.0 * np.pi))
                if theta > np.pi:
                    theta = 2.0 * np.pi - theta
                    phi = float((phi + np.pi) % (2.0 * np.pi))

                T = _mzi_matrix_np(theta, phi)
                U_curr[[m, n], :] = T @ U_curr[[m, n], :]
                left_ops.append((m, n, theta, phi))

    # Remaining diagonal phase screen
    diag_phases = np.angle(np.diag(U_curr))

    # Commute left_ops through diagonal phase screen D:
    # U = (L_1^-1 @ ... @ L_k^-1) @ D @ (R_m @ ... @ R_1)
    # Using identity: T_{m,n}^-1(theta, phi) @ D = D' @ T_{m,n}(theta, phi')
    current_diag = diag_phases.copy()
    commuted_left_ops = []
    for (m, n, theta, phi) in reversed(left_ops):
        alpha = current_diag[m]
        beta = current_diag[n]
        theta_p = theta
        phi_p = float((alpha - beta) % (2.0 * np.pi))
        alpha_p = float((beta - phi - theta - np.pi) % (2.0 * np.pi))
        beta_p = float((beta - theta - np.pi) % (2.0 * np.pi))
        current_diag[m] = alpha_p
        current_diag[n] = beta_p
        commuted_left_ops.append((m, n, theta_p, phi_p))

    # Full forward optical wave cascade from input to output:
    # First right_ops, followed by commuted_left_ops
    ops_input_to_output = list(right_ops) + list(commuted_left_ops)

    # Build target digital twin topology pairs
    target_pairs = []
    for col in range(N):
        if col % 2 == 0:
            pairs = [(2 * k, 2 * k + 1) for k in range(N // 2)]
        else:
            pairs = [(2 * k + 1, 2 * k + 2) for k in range((N - 2) // 2)]
        target_pairs.extend(pairs)

    # Commute disjoint adjacent gates to match the digital twin's canonical column schedule
    curr_ops = list(ops_input_to_output)
    for i in range(len(target_pairs)):
        desired = target_pairs[i]
        found_idx = -1
        for j in range(i, len(curr_ops)):
            if (curr_ops[j][0], curr_ops[j][1]) == desired:
                found_idx = j
                break
        assert found_idx != -1, f"Clements topology mismatch: could not find pair {desired}"
        for k in range(found_idx, i, -1):
            pair_k = (curr_ops[k][0], curr_ops[k][1])
            pair_prev = (curr_ops[k - 1][0], curr_ops[k - 1][1])
            assert len(set(pair_k) & set(pair_prev)) == 0, (
                f"Cannot commute overlapping gates {pair_k} and {pair_prev}"
            )
            curr_ops[k], curr_ops[k - 1] = curr_ops[k - 1], curr_ops[k]

    thetas = np.array([op[2] for op in curr_ops], dtype=np.float64)
    phis = np.array([op[3] for op in curr_ops], dtype=np.float64)
    final_diag = np.mod(current_diag, 2.0 * np.pi)

    return thetas, phis, final_diag


def clements_reconstruct_torch(
    thetas: torch.Tensor,
    phis: torch.Tensor,
    diag_phases: Optional[torch.Tensor] = None,
    n_modes: Optional[int] = None,
    complex_dtype: torch.dtype = torch.complex128
) -> torch.Tensor:
    """
    Analytically reconstructs the unitary matrix U in PyTorch from Clements phase parameters.

    Args:
        thetas: Internal phase shifts of shape (..., M).
        phis: External phase shifts of shape (..., M).
        diag_phases: Output diagonal phase screen of shape (..., N).
        n_modes: Optical channel dimension N. Inferred from M if None.
        complex_dtype: Complex floating point dtype.

    Returns:
        U: Unitary matrix tensor of shape (..., N, N).
    """
    M = thetas.shape[-1]
    if n_modes is None:
        # Solve M = N*(N-1)/2 => N^2 - N - 2M = 0 => N = (1 + sqrt(1 + 8M)) / 2
        n_modes = int(round((1.0 + math.sqrt(1.0 + 8.0 * M)) / 2.0))

    device = thetas.device
    batch_shape = thetas.shape[:-1]

    # Build Clements column structure
    layer_info = []
    global_idx = 0
    for col in range(n_modes):
        if col % 2 == 0:
            pairs = [(2 * k, 2 * k + 1) for k in range(n_modes // 2)]
        else:
            pairs = [(2 * k + 1, 2 * k + 2) for k in range((n_modes - 2) // 2)]
        mzi_indices = list(range(global_idx, global_idx + len(pairs)))
        global_idx += len(pairs)
        p_idx = torch.tensor([p for p, q in pairs], dtype=torch.long, device=device)
        q_idx = torch.tensor([q for p, q in pairs], dtype=torch.long, device=device)
        layer_info.append({
            "p_indices": p_idx,
            "q_indices": q_idx,
            "mzi_indices": torch.tensor(mzi_indices, dtype=torch.long, device=device)
        })

    # Initialize U_cumul = Identity
    eye = torch.eye(n_modes, device=device, dtype=complex_dtype)
    if batch_shape:
        U_cumul = eye.repeat(*batch_shape, 1, 1)
    else:
        U_cumul = eye.clone()

    half_theta = 0.5 * thetas.to(dtype=torch.float64)
    sin_half = torch.sin(half_theta).to(complex_dtype)
    cos_half = torch.cos(half_theta).to(complex_dtype)
    exp_i_phi = torch.exp(1.0j * phis.to(complex_dtype))
    global_phase = 1.0j * torch.exp(1.0j * half_theta.to(complex_dtype))

    t00 = global_phase * exp_i_phi * sin_half
    t01 = global_phase * cos_half
    t10 = global_phase * exp_i_phi * cos_half
    t11 = -global_phase * sin_half

    for layer in layer_info:
        p_idx = layer["p_indices"]
        q_idx = layer["q_indices"]
        mzi_idx = layer["mzi_indices"]

        if batch_shape:
            U_col = eye.repeat(*batch_shape, 1, 1).clone()
        else:
            U_col = eye.clone()

        U_col[..., p_idx, p_idx] = t00[..., mzi_idx]
        U_col[..., p_idx, q_idx] = t01[..., mzi_idx]
        U_col[..., q_idx, p_idx] = t10[..., mzi_idx]
        U_col[..., q_idx, q_idx] = t11[..., mzi_idx]

        U_cumul = torch.matmul(U_col, U_cumul)

    # Apply output diagonal phase screen if provided
    if diag_phases is not None:
        D = torch.diag_embed(torch.exp(1.0j * diag_phases.to(complex_dtype)))
        U_cumul = torch.matmul(D, U_cumul)

    return U_cumul


def generate_random_unitary(
    N: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.complex128
) -> torch.Tensor:
    """
    Generates a uniformly distributed random N x N unitary matrix from the Haar measure
    using the QR decomposition of a complex standard Gaussian matrix.
    """
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    X = (torch.randn(N, N, device=device, dtype=real_dtype) +
         1.0j * torch.randn(N, N, device=device, dtype=real_dtype)) / math.sqrt(2.0)
    Q, R = torch.linalg.qr(X)
    diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
    phase = diag_R / torch.abs(diag_R)
    U = Q * phase.unsqueeze(-2)
    return U.to(dtype=dtype)
