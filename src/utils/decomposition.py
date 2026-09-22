"""
Exact Analytical Clements Planar Unitary Matrix Decomposition.
Mathematical reference:
- Clements et al. (2016), Optica 3(12), 1460-1466
Decomposes an arbitrary unitary matrix U in U(N) into N*(N-1)/2 MZI angles
and a final diagonal phase screen.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict


def clements_decompose_np(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs the Clements decomposition on an N x N unitary matrix U using NumPy.

    Returns:
        thetas: Array of length M = N*(N-1)/2 containing internal MZI phases in [0, pi].
        phis: Array of length M containing external MZI phases in [0, 2*pi).
        diag_phases: Array of length N containing residual diagonal phases in [0, 2*pi).
    """
    U_curr = np.array(U, dtype=np.complex128, copy=True)
    N = U_curr.shape[0]
    M = (N * (N - 1)) // 2

    # Storage for MZI transformations
    # Each record: (col, p, q, theta, phi, is_left)
    operations = []

    for col in range(N - 1):
        if col % 2 == 0:
            # Even step: null from bottom-left
            for row in range(col + 1):
                p = col - row
                q = p + 1
                # Null element U_curr[N - 1 - row, p] using column p and q
                target_r = N - 1 - row
                u1 = U_curr[target_r, p]
                u2 = U_curr[target_r, q]

                if np.abs(u1) < 1e-12 and np.abs(u2) < 1e-12:
                    theta = 0.0
                    phi = 0.0
                else:
                    phi = np.angle(u2) - np.angle(u1) - np.pi
                    theta = 2.0 * np.arctan2(np.abs(u1), np.abs(u2))

                # Normalize phi to [0, 2*pi)
                phi = float(np.mod(phi, 2.0 * np.pi))
                theta = float(np.mod(theta, 2.0 * np.pi))
                if theta > np.pi:
                    theta = 2.0 * np.pi - theta
                    phi = float(np.mod(phi + np.pi, 2.0 * np.pi))

                # Construct 2x2 inverse rotation on columns
                # T = i * exp(i*theta/2) * [ [ exp(i*phi)*sin(theta/2),  cos(theta/2)  ],
                #                            [ exp(i*phi)*cos(theta/2), -sin(theta/2) ] ]
                st = np.sin(theta / 2.0)
                ct = np.cos(theta / 2.0)
                ep = np.exp(1.0j * phi)
                gp = 1.0j * np.exp(1.0j * theta / 2.0)
                T = gp * np.array([[ep * st, ct], [ep * ct, -st]], dtype=np.complex128)

                # Right multiply: U_curr = U_curr @ T_inv = U_curr @ T^dagger
                U_curr[:, [p, q]] = U_curr[:, [p, q]] @ np.conjugate(T.T)
                operations.append((col, p, q, theta, phi, True))
        else:
            # Odd step: null from top-right
            for row in range(col + 1):
                p = N - 1 - (col - row)
                q = p + 1
                target_c = row
                u1 = U_curr[p - 1, target_c]
                u2 = U_curr[p, target_c]

                if np.abs(u1) < 1e-12 and np.abs(u2) < 1e-12:
                    theta = 0.0
                    phi = 0.0
                else:
                    phi = np.angle(u1) - np.angle(u2)
                    theta = 2.0 * np.arctan2(np.abs(u2), np.abs(u1))

                phi = float(np.mod(phi, 2.0 * np.pi))
                theta = float(np.mod(theta, 2.0 * np.pi))
                if theta > np.pi:
                    theta = 2.0 * np.pi - theta
                    phi = float(np.mod(phi + np.pi, 2.0 * np.pi))

                st = np.sin(theta / 2.0)
                ct = np.cos(theta / 2.0)
                ep = np.exp(1.0j * phi)
                gp = 1.0j * np.exp(1.0j * theta / 2.0)
                T = gp * np.array([[ep * st, ct], [ep * ct, -st]], dtype=np.complex128)

                # Left multiply: U_curr = T @ U_curr
                U_curr[[p - 1, p], :] = T @ U_curr[[p - 1, p], :]
                operations.append((col, p - 1, p, theta, phi, False))

    # Extract final diagonal phase matrix
    diag_phases = np.angle(np.diag(U_curr))
    thetas = np.array([op[3] for op in operations], dtype=np.float32)
    phis = np.array([op[4] for op in operations], dtype=np.float32)

    return thetas, phis, diag_phases


def generate_random_unitary(N: int, device: str = "cpu") -> torch.Tensor:
    """
    Generates a uniformly distributed random N x N unitary matrix from the Haar measure
    using the QR decomposition of a complex standard Gaussian matrix.
    """
    X = (torch.randn(N, N, device=device) + 1.0j * torch.randn(N, N, device=device)) / math.sqrt(2.0)
    Q, R = torch.linalg.qr(X)
    # Ensure Haar distribution by adjusting phase of R's diagonal
    diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
    phase = diag_R / torch.abs(diag_R)
    U = Q * phase.unsqueeze(-2)
    return U


import math
