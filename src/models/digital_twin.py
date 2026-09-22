"""
Differentiable Digital Twin of an N-Mode Coherent Silicon Photonic Processor.
Clements Planar Mesh Architecture with Complete SOI Physics.
Mathematical references:
- Clements et al. (2016), Optica 3(12), 1460-1466
- Fang et al. (2019), Optics Express 27(10), 14009-14029
- Zhu et al. (2020), IEEE/ACM ICCAD 2020, pp. 1-7
- Bogaerts et al. (2020), Nature 586, 207-216
- Hamerly et al. (2019), Phys. Rev. X 9, 021023
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Union

from src.config import PhotonicConfig
from src.physics.mzi import mzi_transfer_matrix
from src.physics.thermal import ThermalCrosstalkModel
from src.physics.spatial_wafer import SpatialWaferMap
from src.physics.routing_loss import ClementsPhysicalRouting
from src.physics.loss_model import OpticalLossModel
from src.physics.photodiode import PhotodetectorArray
from src.nn.quantizer import DACQuantizer
from src.physics.backreflection import FabryPerotBackreflection
from src.physics.nonlinear_optics import SiliconNonlinearOptics
from src.physics.bend_loss import WaveguideBendLossModel
from src.physics.polarization import JonesPolarizationModel


class PhotonicMeshDigitalTwin(nn.Module):
    """
    GPU-accelerated, fully differentiable digital twin simulator for an N-mode
    coherent optical matrix multiplier (Clements planar MZI mesh).

    Foundry-Grade Realism:
    1. Spatially correlated 2D wafer maps (Gaussian Random Fields with Cholesky factorization)
    2. Directional coupler split deviations (kappa = 0.5 +/- eps) & intrinsic arm phase bias
    3. Broadband optical dispersion (lambda-dependent kappa and propagation phase delay)
    4. Multi-heater electro-thermal Joule dissipation with TCR and packaging substrate heating
    5. Planar waveguide routing (crossings loss & optical crosstalk, path-length propagation skew)
    6. Finite DAC quantization with DNL/INL non-linearities and analog phase jitter (STE)
    7. Balanced homodyne photodetection (BPD), TIA saturation non-linearity, and ADC digitization
    """
    def __init__(self, config: Optional[PhotonicConfig] = None):
        super().__init__()
        if config is None:
            config = PhotonicConfig()
        self.config = config
        self.n_modes = config.n_modes
        self.total_mzis = config.total_mzis
        self.num_layers = config.num_layers
        self.ideal_mode = config.ideal_mode
        self.device = config.device
        self.complex_dtype = config.complex_dtype

        # ---------------- 1. Thermal Crosstalk & Electro-Thermal Model ----------------
        self.thermal_model = ThermalCrosstalkModel(config)

        # ---------------- 2. Spatially Correlated Wafer Map ----------------
        self.spatial_wafer = SpatialWaferMap(self.thermal_model.coords, config)
        self.register_buffer("coupler_eps1", self.spatial_wafer.coupler_eps1)
        self.register_buffer("coupler_eps2", self.spatial_wafer.coupler_eps2)
        self.register_buffer("phi_intrinsic", self.spatial_wafer.phi_intrinsic)

        # ---------------- 3. Physical Waveguide Routing & Crossings ----------------
        self.routing_model = ClementsPhysicalRouting(config)

        # ---------------- 4. DAC Phase Quantizers (STE, DNL/INL, Jitter) ----------------
        self.dac_theta = DACQuantizer(
            bits=config.dac_bits,
            val_min=0.0,
            val_max=math.pi,
            enabled=config.enable_quantization and not config.ideal_mode,
            dnl_lsb=config.dac_dnl_lsb,
            inl_lsb=config.dac_inl_lsb,
            phase_jitter_std=config.phase_jitter_std,
            enable_phase_jitter=config.enable_phase_jitter and not config.ideal_mode
        )
        self.dac_phi = DACQuantizer(
            bits=config.dac_bits,
            val_min=0.0,
            val_max=2.0 * math.pi,
            enabled=config.enable_quantization and not config.ideal_mode,
            dnl_lsb=config.dac_dnl_lsb,
            inl_lsb=config.dac_inl_lsb,
            phase_jitter_std=config.phase_jitter_std,
            enable_phase_jitter=config.enable_phase_jitter and not config.ideal_mode
        )
        self.dac_phi_diag = DACQuantizer(
            bits=getattr(config, "dac_phi_diag_bits", config.dac_bits),
            val_min=0.0,
            val_max=2.0 * math.pi,
            enabled=config.enable_quantization and not config.ideal_mode,
            dnl_lsb=config.dac_dnl_lsb,
            inl_lsb=config.dac_inl_lsb,
            phase_jitter_std=config.phase_jitter_std,
            enable_phase_jitter=config.enable_phase_jitter and not config.ideal_mode
        )

        # ---------------- 5. Progressive Optical Loss Model ----------------
        self.loss_model = OpticalLossModel(config)

        # ---------------- 6. Photodetector Readout & Noise ----------------
        self.photodetector = PhotodetectorArray(config)

        # ---------------- 7. Mesh Topology Structure ----------------
        self._build_clements_topology()

        # ---------------- 8. Second-Order Physics Extensions ----------------
        self.backreflection_model = FabryPerotBackreflection(config)
        self.nonlinear_model = SiliconNonlinearOptics(config)
        self.bend_loss_model = WaveguideBendLossModel(config)
        self.polarization_model = JonesPolarizationModel(config)

        # Move buffers to configured device
        self.to(self.device)

    def _build_clements_topology(self):
        """
        Precomputes the layered Clements planar mesh structure.
        For each column l in [0, N-1]:
        - Identifies active mode pairs (p, q)
        - Maps global MZI linear indices m in [0, M-1]
        """
        self.layer_info = []
        global_mzi_idx = 0

        for col in range(self.n_modes):
            if col % 2 == 0:
                # Even column: pairs (0, 1), (2, 3), ..., (N-2, N-1)
                pairs = [(2 * k, 2 * k + 1) for k in range(self.n_modes // 2)]
            else:
                # Odd column: pairs (1, 2), (3, 4), ..., (N-3, N-2)
                pairs = [(2 * k + 1, 2 * k + 2) for k in range((self.n_modes - 2) // 2)]

            mzi_indices = list(range(global_mzi_idx, global_mzi_idx + len(pairs)))
            global_mzi_idx += len(pairs)

            p_indices = torch.tensor([p for p, q in pairs], dtype=torch.long)
            q_indices = torch.tensor([q for p, q in pairs], dtype=torch.long)
            mzi_idx_tensor = torch.tensor(mzi_indices, dtype=torch.long)

            self.layer_info.append({
                "col": col,
                "p_indices": p_indices,
                "q_indices": q_indices,
                "mzi_indices": mzi_idx_tensor,
                "num_mzis": len(pairs)
            })

        assert global_mzi_idx == self.total_mzis, (
            f"Expected {self.total_mzis} total MZIs, but configured {global_mzi_idx}"
        )

    def set_ideal_mode(self, ideal: bool = True):
        """Dynamically toggles between ideal mathematical mode and realistic physical mode."""
        self.ideal_mode = ideal
        self.config.ideal_mode = ideal
        self.thermal_model.enabled = not ideal and self.config.enable_thermal_crosstalk
        self.thermal_model.enable_tcr = not ideal and self.config.enable_tcr
        self.routing_model.enabled = not ideal and self.config.enable_physical_routing
        self.loss_model.enabled = not ideal and self.config.enable_loss
        self.photodetector.enabled = not ideal and self.config.enable_noise
        self.photodetector.enable_adc = not ideal and self.config.enable_adc
        self.dac_theta.enabled = not ideal and self.config.enable_quantization
        self.dac_theta.enable_phase_jitter = not ideal and self.config.enable_phase_jitter
        self.dac_phi.enabled = not ideal and self.config.enable_quantization
        self.dac_phi.enable_phase_jitter = not ideal and self.config.enable_phase_jitter
        self.dac_phi_diag.enabled = not ideal and self.config.enable_quantization
        self.dac_phi_diag.enable_phase_jitter = not ideal and self.config.enable_phase_jitter

    def prepare_physical_phases(
        self,
        theta: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        diag_phases: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Processes command phases through DAC quantization, DNL/INL, jitter,
        and non-local thermal diffusion with TCR electro-thermal feedback.
        """
        if phi is None:
            phi = torch.zeros_like(theta)

        if self.ideal_mode:
            return theta, phi, diag_phases

        # 1. DAC Discretization with non-linearity and phase jitter
        theta_quant = self.dac_theta(theta)
        phi_quant = self.dac_phi(phi)
        diag_quant = self.dac_phi_diag(diag_phases) if diag_phases is not None else None

        # 2. Electro-Thermal Diffusion Crosstalk on internal phases
        theta_phys = self.thermal_model(theta_quant)
        phi_phys = phi_quant
        diag_phys = diag_quant

        return theta_phys, phi_phys, diag_phys

    def compute_transfer_matrix(
        self,
        theta: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        diag_phases: Optional[torch.Tensor] = None,
        wavelength: Optional[Union[float, torch.Tensor]] = None,
        override_eps1: Optional[torch.Tensor] = None,
        override_eps2: Optional[torch.Tensor] = None,
        override_phi_intrinsic: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Synthesizes the full cumulative N x N optical transfer matrix T_PIC.

        Args:
            theta: Internal phase tensor of shape (..., M).
            phi: External phase tensor of shape (..., M).
            diag_phases: Optional output diagonal phase screen tensor of shape (..., N).
            wavelength: Operating optical wavelength in meters.
            override_eps1: Optional tensor overriding coupler_eps1 (for parameter estimation).
            override_eps2: Optional tensor overriding coupler_eps2 (for parameter estimation).
            override_phi_intrinsic: Optional tensor overriding phi_intrinsic.

        Returns:
            T_PIC: Complex matrix tensor of shape (..., N, N).
        """
        batch_shape = theta.shape[:-1]
        device = theta.device

        # Process through DAC and thermal non-idealities
        theta_phys, phi_phys, diag_phys = self.prepare_physical_phases(theta, phi, diag_phases)

        # Initialize cumulative mesh transfer matrix as Identity: shape (..., N, N)
        eye_N = torch.eye(self.n_modes, device=device, dtype=self.complex_dtype)
        if batch_shape:
            T_cumul = eye_N.repeat(*batch_shape, 1, 1)
        else:
            T_cumul = eye_N.clone()

        # Step through Clements columns sequentially
        for layer in self.layer_info:
            col = layer["col"]
            mzi_idx = layer["mzi_indices"].to(device)
            p_idx = layer["p_indices"].to(device)
            q_idx = layer["q_indices"].to(device)

            # Extract phase parameters for MZIs in this column
            th_col = theta_phys[..., mzi_idx]  # (..., num_mzis)
            ph_col = phi_phys[..., mzi_idx]    # (..., num_mzis)

            if override_eps1 is not None and override_eps2 is not None:
                eps1_col = override_eps1.to(device)[mzi_idx]
                eps2_col = override_eps2.to(device)[mzi_idx]
                phi_int_col = override_phi_intrinsic.to(device)[mzi_idx] if override_phi_intrinsic is not None else None
            elif self.ideal_mode or not self.config.enable_coupler_errors:
                eps1_col = None
                eps2_col = None
                phi_int_col = None
            else:
                eps1_col = self.coupler_eps1.to(device)[mzi_idx]
                eps2_col = self.coupler_eps2.to(device)[mzi_idx]
                phi_int_col = self.phi_intrinsic.to(device)[mzi_idx]

            # Compute 2x2 MZI transfer matrices: (..., num_mzis, 2, 2)
            T_mzis = mzi_transfer_matrix(
                theta=th_col,
                phi=ph_col,
                epsilon1=eps1_col,
                epsilon2=eps2_col,
                phi_intrinsic=phi_int_col,
                wavelength=wavelength if self.config.enable_dispersion else None,
                excess_loss_db=self.config.coupler_excess_loss_db if not self.ideal_mode else 0.0,
                ideal_mode=self.ideal_mode,
                complex_dtype=self.complex_dtype
            )

            # Construct column matrix U_col of shape (..., N, N)
            if batch_shape:
                U_col = eye_N.repeat(*batch_shape, 1, 1).clone()
            else:
                U_col = eye_N.clone()

            # Insert 2x2 blocks into U_col
            U_col[..., p_idx, p_idx] = T_mzis[..., 0, 0]
            U_col[..., p_idx, q_idx] = T_mzis[..., 0, 1]
            U_col[..., q_idx, p_idx] = T_mzis[..., 1, 0]
            U_col[..., q_idx, q_idx] = T_mzis[..., 1, 1]

            # Apply progressive stage insertion loss across column stage (Clements balanced loss)
            if self.config.enable_loss and not self.ideal_mode:
                a_l = self.loss_model.get_layer_attenuation(col).to(device=device, dtype=self.complex_dtype)
                U_col = U_col * a_l

            # Forward cascade multiplication: T_cumul = U_col @ T_cumul
            T_cumul = torch.matmul(U_col, T_cumul)

            # Apply inter-column physical crossings if routing is enabled
            if self.config.enable_physical_routing and not self.ideal_mode and col % 2 == 0:
                p_cr = self.routing_model.crossing_p.to(device)
                q_cr = self.routing_model.crossing_q.to(device)
                if len(p_cr) > 0:
                    t_b = self.routing_model.t_bar.to(device=device, dtype=self.complex_dtype)
                    t_c = (1.0j * self.routing_model.t_cross).to(device=device, dtype=self.complex_dtype)
                    C_cross = eye_N.clone()
                    C_cross[p_cr, p_cr] = t_b
                    C_cross[p_cr, q_cr] = t_c
                    C_cross[q_cr, p_cr] = t_c
                    C_cross[q_cr, q_cr] = t_b
                    T_cumul = torch.matmul(C_cross, T_cumul)

        # Apply output diagonal phase screen (necessary for universal U(N) synthesis)
        if diag_phys is not None:
            D_diag = torch.diag_embed(torch.exp(1.0j * diag_phys.to(self.complex_dtype)))
            T_cumul = torch.matmul(D_diag, T_cumul)

        # Apply channel physical propagation loss and phase delay
        if self.config.enable_physical_routing and not self.ideal_mode:
            lam = wavelength if wavelength is not None else self.config.wavelength
            beta = 2.0 * math.pi * self.routing_model.n_eff / lam
            prop_phase = torch.exp(1.0j * beta * self.routing_model.route_lengths.to(device)).to(self.complex_dtype)
            prop_atten = self.routing_model.route_attenuations.to(device=device, dtype=self.complex_dtype)
            D_prop = torch.diag_embed(prop_phase * prop_atten)
            T_cumul = torch.matmul(D_prop, T_cumul)

        # Apply bend radiation and mode-mismatch loss across mesh
        if self.config.enable_bend_loss and not self.ideal_mode:
            D_bend = torch.diag_embed(self.bend_loss_model.bend_transmission.to(device=device, dtype=self.complex_dtype))
            T_cumul = torch.matmul(D_bend, T_cumul)

        # Apply multi-cavity Fabry-Perot backreflection ripple
        if self.config.enable_backreflection and not self.ideal_mode:
            H_fp = self.backreflection_model.compute_transfer_filter(wavelength=wavelength, device=device).to(self.complex_dtype)
            D_fp = torch.diag_embed(H_fp)
            T_cumul = torch.matmul(D_fp, T_cumul)

        return T_cumul

    def propagate_field(
        self,
        E_in: torch.Tensor,
        theta: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        diag_phases: Optional[torch.Tensor] = None,
        wavelength: Optional[Union[float, torch.Tensor]] = None,
        override_eps1: Optional[torch.Tensor] = None,
        override_eps2: Optional[torch.Tensor] = None,
        override_phi_intrinsic: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Propagates complex optical field vector E_in stage-by-stage through the mesh.
        Optimized O(N) memory complexity per layer without dense N x N matrix allocation.

        Args:
            E_in: Complex optical field of shape (..., N).
            theta: Internal phases of shape (..., M).
            phi: External phases of shape (..., M).
            diag_phases: Optional output diagonal phase screen of shape (..., N).
            wavelength: Operating optical wavelength in meters.
            override_eps1: Optional tensor overriding coupler_eps1.
            override_eps2: Optional tensor overriding coupler_eps2.
            override_phi_intrinsic: Optional tensor overriding phi_intrinsic.

        Returns:
            E_out: Transmitted complex field of shape (..., N).
        """
        device = E_in.device
        theta_phys, phi_phys, diag_phys = self.prepare_physical_phases(theta, phi, diag_phases)

        # Check if full Jones vector polarization tracking is active
        use_pol = self.config.enable_polarization and not self.ideal_mode

        # Clone input field and apply input grating coupler & laser phase noise
        E_curr = E_in.to(self.complex_dtype).clone()
        if not self.ideal_mode:
            E_curr = self.photodetector.apply_grating_coupler(E_curr, wavelength)
            E_curr = self.photodetector.apply_laser_phase_noise(E_curr)

        if use_pol and (E_curr.ndim == 0 or E_curr.shape[-1] != 2):
            # Promote scalar field to Jones vector [E_TE, 0.0]
            zeros = torch.zeros_like(E_curr)
            E_curr = torch.stack([E_curr, zeros], dim=-1)

        for layer in self.layer_info:
            col = layer["col"]
            mzi_idx = layer["mzi_indices"].to(device)
            p_idx = layer["p_indices"].to(device)
            q_idx = layer["q_indices"].to(device)

            th_col = theta_phys[..., mzi_idx]
            ph_col = phi_phys[..., mzi_idx]

            if override_eps1 is not None and override_eps2 is not None:
                eps1_col = override_eps1.to(device)[mzi_idx]
                eps2_col = override_eps2.to(device)[mzi_idx]
                phi_int_col = override_phi_intrinsic.to(device)[mzi_idx] if override_phi_intrinsic is not None else None
            elif self.ideal_mode or not self.config.enable_coupler_errors:
                eps1_col = None
                eps2_col = None
                phi_int_col = None
            else:
                eps1_col = self.coupler_eps1.to(device)[mzi_idx]
                eps2_col = self.coupler_eps2.to(device)[mzi_idx]
                phi_int_col = self.phi_intrinsic.to(device)[mzi_idx]

            # 2x2 MZI transfer matrices: (..., num_mzis, 2, 2)
            T_mzis = mzi_transfer_matrix(
                theta=th_col,
                phi=ph_col,
                epsilon1=eps1_col,
                epsilon2=eps2_col,
                phi_intrinsic=phi_int_col,
                wavelength=wavelength if self.config.enable_dispersion else None,
                excess_loss_db=self.config.coupler_excess_loss_db if not self.ideal_mode else 0.0,
                ideal_mode=self.ideal_mode,
                complex_dtype=self.complex_dtype
            )

            # Extract pair field amplitudes
            Ep = E_curr[..., p_idx, :] if use_pol else E_curr[..., p_idx]
            Eq = E_curr[..., q_idx, :] if use_pol else E_curr[..., q_idx]

            if use_pol:
                T00 = T_mzis[..., 0, 0].unsqueeze(-1)
                T01 = T_mzis[..., 0, 1].unsqueeze(-1)
                T10 = T_mzis[..., 1, 0].unsqueeze(-1)
                T11 = T_mzis[..., 1, 1].unsqueeze(-1)
                Ep_new = T00 * Ep + T01 * Eq
                Eq_new = T10 * Ep + T11 * Eq
            else:
                Ep_new = T_mzis[..., 0, 0] * Ep + T_mzis[..., 0, 1] * Eq
                Eq_new = T_mzis[..., 1, 0] * Ep + T_mzis[..., 1, 1] * Eq
            # Write updated fields back into optical modes
            if use_pol:
                E_curr[..., p_idx, :] = Ep_new
                E_curr[..., q_idx, :] = Eq_new
            else:
                E_curr[..., p_idx] = Ep_new
                E_curr[..., q_idx] = Eq_new

            # Apply stage insertion loss across entire column stage (Clements balanced loss)
            if self.config.enable_loss and not self.ideal_mode:
                a_l = self.loss_model.get_layer_attenuation(col).to(device=device, dtype=self.complex_dtype)
                if use_pol:
                    a_l = a_l.unsqueeze(-1)
                E_curr = E_curr * a_l

            # Apply nonlinear optical effects (TPA, FCA, SPM, FCD)
            if self.config.enable_nonlinear_optics and not self.ideal_mode:
                E_curr = self.nonlinear_model(E_curr)

            # Apply polarization bend cross-coupling
            if use_pol:
                E_curr = self.polarization_model.apply_bend_coupling(E_curr)

            # Apply inter-column waveguide crossings (non-polarization scalar path)
            if self.config.enable_physical_routing and not self.ideal_mode and not use_pol:
                E_curr = self.routing_model.apply_inter_stage_crossings(E_curr, col)

        # Apply cumulative polarization crossing PDL once after the layer loop
        if self.config.enable_physical_routing and not self.ideal_mode and use_pol:
            crossings = self.routing_model.crossing_counts.to(device)
            E_curr = self.polarization_model.apply_crossing_pdl(E_curr, crossings)

        # Apply output diagonal phase screen (necessary for universal U(N) synthesis)
        if diag_phys is not None:
            D_factor = torch.exp(1.0j * diag_phys.to(self.complex_dtype))
            if use_pol:
                D_factor = D_factor.unsqueeze(-1)
            E_curr = E_curr * D_factor

        # Apply route-dependent physical propagation loss and phase delay
        if self.config.enable_physical_routing and not self.ideal_mode and not use_pol:
            E_curr = self.routing_model.apply_channel_propagation(E_curr, wavelength)

        # Apply bend radiation and mode-mismatch loss
        if self.config.enable_bend_loss and not self.ideal_mode:
            E_curr = self.bend_loss_model(E_curr)

        # Apply multi-cavity Fabry-Perot backreflection ripple
        if self.config.enable_backreflection and not self.ideal_mode:
            E_curr = self.backreflection_model(E_curr, wavelength=wavelength)

        # Apply output grating coupler
        if not self.ideal_mode:
            E_curr = self.photodetector.apply_grating_coupler(E_curr, wavelength)

        return E_curr

    def forward(
        self,
        E_in: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
        diag_phases: Optional[torch.Tensor] = None,
        wavelength: Optional[Union[float, torch.Tensor]] = None,
        add_noise: bool = True,
        balanced: Optional[bool] = None,
        readout_mode: Optional[str] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Flexible forward pass supporting both field vector propagation and
        full transfer matrix computation.

        If E_in is provided:
            Returns (E_out, measured_output) where:
                E_out: Transmitted complex field of shape (..., N) or (..., N, 2)
                measured_output: Measured photocurrent or digitized ADC voltage.
        If E_in is None:
            Returns T_PIC: Mesh transfer matrix of shape (..., N, N)
        """
        if theta is None:
            raise ValueError("Internal phase tensor 'theta' must be provided.")

        if E_in is None:
            # Return transfer matrix T_PIC
            return self.compute_transfer_matrix(
                theta=theta,
                phi=phi,
                diag_phases=diag_phases,
                wavelength=wavelength
            )

        # Propagate optical field through Clements mesh
        E_out = self.propagate_field(
            E_in=E_in,
            theta=theta,
            phi=phi,
            diag_phases=diag_phases,
            wavelength=wavelength
        )

        use_pol = self.config.enable_polarization and not self.ideal_mode and E_out.ndim > 1 and E_out.shape[-1] == 2

        # Detect at photodiode readout layer
        effective_mode = readout_mode or getattr(self.config, "readout_mode", "direct")
        if balanced is not None:
            effective_mode = "dual_rail" if balanced else "direct"

        if effective_mode == "dual_rail" and not self.ideal_mode and self.n_modes % 2 == 0:
            if use_pol:
                E_pos = torch.sqrt(self.polarization_model.total_optical_power(E_out[..., 0::2, :])).to(self.complex_dtype)
                E_neg = torch.sqrt(self.polarization_model.total_optical_power(E_out[..., 1::2, :])).to(self.complex_dtype)
            else:
                E_pos = E_out[..., 0::2]
                E_neg = E_out[..., 1::2]
            I_diff = self.photodetector.detect_balanced(E_pos, E_neg, add_noise=add_noise)
            readout = self.photodetector.readout_electronics(I_diff)
            return E_out, readout

        if effective_mode in ("homodyne_i", "homodyne_q") and not self.ideal_mode:
            if use_pol:
                E_eff = torch.sqrt(self.polarization_model.total_optical_power(E_out)).to(self.complex_dtype)
            else:
                E_eff = E_out
            I_hom = self.photodetector.detect_homodyne(E_eff, mode=effective_mode, add_noise=add_noise)
            readout = self.photodetector.readout_electronics(I_hom)
            return E_out, readout

        # Single-ended detection
        if use_pol:
            E_eff = torch.sqrt(self.polarization_model.total_optical_power(E_out)).to(self.complex_dtype)
            I_out = self.photodetector(E_out=E_eff, add_noise=add_noise and not self.ideal_mode, readout_mode="direct")
        else:
            I_out = self.photodetector(E_out=E_out, add_noise=add_noise and not self.ideal_mode, readout_mode="direct")
        readout = self.photodetector.readout_electronics(I_out)
        return E_out, readout
