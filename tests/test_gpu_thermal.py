"""
Unit and integration tests for the Apple Metal GPU-accelerated voxel heat equation solver.
"""

import pytest
import numpy as np

from simulation.voxel_thermal_metal import is_metal_available as is_voxel_thermal_metal_available, run_thermal3d_substeps_metal
from simulation.voxel_thermal_3d import _thermal3d_step_explicit, VoxelThermalSolver3D


@pytest.mark.skipif(not is_voxel_thermal_metal_available(), reason="Apple Metal GPU not available on this platform")
class TestGpuVoxelBioheatDiffusion:
    """Tests for Metal GPU-accelerated 3D Cartesian Bioheat Diffusion Stencil."""

    def test_voxel_thermal_substeps_metal_exact_parity(self):
        """Verify Metal 3D stencil matches CPU Numba kernel across multi-step heterogeneous diffusion."""
        Nx, Ny, Nz = 12, 12, 10
        dx, dy, dz = 0.05, 0.05, 0.03
        dt = 2e-5

        T_cpu = np.full((Nx, Ny, Nz), 25.0, dtype=np.float64)
        T_next = np.empty_like(T_cpu)
        Q = np.zeros_like(T_cpu)
        Q[4:8, 4:8, 0:3] = 800.0  # Localized volumetric heat source

        # Heterogeneous properties (two distinct strata)
        k_grid = np.full((Nx, Ny, Nz), 0.25e-3, dtype=np.float64)
        k_grid[:, :, 5:] = 0.50e-3
        rho_cp = np.full((Nx, Ny, Nz), 1.8e-3, dtype=np.float64)
        rho_cp[:, :, 5:] = 3.5e-3

        h_eff = 12.0
        t_ambient = 25.0
        n_steps = 10

        # CPU Reference
        for _ in range(n_steps):
            _thermal3d_step_explicit(T_cpu, T_next, Q, k_grid, rho_cp, Nx, Ny, Nz, dx, dy, dz, dt, h_eff, t_ambient, 1)
            T_cpu, T_next = T_next, T_cpu

        # GPU Metal
        T_gpu = run_thermal3d_substeps_metal(
            np.full((Nx, Ny, Nz), 25.0), Q, k_grid, rho_cp,
            Nx, Ny, Nz, dx, dy, dz, dt, n_steps, h_eff, t_ambient, 1
        )

        max_abs_diff = float(np.max(np.abs(T_cpu - T_gpu)))
        assert max_abs_diff < 1e-4, f"Thermal 3D GPU vs CPU max difference {max_abs_diff:.2e} exceeds tolerance."
        assert abs(np.max(T_cpu) - np.max(T_gpu)) < 1e-3

    def test_voxel_thermal_solver_3d_gpu_dispatch(self):
        """Verify VoxelThermalSolver3D.solve runs with engine='gpu'."""
        gx = np.linspace(-0.3, 0.3, 8)
        gy = np.linspace(-0.3, 0.3, 8)
        gz = np.linspace(0.0, 0.4, 6)
        k_grid = np.full((8, 8, 6), 0.25e-3)
        rhocp_grid = np.full((8, 8, 6), 1.9e-3)

        solver = VoxelThermalSolver3D(
            grid_x=gx, grid_y=gy, grid_z=gz,
            k_grid=k_grid, rho_cp_grid=rhocp_grid,
            h_conv=10.0, emissivity=0.9, t_ambient=22.0
        )

        Q_3d = np.zeros((8, 8, 6))
        Q_3d[3:5, 3:5, 0:2] = 50.0

        res_gpu = solver.solve(Q_3d=Q_3d, pulse_duration=0.005, total_time=0.010, dt=0.001, engine='gpu')
        res_cpu = solver.solve(Q_3d=Q_3d, pulse_duration=0.005, total_time=0.010, dt=0.001, engine='cpu')

        assert res_gpu['engine'] == 'gpu'
        assert res_cpu['engine'] == 'cpu'
        assert abs(res_gpu['peak_surface_T'] - res_cpu['peak_surface_T']) < 0.05

    def test_fused_ablation_exact_parity(self):
        """Verify Fused Metal multi-phase ablation kernel matches CPU across T, H_acc, Omega, and chi."""
        from simulation.metal_ablation_solver import is_metal_available as is_abl_avail, MetalAblationContext
        from simulation.coupled_ablation import _accumulate_arrhenius, _accumulate_char
        from simulation.voxel_thermal_3d import _apply_latent_heat_plateau

        assert is_abl_avail()

        Nx, Ny, Nz = 14, 14, 12
        dx, dy, dz = 0.03, 0.03, 0.02
        dt = 2.5e-5
        h_eff = 15.0e-6
        t_amb = 20.0
        t_abl = 100.0
        t_void = 20.0
        arr_A, arr_Ea_R = 1.0e7, 65000.0 / 8.314
        char_A, char_Ea_R = 1.0e7, 65000.0 / 8.314

        T_cpu = np.full((Nx, Ny, Nz), 25.0, dtype=np.float64)
        T_next = np.empty_like(T_cpu)
        Q = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        Q[5:9, 5:9, 0:3] = 750.0
        k = np.full((Nx, Ny, Nz), 0.3e-3, dtype=np.float64)
        k[:, :, 6:] = 0.5e-3
        rho = np.full((Nx, Ny, Nz), 2.0e-3, dtype=np.float64)
        rho[:, :, 6:] = 3.5e-3

        H_acc_cpu = np.zeros_like(T_cpu)
        Omega_cpu = np.zeros_like(T_cpu)
        chi_cpu = np.zeros_like(T_cpu)
        dry_cpu = np.zeros((Nx, Ny, Nz), dtype=np.bool_)
        dry_cpu[6:8, 6:8, 0:2] = True
        ablated_cpu = np.zeros((Nx, Ny, Nz), dtype=np.bool_)

        n_steps = 30

        # CPU Run
        for _ in range(n_steps):
            _thermal3d_step_explicit(T_cpu, T_next, Q, k, rho, Nx, Ny, Nz, dx, dy, dz, dt, h_eff, t_amb, 1)
            _apply_latent_heat_plateau(T_next, H_acc_cpu, rho, ablated_cpu, t_abl, Nx, Ny, Nz, dry_cpu, t_void)
            _accumulate_arrhenius(T_next, Omega_cpu, ablated_cpu, arr_A, arr_Ea_R, dt, Nx, Ny, Nz)
            _accumulate_char(T_next, chi_cpu, dry_cpu, ablated_cpu, char_A, char_Ea_R, dt, Nx, Ny, Nz)
            T_cpu, T_next = T_next, T_cpu

        # GPU Run
        ctx = MetalAblationContext(
            Nx, Ny, Nz, dx, dy, dz, dt, h_eff, t_amb, t_abl, t_void,
            arrhenius=(arr_A, arr_Ea_R),
            char_params=(char_A, char_Ea_R)
        )
        ctx.set_state(
            np.full((Nx, Ny, Nz), 25.0), np.zeros((Nx, Ny, Nz)), np.zeros((Nx, Ny, Nz)),
            np.zeros((Nx, Ny, Nz)), dry_cpu, ablated_cpu, Q, k, rho
        )
        ctx.advance_substeps(n_steps, is_active=1)
        T_gpu, H_acc_gpu, Omega_gpu, chi_gpu = ctx.get_state()

        assert float(np.max(np.abs(T_cpu - T_gpu))) < 1e-4
        assert float(np.max(np.abs(H_acc_cpu - H_acc_gpu))) < 1e-5
        assert float(np.max(np.abs(Omega_cpu - Omega_gpu))) < 1e-5
        assert float(np.max(np.abs(chi_cpu - chi_gpu))) < 1e-5

    def test_coupled_ablation_engine_gpu_parity(self):
        """Verify CoupledAblationEngine produces identical ablation metrics on GPU vs CPU."""
        from simulation.coupled_ablation import CoupledAblationEngine

        shape = (12, 12, 10)
        opt_props = {
            'mu_a': np.full(shape, 0.5),
            'mu_s': np.full(shape, 10.0),
            'g': np.full(shape, 0.9),
            'n': np.full(shape, 1.4),
        }
        k_grid = np.full(shape, 0.3e-3)
        rho_cp_grid = np.full(shape, 2.0e-3)
        h_lat_grid = np.full(shape, 2.26)

        dry_state = {
            'mu_a': 0.8, 'mu_s': 12.0, 'g': 0.9, 'n': 1.45,
            'k': 0.25e-3, 'rho_cp': 1.5e-3, 't_remove': 350.0
        }

        eng_gpu = CoupledAblationEngine(
            optical_props=opt_props, k_grid=k_grid, rho_cp_grid=rho_cp_grid,
            h_lat_grid=h_lat_grid, Lx=1.2, Ly=1.2, Lz=1.0,
            power=50.0, spot_dia=0.4, t_abl=100.0, t_ambient=25.0,
            dt_epoch=0.002, dt_sub=1e-4, n_photons=10000,
            dry_state=dry_state, thermal_engine='gpu'
        )

        eng_cpu = CoupledAblationEngine(
            optical_props=opt_props, k_grid=k_grid, rho_cp_grid=rho_cp_grid,
            h_lat_grid=h_lat_grid, Lx=1.2, Ly=1.2, Lz=1.0,
            power=50.0, spot_dia=0.4, t_abl=100.0, t_ambient=25.0,
            dt_epoch=0.002, dt_sub=1e-4, n_photons=10000,
            dry_state=dry_state, thermal_engine='cpu'
        )

        assert eng_gpu.use_gpu_thermal is True
        assert eng_cpu.use_gpu_thermal is False

        # Advance 2 epochs
        d_gpu = [eng_gpu.step_epoch() for _ in range(2)]
        d_cpu = [eng_cpu.step_epoch() for _ in range(2)]

        # Peak temperatures match within 0.1 °C
        assert abs(np.max(eng_gpu.T) - np.max(eng_cpu.T)) < 0.1

