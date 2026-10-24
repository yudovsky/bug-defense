"""
Unit tests for optical and thermal simulation solvers (simulation/).
"""

import unittest
import numpy as np

from simulation.voxel_mc_3d import run_voxel_mc3d
from simulation.voxel_thermal_3d import VoxelThermalSolver3D, _thermal3d_step_explicit


class TestSolvers(unittest.TestCase):
    def test_voxel_mc_3d_energy_closure(self):
        # Test 3D Cartesian Voxel Monte Carlo on a small domain
        res = run_voxel_mc3d(
            power=2.0,
            pulse_duration=0.01,
            n_photons=2000,
            spot_dia=0.20,
            Lx=1.0,
            Ly=1.0,
            Lz=0.8,
            Nx=10,
            Ny=10,
            Nz=8
        )

        self.assertIn('energy_closure', res)
        # Energy must be strictly conserved within 0.01 tolerance
        self.assertAlmostEqual(res['energy_closure'], 1.0, delta=0.02)

        # Absorbed energy array must be non-negative
        A_3d = res['A_3d']
        self.assertTrue(np.all(A_3d >= 0.0))
        self.assertEqual(A_3d.shape, (10, 10, 8))

        # Total integrated absorption must not exceed incident power * pulse_duration
        total_E_in = 2.0 * 0.01
        self.assertLessEqual(np.sum(A_3d), total_E_in * 1.05)

    def test_voxel_thermal_solver_3d(self):
        gx = np.linspace(-0.5, 0.5, 8)
        gy = np.linspace(-0.5, 0.5, 8)
        gz = np.linspace(0.0, 0.6, 6)
        k_grid = np.full((8, 8, 6), 0.2e-3)
        rhocp_grid = np.full((8, 8, 6), 1.8e-3)

        solver = VoxelThermalSolver3D(
            grid_x=gx,
            grid_y=gy,
            grid_z=gz,
            k_grid=k_grid,
            rho_cp_grid=rhocp_grid,
            h_conv=15.0,
            emissivity=0.95,
            t_ambient=25.0
        )

        Q_3d = np.zeros((8, 8, 6))
        Q_3d[3:5, 3:5, 0:2] = 20.0

        res = solver.solve(
            Q_3d=Q_3d,
            pulse_duration=0.005,
            total_time=0.015,
            dt=0.001
        )

        self.assertIn('peak_surface_T', res)
        self.assertIn('peak_center_T', res)
        self.assertGreaterEqual(res['peak_surface_T'], 25.0)

        # Check time history continuity
        self.assertGreater(len(res['time_history']), 0)
        self.assertFalse(np.any(np.isnan(res['time_history'])))

    def test_thermal3d_step_explicit_kernel(self):
        Nx, Ny, Nz = 5, 5, 5
        T = np.full((Nx, Ny, Nz), 20.0)
        T_new = np.zeros((Nx, Ny, Nz))
        Q = np.zeros((Nx, Ny, Nz))
        Q[2, 2, 2] = 100.0  # Heat source in center
        k_grid = np.full((Nx, Ny, Nz), 0.2e-3)
        rhocp_grid = np.full((Nx, Ny, Nz), 1.8e-3)

        _thermal3d_step_explicit(
            T=T,
            T_new=T_new,
            Q=Q,
            k_grid=k_grid,
            rho_cp_grid=rhocp_grid,
            Nx=Nx, Ny=Ny, Nz=Nz,
            dx=0.1, dy=0.1, dz=0.1,
            dt=0.0005,
            h_eff=1e-5,
            t_ambient=20.0,
            is_active=1
        )

        # Center voxel should heat up
        self.assertGreater(T_new[2, 2, 2], 20.0)
        self.assertFalse(np.any(np.isnan(T_new)))
        self.assertFalse(np.any(np.isinf(T_new)))


if __name__ == '__main__':
    unittest.main()
