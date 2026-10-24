import pytest
import numpy as np
from simulation.voxel_mc_metal import is_metal_available, run_voxel_mc_metal
from simulation.voxel_mc_3d import run_voxel_mc3d


@pytest.mark.skipif(not is_metal_available(), reason="Apple Metal GPU not available on this platform")
class TestGPUMonteCarlo:
    """Test suite for Apple Metal GPU-accelerated 3D Voxel Monte Carlo Engine."""

    def test_metal_device_available(self):
        assert is_metal_available() is True

    def test_homogeneous_cube_energy_closure(self):
        """Verify strict energy conservation on a homogeneous scattering cube."""
        Nx, Ny, Nz = 20, 20, 25
        mu_a = np.full((Nx, Ny, Nz), 0.2, dtype=np.float64)
        mu_s = np.full((Nx, Ny, Nz), 10.0, dtype=np.float64)
        g = np.full((Nx, Ny, Nz), 0.80, dtype=np.float64)
        n = np.full((Nx, Ny, Nz), 1.40, dtype=np.float64)
        props = {'mu_a': mu_a, 'mu_s': mu_s, 'g': g, 'n': n}

        res = run_voxel_mc_metal(
            power=5.0,
            n_photons=50_000,
            Lx=1.0, Ly=1.0, Lz=1.0,
            Nx=Nx, Ny=Ny, Nz=Nz,
            custom_3d_props=props,
            theta_inc_deg=0.0,
            phi_inc_deg=0.0,
            spot_dia=0.20,
            base_seed=1234,
        )

        assert abs(res['energy_closure'] - 1.0) < 0.01
        assert res['spectro_raw']['Absorption'] > 0.0
        assert res['spectro_raw']['R_spec'] > 0.0
        assert res['spectro_raw']['R_diff'] > 0.0

    def test_multilayer_non_normal_gaussian_parity(self):
        """
        Verify numerical parity between CPU and GPU on a multilayered geometry
        with an oblique (non-normal) incident Gaussian beam.
        """
        Lx, Ly, Lz = 2.0, 2.0, 1.0
        Nx, Ny, Nz = 30, 30, 40
        mu_a = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        mu_s = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        g = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        n = np.zeros((Nx, Ny, Nz), dtype=np.float64)

        # Layer 1: z in [0, 0.25 mm] (cuticle/stratum corneum)
        mu_a[:, :, 0:10] = 0.60
        mu_s[:, :, 0:10] = 16.0
        g[:, :, 0:10] = 0.85
        n[:, :, 0:10] = 1.52

        # Layer 2: z in [0.25, 0.60 mm] (subelytral / fat cavity)
        mu_a[:, :, 10:24] = 0.05
        mu_s[:, :, 10:24] = 4.0
        g[:, :, 10:24] = 0.80
        n[:, :, 10:24] = 1.34

        # Layer 3: z in [0.60, 1.00 mm] (dense muscle / visceral core)
        mu_a[:, :, 24:40] = 0.70
        mu_s[:, :, 24:40] = 18.0
        g[:, :, 24:40] = 0.88
        n[:, :, 24:40] = 1.42

        props = {'mu_a': mu_a, 'mu_s': mu_s, 'g': g, 'n': n}

        theta_deg = 30.0
        phi_deg = 45.0
        n_phot = 100_000

        # CPU Reference
        res_cpu = run_voxel_mc3d(
            power=10.0,
            n_photons=n_phot,
            Lx=Lx, Ly=Ly, Lz=Lz,
            Nx=Nx, Ny=Ny, Nz=Nz,
            custom_3d_props=props,
            theta_inc_deg=theta_deg,
            phi_inc_deg=phi_deg,
            spot_dia=0.40,
            beam_center=(1.0, 1.0),
            beam_profile='gaussian',
            engine='cpu',
        )

        # GPU Metal
        res_gpu = run_voxel_mc_metal(
            power=10.0,
            n_photons=n_phot,
            Lx=Lx, Ly=Ly, Lz=Lz,
            Nx=Nx, Ny=Ny, Nz=Nz,
            custom_3d_props=props,
            theta_inc_deg=theta_deg,
            phi_inc_deg=phi_deg,
            spot_dia=0.40,
            beam_center=(1.0, 1.0),
            base_seed=777,
        )

        # 1. Exact unpolarized Fresnel reflectance check at entrance
        cos_i = np.cos(np.radians(theta_deg))
        n1, n2 = 1.0, 1.52
        sin_t = (n1 / n2) * np.sin(np.radians(theta_deg))
        cos_t = np.sqrt(1.0 - sin_t**2)
        rs = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)
        rp = (n1*cos_t - n2*cos_i) / (n1*cos_t + n2*cos_i)
        r_fresnel_exact = 0.5 * (rs**2 + rp**2)

        assert abs(res_gpu['spectro_raw']['R_spec'] - r_fresnel_exact) < 0.002
        assert abs(res_gpu['energy_closure'] - 1.0) < 0.01

        # 2. Spectrophotometric channels within Monte Carlo statistical scatter
        assert abs(res_gpu['spectro_raw']['Absorption'] - res_cpu['spectro_raw']['Absorption']) < 0.015
        assert abs(res_gpu['spectro_raw']['R_diff'] - res_cpu['spectro_raw']['R_diff']) < 0.015
        assert abs(res_gpu['spectro_raw']['Transmission'] - res_cpu['spectro_raw']['Transmission']) < 0.015

        # 3. Spatial Absorption Profile Correlation
        A_cpu_z = np.sum(res_cpu['A_3d'], axis=(0, 1))
        A_gpu_z = np.sum(res_gpu['A_3d'], axis=(0, 1))
        corr = np.corrcoef(A_cpu_z, A_gpu_z)[0, 1]
        assert corr > 0.995, f"Spatial correlation {corr:.5f} is below 0.995"

    def test_run_voxel_mc3d_auto_dispatches_to_gpu(self):
        """Verify that run_voxel_mc3d with default engine='auto' uses Metal GPU."""
        res = run_voxel_mc3d(
            power=5.0,
            n_photons=10_000,
            spot_dia=0.30,
            Lx=1.0, Ly=1.0, Lz=0.5,
            Nx=20, Ny=20, Nz=15,
        )
        assert res['engine'] == 'gpu'
        assert abs(res['energy_closure'] - 1.0) < 0.01

    def test_run_voxel_mc3d_explicit_cpu_override(self):
        """Verify that engine='cpu' explicitly forces CPU Numba execution."""
        res = run_voxel_mc3d(
            power=5.0,
            n_photons=10_000,
            spot_dia=0.30,
            Lx=1.0, Ly=1.0, Lz=0.5,
            Nx=20, Ny=20, Nz=15,
            engine='cpu',
        )
        assert res['engine'] == 'cpu'
        assert abs(res['energy_closure'] - 1.0) < 0.01

    def test_run_voxel_mc3d_auto_fallback_conditions(self):
        """Verify fallback to CPU on unsupported GPU beam profiles or launch axes."""
        # Top-hat beam: falls back to CPU
        res_tophat = run_voxel_mc3d(
            power=5.0,
            n_photons=5_000,
            spot_dia=0.30,
            Lx=1.0, Ly=1.0, Lz=0.5,
            Nx=15, Ny=15, Nz=10,
            beam_profile='tophat',
            engine='auto',
        )
        assert res_tophat['engine'] == 'cpu'

        # Launch axis 'x': falls back to CPU
        res_x = run_voxel_mc3d(
            power=5.0,
            n_photons=5_000,
            spot_dia=0.30,
            Lx=0.5, Ly=1.0, Lz=1.0,
            Nx=10, Ny=15, Nz=15,
            launch_axis='x',
            engine='auto',
        )
        assert res_x['engine'] == 'cpu'

    def test_coupled_ablation_engine_gpu_integration(self):
        """Verify CoupledAblationEngine runs dynamic ablation using GPU MC."""
        from simulation.coupled_ablation import CoupledAblationEngine

        Nx, Ny, Nz = 15, 15, 12
        opt = {
            'mu_a': np.full((Nx, Ny, Nz), 1.0),
            'mu_s': np.full((Nx, Ny, Nz), 10.0),
            'g': np.full((Nx, Ny, Nz), 0.8),
            'n': np.full((Nx, Ny, Nz), 1.4),
        }
        k_grid = np.full((Nx, Ny, Nz), 0.3e-3)
        rho_cp_grid = np.full((Nx, Ny, Nz), 2.0e-3)
        h_lat_grid = np.full((Nx, Ny, Nz), 2.5)

        engine = CoupledAblationEngine(
            optical_props=opt,
            k_grid=k_grid,
            rho_cp_grid=rho_cp_grid,
            h_lat_grid=h_lat_grid,
            Lx=1.5, Ly=1.5, Lz=1.2,
            power=20.0,
            spot_dia=0.40,
            t_abl=100.0,
            t_ambient=20.0,
            dt_epoch=0.005,
            dt_sub=1e-4,
            n_photons=20_000,
            mc_engine='auto',
        )

        diag1 = engine.step_epoch()
        assert engine.last_mc is not None
        assert engine.last_mc['engine'] == 'gpu'
        assert diag1['t_optical_s'] < 0.10  # GPU epoch optical step is sub-100ms
        assert float(np.max(engine.T)) > 20.0

