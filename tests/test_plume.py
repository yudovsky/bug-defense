"""
Unit tests for simulation.plume.PlumeAbsorptionField.
"""

import numpy as np
import pytest
from simulation.plume import PlumeAbsorptionField, MultispectralPlumeField


def test_plume_kernel_anisotropy_and_normalization():
    """Verify that kernel peak is normalized to 1.0 and accounts for anisotropic voxel dimensions."""
    shape = (21, 21, 21)
    dx, dy, dz = 0.030, 0.030, 0.020  # mm
    sigma = 0.040  # mm
    plume = PlumeAbsorptionField(shape=shape, dx_mm=dx, dy_mm=dy, dz_mm=dz, sigma_mm=sigma)
    
    # Kernel maximum must be 1.0 exactly at the center
    center_idx = tuple(s // 2 for s in plume.kernel.shape)
    assert plume.kernel[center_idx] == pytest.approx(1.0)
    assert np.all(plume.kernel >= 0.0)
    assert np.all(plume.kernel <= 1.0)


def test_plume_temporal_decay():
    """Verify exact exponential decay mu_a(t+dt) = mu_a(t) * exp(-dt / tau)."""
    shape = (10, 10, 10)
    tau = 0.020  # 20 ms
    plume = PlumeAbsorptionField(shape=shape, tau_linger_s=tau)
    
    # Manually populate initial plume
    plume.mu_a_plume[5, 5, 5] = 10.0
    void_mask = np.ones(shape, dtype=bool)
    no_ablated = np.zeros(shape, dtype=bool)
    
    dt = 0.010  # 10 ms = 0.5 * tau
    plume.update(dt_s=dt, newly_ablated_mask=no_ablated, void_mask=void_mask)
    
    expected = 10.0 * np.exp(-0.010 / 0.020)
    assert plume.mu_a_plume[5, 5, 5] == pytest.approx(expected, rel=1e-5)


def test_plume_spatial_dispersion_and_void_masking():
    """Verify that plume is deposited into void voxels and strictly masked from solid tissue."""
    shape = (15, 15, 15)
    plume = PlumeAbsorptionField(shape=shape, delta_mu_per_voxel=8.0, sigma_mm=0.05)
    
    newly_ablated = np.zeros(shape, dtype=bool)
    newly_ablated[7, 7, 7] = True
    
    # Void mask: only a small 3x3x3 sphere around the center is void, rest is solid
    void_mask = np.zeros(shape, dtype=bool)
    void_mask[6:9, 6:9, 6:9] = True
    
    plume.update(dt_s=0.001, newly_ablated_mask=newly_ablated, void_mask=void_mask)
    
    # Center void voxel must have plume absorption
    assert plume.mu_a_plume[7, 7, 7] > 0.0
    assert plume.mu_a_plume[7, 7, 7] == pytest.approx(8.0, rel=1e-2)
    
    # Neighbors inside the void mask must also have positive plume absorption
    assert plume.mu_a_plume[7, 7, 8] > 0.0
    assert plume.mu_a_plume[7, 7, 8] < plume.mu_a_plume[7, 7, 7]  # spatial drop-off
    
    # Outside the void mask (solid tissue), plume MUST be identically 0.0
    solid_voxels = ~void_mask
    assert np.all(plume.mu_a_plume[solid_voxels] == 0.0)


def test_plume_clamping():
    """Verify that plume optical absorption is clamped to mu_a_max."""
    shape = (11, 11, 11)
    plume = PlumeAbsorptionField(shape=shape, delta_mu_per_voxel=50.0, mu_a_max=25.0)
    
    newly = np.ones(shape, dtype=bool)
    void = np.ones(shape, dtype=bool)
    
    plume.update(dt_s=0.001, newly_ablated_mask=newly, void_mask=void)
    assert np.all(plume.mu_a_plume <= 25.0)
    assert np.max(plume.mu_a_plume) == pytest.approx(25.0)


def test_multispectral_extinction_scaling():
    """Verify that MultispectralPlumeField applies correct wavelength-dependent extinction ceilings."""
    shape = (15, 15, 15)
    
    expected_ceilings = {
        10600: 0.10,
        1940: 0.25,
        1470: 0.30,
        1064: 0.45,
        450: 1.20,
    }
    
    newly = np.zeros(shape, dtype=bool)
    newly[7, 7, 7] = True
    void = np.ones(shape, dtype=bool)
    
    for wl, expected_max in expected_ceilings.items():
        plume = MultispectralPlumeField(shape=shape, wavelength_nm=wl)
        assert plume.mu_max == pytest.approx(expected_max)
        
        # Heavy deposit to saturate
        for _ in range(150):
            plume.update(dt_s=0.0, newly_ablated_mask=newly, void_mask=void)

        assert np.max(plume.mu_ext_plume) <= expected_max
        assert np.max(plume.mu_ext_plume) == pytest.approx(expected_max, rel=0.01)


def test_plume_diffusion_mode_point_injection_and_void_confinement():
    """Verify that diffusion mode injects at just the voxel location, diffuses in void, and stays 0 in solid."""
    shape = (15, 15, 15)
    plume = MultispectralPlumeField(
        shape=shape,
        model='diffusion',
        delta_mu_per_voxel=5.0,
        D_plume=0.5,
        v_drift=10.0,
        blast_expansion=False,
    )

    # 1. Point injection: only voxel (7, 7, 7) explodes
    newly = np.zeros(shape, dtype=bool)
    newly[7, 7, 7] = True
    plume.inject(newly)

    # Verify injected ONLY at (7, 7, 7)
    assert plume.raw_field[7, 7, 7] == pytest.approx(5.0)
    assert np.sum(plume.raw_field > 0.0) == 1

    # 2. Diffusion in void space: cavity around center, rest solid
    void_mask = np.zeros(shape, dtype=bool)
    void_mask[5:10, 5:10, 5:10] = True

    # Run 10 diffusion sub-steps of 50 us
    for _ in range(10):
        plume.diffuse_substep(void_mask, dt_sub=5e-5)

    # Plume has diffused to neighbors inside void
    assert plume.raw_field[7, 7, 7] < 5.0
    assert plume.raw_field[7, 7, 8] > 0.0

    # Solid tissue voxels must be STRICTLY 0.0
    solid_voxels = ~void_mask
    assert np.all(plume.raw_field[solid_voxels] == 0.0)

    # Boundary voxels must be 0.0
    assert np.all(plume.raw_field[0, :, :] == 0.0)
    assert np.all(plume.raw_field[-1, :, :] == 0.0)


def test_plume_explosive_blast_expansion():
    """Verify that blast_expansion immediately depletes the source voxel and disperses across the void cavity."""
    shape = (25, 25, 25)
    plume = MultispectralPlumeField(
        shape=shape,
        model='diffusion',
        delta_mu_per_voxel=1.0,
        D_plume=2.0,
        v_drift=80.0,
        blast_expansion=True,
    )

    # Void cavity from z=5 to z=20, center at (12, 12)
    void_mask = np.zeros(shape, dtype=bool)
    void_mask[8:17, 8:17, 5:20] = True

    # Explode single voxel at crater bottom (12, 12, 15)
    newly = np.zeros(shape, dtype=bool)
    newly[12, 12, 15] = True
    plume.inject(newly, void_mask=void_mask)

    # Source voxel must be depleted compared to concentrated 1.0 (vapor has expanded)
    assert plume.raw_field[12, 12, 15] < 0.20
    # Blast has expanded to multiple neighboring voxels inside the cavity
    nonzero_in_void = np.sum((plume.raw_field > 1e-4) & void_mask)
    assert nonzero_in_void > 10
    # Plume must be strictly zero outside void (in solid tissue)
    assert np.all(plume.raw_field[~void_mask] == 0.0)


def test_plume_diffusion_thermal_independence():
    """Verify that adding 2-channel plume diffusion leaves thermal state variables completely unperturbed."""
    from simulation.metal_ablation_solver import is_metal_available, MetalAblationContext

    if not is_metal_available():
        pytest.skip("Apple Metal GPU not available on this platform")

    Nx, Ny, Nz = 12, 12, 10
    dx, dy, dz = 0.03, 0.03, 0.02
    dt_sub = 5e-5

    k = np.full((Nx, Ny, Nz), 0.3e-3, dtype=np.float32)
    rho = np.full((Nx, Ny, Nz), 2.0e-3, dtype=np.float32)
    Q = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Q[4:8, 4:8, 2:6] = 50.0
    dry = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
    abl = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
    abl[5:7, 5:7, 0:3] = 1

    T_init = np.full((Nx, Ny, Nz), 25.0, dtype=np.float32)
    H_init = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    O_init = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    chi_init = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    # Run without plume
    ctx1 = MetalAblationContext(Nx, Ny, Nz, dx, dy, dz, dt_sub, 10.0, 20.0, 100.0, 300.0,
                                arrhenius=(1e4, 5000.0 / 8.314), char_params=(1e3, 4000.0 / 8.314))
    ctx1.set_state(T_init, H_init, O_init, chi_init, dry, abl, Q, k, rho)
    ctx1.advance_substeps(50, is_active=1)
    T_1, H_1, O_1, chi_1 = ctx1.get_state()

    # Run with 2-channel plume diffusion active
    ctx2 = MetalAblationContext(Nx, Ny, Nz, dx, dy, dz, dt_sub, 10.0, 20.0, 100.0, 300.0,
                                arrhenius=(1e4, 5000.0 / 8.314), char_params=(1e3, 4000.0 / 8.314),
                                plume_params={'D': 0.5, 'vz': 15.0, 'tau_linger': 0.010})
    plume_init = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    plume_init[5:7, 5:7, 1] = 10.0
    ctx2.set_state(T_init, H_init, O_init, chi_init, dry, abl, Q, k, rho, plume=plume_init)
    ctx2.advance_substeps(50, is_active=1)
    T_2, H_2, O_2, chi_2 = ctx2.get_state()

    # 100% exact numerical unperturbed parity: 0.00000000 deg C
    assert np.max(np.abs(T_1 - T_2)) == 0.0
    assert np.max(np.abs(H_1 - H_2)) == 0.0
    assert np.max(np.abs(O_1 - O_2)) == 0.0
    assert np.max(np.abs(chi_1 - chi_2)) == 0.0


def test_plume_anisotropic_diffusion():
    """Verify that anisotropic diffusion spreads faster along z (D_z=4) than radial xy (D_xy=1)."""
    shape = (31, 31, 31)
    dx = dy = dz = 0.02
    plume = MultispectralPlumeField(
        shape=shape,
        dx_mm=dx, dy_mm=dy, dz_mm=dz,
        model='diffusion',
        D_xy=1.0,
        D_z=4.0,
        v_drift=0.0,
        tau_linger_s=float('inf'),
        blast_expansion=False,
    )

    # All void
    void_mask = np.ones(shape, dtype=bool)

    # Inject concentrated point at center (15, 15, 15)
    center = (15, 15, 15)
    newly = np.zeros(shape, dtype=bool)
    newly[center] = True
    plume.inject(newly)

    # Step 20 substeps within CFL limit (dt_max = 3.33e-5)
    for _ in range(20):
        plume.diffuse_substep(void_mask, dt_sub=2e-5)

    # Compare dispersion: plume[15, 15, 17] (z-offset 2) vs plume[17, 15, 15] (x-offset 2)
    val_z = plume.raw_field[15, 15, 17]
    val_x = plume.raw_field[17, 15, 15]
    val_y = plume.raw_field[15, 17, 15]

    # Equal x and y symmetry
    assert val_x == pytest.approx(val_y, rel=1e-4)
    # Strong axial elongation due to D_z = 4 * D_xy
    assert val_z > val_x * 2.0

    # Also test Metal GPU parity if available
    from simulation.metal_ablation_solver import is_metal_available, MetalAblationContext
    if is_metal_available():
        ctx = MetalAblationContext(
            31, 31, 31, dx, dy, dz, 2e-5, 10.0, 20.0, 100.0, 300.0,
            plume_params={'D_xy': 1.0, 'D_z': 4.0, 'vz': 0.0, 'tau_linger': float('inf')}
        )
        p_init = np.zeros(shape, dtype=np.float32)
        p_init[center] = plume.delta_mu
        T_0 = np.full(shape, 20.0, dtype=np.float32)
        zeros_f = np.zeros(shape, dtype=np.float32)
        ones_u = np.ones(shape, dtype=np.uint8)
        ctx.set_state(T_0, zeros_f, zeros_f, zeros_f, ones_u, ones_u, zeros_f, zeros_f, zeros_f, plume=p_init)
        ctx.advance_substeps(20, is_active=1)
        p_gpu = ctx.get_plume()
        gpu_z = p_gpu[15, 15, 17]
        gpu_x = p_gpu[17, 15, 15]
        assert gpu_z > gpu_x * 2.0
        assert (gpu_z / gpu_x) == pytest.approx(val_z / val_x, rel=1e-3)



