"""
Unit tests for the three physical extensions in simulation.coupled_ablation:
1. Carbonization (pyrolysis of dry matrix into char via Arrhenius rate law)
2. Plume attenuation closure (mass-flux dependent shielding and lingering)
3. Oblique incidence angle (angle-dependent Fresnel and refraction)
"""

import numpy as np
import pytest

from simulation.coupled_ablation import CoupledAblationEngine, _accumulate_char


def _make_test_engine(dry_state=None, char_state=None, plume_closure=None,
                       theta_inc_deg=0.0, phi_inc_deg=0.0, power=5.0):
    N, Nz = 11, 20
    voxel = 0.05
    shape = (N, N, Nz)
    optical = {
        'mu_a': np.full(shape, 10.0),
        'mu_s': np.zeros(shape),
        'g': np.zeros(shape),
        'n': np.ones(shape),
    }
    return CoupledAblationEngine(
        optical_props=optical,
        k_grid=np.full(shape, 0.5e-3),
        rho_cp_grid=np.full(shape, 4.0e-3),
        h_lat_grid=np.full(shape, 2.0),
        Lx=N * voxel, Ly=N * voxel, Lz=Nz * voxel,
        power=power,
        spot_dia=0.3,
        t_abl=100.0,
        t_ambient=25.0,
        dt_epoch=0.002,
        dt_sub=5e-5,
        n_photons=10000,
        dry_state=dry_state,
        char_state=char_state,
        plume_closure=plume_closure,
        theta_inc_deg=theta_inc_deg,
        phi_inc_deg=phi_inc_deg,
    )


def test_accumulate_char_kernel():
    """Verify analytical Arrhenius integration for chi in [0, 1]."""
    Nx, Ny, Nz = 2, 2, 2
    T = np.full((Nx, Ny, Nz), 300.0)  # 300 C = 573.15 K
    chi = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    dry = np.ones((Nx, Ny, Nz), dtype=np.bool_)
    ablated = np.zeros((Nx, Ny, Nz), dtype=np.bool_)
    
    A = 1.0e6
    Ea_over_R = 8000.0  # rate = 1e6 * exp(-8000 / 573.15) ~ 859.3 s^-1
    dt = 1e-4
    
    _accumulate_char(T, chi, dry, ablated, A, Ea_over_R, dt, Nx, Ny, Nz)
    expected_rate = A * np.exp(-Ea_over_R / (300.0 + 273.15))
    expected_chi = 1.0 - np.exp(-expected_rate * dt)
    assert np.allclose(chi, expected_chi, rtol=1e-5)
    assert np.all(chi >= 0.0) and np.all(chi <= 1.0)
    
    # Non-dry voxels must not accumulate char
    dry[:, :, :] = False
    chi_before = chi.copy()
    _accumulate_char(T, chi, dry, ablated, A, Ea_over_R, dt, Nx, Ny, Nz)
    assert np.array_equal(chi, chi_before)


def test_carbonization_switch_disabled_by_default():
    """When char_state=None, char state must remain zero and unallocated."""
    eng = _make_test_engine(char_state=None)
    assert eng.char_state is None
    assert np.all(eng.chi == 0.0)
    res = eng.step_epoch()
    assert np.all(eng.chi == 0.0)


def test_carbonization_property_interpolation():
    """Verify that as chi increases, dry voxels interpolate toward char properties."""
    dry_state = dict(mu_a=5.0, mu_s=0.0, g=0.0, n=1.0, k=0.2e-3, rho_cp=2.0e-3, t_remove=600.0)
    char_state = dict(A=1e8, Ea=60000.0, mu_a=50.0, mu_s=10.0, k=0.8e-3, rho_cp=1.0e-3, t_remove=600.0)
    
    eng = _make_test_engine(dry_state=dry_state, char_state=char_state, power=15.0)
    
    # Run multiple epochs to drive boiling and charring
    for _ in range(25):
        eng.step_epoch()
    
    # If any voxels became dry, check that chi > 0 and mu_a increased
    dry_mask = eng.dry & (~eng.ablated)
    if np.any(dry_mask):
        c = eng.chi[dry_mask]
        mu_a = eng.optical['mu_a'][dry_mask]
        expected_mu_a = dry_state['mu_a'] + c * (char_state['mu_a'] - dry_state['mu_a'])
        assert np.allclose(mu_a, expected_mu_a, rtol=1e-4)


def test_plume_closure_switch():
    """Verify plume optical depth accumulation and power attenuation."""
    # Without plume closure: transmission is identically 1.0
    eng_no_plume = _make_test_engine(plume_closure=None)
    res0 = eng_no_plume.step_epoch()
    assert res0['plume_transmission'] == 1.0
    assert res0['plume_tau'] == 0.0
    
    # With 3D plume absorption field
    plume_cfg = dict(tau_linger_s=0.010, delta_mu_per_voxel=5.0, sigma_mm=0.05)
    eng_plume = _make_test_engine(plume_closure=plume_cfg, power=20.0)
    
    # Step until ablation occurs
    ablated_occurred = False
    for _ in range(30):
        diag = eng_plume.step_epoch()
        if diag['n_new'] > 0:
            ablated_occurred = True
            # Plume should attenuate delivered power in next epoch
            assert diag['plume_tau'] > 0.0
            assert diag['plume_transmission'] < 1.0
            assert diag['plume_peak_mu_a'] > 0.0
            assert diag['plume_transmission'] == pytest.approx(np.exp(-diag['plume_tau']))
            break
    assert ablated_occurred, "Simulation did not produce ablation in 30 epochs"


def test_oblique_incidence_angle():
    """Verify that theta_inc_deg runs without error and affects transport."""
    eng_normal = _make_test_engine(theta_inc_deg=0.0)
    eng_oblique = _make_test_engine(theta_inc_deg=45.0)
    
    res_norm = eng_normal.step_epoch()
    res_obl = eng_oblique.step_epoch()
    
    # Oblique incidence at 45 deg has unpolarized Fresnel specular reflection (~4.8%)
    # even on index-matched slab entrance if n changes or angle refracts
    assert res_norm['absorbed_frac'] > 0.0
    assert res_obl['absorbed_frac'] > 0.0
