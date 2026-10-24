"""
McMillan et al. (2021) Lasers Surg Med 53(5):731-740 reproduction with the production engine.

Inputs (all from the paper, Methods): single microbeam ("pixel") of a fractional CO2 laser,
beam width 250 um, top-hat spatial profile (Fig. 3), triangular temporal pulse whose length scales
with energy (50 mJ -> ~60 ms, constant power), mu_a = 850 cm^-1 with no scattering, 0.6 x 0.6 x 1.2 mm
tissue block, initial temperature 5 C, no surface cooling, tissue = 75 % water + 25 % protein with
rho = 1000/(W + 0.649 Pro), cp = 1000 (4.2 W + 1.09 Pro), kappa = rho (6.28e-4 W + 1.17e-4 Pro),
water boiling at 100 C (energy W rho Lv, absorption/thermal properties change as it boils off) and
removal of the dehydrated voxel at the ablation temperature Ta (best fit in the paper: 500 C).

NOT specified in the paper and therefore a stated assumption here: the absorption coefficient of
the dehydrated (protein-only) tissue. Baseline = 25 % of the wet value (protein mass fraction);
sensitivity cases 0 % (transparent) and 100 % (unchanged) are reported alongside.
Differences from the paper's code: 7.5 um isotropic voxels (paper: 3.75 x 3.75 x 7.5 um), n = 1 (no
Fresnel), 2 ms optical updates, and the engine's explicit scheme at dt = 40 us.
"""

import json
import math
import time

import numpy as np

import bench_common  # noqa: F401
from bench_common import make_result, RAW_DIR
from simulation.coupled_ablation import CoupledAblationEngine

REF = "McMillan et al. (2021) Lasers Surg Med 53(5):731-740"
CACHE = RAW_DIR / "mcmillan_cache.json"

W0, PRO0 = 0.75, 0.25
RHO_WET = 1000.0 / (W0 + 0.649 * PRO0)                      # kg/m^3
CP_WET = 1000.0 * (4.2 * W0 + 1.09 * PRO0)                  # J/(kg K)
K_WET = RHO_WET * (6.28e-4 * W0 + 1.17e-4 * PRO0)           # W/(m K)
RHO_DRY = 1000.0 / 0.649                                    # dehydrated, protein-only (W = 0, Pro = 1)
CP_DRY = 1000.0 * 1.09
K_DRY = RHO_DRY * 1.17e-4
LV = 2.26e6                                                 # J/kg
MU_A_WET = 85.0                                             # mm^-1 (850 cm^-1)
H_LAT = W0 * RHO_WET * LV * 1e-9                            # J/mm^3 (energy to boil off the water)
BEAM_DIA = 0.25                                             # mm
T0 = 5.0


def pulse_profile(E_mJ, shape, n_epochs, dt):
    """Power per epoch [W] for a pulse of energy E whose length is 1.2 ms per mJ (constant power setting)."""
    tau = 1.2e-3 * E_mJ
    t = (np.arange(n_epochs) + 0.5) * dt
    if shape == 'triangular':
        p = 1.0 - np.abs(2.0 * t / tau - 1.0)
    elif shape == 'tophat':
        p = np.ones_like(t)
    else:
        raise ValueError(shape)
    p = np.clip(p, 0.0, None)
    return p * (E_mJ * 1e-3) / (p.sum() * dt)            # normalise to exactly E


def simulate_pixel(E_mJ, shape='triangular', dry_frac=0.25, Ta=500.0, dt_epoch=0.002, dt_sub=4e-5,
                   n_photons=20000, voxel=0.0075, beam_profile='tophat', plume_closure=None):
    N = 81
    Nz = int(round(1.2 / voxel))
    L = N * voxel
    shp = (N, N, Nz)
    n_ep = int(round(1.2e-3 * E_mJ / dt_epoch))
    eng = CoupledAblationEngine(
        optical_props={'mu_a': np.full(shp, MU_A_WET), 'mu_s': np.zeros(shp), 'g': np.zeros(shp), 'n': np.ones(shp)},
        k_grid=np.full(shp, K_WET * 1e-3), rho_cp_grid=np.full(shp, RHO_WET * CP_WET * 1e-9),
        h_lat_grid=np.full(shp, H_LAT), Lx=L, Ly=L, Lz=Nz * voxel, power=1.0, spot_dia=BEAM_DIA,
        t_abl=100.0, t_ambient=T0, dt_epoch=dt_epoch, dt_sub=dt_sub, n_photons=n_photons, h_eff=0.0,
        beam_profile=beam_profile,
        dry_state=dict(mu_a=dry_frac * MU_A_WET, mu_s=0.0, g=0.0, n=1.0, k=K_DRY * 1e-3,
                       rho_cp=RHO_DRY * CP_DRY * 1e-9, t_remove=Ta),
        plume_closure=plume_closure,
    )
    # Temporal pulse profile (clean base shape without plume suffix)
    base_shape = 'tophat' if 'tophat' in shape else 'triangular'
    P = pulse_profile(E_mJ, base_shape, n_ep, dt_epoch)
    ic = N // 2
    t_hist, d_hist = [], []
    t0 = time.time()
    for i in range(n_ep):
        eng.power = float(P[i])
        eng.step_epoch()
        t_hist.append(eng.t)
        d_hist.append(eng.centerline_depth_mm(ic, ic))
    return {
        'depth_mm': eng.centerline_depth_mm(ic, ic),
        't_s': t_hist, 'depth_hist_mm': d_hist,
        'xz_ablated': eng.ablated[:, ic, :].tolist(), 'xz_dry': eng.dry[:, ic, :].tolist(),
        'xz_T': eng.T[:, ic, :].round(1).tolist(),
        'voxel_mm': voxel, 'runtime_s': time.time() - t0,
    }


def _load_dig():
    with open(RAW_DIR / "mcmillan2021_fig3_digitized.json") as f:
        return json.load(f)


def run_mcmillan(energies=(50, 100, 150, 200, 250, 300, 350, 400), use_cache=True, verbose=True):
    dig = _load_dig()
    cache = json.load(open(CACHE)) if (use_cache and CACHE.exists()) else {}
    # no-plume baseline; the plume closure is calibrated and tested separately in plume_calibration.py
    cases = [
        ('triangular', 0.25, None),
        ('triangular', 0.0, None),
        ('triangular', 1.0, None),
        ('tophat', 0.25, None),
    ]
    for shape, frac, pc in cases:
        for E in energies:
            key = f"{shape}_dry{frac}_E{E}"
            if key not in cache:
                cache[key] = simulate_pixel(E, shape, frac, plume_closure=pc)
                with open(CACHE, "w") as f:
                    json.dump(cache, f)
            if verbose:
                print(f"  {shape:18s} dry={frac:4.2f} E={E:3d} mJ -> {cache[key]['depth_mm']*1000:5.0f} um "
                      f"({cache[key]['runtime_s']:.0f} s)", flush=True)

    exp = {p['energy_mJ']: p for p in dig['experiment']}
    sim500 = {p['energy_mJ']: p['depth_mm'] for p in dig['simulation']['500']}
    f7_tri = {p['energy_mJ']: p['depth_mm'] for p in dig['fig7']['triangular_temporal']}
    f7_top = {p['energy_mJ']: p['depth_mm'] for p in dig['fig7']['tophat_temporal']}
    results = []
    for E in energies:
        e = exp[E]
        sd = 0.5 * (e['err_hi_mm'] - e['err_lo_mm'])
        base = cache[f"triangular_dry0.25_E{E}"]['depth_mm']
        series = {
            'energy_mJ': E, 'exp_sd': sd, 'lit_sim_Ta500': sim500[E], 'lit_sim_fig7_tri': f7_tri[E],
            'lit_sim_fig7_tophat': f7_top[E],
            'engine_dry0': cache[f"triangular_dry0.0_E{E}"]['depth_mm'],
            'engine_dry1': cache[f"triangular_dry1.0_E{E}"]['depth_mm'],
            'engine_tophat_temporal': cache[f"tophat_dry0.25_E{E}"]['depth_mm'],
        }
        # 1. Baseline unshielded comparison (retains historical 2.1-2.6x failure visibility)
        results.append(make_result(
            f"M-{E}-exp", "validation", f"MTZ depth, {E} mJ/pixel (OCT experiment)", REF + ", Fig. 3",
            "MTZ depth", "mm", base, e['depth_mm'], 2.0 * sd + 2 * 0.0075,
            "2 x error-bar half-width (as drawn) + 2 voxels (15 um)",
            "triangular pulse, dry-tissue mu_a = 25 % of wet, Ta = 500 C, no plume", series=series))
        results.append(make_result(
            f"MX-{E}-sim", "cross-code", f"MTZ depth, {E} mJ/pixel (McMillan MCRT, Ta = 500 C)", REF + ", Fig. 3",
            "MTZ depth", "mm", base, sim500[E], 0.03,
            "0.03 mm (digitisation + differing voxel sizes)", "McMillan simulation digitised from Fig. 3",
            series=series))
    return results


if __name__ == "__main__":
    for r in run_mcmillan():
        print(f"{r['id']:14s} {'PASS' if r['pass'] else 'FAIL'} engine={r['model']*1000:5.0f} um  ref={r['reference_value']*1000:5.0f} um  "
              f"tol ±{r['tolerance_abs']*1000:.0f}")
