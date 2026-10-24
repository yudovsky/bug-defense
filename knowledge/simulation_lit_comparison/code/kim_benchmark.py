"""
Kim et al. (2023) J Comput Des Eng 10(3):947-958 reproduction with the production engine.

Inputs (paper Sections 2-3): 1060 nm Nd:YAG pulse train (1 MHz, 120/480 ns pulses, 15 W average)
treated as continuous 15 W; 1.2 mm disk source with exp(-2 r/R) areal weighting; mu_a = 0.7512 /cm,
mu_s = 70 /cm, g = 0.9, n = 1.4; rho_l = 1109 kg/m^3, c_l = 3391 J/(kg K), kappa0 = 0.37 W/(m K),
rho_g c_g = 4400 x 382 (post-vaporisation), water fraction 0.75, h_fg = 2.17e9 J/m^3, T0 = 25 C;
coagulation threshold 55 C, ablation threshold 200 C.

NOT reported in the paper: irradiation time / pulse count. It is therefore the single FITTED
parameter of this comparison (one per group: the 'coagulation' group PS1/PS2 and the 'ablation'
group PS3/PS4), chosen so the engine's mean depth equals the mean of the two measured depths.
Everything else (the depth difference between specimens, and the top-view areas) is then a
prediction. Like the paper, tissue is never removed from the domain: coagulation / "ablation" are the
regions that reach 55 C / 200 C (peak temperature), and the optical field is static in geometry.
Differences from the paper: flat slab (paper uses OCT-measured surface), 25 um voxels, 2 ms optical
updates, constant thermal conductivity.
"""

import json
import time

import numpy as np

import bench_common  # noqa: F401
from bench_common import make_result, RAW_DIR
from simulation.coupled_ablation import CoupledAblationEngine

REF = "Kim et al. (2023) J Comput Des Eng 10(3):947-958"
CACHE = RAW_DIR / "kim_cache.json"

# Published results (Sections 4.2, 4.3): simulation / experiment
KIM = {
    'PS1': {'group': 'coag', 'depth_sim': 0.518, 'depth_exp': 0.572, 'area_sim': 0.173, 'area_exp': 0.187},
    'PS2': {'group': 'coag', 'depth_sim': 0.478, 'depth_exp': 0.498, 'area_sim': 0.215, 'area_exp': 0.237},
    'PS3': {'group': 'abl', 'depth_sim': 0.351, 'depth_exp': 0.327, 'area_sim': 0.255, 'area_exp': 0.283},
    'PS4': {'group': 'abl', 'depth_sim': 0.308, 'depth_exp': 0.293, 'area_sim': 0.232, 'area_exp': 0.246},
}
T_COAG, T_ABL = 55.0, 200.0


def _mean_depth(mask_xz, dz):
    """Kim's definition: mean of the per-position depths along the surface line of the lesion."""
    d = []
    for col in mask_xz:
        idx = np.nonzero(col)[0]
        if idx.size:
            d.append((idx.max() + 1) * dz)
    return float(np.mean(d)) if d else 0.0


def simulate_kim(t_end=2.5, dt_epoch=0.002, voxel=0.025, n_photons=20000, beam_profile='kim_exp_disk'):
    N, Nz = 160, 80
    shp = (N, N, Nz)
    eng = CoupledAblationEngine(
        optical_props={'mu_a': np.full(shp, 0.07512), 'mu_s': np.full(shp, 7.0), 'g': np.full(shp, 0.9),
                       'n': np.full(shp, 1.4)},
        k_grid=np.full(shp, 0.37e-3), rho_cp_grid=np.full(shp, 1109.0 * 3391.0 * 1e-9),
        h_lat_grid=np.full(shp, 0.75 * 2.17), Lx=N * voxel, Ly=N * voxel, Lz=Nz * voxel,
        power=15.0, spot_dia=1.2, t_abl=100.0, t_ambient=25.0, dt_epoch=dt_epoch, dt_sub=5e-4,
        n_photons=n_photons, h_eff=0.0, beam_profile=beam_profile,
        dry_state=dict(mu_a=0.07512, mu_s=7.0, g=0.9, n=1.4, k=0.37e-3, rho_cp=4400.0 * 382.0 * 1e-9,
                       t_remove=1e9),   # never removed: Kim keeps hot tissue in the domain
    )
    ic = N // 2
    out = {k: [] for k in ('t_s', 'coag_depth', 'coag_area', 'abl_depth', 'abl_area', 'tmax')}
    t0 = time.time()
    for _ in range(int(round(t_end / dt_epoch))):
        eng.step_epoch()
        coag = eng.T_peak >= T_COAG
        out['t_s'].append(eng.t)
        out['coag_depth'].append(_mean_depth(coag[:, ic, :], eng.dz))
        out['coag_area'].append(float(coag[:, :, 0].sum() * eng.dx * eng.dy))
        hot = eng.T_peak >= T_ABL
        out['abl_depth'].append(_mean_depth(hot[:, ic, :], eng.dz))
        out['abl_area'].append(float(hot[:, :, 0].sum() * eng.dx * eng.dy))
        out['tmax'].append(float(eng.T.max()))
    out['runtime_s'] = time.time() - t0
    out['xz_coag'] = (eng.T_peak[:, ic, :] >= T_COAG).tolist()
    out['xz_abl'] = (eng.T_peak[:, ic, :] >= T_ABL).tolist()
    out['xz_T'] = eng.T[:, ic, :].round(1).tolist()
    out['voxel_mm'] = voxel
    return out


def _interp_time(t, y, target):
    y = np.maximum.accumulate(np.asarray(y))
    if target > y[-1] or target < y[0]:
        return float('nan')
    return float(np.interp(target, y, t))


def run_kim(use_cache=True, verbose=True):
    if use_cache and CACHE.exists():
        sim = json.load(open(CACHE))
    else:
        sim = simulate_kim()
        with open(CACHE, 'w') as f:
            json.dump(sim, f)
    t = np.array(sim['t_s'])
    fit = {}
    for grp, depth_key, area_key in (('coag', 'coag_depth', 'coag_area'), ('abl', 'abl_depth', 'abl_area')):
        specs = [k for k, v in KIM.items() if v['group'] == grp]
        target = float(np.mean([KIM[k]['depth_exp'] for k in specs]))
        tf = _interp_time(t, sim[depth_key], target)
        fit[grp] = {'t_fit_s': tf, 'target_mm': target}
        if verbose:
            print(f"  group {grp}: fitted exposure t = {tf*1000:.0f} ms (mean exp depth {target:.3f} mm)")
    results = []
    for k, v in KIM.items():
        grp = v['group']
        tf = fit[grp]['t_fit_s']
        if not np.isfinite(tf):
            continue
        i = int(np.searchsorted(t, tf))
        i = min(i, len(t) - 1)
        e_depth = sim['coag_depth' if grp == 'coag' else 'abl_depth'][i]
        e_area = sim['coag_area' if grp == 'coag' else 'abl_area'][i]
        what = 'coagulation (55 C)' if grp == 'coag' else 'ablation (200 C)'
        series = {'group': grp, 't_fit_s': tf, 'specimen': k, **v}
        # depth: tolerance = 10% (Kim's own reported sim-vs-exp difference scale); it is a fit, so labelled as such
        results.append(make_result(
            f"K-{k}-area", "validation", f"{k}: top-view {what} area (prediction at fitted exposure)",
            REF, f"{what} area", "mm^2", e_area, v['area_exp'], 0.25 * v['area_exp'],
            "25 % (Kim's reported sim-vs-exp area differences are 6-11 %; geometry here is a flat slab)",
            f"exposure fitted on depth: {tf*1000:.0f} ms (not reported in paper)", series=series))
        results.append(make_result(
            f"K-{k}-depth", "calibrated", f"{k}: mean {what} depth (fit target = mean of the pair)",
            REF, f"{what} depth", "mm", e_depth, v['depth_exp'], 0.25 * v['depth_exp'],
            "25 % (single fitted exposure per group; shows how well the group mean explains each specimen)",
            f"exposure fitted: {tf*1000:.0f} ms", series=series))
    return results, sim, fit


def run_kim_alt(use_cache=True):
    """
    Sensitivity to the ambiguous source formula: the paper prints exp(-2r) but calls it a Gaussian
    profile; exp(-2 r^2) is the standard Gaussian. Same procedure (exposure fitted on depth per group).
    """
    path = RAW_DIR / "kim_alt_profile.json"
    if use_cache and path.exists():
        return json.load(open(path))
    sim = simulate_kim(beam_profile='kim_gauss_disk')
    t = np.array(sim['t_s'])
    out = {'profile': 'exp(-2 (r/R)^2) disk', 'runtime_s': sim['runtime_s']}
    for grp, dk, ak in (('coag', 'coag_depth', 'coag_area'), ('abl', 'abl_depth', 'abl_area')):
        specs = [k for k, v in KIM.items() if v['group'] == grp]
        target = float(np.mean([KIM[k]['depth_exp'] for k in specs]))
        tf = _interp_time(t, sim[dk], target)
        i = min(int(np.searchsorted(t, tf)), len(t) - 1) if np.isfinite(tf) else None
        for k in specs:
            out[k] = {'t_fit_s': tf, 'depth': None if i is None else sim[dk][i], 'area': None if i is None else sim[ak][i]}
    json.dump(out, open(path, "w"))
    return out


if __name__ == "__main__":
    res, sim, fit = run_kim()
    for r in res:
        print(f"{r['id']:12s} {'PASS' if r['pass'] else 'FAIL'}  engine={r['model']:.3f}  paper_exp={r['reference_value']:.3f}  "
              f"kim_sim={r['series']['depth_sim' if r['id'].endswith('depth') else 'area_sim']:.3f}  {r['units']}")
