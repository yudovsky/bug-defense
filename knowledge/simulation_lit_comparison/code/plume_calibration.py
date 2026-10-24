"""
Plume-shielding closure: calibration with held-out testing (McMillan et al. 2021).

Closure (simulation/coupled_ablation.py): tau = C * mdot'', delivered power x exp(-tau), with
mdot'' the ablated mass flux [kg/m^2/s] and C = kappa_eff * h / v_exit [m^2 s / kg].

This is a ONE-PARAMETER CALIBRATION, not a validation. To keep it honest:
  * C is fitted on Fig. 3 experiment at 100, 200, 400 mJ only (log-ratio least squares, grid search);
  * it is then tested, unchanged, on (a) the five Fig. 3 energies not used in the fit and
    (b) Fig. 7 top-hat temporal pulse -- a different pulse shape -- from the paper's own simulation;
  * the fitted C is compared with a physical bracket (assumed ranges, NOT measured):
    kappa_eff 100-1000 m^2/kg, h 0.3-1 mm, v_exit 10-100 m/s  ->  C in [3e-4, 0.1] m^2 s / kg;
  * the fit is repeated with the lingering-plume accumulation switched off, because that term
    re-counts the residence time already contained in h / v_exit.
"""
import json

import numpy as np

import bench_common  # noqa: F401
from bench_common import make_result, RAW_DIR
import mcmillan_benchmark as mb

REF = "McMillan et al. (2021) Lasers Surg Med 53(5):731-740"
CACHE = RAW_DIR / "plume_calibration.json"
FIT_E = (100, 200, 400)
HOLD_E = (50, 150, 250, 300, 350)
ALL_E = (50, 100, 150, 200, 250, 300, 350, 400)
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
LINGER = {'linger_5ms': 0.005, 'no_linger': 1e-6}
C_PHYS = (100 * 0.3e-3 / 100, 1000 * 1.0e-3 / 10)       # [3e-4, 0.1]


def _depth(E, shape, C, tau_linger):
    cfg = dict(c_plume=C, tau_linger_s=tau_linger, rho_solid_kg_m3=1100.0)
    return mb.simulate_pixel(E, shape, 0.25, plume_closure=cfg)['depth_mm']


def calibrate(use_cache=True, verbose=True):
    cache = json.load(open(CACHE)) if (use_cache and CACHE.exists()) else {}
    dig = json.load(open(RAW_DIR / "mcmillan2021_fig3_digitized.json"))
    exp = {p['energy_mJ']: p['depth_mm'] for p in dig['experiment']}

    def get(key, fn):
        if key not in cache:
            cache[key] = fn()
            json.dump(cache, open(CACHE, 'w'))
        return cache[key]

    for lname, tl in LINGER.items():
        for C in C_GRID:
            for E in FIT_E:
                d = get(f"scan|{lname}|{C}|{E}", lambda: _depth(E, 'triangular', C, tl))
                if verbose:
                    print(f"  scan {lname:10s} C={C:<5} E={E} -> {d*1000:6.0f} um (exp {exp[E]*1000:.0f})", flush=True)
        score = {C: float(np.sum([np.log(max(cache[f"scan|{lname}|{C}|{E}"], 1e-4) / exp[E]) ** 2 for E in FIT_E])) for C in C_GRID}
        best = min(score, key=score.get)
        cache[f"best|{lname}"] = best
        cache[f"score|{lname}"] = score
        for E in HOLD_E:
            get(f"hold|{lname}|{E}", lambda: _depth(E, 'triangular', best, tl))
        for E in ALL_E:
            get(f"f7top|{lname}|{E}", lambda: _depth(E, 'tophat', best, tl))
        for E in FIT_E:
            cache[f"hold|{lname}|{E}"] = cache[f"scan|{lname}|{best}|{E}"]
        json.dump(cache, open(CACHE, 'w'))
    return cache, exp, dig


def run_plume(use_cache=True, verbose=True):
    cache, exp, dig = calibrate(use_cache, verbose)
    err = {p['energy_mJ']: 0.5 * (p['err_hi_mm'] - p['err_lo_mm']) for p in dig['experiment']}
    f7top = {p['energy_mJ']: p['depth_mm'] for p in dig['fig7']['tophat_temporal']}
    res = []
    for lname in LINGER:
        best = cache[f"best|{lname}"]
        for E in ALL_E:
            d = cache[f"hold|{lname}|{E}"]
            tier = 'calibrated' if E in FIT_E else 'validation'
            role = 'fit point' if E in FIT_E else 'HELD-OUT'
            res.append(make_result(
                f"PL-{lname}-{E}", tier, f"MTZ depth with plume closure, {E} mJ ({role}; {lname}, C = {best:g})", REF + ", Fig. 3",
                "MTZ depth", "mm", d, exp[E], 2.0 * err[E] + 2 * 0.0075,
                "2 x error-bar half-width + 2 voxels (15 um)", f"C = {best:g} m^2 s/kg fitted on {FIT_E} mJ only",
                series={'energy_mJ': E, 'variant': lname, 'C': best, 'fit_point': E in FIT_E}))
            res.append(make_result(
                f"PX-{lname}-{E}", "cross-code", f"Top-hat temporal pulse, {E} mJ (paper Fig. 7 simulation; {lname}, C = {best:g})",
                REF + ", Fig. 7", "MTZ depth", "mm", cache[f"f7top|{lname}|{E}"], f7top[E], 0.03,
                "0.03 mm (digitisation + differing voxel sizes)", "held-out pulse shape (C fitted on triangular pulse)",
                series={'energy_mJ': E, 'variant': lname, 'C': best}))
    return res, cache


if __name__ == '__main__':
    r, c = run_plume(use_cache=True)
    for lname in LINGER:
        print(lname, 'best C =', c[f"best|{lname}"], 'scores', {k: round(v, 3) for k, v in c[f"score|{lname}"].items()})
