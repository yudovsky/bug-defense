"""
Validation tier: engine vs. published ex vivo experiments.

Shimojo et al. (2023) Sci Rep 13:11898 — 980 nm CW, Gaussian beam 1/e^2 diameter 2.8 mm,
scanned across porcine liver at 0.5/0.75/1.0 mm/s (50 W; vaporization + coagulation) and
1.0/1.25/1.5 mm/s (10 W; coagulation only). Inputs are taken verbatim from Supplementary
Table S1; the experimental means and +-1 SD (n = 3) are digitised from Figs. 5 and 7 by
digitize_shimojo_figures.py (raw_output/shimojo2023_digitized.json). Measurement
definitions follow the paper's Supplementary "Measurement of the coagulation and
vaporization regions".

Not reproduced (documented as excluded in README): Kim et al. 2023 (irradiation time /
pulse count not reported), McMillan et al. 2021 (multi-stage dehydration + 300–500 °C
ablation model not representable by the single-plateau engine), Keller et al. 2016
(mortality endpoint, not a thermal/ablation observable).
"""

import json
import math
import time

import numpy as np

import bench_common  # noqa: F401
from bench_common import make_result, RAW_DIR
from simulation.coupled_ablation import CoupledAblationEngine

SHIMOJO_REF = "Shimojo et al. (2023) Sci Rep 13:11898"

# Supplementary Table S1 (converted to mm units)
LIVER = dict(
    mu_a=0.07,            # mm^-1
    musp_native=0.51,     # mm^-1
    musp_coag=1.85,       # mm^-1
    g=0.9,
    n=1.44,
    k=5.43e-3 / 10.0,     # W/(cm K) -> W/(mm K)
    rho_cp=1.0e-3 * 3.542,  # g/mm^3 * J/(g K)
    A=7.39e37,            # 1/s
    Ea=2.577e5,           # J/mol
    Tv=100.0,
    Lv_vol=1.0e-3 * 2257.0,  # rho * Lv [J/mm^3]
    T0=22.0,
)
BEAM_DIA_MM = 2.8
SCAN_HALF_LENGTH = 10.0  # experiment scanned 20 mm


def _load_digitized():
    with open(RAW_DIR / "shimojo2023_digitized.json") as f:
        return json.load(f)


def _row_extent(mask_1d, d):
    idx = np.nonzero(mask_1d)[0]
    return float((idx.max() - idx.min() + 1) * d) if idx.size else 0.0


def cross_section_metrics(ablated_xz, coag_xz, dx, dz):
    """Shimojo Supplementary definitions applied to an x-z cross-section (index [ix, iz])."""
    out = {}
    abl_cols = [np.nonzero(ablated_xz[i])[0] for i in range(ablated_xz.shape[0])]
    depths = np.array([(c.max() + 1) * dz if c.size else 0.0 for c in abl_cols])
    if depths.max() > 0.0:
        out['vap_depth'] = float(depths.max())
        out['vap_width'] = max(_row_extent(ablated_xz[:, k], dx) for k in range(ablated_xz.shape[1]))
        deepest = np.flatnonzero(depths == depths.max())
        i_axis = int(deepest[len(deepest) // 2])
        k_floor = int(round(depths.max() / dz))  # first intact voxel below the crater on that axis
        below = coag_xz[i_axis, k_floor:]
        n_coag = int(np.argmin(below)) if not below.all() else below.size
        out['coag_depth'] = float(n_coag * dz)
        out['coag_width'] = _row_extent(coag_xz[:, k_floor - 1] | ablated_xz[:, k_floor - 1], dx)
    else:
        out['vap_depth'] = 0.0
        out['vap_width'] = 0.0
        cols = [np.nonzero(coag_xz[i])[0] for i in range(coag_xz.shape[0])]
        out['coag_depth'] = float(max(((c.max() + 1) * dz if c.size else 0.0) for c in cols))
        out['coag_width'] = max(_row_extent(coag_xz[:, k], dx) for k in range(coag_xz.shape[1]))
    return out


def simulate_shimojo_scan(power, speed, dop=True, n_photons=100000, voxel=0.1,
                          Lx=24.0, Ly=12.0, Lz=10.0, scan_margin=1.0, t_off=30.0,
                          dt_sub=0.004, dt_epoch=None, verbose=True):
    """
    Scan the beam along +y from y = scan_margin to Ly - scan_margin across a liver block
    (adiabatic boundaries, 22 °C) and evaluate the cross-section at y = Ly/2 after t_off of
    laser-off cooling. The 20 mm experimental scan is shortened to Ly - 2*margin; the
    measurement plane is >= 2 beam diameters from both scan ends.
    """
    Nx, Ny, Nz = int(round(Lx / voxel)), int(round(Ly / voxel)), int(round(Lz / voxel))
    shape = (Nx, Ny, Nz)
    musp0 = LIVER['musp_native']
    optical = {'mu_a': np.full(shape, LIVER['mu_a']),
               'mu_s': np.full(shape, musp0 / (1.0 - LIVER['g'])),
               'g': np.full(shape, LIVER['g']),
               'n': np.full(shape, LIVER['n'])}
    if dt_epoch is None:
        dt_epoch = min(0.1, voxel / speed)
    dt_sub = min(dt_sub, dt_epoch)
    eng = CoupledAblationEngine(
        optical_props=optical,
        k_grid=np.full(shape, LIVER['k']),
        rho_cp_grid=np.full(shape, LIVER['rho_cp']),
        h_lat_grid=np.full(shape, LIVER['Lv_vol']),
        Lx=Lx, Ly=Ly, Lz=Lz,
        power=power, spot_dia=BEAM_DIA_MM,
        t_abl=LIVER['Tv'], t_ambient=LIVER['T0'],
        dt_epoch=dt_epoch, dt_sub=dt_sub, n_photons=n_photons,
        h_eff=0.0, arrhenius=(LIVER['A'], LIVER['Ea']),
    )
    y0, y1 = scan_margin, Ly - scan_margin
    n_on = int(math.ceil((y1 - y0) / speed / dt_epoch))
    t_start = time.time()
    for e in range(n_on):
        eng.beam_center = (Lx / 2.0, y0 + speed * (e + 0.5) * dt_epoch)
        if dop:
            alpha = 1.0 - np.exp(-eng.Omega)
            musp = (1.0 - alpha) * musp0 + alpha * LIVER['musp_coag']
            intact = ~eng.ablated
            eng.optical['mu_s'][intact] = musp[intact] / (1.0 - LIVER['g'])
        eng.step_epoch()
        if verbose and (e + 1) % 25 == 0:
            print(f"    [{power:.0f} W, {speed} mm/s, DOP={dop}] epoch {e+1}/{n_on}  "
                  f"ablated={int(eng.ablated.sum())}  Tmax={eng.T.max():.1f}  ({time.time()-t_start:.0f} s)")
    # Laser off: thermal relaxation (damage keeps accumulating), 1 s epochs
    eng.power = 0.0
    eng.dt_epoch = 1.0
    eng.n_substeps = int(round(1.0 / eng.dt_sub))
    for _ in range(int(round(t_off))):
        eng.step_epoch()

    iy = int(round((Ly / 2.0) / voxel - 0.5))
    abl = eng.ablated[:, iy, :]
    coag = (eng.Omega[:, iy, :] >= 1.0) & ~abl
    m = cross_section_metrics(abl, coag, eng.dx, eng.dz)
    m['runtime_s'] = time.time() - t_start
    m['xz_ablated'] = abl
    m['xz_coag'] = coag
    return m


CACHE_PATH = RAW_DIR / "shimojo_case_cache.json"
DEFAULT_DT_EPOCH = 0.002   # s, optical update interval of the main benchmark run


def _case_worker(args):
    power, v, dop, n_photons, t_off, dom, dt_epoch = args
    mm = simulate_shimojo_scan(power, v, dop=dop, n_photons=n_photons, t_off=t_off, verbose=True,
                               dt_epoch=dt_epoch, dt_sub=dt_epoch, **dom)
    return {k: (val.tolist() if isinstance(val, np.ndarray) else val) for k, val in mm.items()}


def _cache_key(power, v, dop, n_photons, dom, t_off, dt_epoch):
    key = f"P{power}_v{v}_dop{dop}_n{n_photons}_L{dom['Lx']}x{dom['Ly']}x{dom['Lz']}_toff{t_off}"
    return key if dt_epoch is None else key + f"_dt{dt_epoch}"


def run_shimojo(n_photons=20000, dop=True, verbose=True, use_cache=True, dt_epoch=DEFAULT_DT_EPOCH,
                workers=6):
    """
    Domains: tissue depth 10 mm as in Shimojo (adiabatic boundaries). 50 W cases use a 24 mm
    wide block so the coagulation zone (up to ~12 mm wide) does not reach the side walls;
    10 W cases (coagulation < 4 mm) use 12 mm. dt_epoch is the optical update interval (the beam
    moves dt_epoch*speed between photon-transport updates); dt_epoch=None uses the coarse
    legacy schedule min(0.1 s, voxel/speed). Cases run in parallel processes and are cached
    per parameter set in raw_output/shimojo_case_cache.json.
    """
    dig = _load_digitized()
    results = []
    cases = [('fig5_50W', 50.0, [0.5, 0.75, 1.0], 30.0, ['vap_depth', 'vap_width', 'coag_depth', 'coag_width'],
              dict(Lx=24.0, Ly=12.0, Lz=10.0)),
             ('fig7_10W', 10.0, [1.0, 1.25, 1.5], 10.0, ['coag_depth', 'coag_width'],
              dict(Lx=12.0, Ly=12.0, Lz=10.0))]
    cache = {}
    if use_cache and CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    todo = []
    for fig, power, speeds, t_off, metrics, dom in cases:
        for v in speeds:
            key_c = _cache_key(power, v, dop, n_photons, dom, t_off, dt_epoch)
            if key_c not in cache:
                todo.append((key_c, (power, v, dop, n_photons, t_off, dom, dt_epoch)))
    if todo:
        import os
        from concurrent.futures import ProcessPoolExecutor
        threads = max(1, (os.cpu_count() or 6) // max(1, min(workers, len(todo))))
        os.environ['NUMBA_NUM_THREADS'] = str(threads)   # inherited by the worker processes
        print(f"  running {len(todo)} cases in {min(workers, len(todo))} processes x {threads} threads", flush=True)
        with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as ex:
            for (key_c, _), out in zip(todo, ex.map(_case_worker, [a for _, a in todo])):
                cache[key_c] = out
                with open(CACHE_PATH, "w") as f:
                    json.dump(cache, f)
    sections = {}
    for fig, power, speeds, t_off, metrics, dom in cases:
        for si, v in enumerate(speeds):
            m = cache[_cache_key(power, v, dop, n_photons, dom, t_off, dt_epoch)]
            legacy = cache.get(_cache_key(power, v, dop, 200000, dom, t_off, None))
            sections[f"{power:.0f}W_{v}"] = {'ablated': m['xz_ablated'], 'coag': m['xz_coag'],
                                             'Lx': dom['Lx'], 'Lz': dom['Lz']}
            if verbose:
                print(f"  {power:.0f} W @ {v} mm/s: " + ", ".join(f"{k}={m[k]:.2f}" for k in metrics)
                      + f"  ({m['runtime_s']:.0f} s)")
            for key in metrics:
                e = dig[fig][key]['exp'][si]
                sd = 0.5 * (e['bar_hi'] - e['bar_lo'])
                lit_dop = dig[fig][key]['dop'][si]['value']
                lit_nodop = dig[fig][key]['nodop'][si]['value']
                label = key.replace('vap', 'Vaporization').replace('coag', 'Coagulation').replace('_', ' ')
                fig_ref = SHIMOJO_REF + (", Fig. 5" if power > 20 else ", Fig. 7")
                series = {'speed_mm_s': v, 'power_W': power, 'exp_sd': sd,
                          'lit_sim_dop': lit_dop, 'lit_sim_nodop': lit_nodop,
                          'coarse_epoch_model': None if legacy is None else legacy[key],
                          'dt_epoch_s': dt_epoch}
                results.append(make_result(
                    f"S-{power:.0f}W-{v}-{key}", "validation",
                    f"{label}, {power:.0f} W, {v} mm/s (experiment)", fig_ref,
                    label, "mm", m[key], e['value'], 2.0 * sd + 0.1,
                    "2 x experimental SD (n = 3, digitised) + 1 voxel (0.1 mm)",
                    f"this engine run with DOP={dop}, {n_photons:,} photons per {dt_epoch*1000:.0f} ms optical update",
                    series=series))
                results.append(make_result(
                    f"X-{power:.0f}W-{v}-{key}", "cross-code",
                    f"{label}, {power:.0f} W, {v} mm/s (Shimojo DOP simulation)", fig_ref,
                    label, "mm", m[key], lit_dop, 0.2,
                    "2 voxels (0.1 mm discretisation in each code)",
                    f"Shimojo MCX+FDM with DOP (digitised); without DOP they report {lit_nodop:.2f} mm",
                    series=series))
    with open(RAW_DIR / "shimojo_cross_sections.json", "w") as f:
        json.dump(sections, f)
    return results


def run_all_validation(use_cache=True):
    return run_shimojo(use_cache=use_cache)


if __name__ == "__main__":
    for r in run_all_validation():
        print(f"{r['id']:28s} {'PASS' if r['pass'] else 'FAIL'}  model={r['model']:.2f}  exp={r['reference_value']:.2f} "
              f"± tol {r['tolerance_abs']:.2f} mm  | {r['notes']}")
