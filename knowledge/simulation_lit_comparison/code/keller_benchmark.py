"""
Keller et al. (2016) Sci Rep 6:20936 — what the engine can and cannot say.

The paper measures mosquito (Anopheles stephensi) mortality; Table 1 gives the LD90 fluence for 10
exposure conditions. The engine has no mosquito optical model or mortality endpoint, so mortality
itself is NOT reproduced. What can be tested is the paper's physical interpretation: that lethal
fluence is set by thermal damage and depends on pulse length through thermal confinement.

Test: for the three 10.6 um (CO2, water-absorbed) conditions in Table 1 -- the only wavelength where
the absorber (water, mu_a ~ 850 /cm, the value used by McMillan et al. 2021) is known independently
of the mosquito -- compute with the engine's explicit bioheat solver the fluence at which an
Arrhenius damage integral first reaches Omega = 1 at the irradiated surface of a water-like body
(rho c = 4.18 J/cm^3/K, k = 0.6 W/m/K), 1-D, 25 C start, including post-pulse cooling. Two published
Arrhenius parameter sets are used to show how sensitive the prediction is. Nothing is tuned.
"""

import json

import numpy as np
from numba import njit

import bench_common  # noqa: F401
from bench_common import make_result, RAW_DIR
from simulation.voxel_thermal_3d import _thermal3d_step_explicit
from simulation.coupled_ablation import _accumulate_arrhenius, R_GAS

REF = "Keller et al. (2016) Sci Rep 6:20936, Table 1"

# Table 1 (verified against the PDF): wavelength nm, beam dia mm, pulse ms, LD90 fluence J/cm^2,
# LD90 pulse energy mJ, LD90 power W
TABLE1 = [
    (532, 2.3, 25.0, 1.8, 74, 3.0),
    (532, 2.0, 5.7, 1.7, 48, 8.5),
    (532, 5.8, 20.0, 0.6, 170, 8.5),
    (532, 2.5, 1e-5, 0.3, 10, 1.0e6),
    (1064, 2.5, 1e-5, 0.5, 26, 2.6e6),
    (1064, 2.5, 0.09, 1.8, 90, 1.0e3),
    (976, 2.5, 25.0, 5.6, 273, 10.9),
    (10600, 2.5, 25.0, 1.1, 52, 2.1),
    (10600, 2.5, 4.3, 0.9, 44, 10.2),
    (10600, 5.8, 25.0, 0.8, 211, 8.4),
    (1470, 2.5, 25.0, 4.0, 197, 7.9),
]
CO2_ROWS = [r for r in TABLE1 if r[0] == 10600]

MU_A_10P6 = 85.0                 # mm^-1 (850 /cm)
K_WATER = 0.6e-3                 # W/(mm K)
RHO_CP_WATER = 4.18e-3           # J/(mm^3 K)
T0 = 25.0
ARRHENIUS = {'Shimojo 2023 (liver)': (7.39e37, 2.577e5),
             'Henriques-Moritz (skin; McMillan 2021)': (3.1e98, 6.3e5)}


def omega_after_pulse(I_peak, t_pulse, A, Ea, mu_a=MU_A_10P6, dz=0.003, Nz=133, cool=0.06):
    """Max surface Omega for a rectangular pulse of peak irradiance I_peak [W/mm^2]."""
    Nx = Ny = 4
    dx = 0.05
    shape = (Nx, Ny, Nz)
    z = (np.arange(Nz) + 0.5) * dz
    Q = np.zeros(shape)
    Q[:, :, :] = (I_peak * mu_a * np.exp(-mu_a * z))[None, None, :]
    k = np.full(shape, K_WATER)
    rc = np.full(shape, RHO_CP_WATER)
    T = np.full(shape, T0)
    Tn = np.empty_like(T)
    om = np.zeros(shape)
    ablated = np.zeros(shape, dtype=np.bool_)
    dt = 1.5e-5   # explicit limit for dz = 3 um, dx = 50 um: 0.5/(alpha * sum 1/h^2) = 3.1e-5 s
    n_on = int(round(t_pulse / dt))
    n_cool = int(round(cool / dt))
    for i in range(n_on):
        _thermal3d_step_explicit(T, Tn, Q, k, rc, Nx, Ny, Nz, dx, dx, dz, dt, 0.0, T0, True)
        _accumulate_arrhenius(Tn, om, ablated, A, Ea / R_GAS, dt, Nx, Ny, Nz)
        T, Tn = Tn, T
    for i in range(n_cool):          # post-pulse relaxation (damage keeps accumulating)
        _thermal3d_step_explicit(T, Tn, Q, k, rc, Nx, Ny, Nz, dx, dx, dz, dt, 0.0, T0, False)
        _accumulate_arrhenius(Tn, om, ablated, A, Ea / R_GAS, dt, Nx, Ny, Nz)
        T, Tn = Tn, T
    return float(om[:, :, 0].max()), float(T0 + 0)  # (omega, unused)


def threshold_fluence(t_pulse_ms, A, Ea):
    """Fluence [J/cm^2] at which the surface Omega reaches 1 (bisection on a geometric bracket)."""
    t = t_pulse_ms * 1e-3
    lo, hi = 1e-4, 1.0     # J/mm^2  (= 0.01 .. 100 J/cm^2)
    for _ in range(28):
        mid = np.sqrt(lo * hi)
        om, _ = omega_after_pulse(mid / t, t, A, Ea)
        if om >= 1.0:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi) * 100.0)   # J/mm^2 -> J/cm^2


def run_keller(use_cache=True, verbose=True):
    cache_path = RAW_DIR / "keller_cache.json"
    cache = json.load(open(cache_path)) if (use_cache and cache_path.exists()) else {}
    results = []
    for name, (A, Ea) in ARRHENIUS.items():
        for (wl, d, tp, f_ld90, E, P) in CO2_ROWS:
            key = f"{name}|{tp}"
            if key not in cache:
                cache[key] = threshold_fluence(tp, A, Ea)
                json.dump(cache, open(cache_path, "w"))
            F_pred = cache[key]
            if verbose:
                print(f"  {name:42s} t={tp:5.1f} ms (beam {d} mm): predicted Omega=1 fluence {F_pred:5.2f} J/cm2 vs LD90 {f_ld90}")
            # Gaussian beams: the target sits at the peak of the beam (2x the nominal mean fluence)
            results.append(make_result(
                f"KE-{name.split()[0]}-{d}mm-{tp}ms", "validation",
                f"CO2 lethal fluence, {d} mm beam, {tp} ms ({name})", REF,
                "Fluence for thermal damage (Omega = 1) vs measured LD90", "J/cm^2", F_pred, f_ld90,
                0.5 * f_ld90, "50 % (factor-2 level; mosquito is not a semi-infinite water body)",
                "engine: 1-D explicit bioheat + Arrhenius; mortality NOT modelled; nominal fluence = E/(pi r^2)",
                series={'t_pulse_ms': tp, 'beam_mm': d, 'kinetics': name, 'ld90': f_ld90}))
    return results


if __name__ == "__main__":
    for r in run_keller():
        print(f"{r['id']:28s} {'PASS' if r['pass'] else 'FAIL'}  engine={r['model']:.2f}  LD90={r['reference_value']:.2f}")
