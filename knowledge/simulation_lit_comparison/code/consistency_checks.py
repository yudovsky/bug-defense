"""
Internal-consistency checks for the beetle-relevant engine features.

These are NOT literature validations: no paper data are used. Each compares the engine with an
exact / independent calculation, so the reference value is derived from the stated inputs, never
typed in. They replace the earlier 'Guo 2014' and 'Ospina-Rozo 2022' benchmarks, whose reference
values could not be traced to a source (see references/README.md, errata).

C1  Layered cuticle stack, transient conduction -- engine vs an independent ODE solve of the same
    finite-volume system (scipy BDF). Context only: series-resistance steady-state ratio.
C2  Radiant heating of a small body -- engine vs the lumped-capacitance solution that uses the
    engine's actual boundary geometry (surface loss on the two z faces only), run long enough to
    reach steady state.
C3  Oblique incidence on an absorbing, index-mismatched half-space -- engine vs exact unpolarised
    Fresnel reflectance and the exact mean absorption depth cos(theta_t)/mu_a of the refracted ray.
C4  Carbonisation kinetics -- engine kernel vs the closed-form isothermal solution and a
    high-accuracy ODE integration of a temperature ramp.
"""

import json

import numpy as np
from scipy.integrate import solve_ivp

import bench_common  # noqa: F401
from bench_common import make_result, RAW_DIR
from simulation.voxel_thermal_3d import _thermal3d_step_explicit
from simulation.voxel_mc_3d import run_voxel_mc3d
from simulation.coupled_ablation import _accumulate_char, R_GAS

CACHE = RAW_DIR / "consistency_cache.json"


# ---------------------------------------------------------------------------------------------
# C1: layered stack
# ---------------------------------------------------------------------------------------------
def _stack_properties(sandwich, Nz=50):
    k = np.full(Nz, 0.38e-3)            # W/(mm K): solid chitin
    rc = np.full(Nz, 2.4e-3)            # J/(mm^3 K)
    if sandwich:                        # 160 um air-filled trabecular core, assumed effective k and rho c
        k[8:40] = 0.08e-3
        rc[8:40] = 1.0e-3
    return k, rc


def engine_layered(sandwich, pulse=0.5, total=1.0, T_hot=50.0, T0=25.0, dt=5e-5, dz=0.005):
    Nx = Ny = 4
    Nz = 50
    k1, rc1 = _stack_properties(sandwich, Nz)
    shape = (Nx, Ny, Nz)
    k = np.broadcast_to(k1, shape).copy()
    rc = np.broadcast_to(rc1, shape).copy()
    T = np.full(shape, T0)
    Tn = np.empty_like(T)
    Q = np.zeros(shape)
    q_cum = 0.0
    for step in range(int(round(total / dt))):
        T[:, :, 0] = T_hot if step * dt < pulse else T0
        _thermal3d_step_explicit(T, Tn, Q, k, rc, Nx, Ny, Nz, 0.05, 0.05, dz, dt, 1.5e-5, T0, False)
        q_cum += float(k[0, 0, -1] * (Tn[:, :, -2].mean() - T0) / dz) * dt
        Tn[:, :, -1] = T0
        T, Tn = Tn, T
    return q_cum


def independent_layered(sandwich, pulse=0.5, total=1.0, T_hot=50.0, T0=25.0, dz=0.005):
    k, rc = _stack_properties(sandwich)
    kf = 2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:])
    N = len(k)

    def rhs(t, y):
        T = y[:N]
        T = T.copy()
        T[0] = T_hot if t < pulse else T0
        T[-1] = T0
        flux = kf * (T[1:] - T[:-1]) / dz ** 2          # W/mm^3 contribution across each face
        dT = np.zeros(N)
        dT[1:-1] = (flux[1:] - flux[:-1]) / rc[1:-1]
        dq = k[-1] * (T[-2] - T0) / dz
        return np.concatenate([dT, [dq]])

    y0 = np.concatenate([np.full(N, T0), [0.0]])
    # integrate the hot phase and the relaxation phase separately (the boundary switches at t = pulse)
    s1 = solve_ivp(rhs, [0, pulse], y0, method='BDF', rtol=1e-10, atol=1e-12)
    s2 = solve_ivp(rhs, [pulse, total], s1.y[:, -1], method='BDF', rtol=1e-10, atol=1e-12)
    return float(s2.y[-1, -1])


def check_layered():
    out = {}
    for name, sw in (('solid', False), ('sandwich', True)):
        out[name] = {'engine': engine_layered(sw), 'independent': independent_layered(sw)}
    k, _ = _stack_properties(False)
    ks, _ = _stack_properties(True)
    L = 0.005
    out['R_ratio'] = float((L * (1 / ks)).sum() / (L * (1 / k)).sum())
    out['engine_ratio'] = out['solid']['engine'] / out['sandwich']['engine']
    return out


# ---------------------------------------------------------------------------------------------
# C2: lumped radiant heating
# ---------------------------------------------------------------------------------------------
def engine_radiant(alpha, I=0.001, duration=400.0, dt=0.005):
    N = 10
    dx = 0.1
    shape = (N, N, N)
    k = np.full(shape, 0.35e-3)
    rc = np.full(shape, 2.8e-3)
    T0 = 22.0
    T = np.full(shape, T0)
    Tn = np.empty_like(T)
    Q = np.zeros(shape)
    Q[:, :, 0] = alpha * I / dx
    h = 2.05e-5
    ts, Ts = [], []
    for step in range(int(round(duration / dt))):
        _thermal3d_step_explicit(T, Tn, Q, k, rc, N, N, N, dx, dx, dx, dt, h, T0, True)
        T, Tn = Tn, T
        if step % int(round(1.0 / dt)) == 0:
            ts.append(step * dt)
            Ts.append(float(T.mean()) - T0)
    return np.array(ts), np.array(Ts), h, 2.8e-3, 1.0


def check_radiant():
    out = {}
    for name, a in (('dark', 0.92), ('shiny', 0.48)):
        t, dT, h, rc, L = engine_radiant(a)
        tau = rc * L / (2 * h)                       # two z faces lose heat; lateral faces are adiabatic
        dT_ss = a * 0.001 / (2 * h)
        out[name] = {'t': t.tolist(), 'dT': dT.tolist(), 'tau': tau, 'dT_ss': dT_ss,
                     'analytic': (dT_ss * (1 - np.exp(-t / tau))).tolist()}
    return out


# ---------------------------------------------------------------------------------------------
# C3: oblique incidence
# ---------------------------------------------------------------------------------------------
def fresnel_unpolarised(n, theta):
    st = np.sin(theta) / n
    ct = np.sqrt(1 - st * st)
    ci = np.cos(theta)
    rs = ((ci - n * ct) / (ci + n * ct)) ** 2
    rp = ((ct - n * ci) / (ct + n * ci)) ** 2
    return 0.5 * (rs + rp)


def check_oblique(angles=(0, 15, 30, 45, 60), n=1.55, mu_a=50.0):
    rows = []
    dz = 0.002
    Nz = 250
    for ang in angles:
        r = run_voxel_mc3d(n_photons=600000, spot_dia=0.2, theta_inc_deg=float(ang), Lx=4.0, Ly=4.0, Lz=Nz * dz,
                           Nx=40, Ny=40, Nz=Nz,
                           layers_config=[{'z_end': Nz * dz, 'mu_a': mu_a, 'mu_s': 0.0, 'g': 0.0, 'n': n}])
        A = r['A_3d'].sum(axis=(0, 1))
        z = r['grid_z']
        th = np.radians(ang)
        ct = np.sqrt(1 - (np.sin(th) / n) ** 2)
        rows.append({'angle': ang, 'R_engine': r['spectro_raw']['R_spec'], 'R_exact': float(fresnel_unpolarised(n, th)),
                     'zbar_engine': float((A * z).sum() / A.sum()), 'zbar_exact': float(ct / mu_a),
                     'N': 600000})
    return rows


# ---------------------------------------------------------------------------------------------
# C4: char kinetics
# ---------------------------------------------------------------------------------------------
def check_char(A=1e10, Ea=1.2e5):
    shape = (2, 2, 2)
    dry = np.ones(shape, dtype=np.bool_)
    ab = np.zeros(shape, dtype=np.bool_)
    # isothermal
    T = np.full(shape, 400.0)
    chi = np.zeros(shape)
    dt = 1e-3
    for _ in range(500):
        _accumulate_char(T, chi, dry, ab, A, Ea / R_GAS, dt, 2, 2, 2)
    k = A * np.exp(-Ea / (R_GAS * (400.0 + 273.15)))
    iso = {'engine': float(chi[0, 0, 0]), 'exact': float(1 - np.exp(-k * 0.5)), 'rate_per_s': float(k)}
    # temperature ramp 300 -> 500 C over 2 s
    chi = np.zeros(shape)
    nst = 2000
    dtr = 2.0 / nst
    for i in range(nst):
        T[:] = 300.0 + 200.0 * ((i + 0.5) / nst)
        _accumulate_char(T, chi, dry, ab, A, Ea / R_GAS, dtr, 2, 2, 2)
    sol = solve_ivp(lambda t, y: (1 - y) * A * np.exp(-Ea / (R_GAS * (300.0 + 100.0 * t + 273.15))), [0, 2.0], [0.0],
                    rtol=1e-12, atol=1e-14)
    ramp = {'engine': float(chi[0, 0, 0]), 'exact': float(sol.y[0, -1])}
    return {'iso': iso, 'ramp': ramp, 'A': A, 'Ea': Ea}


# ---------------------------------------------------------------------------------------------
def run_consistency(use_cache=True, verbose=True):
    if use_cache and CACHE.exists():
        d = json.load(open(CACHE))
    else:
        d = {'layered': check_layered(), 'radiant': check_radiant(), 'oblique': check_oblique(), 'char': check_char()}
        json.dump(d, open(CACHE, 'w'))
    res = []
    L = d['layered']
    for name in ('solid', 'sandwich'):
        e, i = L[name]['engine'], L[name]['independent']
        res.append(make_result(f"C1-{name}", "verification", f"Layered cuticle stack, cumulative ventral heat ({name})",
                               "Independent ODE solve (scipy BDF) of the same finite-volume system", "Heat into body cavity",
                               "mJ/mm^2", e * 1e3, i * 1e3, 0.005 * i * 1e3, "0.5 % (explicit-Euler time-step error)",
                               "k, rho c are assumed inputs (0.38 / 0.08 W/(m K) core); no literature value is compared",
                               series={'name': name}))
    R = d['radiant']
    for name in ('dark', 'shiny'):
        t = np.array(R[name]['t'])
        dT = np.array(R[name]['dT'])
        i_tau = int(np.argmin(abs(t - R[name]['tau'])))
        res.append(make_result(f"C2-{name}-tau", "verification", f"Radiant heating at t = tau ({name} body)",
                               "Lumped-capacitance solution with the engine's boundary geometry", "dT at t = tau", "K",
                               float(dT[i_tau]), float(R[name]['analytic'][i_tau]), 0.03 * float(R[name]['analytic'][i_tau]),
                               "3 % (Biot number 0.06: lumped model error)", f"tau = rho c L / (2 h) = {R[name]['tau']:.1f} s (two faces)"))
        res.append(make_result(f"C2-{name}-end", "verification", f"Radiant heating at t = 400 s ({name} body)",
                               "Lumped-capacitance solution with the engine's boundary geometry", "dT at 400 s", "K",
                               float(dT[-1]), float(R[name]['analytic'][-1]), 0.03 * float(R[name]['analytic'][-1]),
                               "3 % (Biot number 0.06)", f"steady state alpha I / (2h) = {R[name]['dT_ss']:.1f} K",
                               series={'t': t.tolist(), 'dT': dT.tolist(), 'analytic': R[name]['analytic'], 'name': name}))
    for r in d['oblique']:
        res.append(make_result(f"C3-R-{r['angle']}", "verification", f"Specular reflectance at {r['angle']} deg (n = 1.55)",
                               "Exact unpolarised Fresnel", "R_spec", "-", r['R_engine'], r['R_exact'], 1e-9 + 1e-12,
                               "deterministic", "", series={'angle': r['angle']}))
        res.append(make_result(f"C3-z-{r['angle']}", "verification", f"Mean absorption depth at {r['angle']} deg",
                               "Exact: cos(theta_t) / mu_a along the refracted ray", "Mean depth of absorbed energy", "mm",
                               r['zbar_engine'], r['zbar_exact'], 0.02 * r['zbar_exact'],
                               "2 % (Monte Carlo + 2 um voxel binning)", f"{r['N']:,} photons", series={'angle': r['angle']}))
    C = d['char']
    res.append(make_result("C4-iso", "verification", "Char fraction after 0.5 s at 400 C (isothermal)",
                           "Closed form 1 - exp(-k t)", "chi", "-", C['iso']['engine'], C['iso']['exact'], 1e-6,
                           "1e-6", f"test kinetics A = {C['A']:.0e} /s, Ea = {C['Ea']:.1e} J/mol (NOT measured values)"))
    res.append(make_result("C4-ramp", "verification", "Char fraction after a 300 -> 500 C ramp",
                           "High-accuracy ODE integration", "chi", "-", C['ramp']['engine'], C['ramp']['exact'], 1e-3,
                           "1e-3 (time-step error)", "same test kinetics"))
    if verbose:
        for r in res:
            print(f"  {r['id']:14s} {'PASS' if r['pass'] else 'FAIL'} engine={r['model']:.5g} ref={r['reference_value']:.5g}")
    return res, d


if __name__ == '__main__':
    run_consistency(use_cache=False)
