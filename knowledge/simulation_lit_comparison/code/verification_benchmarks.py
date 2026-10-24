"""
Verification tier: engine results vs. exact analytic solutions and canonical
radiative-transfer tables. Every model number is produced by the production engine
(simulation.voxel_mc_3d, simulation.voxel_thermal_3d, simulation.coupled_ablation)
with no calibration, clipping, or fallback values.

V1  Slab radiative transfer (van de Hulst tables; MCML paper, Wang et al. 1995)
V2  Semi-infinite diffuse reflectance (exact, Chandrasekhar H-function)
V3  Transient heating by Beer-Lambert volumetric absorption, insulated surface
    (closed-form solution, Carslaw & Jaeger-type)
V4  Steady-state ablation-front velocity, v = I / (rho*c*(T_v - T_0) + rho*L)
    (energy-balance / Stefan steady state)
"""

import numpy as np
from scipy.special import erfcx, erfc
from scipy.optimize import brentq

import bench_common  # noqa: F401  (sets sys.path)
from bench_common import make_result
from simulation.voxel_mc_3d import run_voxel_mc3d
from simulation.coupled_ablation import CoupledAblationEngine

# ---------------------------------------------------------------------------
# Canonical reference values. See references/README.md for sources.
# ---------------------------------------------------------------------------
# Slab: mu_a = 10 /cm, mu_s = 90 /cm, g = 0.75, d = 0.02 cm (optical thickness 2, albedo 0.9)
SLAB_MATCHED_RD = 0.09739   # van de Hulst (1980), as tabulated in Wang, Jacques & Zheng (1995)
SLAB_MATCHED_TT = 0.66096


def semi_infinite_isotropic_rd(albedo, mu0=1.0, n_quad=400):
    """
    Exact diffuse reflectance of an index-matched semi-infinite isotropic scatterer for a
    collimated beam at direction cosine mu0: Rd = 1 - H(mu0) sqrt(1 - albedo), where H is
    Chandrasekhar's H-function (Chandrasekhar 1960, Radiative Transfer, Ch. V), solved here
    by fixed-point iteration on a Gauss-Legendre grid.
    """
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n_quad)
    mu = 0.5 * (x + 1.0)
    wt = 0.5 * w
    H = np.ones_like(mu)
    for _ in range(5000):
        H_new = 1.0 / (1.0 - 0.5 * albedo * mu * (wt * H / (mu[:, None] + mu[None, :])).sum(axis=1))
        if np.max(np.abs(H_new - H)) < 1e-14:
            H = H_new
            break
        H = H_new
    H_mu0 = 1.0 / (1.0 - 0.5 * albedo * mu0 * np.sum(wt * H / (mu0 + mu)))
    return float(1.0 - H_mu0 * np.sqrt(1.0 - albedo))


def _mc_batches(n_batches, n_photons, **kwargs):
    rows = []
    for _ in range(n_batches):
        r = run_voxel_mc3d(n_photons=n_photons, **kwargs)
        rows.append(r)
    return rows


def _mean_se(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def bench_v1_slab_matched(n_batches=10, n_photons=200000):
    """Optical thickness 2 slab, index matched (n = 1): Rd and Tt vs van de Hulst."""
    common = dict(spot_dia=0.01, Lx=20.0, Ly=20.0, Lz=0.2, Nx=20, Ny=20, Nz=10,
                  layers_config=[{'z_end': 0.2, 'mu_a': 1.0, 'mu_s': 9.0, 'g': 0.75, 'n': 1.0}])
    runs = _mc_batches(n_batches, n_photons, **common)
    rd, rd_se = _mean_se([r['spectro_raw']['R_diff'] for r in runs])
    tt, tt_se = _mean_se([r['spectro_raw']['Transmission'] for r in runs])
    lat = np.mean([r['spectro']['Lateral_loss'] for r in runs])
    ref = "Wang, Jacques & Zheng (1995) Table (van de Hulst 1980)"
    note = f"{n_batches}x{n_photons:,} photons; tolerance = 3 x batch standard error; lateral loss {lat:.3f}%"
    return [
        make_result("V1a", "verification", "Slab Rd (n=1, tau=2, a=0.9, g=0.75)", ref,
                    "Diffuse reflectance Rd", "-", rd, SLAB_MATCHED_RD, 3 * rd_se + 1e-5,
                    "3 x MC standard error", note, series={'se': rd_se, 'n_batches': n_batches, 'n_photons': n_photons}),
        make_result("V1b", "verification", "Slab Tt (n=1, tau=2, a=0.9, g=0.75)", ref,
                    "Total transmittance Tt", "-", tt, SLAB_MATCHED_TT, 3 * tt_se + 1e-5,
                    "3 x MC standard error", note, series={'se': tt_se, 'n_batches': n_batches, 'n_photons': n_photons}),
    ]


def bench_v2_semi_infinite(n_batches=10, n_photons=200000):
    """Semi-infinite isotropic scatterer, albedo 0.9, index matched: Rd vs exact H-function result."""
    common = dict(spot_dia=0.01, Lx=20.0, Ly=20.0, Lz=4.0, Nx=20, Ny=20, Nz=8,
                  layers_config=[{'z_end': 4.0, 'mu_a': 1.0, 'mu_s': 9.0, 'g': 0.0, 'n': 1.0}])
    runs = _mc_batches(n_batches, n_photons, **common)
    rd, rd_se = _mean_se([r['spectro_raw']['R_diff'] for r in runs])
    leak = np.mean([r['spectro']['Transmission'] + r['spectro']['Lateral_loss'] for r in runs])
    return [make_result(
        "V2", "verification", "Semi-infinite Rd (n=1, a=0.9, g=0)",
        "Chandrasekhar (1960) H-function, exact",
        "Diffuse reflectance Rd", "-", rd, semi_infinite_isotropic_rd(0.9), 3 * rd_se + 1e-5,
        "3 x MC standard error",
        f"{n_batches}x{n_photons:,} photons; transmission+lateral leakage {leak:.4f}%")]


def bench_v1c_fresnel_slab(n_photons=400000):
    """
    Absorbing, non-scattering n = 1.5 slab in air at normal incidence. Exact (incoherent
    multiple internal reflection): T = (1-R)^2 e^-tau / (1 - R^2 e^-2tau),
    Rd (excl. specular) = R (1-R)^2 e^-2tau / (1 - R^2 e^-2tau), R = ((n-1)/(n+1))^2.
    A one-voxel air layer below the slab provides the rear slab/air interface.
    """
    n, mu_a, d = 1.5, 1.0, 0.5
    R = ((n - 1.0) / (n + 1.0)) ** 2
    e1, e2 = np.exp(-mu_a * d), np.exp(-2.0 * mu_a * d)
    T_ref = (1 - R) ** 2 * e1 / (1 - R ** 2 * e2)
    Rd_ref = R * (1 - R) ** 2 * e2 / (1 - R ** 2 * e2)
    r = run_voxel_mc3d(n_photons=n_photons, spot_dia=0.01, Lx=2.0, Ly=2.0, Lz=0.55, Nx=4, Ny=4, Nz=11,
                       layers_config=[{'z_end': 0.5, 'mu_a': mu_a, 'mu_s': 0.0, 'g': 0.0, 'n': n},
                                      {'z_end': 0.55, 'mu_a': 0.0, 'mu_s': 0.0, 'g': 0.0, 'n': 1.0}])
    s = r['spectro_raw']
    se_T = np.sqrt(T_ref * (1 - T_ref) / n_photons)
    se_R = np.sqrt(Rd_ref * (1 - Rd_ref) / n_photons)
    ref = "Exact incoherent multiple-reflection slab formula (Fresnel, normal incidence)"
    return [
        make_result("V1c", "verification", "Fresnel slab transmittance (n=1.5, tau=0.5)", ref,
                    "Total transmittance", "-", s['Transmission'], T_ref, 4 * se_T, "4 x binomial SE"),
        make_result("V1d", "verification", "Fresnel slab internal-reflection return (n=1.5)", ref,
                    "Diffuse reflectance (excl. specular)", "-", s['R_diff'], Rd_ref, 4 * se_R, "4 x binomial SE"),
        make_result("V1e", "verification", "Entrance specular reflection (n=1.5)", ref,
                    "Specular reflectance", "-", s['R_spec'], R, 1e-9, "deterministic"),
    ]


def bench_v1f_oblique_refraction(n_photons=400000):
    """
    45 deg incidence on an absorbing n = 1.5 half-space (index-matched rear boundary):
    T = (1 - R_45) exp(-mu_a d / cos(theta_t)), Snell refraction + unpolarised Fresnel.
    """
    n, mu_a, d, th = 1.5, 1.0, 1.0, np.radians(45.0)
    st = np.sin(th) / n
    ct = np.sqrt(1 - st ** 2)
    ci = np.cos(th)
    rs = ((ci - n * ct) / (ci + n * ct)) ** 2
    rp = ((ct - n * ci) / (ct + n * ci)) ** 2
    R45 = 0.5 * (rs + rp)
    T_ref = (1 - R45) * np.exp(-mu_a * d / ct)
    r = run_voxel_mc3d(n_photons=n_photons, spot_dia=0.2, theta_inc_deg=45.0, Lx=8.0, Ly=8.0, Lz=d,
                       Nx=40, Ny=40, Nz=20,
                       layers_config=[{'z_end': d, 'mu_a': mu_a, 'mu_s': 0.0, 'g': 0.0, 'n': n}])
    se = np.sqrt(T_ref * (1 - T_ref) / n_photons)
    return [make_result("V1f", "verification", "45 deg refraction into absorber (n=1.5)",
                        "Snell's law + unpolarised Fresnel, exact", "Transmittance", "-",
                        r['spectro_raw']['Transmission'], T_ref, 4 * se, "4 x binomial SE",
                        f"theta_t = {np.degrees(np.arcsin(st)):.2f} deg, R_45 = {R45:.5f}")]


def bench_v2b_semi_infinite_mismatched(n_batches=10, n_photons=200000):
    """
    Semi-infinite isotropic scatterer, albedo 0.9, n = 1.5 in air: Rd vs MCML manual Table 6.2
    (Giovanelli 1955: 0.2600; MCML 0.25907 +- 0.00170; Prahl et al. 1989 0.26079 +- 0.00079).
    Exercises angle-resolved Fresnel reflection / TIR at the exit surface.
    """
    common = dict(spot_dia=0.01, Lx=20.0, Ly=20.0, Lz=4.0, Nx=20, Ny=20, Nz=8,
                  layers_config=[{'z_end': 4.0, 'mu_a': 1.0, 'mu_s': 9.0, 'g': 0.0, 'n': 1.5}])
    runs = _mc_batches(n_batches, n_photons, **common)
    rd, rd_se = _mean_se([r['spectro_raw']['R_diff'] + r['spectro_raw']['R_spec'] for r in runs])
    rd_diffuse = float(np.mean([r['spectro_raw']['R_diff'] for r in runs]))
    return [make_result(
        "V2b", "verification", "Semi-infinite total R (n=1.5, a=0.9, g=0)",
        "Wang & Jacques (1992) MCML manual Table 6.2 (Giovanelli 1955)",
        "Total reflectance (specular + diffuse)", "-", rd, 0.2600, 3 * rd_se + 0.0017,
        "3 x MC standard error + 0.0017 (spread between published reference values)",
        series={'se': rd_se, 'diffuse_only': rd_diffuse, 'n_batches': n_batches, 'n_photons': n_photons},
        notes=f"{n_batches}x{n_photons:,} photons; diffuse-only Rd = {rd_diffuse:.5f}, specular 0.04. "
        "Table 6.2 values include the specular term: an independent 1D MC gives diffuse 0.21982 "
        "(+0.04 = 0.2598) and exact 0.4148 for n = 1. Published: MCML 0.25907+-0.00170, "
        "Prahl 0.26079+-0.00079")]


def beer_lambert_surface_temperature_rise(I, mu_a, k, rho_cp, t):
    """
    Surface temperature rise of an insulated semi-infinite medium with volumetric
    source Q(z) = I mu_a exp(-mu_a z) switched on at t = 0:
        dT(0,t) = (I/k) [ 2 sqrt(alpha t / pi) - (1/mu_a)(1 - exp(mu_a^2 alpha t) erfc(mu_a sqrt(alpha t))) ]
    """
    alpha = k / rho_cp
    t = np.asarray(t, dtype=float)
    x = mu_a * np.sqrt(alpha * t)
    return (I / k) * (2.0 * np.sqrt(alpha * t / np.pi) - (1.0 - erfcx(x)) / mu_a)


def _quasi_1d_engine(mu_a, k, rho_cp, h_lat, t_abl, Lz, Nz, I_target, dt_epoch, dt_sub, n_photons):
    """
    Quasi-1D column (0.2 x 0.2 mm, insulated sides) under a beam much wider than the column
    (w0 = 10x the half-width), non-scattering index-matched absorber.
    """
    Nx = Ny = 4
    Lx = Ly = 0.2
    w0 = 10.0 * (Lx / 2.0)
    spot_dia = 2.0 * w0
    power = I_target * np.pi * w0 ** 2 / 2.0  # peak irradiance 2P/(pi w0^2) = I_target
    shape = (Nx, Ny, Nz)
    optical = {'mu_a': np.full(shape, mu_a), 'mu_s': np.zeros(shape),
               'g': np.zeros(shape), 'n': np.ones(shape)}
    return CoupledAblationEngine(
        optical_props=optical,
        k_grid=np.full(shape, k), rho_cp_grid=np.full(shape, rho_cp),
        h_lat_grid=np.full(shape, h_lat),
        Lx=Lx, Ly=Ly, Lz=Lz, power=power, spot_dia=spot_dia,
        t_abl=t_abl, t_ambient=20.0, dt_epoch=dt_epoch, dt_sub=dt_sub,
        n_photons=n_photons, h_eff=0.0,
    )


def bench_v3_beer_lambert_heating():
    """Transient surface heating vs closed form (no phase change)."""
    mu_a, k, rho_cp = 10.0, 0.53e-3, 4.18e-3   # mm^-1, W/(mm K), J/(mm^3 K)
    I_target = 0.5                                # W/mm^2
    eng = _quasi_1d_engine(mu_a, k, rho_cp, h_lat=1e9, t_abl=1e9, Lz=1.5, Nz=150,
                           I_target=I_target, dt_epoch=0.05, dt_sub=1e-4, n_photons=2_000_000)
    times, dT_model, I_eff_hist = [], [], []
    absorbed_J = 0.0
    for _ in range(10):  # 0.5 s
        eng.step_epoch()
        I_eff = float(np.sum(eng.Q) * eng.voxel_vol / (eng.Lx * eng.Ly))  # delivered W/mm^2
        I_eff_hist.append(I_eff)
        absorbed_J += np.sum(eng.Q) * eng.voxel_vol * eng.dt_epoch
        times.append(eng.t)
        dT_model.append(float(np.mean(eng.T[:, :, 0]) - 20.0))
    I_eff = float(np.mean(I_eff_hist))
    times = np.array(times)
    dT_model = np.array(dT_model)
    dT_ref = beer_lambert_surface_temperature_rise(I_eff, mu_a, k, rho_cp, times)
    stored_J = float(np.sum(rho_cp * (eng.T - 20.0)) * eng.voxel_vol)
    profile = {'z_mm': ((np.arange(eng.Nz) + 0.5) * eng.dz).tolist(),
               'dT_engine': (eng.T.mean(axis=(0, 1)) - 20.0).tolist(), 'I_eff': I_eff, 'mu_a': mu_a,
               'k': k, 'rho_cp': rho_cp, 't_end': float(times[-1])}

    tol_rel = 0.02
    res = [
        make_result("V3a", "verification", "Beer-Lambert surface heating, t = 0.5 s",
                    "Closed-form solution (Carslaw & Jaeger type)", "Surface dT", "K",
                    dT_model[-1], dT_ref[-1], tol_rel * dT_ref[-1],
                    "2% (dz = 10 um, cell-averaged surface value)",
                    f"mu_a = {mu_a} /mm, I_delivered = {I_eff:.4f} W/mm^2 (measured from MC deposition)",
                    series={'t_s': times.tolist(), 'model_dT': dT_model.tolist(), 'ref_dT': dT_ref.tolist(), 'profile': profile}),
        make_result("V3b", "verification", "Thermal energy conservation (insulated)",
                    "Energy balance", "Stored / absorbed energy", "J",
                    stored_J, absorbed_J, 1e-6 * absorbed_J, "1 ppm (round-off)",
                    "sum(rho_cp dT V) vs sum(Q V dt) with insulated boundaries"),
    ]
    return res


def bench_v4_ablation_velocity(dt_epochs=(0.010, 0.002)):
    """Steady-state ablation front velocity vs energy balance, for two optical update intervals."""
    mu_a, k, rho_cp = 60.0, 0.53e-3, 4.18e-3
    T0, Tv = 20.0, 100.0
    h_lat = 2.257  # J/mm^3, water
    I_target = 0.5
    t_end = 1.5
    results = []
    for dt_epoch in dt_epochs:
        eng = _quasi_1d_engine(mu_a, k, rho_cp, h_lat=h_lat, t_abl=Tv, Lz=0.6, Nz=120,
                               I_target=I_target, dt_epoch=dt_epoch, dt_sub=5e-5, n_photons=1_000_000)
        times, depth, Ieff, t_surf = [], [], [], []
        n_ep = int(round(t_end / dt_epoch))
        for _ in range(n_ep):
            eng.step_epoch()
            Ieff.append(float(np.sum(eng.Q) * eng.voxel_vol / (eng.Lx * eng.Ly)))
            times.append(eng.t)
            depth.append(float(np.sum(eng.ablated) / (eng.Nx * eng.Ny) * eng.dz))
            col = eng.T[:, :, :]
            surf = [col[i, j, np.argmax(~eng.ablated[i, j, :])] for i in range(eng.Nx) for j in range(eng.Ny)]
            t_surf.append(float(np.mean(surf)))
        times = np.array(times)
        depth = np.array(depth)
        I_eff = float(np.mean(Ieff))
        v_ref = I_eff / (rho_cp * (Tv - T0) + h_lat)
        fit = times >= times[-1] - 0.5
        v_model = float(np.polyfit(times[fit], depth[fit], 1)[0])
        # Reference line: energy-balance slope through the centroid of the engine's fit window
        # (only the steady-state slope is tested; the onset transient has no closed form here).
        t_c, d_c = float(times[fit].mean()), float(depth[fit].mean())
        ref_line = np.where(times >= times[fit][0] - 0.3, d_c + v_ref * (times - t_c), np.nan)

        t_boil_ref = brentq(lambda t: beer_lambert_surface_temperature_rise(I_eff, mu_a, k, rho_cp, t) - (Tv - T0), 1e-6, 10.0)
        results.append(make_result(
            f"V4-{int(dt_epoch*1000)}ms", "verification",
            f"Steady ablation velocity (optical update every {dt_epoch*1000:.0f} ms)",
            "Energy balance: v = I / (rho c (Tv - T0) + rho L)", "Front velocity", "mm/s",
            v_model, v_ref, 0.05 * v_ref,
            "5% (dz = 5 um front quantisation + epoch-end ablation lag)",
            f"I_delivered = {I_eff:.4f} W/mm^2; analytic boiling onset t_b = {t_boil_ref*1000:.1f} ms",
            series={'t_s': times.tolist(), 'depth_mm': depth.tolist(), 'surface_T': t_surf,
                    'ref_depth_mm': [None if np.isnan(v) else float(v) for v in ref_line],
                    'fit_window_s': [float(times[fit][0]), float(times[-1])]}))
    return results


def beer_lambert_profile(z, t, I, mu_a, k, rho_cp, n_tau=4000):
    """
    Closed-form temperature rise T(z, t) - T0 of an insulated semi-infinite solid with source
    Q(z) = I mu exp(-mu z) switched on at t = 0 (Carslaw & Jaeger image-source solution):
      dT = (I mu / rho c) * int_0^t 1/2 exp(a^2)[exp(-mu z) erfc(a - b) + exp(mu z) erfc(a + b)] dtau,
      a = mu sqrt(alpha tau), b = z / (2 sqrt(alpha tau)).
    """
    from scipy.special import erfc
    alpha = k / rho_cp
    tau = (np.arange(n_tau) + 0.5) / n_tau * t
    out = []
    for zz in np.atleast_1d(z):
        a = mu_a * np.sqrt(alpha * tau)
        b = zz / (2.0 * np.sqrt(alpha * tau))
        f = 0.5 * (np.exp(a * a - mu_a * zz) * erfc(a - b) + np.exp(-b * b) * erfcx(a + b))
        out.append(I * mu_a / rho_cp * np.sum(f) * (t / n_tau))
    return np.array(out)


def bench_cj_mu_sweep(mu_list=(1.0, 3.0, 10.0, 30.0), t_end=0.5):
    """Surface heating vs absorption coefficient (closed form: Carslaw & Jaeger), engine vs exact."""
    k, rho_cp = 0.53e-3, 4.18e-3
    results = []
    for mu in mu_list:
        dz = 0.01 if mu <= 10 else 0.005
        Nz = int(round(1.5 / dz))
        eng = _quasi_1d_engine(mu, k, rho_cp, h_lat=1e9, t_abl=1e9, Lz=Nz * dz, Nz=Nz,
                               I_target=0.5, dt_epoch=0.05, dt_sub=5e-5, n_photons=400_000)
        Is = []
        for _ in range(int(round(t_end / 0.05))):
            eng.step_epoch()
            Is.append(float(np.sum(eng.Q) * eng.voxel_vol / (eng.Lx * eng.Ly)))
        # closed form needs the INCIDENT flux; the deposited flux is smaller by the fraction that
        # leaves through the bottom of the finite column
        I_eff = float(np.mean(Is)) / (1.0 - np.exp(-mu * Nz * dz))
        ref = float(beer_lambert_profile(np.array([0.5 * dz]), t_end, I_eff, mu, k, rho_cp)[0])
        model = float(np.mean(eng.T[:, :, 0]) - 20.0)
        results.append(make_result(
            f"V3c-mu{mu:g}", "verification", f"Cell-centre heating, mu_a = {mu:g} /mm, t = {t_end} s",
            "Carslaw & Jaeger image-source solution", "dT at first cell centre", "K", model, ref,
            0.03 * ref, f"3 % (cell-centred sampling, dz = {dz * 1000:g} um)",
            f"I_incident = {I_eff:.4f} W/mm^2 (deposited flux / absorbed fraction)", series={'mu_a': mu, 'dz': dz}))
    return results


def bench_chandrasekhar_sweep(albedos=(0.5, 0.7, 0.9, 0.95, 0.99), n_batches=5, n_photons=100000):
    """Semi-infinite isotropic scatterer, n = 1, normal incidence: Rd vs exact H-function across albedo."""
    results = []
    for a in albedos:
        mu_t = 10.0
        runs = _mc_batches(n_batches, n_photons, spot_dia=0.01, Lx=60.0, Ly=60.0, Lz=30.0, Nx=6, Ny=6, Nz=30,
                           layers_config=[{'z_end': 30.0, 'mu_a': mu_t * (1 - a), 'mu_s': mu_t * a, 'g': 0.0, 'n': 1.0}])
        rd, se = _mean_se([r['spectro_raw']['R_diff'] for r in runs])
        ref = semi_infinite_isotropic_rd(a)
        results.append(make_result(
            f"V2c-a{a}", "verification", f"Semi-infinite Rd, albedo {a} (n=1, g=0)",
            "Chandrasekhar H-function (exact)", "Diffuse reflectance Rd", "-", rd, ref,
            3 * se + 2e-4, "3 x MC standard error (+2e-4 for depth truncation)",
            f"{n_batches}x{n_photons:,} photons", series={'albedo': a, 'se': se}))
    return results


from numba import njit as _njit


@_njit
def _fresnel_1d(n1, n2, ci):
    if ci > 0.999999:
        r = (n1 - n2) / (n1 + n2)
        return r * r
    st2 = (n1 / n2) ** 2 * (1 - ci * ci)
    if st2 >= 1:
        return 1.0
    ct = np.sqrt(1 - st2)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci)
    return 0.5 * (rs * rs + rp * rp)


@_njit
def independent_semi_infinite_mc(N, albedo, n, seed):
    """From-scratch 1-D isotropic-scattering Monte Carlo (independent of the voxel code): total R."""
    np.random.seed(seed)
    rsp = ((n - 1) / (n + 1)) ** 2
    R = 0.0
    for _ in range(N):
        z = 0.0
        uz = 1.0
        w = 1.0 - rsp
        while True:
            z_new = z - np.log(np.random.random()) * uz     # unit mu_t
            if z_new < 0:
                if np.random.random() < _fresnel_1d(n, 1.0, abs(uz)):
                    z_new = -z_new
                    uz = -uz
                else:
                    R += w
                    break
            z = z_new
            w *= albedo
            if w < 1e-4:
                if np.random.random() < 0.1:
                    w *= 10
                else:
                    break
            uz = 2 * np.random.random() - 1
    return R / N + rsp


def bench_giovanelli_sweep(albedos=(0.5, 0.7, 0.9, 0.99), n_batches=5, n_photons=100000):
    """Mismatched (n = 1.5) semi-infinite total reflectance vs albedo: voxel engine vs independent 1-D MC."""
    results = []
    for a in albedos:
        mu_t = 10.0
        runs = _mc_batches(n_batches, n_photons, spot_dia=0.01, Lx=60.0, Ly=60.0, Lz=30.0, Nx=6, Ny=6, Nz=30,
                           layers_config=[{'z_end': 30.0, 'mu_a': mu_t * (1 - a), 'mu_s': mu_t * a, 'g': 0.0, 'n': 1.5}])
        tot, se = _mean_se([r['spectro_raw']['R_diff'] + r['spectro_raw']['R_spec'] for r in runs])
        ind = [independent_semi_infinite_mc(400000, a, 1.5, s) for s in range(8)]
        ref, se_ref = float(np.mean(ind)), float(np.std(ind, ddof=1) / np.sqrt(len(ind)))
        results.append(make_result(
            f"V2d-a{a}", "verification", f"Semi-infinite total R, albedo {a} (n=1.5, g=0)",
            "Independent from-scratch 1-D Monte Carlo (same physics as Giovanelli's case)", "Total reflectance", "-",
            tot, ref, 3 * np.hypot(se, se_ref) + 2e-4, "3 x combined standard error (+2e-4 depth truncation)",
            f"{n_batches}x{n_photons:,} photons vs 8x400,000 photons", series={'albedo': a, 'se': se, 'se_ref': se_ref}))
    return results


def run_all_verification():
    out = []
    out += bench_v1_slab_matched()
    out += bench_v1c_fresnel_slab()
    out += bench_v1f_oblique_refraction()
    out += bench_v2_semi_infinite()
    out += bench_v2b_semi_infinite_mismatched()
    out += bench_giovanelli_sweep()
    out += bench_v3_beer_lambert_heating()
    out += bench_cj_mu_sweep()
    out += bench_chandrasekhar_sweep()
    out += bench_v4_ablation_velocity()
    return out


if __name__ == "__main__":
    for r in run_all_verification():
        print(f"{r['id']:10s} {'PASS' if r['pass'] else 'FAIL'}  model={r['model']:.5g}  ref={r['reference_value']:.5g}  "
              f"err={r['rel_error_pct']:.2f}%  tol={r['tolerance_abs']:.3g}  | {r['notes']}")
