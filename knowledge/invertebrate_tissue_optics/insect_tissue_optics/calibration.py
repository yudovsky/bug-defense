"""
Calibration module for insect tissue optics against spectrophotometric data.

Scope limit
-----------
The fit is valid for sclerotised, melanised beetle cuticle, 185-1400 nm.
It does not transfer to:
- larval cuticle (unsclerotised, different microstructure)
- mucus or gastropod integument
- CaCO3 shell (mineral, crossed-lamellar, wildly different scatterer)
- anything beyond 1400 nm, where water dominates and the source has no data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import scipy.optimize as opt
import iadpython

from . import basis


# ----------------------------------------------------------------------------
# Authoritative raw spectrophotometric means from Cuesta & Lobo (2019)
# PeerJ 7:e8104. Excised elytron slabs measured in integrating sphere.
# ----------------------------------------------------------------------------

CUESTA_LOBO_2019 = {
    "O_coenobita": {
        "thickness_um": 82.10,
        "thickness_sd_um": 7.52,
        "bands": {
            # UV: A reported as 95-98% range (point mean 96.5%). R and T near zero.
            "UV":  {"range_nm": (185, 385), "R": None, "T": None, "A": 96.5},
            "VIS": {"range_nm": (390, 745), "R": 4.2,  "T": 17.0, "A": 78.8},
            "NIR": {"range_nm": (750, 1400), "R": 10.0, "T": 46.4, "A": 43.6},
        },
    },
    "O_medius": {
        "thickness_um": 100.10,
        "thickness_sd_um": 9.90,
        "bands": {
            "UV":  {"range_nm": (185, 385), "R": None, "T": None, "A": 96.5},
            "VIS": {"range_nm": (390, 745), "R": 4.7,  "T": 14.0, "A": 81.3},
            "NIR": {"range_nm": (750, 1400), "R": 11.8, "T": 43.5, "A": 44.7},
        },
    },
}


@dataclass
class FitResult:
    """Container for RTA calibration results."""
    params: Dict[str, float]
    ci: Dict[str, Dict[str, float]]
    fit_report: Dict[str, Any]

    def __getitem__(self, item):
        if item == "params":
            return self.params
        if item == "ci":
            return self.ci
        if item == "fit_report":
            return self.fit_report
        raise KeyError(item)


def band_mean(model_fn, lo: float, hi: float, step: float = 5.0) -> np.ndarray:
    """
    Unweighted mean of forward model on native 5 nm grid across [lo, hi].
    Matches the spectrophotometer band aggregation methodology.
    """
    nm = np.arange(lo, hi + 0.1 * step, step)
    vals = [model_fn(w) for w in nm]
    return np.mean(vals, axis=0)


def _eval_elytron_slab(
    thickness_um: float,
    melanin_frac: float,
    a_prime_scale: float,
    b_mie_global: float,
    nm_arr: np.ndarray,
    mu_a_scler: float = 180.0,
    f_rayleigh: float = 0.20,
    g_eff: float = 0.85,
    n_slab: float = 1.473,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward radiative transport model for an excised beetle elytron slab.

    Geometry matching:
    Spectrophotometer measurements with A = 100 - (T + R) require TOTAL
    (hemispherical) R and TOTAL (hemispherical) T at normal incidence, NOT
    collimated-only transmission. We use Prahl's Inverse Adding-Doubling (IAD)
    forward solver (`iadpython.Sample.rt()`) with dielectric Fresnel boundaries
    at n_above=1.0 and n_below=1.0.

    Cuticle architecture:
    - Epicuticle wax: ~5% thickness, low scattering, n ~ 1.473 setting Fresnel step.
    - Exocuticle: ~40% thickness, quinone-tanned and melanised (melanin + sclerotin).
    - Endocuticle: ~55% thickness, unpigmented chitin-protein matrix.
    """
    scale = thickness_um / 100.0
    d_cm = thickness_um * 1e-4
    d_epi = 5.0 * scale
    d_exo = 40.0 * scale
    d_endo = 55.0 * scale

    ws = np.atleast_1d(np.asarray(nm_arr, dtype=float))

    # Absorption:
    # Exocuticle: eumelanin power law + sclerotin broad polymeric absorption + baseline
    mel_term = melanin_frac * basis.MELANIN_MUA_500 * (ws / 500.0) ** (-basis.MELANIN_EXPONENT)
    base_term = 1.321 * (ws / 500.0) ** (-3.0)
    mua_exo = mel_term + mu_a_scler + base_term

    # Endocuticle: chitin-protein bound water/overtones + baseline
    mua_endo = 0.30 * basis.water_mua_simple(ws) + basis._ch_overtone(ws) + 0.05 + base_term
    mua_epi = np.full_like(ws, 0.5)

    mua_bar = (d_epi * mua_epi + d_exo * mua_exo + d_endo * mua_endo) / thickness_um

    # Scattering (scaled by a_prime_scale):
    # Two-term Rayleigh + Mie dispersion:
    disp = f_rayleigh * (500.0 / ws) ** 4.0 + (1.0 - f_rayleigh) * (500.0 / ws) ** b_mie_global
    musp_exo = 45.0 * a_prime_scale * disp
    musp_endo = 75.0 * a_prime_scale * disp
    musp_epi = 15.0 * a_prime_scale * disp

    musp_bar = (d_epi * musp_epi + d_exo * musp_exo + d_endo * musp_endo) / thickness_um
    mus_bar = musp_bar / max(1.0 - g_eff, 1e-4)

    # Adding-doubling parameters:
    b = (mua_bar + mus_bar) * d_cm
    a = mus_bar / np.maximum(mua_bar + mus_bar, 1e-12)

    # Vectorized / looped IAD Sample:
    ur1 = np.empty_like(ws)
    ut1 = np.empty_like(ws)
    for i in range(len(ws)):
        s = iadpython.Sample(
            a=float(a[i]),
            b=float(b[i]),
            g=float(g_eff),
            n=float(n_slab),
            n_above=1.0,
            n_below=1.0,
        )
        r, t, _, _ = s.rt()
        ur1[i] = float(r)
        ut1[i] = float(t)

    return ur1 * 100.0, ut1 * 100.0


def _evaluate_bands(
    thickness_um: float,
    melanin_frac: float,
    a_prime_scale: float,
    b_mie_global: float,
    step: float = 5.0,
) -> Dict[str, Dict[str, float]]:
    """Evaluates UV, VIS, and NIR band means at given step."""
    # UV integration floor clipped to 200 nm per water table floor
    uv_nm = np.arange(200.0, 385.1, step)
    vis_nm = np.arange(390.0, 745.1, step)
    nir_nm = np.arange(750.0, 1400.1, step)

    r_uv, t_uv = _eval_elytron_slab(thickness_um, melanin_frac, a_prime_scale, b_mie_global, uv_nm)
    r_vis, t_vis = _eval_elytron_slab(thickness_um, melanin_frac, a_prime_scale, b_mie_global, vis_nm)
    r_nir, t_nir = _eval_elytron_slab(thickness_um, melanin_frac, a_prime_scale, b_mie_global, nir_nm)

    def _pack(r_arr, t_arr):
        R = float(np.mean(r_arr))
        T = float(np.mean(t_arr))
        A = float(100.0 - R - T)
        return {"R_model": R, "T_model": T, "A_model": A}

    return {
        "UV": _pack(r_uv, t_uv),
        "VIS": _pack(r_vis, t_vis),
        "NIR": _pack(r_nir, t_nir),
    }


def fit_to_RTA(
    preset_name: str = "beetle_elytron",
    dataset: Dict[str, Any] = CUESTA_LOBO_2019,
    forward: str = "iad",
    free: Sequence[str] = ("melanin_frac", "a_prime_scale", "b_mie_global"),
    holdout: str = "O_medius",
    n_bootstrap: int = 200,
    seed: int = 0,
) -> FitResult:
    """
    Fits three constitutive parameters of beetle cuticle against Cuesta & Lobo (2019).

    Parameters
    ----------
    preset_name : str
        Preset morphology name to calibrate (default "beetle_elytron").
    dataset : dict
        Spectrophotometric targets (raw means).
    forward : str
        Transport forward model ("iad" or "mcml"). Default "iad".
    free : sequence of str
        Parameters to fit: ("melanin_frac", "a_prime_scale", "b_mie_global").
    holdout : str
        Held-out species for generalization check (default "O_medius").
    n_bootstrap : int
        Bootstrap resamples over thickness distribution (default 200).
    seed : int
        RNG seed for deterministic optimization and bootstrapping.

    Returns
    -------
    FitResult with attributes:
        params      : dict, best-fit values
        ci          : dict, 2.5 and 97.5 percentile per parameter from bootstrap
        fit_report  : dict with per-band modelled vs observed R/T/A,
                      in-sample and held-out chi2, and a `warnings` list
    """
    if forward != "iad":
        raise NotImplementedError(f"Forward model {forward!r} not supported (use 'iad')")

    rng = np.random.default_rng(seed)
    in_species = [s for s in dataset if s != holdout][0]

    # Target data
    in_data = dataset[in_species]
    d_mean = in_data["thickness_um"]
    d_sd = in_data["thickness_sd_um"]

    sigma = 3.0  # 3 percentage points standard error

    def _calc_chi2(obs_bands, mod_bands):
        chi2 = 0.0
        for b in ["VIS", "NIR"]:
            r_obs = obs_bands[b]["R"]
            t_obs = obs_bands[b]["T"]
            r_mod = mod_bands[b]["R_model"]
            t_mod = mod_bands[b]["T_model"]
            chi2 += ((r_mod - r_obs) / sigma) ** 2
            chi2 += ((t_mod - t_obs) / sigma) ** 2
        # UV one-sided inequality constraint: penalty only if A < 95.0
        uv_A = mod_bands["UV"]["A_model"]
        if uv_A < 95.0:
            chi2 += ((95.0 - uv_A) / sigma) ** 2
        return chi2

    def _objective(x, d_curr, step=15.0):
        mf, ap, bm = x
        mod_bands = _evaluate_bands(d_curr, mf, ap, bm, step=step)
        return _calc_chi2(in_data["bands"], mod_bands)

    bounds = [(0.01, 0.25), (0.1, 3.0), (0.3, 2.0)]
    x0 = [0.08, 0.30, 1.0]

    # Primary point fit on in-sample species mean thickness
    opt_res = opt.minimize(
        _objective,
        x0,
        args=(d_mean, 15.0),
        bounds=bounds,
        method="L-BFGS-B",
    )
    best_x = opt_res.x

    best_params = {
        "melanin_frac": float(best_x[0]),
        "a_prime_scale": float(best_x[1]),
        "b_mie_global": float(best_x[2]),
    }

    # Evaluate full 5 nm band means for both in-sample and held-out species
    in_mod_bands = _evaluate_bands(d_mean, *best_x, step=5.0)
    hold_data = dataset[holdout]
    hold_mod_bands = _evaluate_bands(hold_data["thickness_um"], *best_x, step=5.0)

    insample_chi2 = float(_calc_chi2(in_data["bands"], in_mod_bands))
    holdout_chi2 = float(_calc_chi2(hold_data["bands"], hold_mod_bands))

    # Bootstrap thickness uncertainty
    boot_params = []
    # Use fast step=25.0 during bootstrap iterations for computational efficiency
    for _ in range(n_bootstrap):
        d_boot = float(rng.normal(d_mean, d_sd))
        d_boot = max(20.0, d_boot)  # physical guard
        res_b = opt.minimize(
            _objective,
            best_x,
            args=(d_boot, 25.0),
            bounds=bounds,
            method="L-BFGS-B",
        )
        if res_b.success:
            boot_params.append(res_b.x)
        else:
            boot_params.append(best_x)

    boot_arr = np.array(boot_params)
    ci = {}
    param_names = ["melanin_frac", "a_prime_scale", "b_mie_global"]
    for idx, name in enumerate(param_names):
        p2_5, p97_5 = np.percentile(boot_arr[:, idx], [2.5, 97.5])
        ci[name] = {"2.5": float(p2_5), "97.5": float(p97_5)}

    warnings = [
        "UV integration floor clipped to 200 nm (water absorption table floor).",
        "Scope limited strictly to sclerotised adult beetle cuticle (185-1400 nm).",
    ]
    if holdout_chi2 >= 3.0 * insample_chi2:
        warnings.append(
            f"Held-out chi2 ({holdout_chi2:.2f}) exceeds 3x in-sample chi2 ({insample_chi2:.2f}); generalisation degraded."
        )

    # Build fit_report table with observed vs modelled comparisons
    def _enrich_report(mod_b, obs_data):
        rep = {}
        for b_name in ["UV", "VIS", "NIR"]:
            obs_b = obs_data["bands"][b_name]
            rep[b_name] = {
                "R_model": mod_b[b_name]["R_model"],
                "R_observed": obs_b["R"],
                "T_model": mod_b[b_name]["T_model"],
                "T_observed": obs_b["T"],
                "A_model": mod_b[b_name]["A_model"],
                "A_observed": obs_b["A"],
            }
        return rep

    fit_report = {
        in_species: _enrich_report(in_mod_bands, in_data),
        holdout: _enrich_report(hold_mod_bands, hold_data),
        "insample_chi2": insample_chi2,
        "holdout_chi2": holdout_chi2,
        "chi2_ratio": holdout_chi2 / max(insample_chi2, 1e-12),
        "warnings": warnings,
    }

    return FitResult(params=best_params, ci=ci, fit_report=fit_report)
