"""
Tabulation of the closure laws. No transport solving happens here.

`spectral_frame` produces a tidy table with one row per (layer, wavelength),
carrying everything an RTE solver needs plus a few derived diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def spectral_frame(morph, nm, include_optional: bool = True) -> pd.DataFrame:
    """Tidy (layer x wavelength) table of optical properties."""
    nm = np.atleast_1d(np.asarray(nm, float))
    frames = []
    depth_to = 0.0
    for idx, L in enumerate(morph.active_layers(include_optional)):
        depth_from = depth_to
        depth_to = depth_from + L.thickness_um
        mua, mus, g = L.mua(nm), L.mus(nm), L.g(nm)
        frames.append(pd.DataFrame({
            "morphology": morph.name,
            "layer_index": idx + 1,
            "layer": L.name,
            "wavelength_nm": nm,
            "thickness_um": L.thickness_um,
            "thickness_cm": L.thickness_um * 1e-4,
            "depth_from_um": depth_from,
            "depth_to_um": depth_to,
            "n": L.n(nm),
            "mua_cm-1": mua,
            "mus_cm-1": mus,
            "g": g,
            "musp_cm-1": L.musp(nm),
            "albedo": mus / np.maximum(mus + mua, 1e-30),
            "mu_eff_cm-1": L.mu_eff(nm),
            "penetration_depth_mm": L.penetration_depth_mm(nm),
            "tau_abs": L.absorption_optical_depth(nm),
            "tau_total": L.optical_thickness(nm),
        }))
    out = pd.concat(frames, ignore_index=True)
    out.attrs["n_above"] = morph.n_above
    out.attrs["n_below"] = morph.n_below
    return out


def fresnel_normal(n1, n2):
    """Normal-incidence power reflectance at a dielectric interface."""
    n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
    return ((n1 - n2) / (n1 + n2)) ** 2


def interface_reflectance(morph, nm, include_optional: bool = True):
    """
    Normal-incidence Fresnel reflectance at every internal interface.

    This is where the air gaps announce themselves: a cuticle/air step is an
    order of magnitude more reflective than any tissue/tissue step.
    """
    nm = np.atleast_1d(np.asarray(nm, float))
    Ls = morph.active_layers(include_optional)
    rows = []
    n_prev = np.full_like(nm, morph.n_above)
    names = ["ambient"] + [L.name for L in Ls] + ["backing"]
    n_list = [n_prev] + [L.n(nm) for L in Ls] + \
             [np.full_like(nm, morph.n_below)]
    for i in range(len(n_list) - 1):
        rows.append(pd.DataFrame({
            "morphology": morph.name,
            "interface": f"{names[i]} | {names[i+1]}",
            "wavelength_nm": nm,
            "n_above": n_list[i],
            "n_below": n_list[i + 1],
            "R_normal": fresnel_normal(n_list[i], n_list[i + 1]),
        }))
    return pd.concat(rows, ignore_index=True)


def summary_at(morph, nm_scalar, include_optional: bool = True) -> pd.DataFrame:
    """Single-wavelength layer table, rounded for reading."""
    df = pd.DataFrame(morph.table(nm_scalar, include_optional))
    for c, p in [("n", 4), ("mua_cm-1", 4), ("mus_cm-1", 1), ("g", 3),
                 ("musp_cm-1", 2), ("albedo", 5), ("tau_abs", 5),
                 ("tau_total", 3)]:
        df[c] = df[c].round(p)
    return df
