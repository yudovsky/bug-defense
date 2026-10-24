"""
Layer definition and composition -> optical property mixing.

A Layer is a thickness plus a composition (volume fractions of matrix
components) plus a scattering parameterisation. Optical properties are
evaluated spectrally on demand.

MIXING RULES
------------
Absorption:  mu_a(lam) = sum_i  f_i * mu_a,i(lam)
             Beer-Lambert additivity over volume fractions. Valid when
             chromophores are well dispersed. NOT valid inside intact
             organelles (see `package_factor` in basis.plant_pigment_mua).

Refractive index:  n(lam) = sum_i f_i * n_i(lam)
             Volume-weighted linear mixing. This is the first-order
             Lorentz-Lorenz result and is accurate to <0.5% for the index
             contrasts present in soft tissue.

Scattering:  parameterised directly as a reduced-scattering power law rather
             than derived from composition, because mu_s' depends on
             microstructure (lamella spacing, droplet size) and not on
             chemical composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from . import basis


# Components whose volume fraction can appear in a layer composition.
# Each maps to (absorption function, refractive index function or constant).
COMPONENTS = {
    "water":        (basis.water_mua, basis.water_n),
    "chitin":       (basis.chitin_mua, basis.N_CHITIN),
    "protein":      (basis.chitin_mua, basis.N_PROTEIN),
    "lipid":        (basis.lipid_mua, basis.N_LIPID),
    "melanin":      (basis.melanin_mua, 1.70),
    "ommochrome":   (basis.ommochrome_mua, 1.60),
    "plant_pigment": (basis.plant_pigment_mua, 1.50),
    "hemocyanin":   (basis.hemocyanin_mua, 1.55),
    "aragonite":    (basis.aragonite_mua, basis.aragonite_n),
    "conchiolin":   (basis.conchiolin_mua, 1.56),
    "mucin":        (basis.chitin_mua, basis.N_MUCIN_SOLIDS),
    "hemoglobin":   (basis.hemoglobin_mua, 1.40),
    "air":          (lambda nm: np.zeros_like(np.atleast_1d(
                        np.asarray(nm, float))), 1.0003),
}


@dataclass
class Layer:
    """One MCML layer."""

    name: str
    thickness_um: float
    composition: Dict[str, float]
    a_prime: float          # mu_s' at 500 nm, cm^-1
    f_rayleigh: float       # Rayleigh fraction of the scattering
    b_mie: float            # Mie exponent
    g500: float             # anisotropy at 500 nm
    note: str = ""
    optional: bool = False
    baseline_scale: float = 1.0  # Background cellular baseline (Jacques 2013)
    # per-layer overrides passed to chromophore basis functions
    chromophore_kwargs: Dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        total = sum(self.composition.values())
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(
                f"Layer {self.name!r}: volume fractions sum to {total:.4f}, "
                "must sum to 1.0"
            )
        for k in self.composition:
            if k not in COMPONENTS:
                raise ValueError(f"Layer {self.name!r}: unknown component {k!r}")

    # -- optical properties ------------------------------------------------

    def mua(self, nm) -> np.ndarray:
        nm = np.atleast_1d(np.asarray(nm, float))
        out = np.zeros_like(nm)
        for comp, frac in self.composition.items():
            if frac == 0.0 or comp == 'air':
                continue
            fn = COMPONENTS[comp][0]
            kw = self.chromophore_kwargs.get(comp, {})
            out = out + frac * np.asarray(fn(nm, **kw), float)
        # Add non-pigmented baseline cellular absorption (Jacques 2013)
        if self.composition.get('air', 0.0) < 0.99 and self.baseline_scale > 0.0:
            out = out + self.baseline_scale * basis.baseline_tissue_mua(nm)
        return out

    def n(self, nm) -> np.ndarray:
        nm = np.atleast_1d(np.asarray(nm, float))
        out = np.zeros_like(nm)
        for comp, frac in self.composition.items():
            if frac == 0.0:
                continue
            nfun = COMPONENTS[comp][1]
            vals = nfun(nm) if callable(nfun) else np.full_like(nm, float(nfun))
            out = out + frac * np.asarray(vals, float)
        return out

    def g(self, nm) -> np.ndarray:
        return basis.anisotropy(nm, self.g500)

    def musp(self, nm) -> np.ndarray:
        return basis.mus_reduced(nm, self.a_prime, self.f_rayleigh, self.b_mie)

    def mus(self, nm) -> np.ndarray:
        """Scattering coefficient for MCML (NOT reduced), safely clipped."""
        g_val = np.minimum(self.g(nm), 0.99)
        return self.musp(nm) / (1.0 - g_val)

    # -- derived -----------------------------------------------------------

    def albedo(self, nm) -> np.ndarray:
        mus, mua = self.mus(nm), self.mua(nm)
        return mus / np.maximum(mus + mua, 1e-30)

    def mu_eff(self, nm) -> np.ndarray:
        """Diffusion-approximation effective attenuation, cm^-1."""
        mua, musp = self.mua(nm), self.musp(nm)
        return np.sqrt(3.0 * mua * (mua + musp))

    def penetration_depth_mm(self, nm) -> np.ndarray:
        """1/e optical penetration depth in the diffusion limit, mm."""
        return 10.0 / np.maximum(self.mu_eff(nm), 1e-30)

    def optical_thickness(self, nm) -> np.ndarray:
        """Total (absorption + scattering) optical depth of this layer."""
        d_cm = self.thickness_um * 1e-4
        return (self.mua(nm) + self.mus(nm)) * d_cm

    def absorption_optical_depth(self, nm) -> np.ndarray:
        return self.mua(nm) * self.thickness_um * 1e-4


@dataclass
class Morphology:
    """An ordered stack of layers plus the surrounding media."""

    name: str
    layers: list
    n_above: float = 1.0003     # air
    n_below: float = 1.40       # semi-infinite tissue backing
    description: str = ""

    def active_layers(self, include_optional: bool = True):
        return [L for L in self.layers if include_optional or not L.optional]

    def total_thickness_um(self, include_optional: bool = True) -> float:
        return sum(L.thickness_um
                   for L in self.active_layers(include_optional))

    def table(self, nm_scalar: float, include_optional: bool = True):
        """Property table at one wavelength. Returns a list of dicts."""
        rows = []
        cum = 0.0
        for L in self.active_layers(include_optional):
            mua = float(L.mua(nm_scalar)[0])
            mus = float(L.mus(nm_scalar)[0])
            g = float(L.g(nm_scalar)[0])
            n = float(L.n(nm_scalar)[0])
            cum += L.thickness_um
            rows.append({
                "layer": L.name,
                "thickness_um": L.thickness_um,
                "depth_to_um": cum,
                "n": n,
                "mua_cm-1": mua,
                "mus_cm-1": mus,
                "g": g,
                "musp_cm-1": mus * (1 - g),
                "albedo": mus / max(mus + mua, 1e-30),
                "tau_abs": mua * L.thickness_um * 1e-4,
                "tau_total": (mua + mus) * L.thickness_um * 1e-4,
            })
        return rows
