"""
Spectral closure laws for invertebrate tissue optics.

Supplies the constitutive relations n(lam), mu_a(lam), mu_s(lam), g(lam) for
a 1-D multilayer invertebrate, in the form a cylindrically-symmetric RTE
solver (MCML and relatives) consumes. It does NOT solve the transport
equation.

Quick start
-----------
    import numpy as np
    from insect_tissue_optics import morphology, export

    m = morphology.larva()
    for row in m.table(1470.0):
        print(row["layer"], row["n"], row["mua_cm-1"], row["mus_cm-1"], row["g"])

    nm = np.arange(400, 2001, 5.0)
    export.spectral_frame(m, nm).to_csv("larva.csv", index=False)

Units
-----
    wavelength      nm
    mu_a, mu_s      cm^-1        (MCML convention)
    thickness       um in this API, cm in MCML input
    g               dimensionless Henyey-Greenstein asymmetry
    n               dimensionless
"""

from . import basis, layers, morphology, export, calibration  # noqa: F401
from .calibration import fit_to_RTA, CUESTA_LOBO_2019

__all__ = ["basis", "layers", "morphology", "export", "calibration", "fit_to_RTA", "CUESTA_LOBO_2019"]
__version__ = "0.2.0"
