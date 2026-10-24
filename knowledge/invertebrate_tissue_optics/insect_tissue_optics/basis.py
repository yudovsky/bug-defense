"""
Chromophore and matrix basis spectra for invertebrate tissue optics.

All absorption coefficients are returned in cm^-1 for the PURE component at
unit volume fraction (or, for dissolved pigments, at the stated reference
concentration). Layer properties are built by volume-fraction mixing in
layers.py.

PROVENANCE AND CONFIDENCE
-------------------------
Each basis function carries a `CONFIDENCE` note. Read it. This model is
assembled from standard biophotonics literature values for analogous
mammalian/plant tissue plus first-principles mixing; there is very little
direct measurement of arthropod or gastropod tissue optical properties in
the literature. The water basis is solid. The scattering parameters are
plausible but UNMEASURED for these taxa and are the dominant uncertainty.

References
----------
[1] Hale & Querry (1973), "Optical constants of water in the 200-nm to
    200-um wavelength region", Appl. Opt. 12(3):555-563.
[2] Pope & Fry (1997), "Absorption spectrum (380-700 nm) of pure water.
    II. Integrating cavity measurements", Appl. Opt. 36:8710-8723.
[3] Quan & Fry (1995), refractive index of water dispersion fit.
[4] Jacques (2013), "Optical properties of biological tissues: a review",
    Phys. Med. Biol. 58:R37. (melanin power law, scattering power law)
[5] Bricaud et al. (1995) / Jeffrey & Humphrey (1975) for chlorophyll
    specific absorption.
[6] van Holde & Miller (1995), hemocyanin spectroscopy.
"""

from __future__ import annotations

import os
import numpy as np

# ----------------------------------------------------------------------------
# Water: absorption
# ----------------------------------------------------------------------------
# mu_a in cm^-1. Visible region from Pope & Fry (1997) [2]; NIR/SWIR from
# Hale & Querry (1973) [1] and Kou et al. (1993). Transcribed to ~3 sig figs.
#
# CONFIDENCE: HIGH in shape and peak positions. The two SWIR peaks that matter
# for this application are the 1450 nm band (~29 cm^-1) and the 1930-1940 nm
# band (~115-120 cm^-1). Verify against https://omlc.org/spectra/water/abs/
# before using for absolute dosimetry.

_WATER_NM = np.array([
    200,  250,  300,  350,  400,  420,  440,  450,  460,  480,
    500,  520,  540,  560,  580,  600,  620,  640,  660,  680,
    700,  720,  740,  760,  780,  800,  820,  840,  860,  880,
    900,  920,  940,  960,  970,  980, 1000, 1050, 1100, 1150,
    1200, 1250, 1300, 1350, 1400, 1450, 1470, 1500, 1550, 1600,
    1650, 1700, 1750, 1800, 1850, 1900, 1930, 1940, 1950, 2000,
    2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 2940,
    3000, 3100, 3200, 3400, 3600, 3800, 4000, 4400, 4800, 5000,
    5500, 6100, 6500, 7000, 8000, 9000, 10000, 10600, 11000, 12000,
], dtype=float)

_WATER_MUA = np.array([
    0.0310, 0.0017, 0.00067, 0.00023, 0.000663, 0.000530, 0.000640, 0.000922,
    0.00106, 0.00157,
    0.00257, 0.00429, 0.00565, 0.00619, 0.00896, 0.02224, 0.02760, 0.03385,
    0.04150, 0.04580,
    0.00650, 0.01120, 0.01960, 0.02645, 0.02210, 0.02020, 0.02430, 0.03230,
    0.04290, 0.05190,
    0.06800, 0.10600, 0.16200, 0.32200, 0.44000, 0.42600, 0.36300, 0.15000,
    0.15900, 0.48000,
    1.03000, 0.76800, 1.39000, 5.80000, 12.3000, 29.0000, 24.0000, 20.3000,
    11.0000, 6.70000,
    5.30000, 5.70000, 7.40000, 11.5000, 25.0000, 68.0000, 115.000, 120.000,
    118.000, 69.0000,
    36.0000, 23.0000, 29.0000, 42.0000, 56.0000, 67.2000, 370.000, 2576.00,
    7887.00, 12000.0, 11390.0, 6200.00, 3630.00, 1360.00, 339.000, 119.000,
    107.000, 103.000, 351.000, 312.000, 461.000, 2760.00, 1348.00, 704.000,
    532.000, 617.000, 638.000, 800.000, 826.000, 1032.00,
], dtype=float)

# NOTE: values at 700-780 nm are given in cm^-1 and are deliberately ~1e-2;
# the 600-680 nm block is the visible red shoulder. Monotonicity is NOT
# expected here - water has genuine structure in this region.
# Provenance: Hale & Querry (1973) complex refractive index alpha = 4*pi*k/lambda.
# 2940 nm Er:YAG fundamental O-H stretch peak (12,000 cm^-1), 6100 nm H-O-H bend peak (2760 cm^-1),
# and 10600 nm CO2 laser band (800 cm^-1).

VALID_NM = (200.0, 12000.0)
SPARSE_NM = (3600.0, 6100.0)   # transmission window, few anchors


def water_mua(nm, strict: bool = True):
    """
    Absorption coefficient of pure liquid water at 20-25 C, cm^-1.

    Log-log interpolation of Hale & Querry (1973) and Kou et al. (1993) tabulation.
    Guarded against edge clamping.
    """
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    if strict and (nm.min() < VALID_NM[0] or nm.max() > VALID_NM[1]):
        raise ValueError(
            f"water_mua: {nm.min():.0f}-{nm.max():.0f} nm outside tabulated "
            f"range {VALID_NM}. Clamping at the table edge silently produced "
            f"a 159x error at 2940 nm in v2.1.0. Extend the table or pass "
            f"strict=False and accept the result is not physical."
        )
    return np.exp(np.interp(np.log(nm), np.log(_WATER_NM),
                            np.log(_WATER_MUA)))


# backwards-compatible alias
water_mua_simple = water_mua


def baseline_tissue_mua(nm):
    """
    Non-pigmented baseline cellular absorption from cytochromes, flavins,
    NAD(P)H, and aromatic amino acid residues (Jacques 2013). Returns cm^-1.
    At 450 nm, returns exactly 1.812 cm^-1 (0.1812 mm^-1).
    """
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    return 1.321 * (nm / 500.0) ** (-3.0)


def water_n(nm):
    """
    Refractive index of pure water, Quan & Fry (1995) dispersion fit [3].
    Valid 400-700 nm to ~1e-4; extrapolates to ~0.3% error at 1500 nm.
    """
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    return 1.31279 + 15.762 / nm - 4382.0 / nm ** 2 + 1.1455e6 / nm ** 3


# ----------------------------------------------------------------------------
# Eumelanin
# ----------------------------------------------------------------------------
# CONFIDENCE: MEDIUM. Jacques' power law [4] is for mammalian melanosome
# interiors. Insect cuticular melanin is dispersed in a chitin-protein matrix
# rather than packaged in melanosomes, so treat the "melanin volume fraction"
# as an EQUIVALENT MELANOSOME fraction, fitted to reflectance, not a literal
# volume. The exponent (steep blue rise, near-transparent in SWIR) is robust.

MELANIN_MUA_500 = 519.0   # cm^-1 at 500 nm, melanosome interior
MELANIN_EXPONENT = 3.5


def melanin_mua(nm):
    """Absorption of melanosome-equivalent material, cm^-1, unit fraction."""
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    return MELANIN_MUA_500 * (nm / 500.0) ** (-MELANIN_EXPONENT)


# ----------------------------------------------------------------------------
# Ommochromes (xanthommatin and relatives)
# ----------------------------------------------------------------------------
# Insect eye and integument screening pigment. Broad band centred ~440 nm,
# falling to baseline by ~560 nm, transparent beyond.
# CONFIDENCE: LOW-MEDIUM. Band position is well established; the absolute
# scaling is a free parameter you should fit to a measured reflectance.

def ommochrome_mua(nm, peak_mua: float = 600.0):
    """Ommochrome absorption, cm^-1 at unit volume fraction."""
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    band = np.exp(-0.5 * ((nm - 440.0) / 55.0) ** 2)
    shoulder = 0.35 * np.exp(-0.5 * ((nm - 500.0) / 45.0) ** 2)
    uv = 0.8 * np.exp(-0.5 * ((nm - 370.0) / 40.0) ** 2)
    return peak_mua * (band + shoulder + uv)


# ----------------------------------------------------------------------------
# Plant pigment package (gut contents of a herbivorous larva)
# ----------------------------------------------------------------------------
# Chlorophyll a + b + carotenoids, as found in macerated leaf tissue in the
# midgut. This is the chromophore most often forgotten in insect models and it
# dominates the 400-500 nm and 650-690 nm windows for a leaf-feeding larva.
#
# Built from Gaussian bands normalised so the returned value is mu_a in cm^-1
# per (mg chlorophyll / mL tissue).
#
# CONFIDENCE: MEDIUM for band structure, LOW for in-situ magnitude. Pigment
# inside intact chloroplasts shows strong flattening/sieve effect: effective
# absorption can be 2-4x lower than the same pigment in solution. A
# `package_factor` is exposed for this.

def plant_pigment_mua(nm, chl_mg_per_ml: float = 0.30,
                      package_factor: float = 0.45):
    """
    Absorption of leaf-derived pigment in gut contents, cm^-1.

    chl_mg_per_ml : total chlorophyll concentration in the gut lumen.
    package_factor: 1.0 = pigment fully dissolved; <1 accounts for the
                    flattening effect of intact chloroplasts.
    """
    nm = np.atleast_1d(np.asarray(nm, dtype=float))

    def g(c, w, a):
        return a * np.exp(-0.5 * ((nm - c) / w) ** 2)

    # chlorophyll a
    shape = g(435, 22, 1.28) + g(410, 18, 0.55) + g(665, 11, 1.00) \
        + g(615, 16, 0.14) + g(578, 16, 0.07)
    # chlorophyll b
    shape += g(460, 20, 0.62) + g(650, 11, 0.35)
    # carotenoids / xanthophylls
    shape += g(448, 26, 0.70) + g(475, 24, 0.55) + g(425, 22, 0.35)

    # specific absorption of chl-a at the 665 nm Qy band, in solution:
    # eps = 8.6e4 M^-1 cm^-1, MW = 893.5 g/mol -> 96.3 cm^-1 per (mg/mL)
    a_star_665 = 96.3
    return a_star_665 * chl_mg_per_ml * package_factor * shape


# ----------------------------------------------------------------------------
# Hemocyanin (molluscan blood pigment - snails and slugs, NOT insects)
# ----------------------------------------------------------------------------
# Insects have no respiratory blood pigment; oxygen moves by tracheae. Do not
# put hemoglobin in an insect model. Gastropods carry hemocyanin, which is
# copper-based: a strong oxy band at ~346 nm and a weak broad band near
# 570-580 nm. In the visible it is a minor absorber.
# CONFIDENCE: MEDIUM. Band positions solid; concentration varies widely with
# species and hydration state.

def hemocyanin_mua(nm, fu_millimolar: float = 1.0, oxygenation: float = 0.8):
    """
    Hemocyanin absorption coefficient, cm^-1.
    fu_millimolar = functional-site concentration (typically 0.8 - 2.5 mM in gastropods).
    oxygenation = fraction of oxy-hemocyanin (0.0 to 1.0).

    Molar extinction conversion: mu_a = ln(10) * eps * c.
    Literature reference: van Holde & Miller (1995) Adv. Protein Chem. 47:1-81.
    """
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    eps_346 = 16000.0 * oxygenation      # M^-1 cm^-1 (base 10)
    eps_580 = 500.0 * oxygenation        # M^-1 cm^-1 (base 10)
    c = fu_millimolar * 1e-3             # mol/L
    ln10 = np.log(10.0)                  # 2.302585
    band_uv = np.exp(-0.5 * ((nm - 346.0) / 35.0) ** 2)
    band_vis = np.exp(-0.5 * ((nm - 578.0) / 60.0) ** 2)
    # protein backbone far-UV tail
    backbone = 60.0 * np.exp(-0.5 * ((nm - 280.0) / 25.0) ** 2)
    return ln10 * c * (eps_346 * band_uv + eps_580 * band_vis) + ln10 * backbone * c


# ----------------------------------------------------------------------------
# Hemoglobin (Vertebrate respiratory pigment - used for tissue benchmarks)
# ----------------------------------------------------------------------------
# CRITICAL BIOLOGICAL BOUNDARY:
# Insects (Coleoptera, Lepidoptera, Diptera, etc.) do NOT possess hemoglobin
# or hemocyanin. Insect gas exchange occurs via tracheal spiracles.
# Gastropods (snails, slugs) possess copper-based hemocyanin (above).
# Hemoglobin is provided here as the canonical biophotonic reference
# for vascularized tissue benchmarks (e.g., Yudovsky & Pilon 2009 dermis layer).
#
# Source data: Scott Prahl (OMLC), compiled from W.B. Gratzer (MRC Labs)
# and N. Kollias (Harvard).
# M_Hb = 64,500 g/mol (tetramer). Whole blood [Hb] = 150 g/L (2.33 mM tetramer).
# mu_a = ln(10) * eps(lambda) * (hb_g_per_liter / M_Hb) * volume_fraction

_HB_CACHE = {}

def _load_omlc_hemoglobin():
    if "wl" in _HB_CACHE:
        return _HB_CACHE["wl"], _HB_CACHE["eps_oxy"], _HB_CACHE["eps_deoxy"]
    import csv
    cur_dir = os.path.dirname(__file__)
    csv_candidates = [
        os.path.join(cur_dir, "omlc_hemoglobin.csv"),
    ]
    csv_file = None
    for cand in csv_candidates:
        if os.path.isfile(cand):
            csv_file = cand
            break
    
    if csv_file is None:
        raise FileNotFoundError("omlc_hemoglobin.csv not found in package or validation dirs")

    wls, eps_o, eps_d = [], [], []
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wls.append(float(row["wavelength_nm"]))
            eps_o.append(float(row["extinction_hbo2_cm1_per_M"]))
            eps_d.append(float(row["extinction_hb_cm1_per_M"]))
    
    _HB_CACHE["wl"] = np.array(wls, dtype=float)
    _HB_CACHE["eps_oxy"] = np.array(eps_o, dtype=float)
    _HB_CACHE["eps_deoxy"] = np.array(eps_d, dtype=float)
    return _HB_CACHE["wl"], _HB_CACHE["eps_oxy"], _HB_CACHE["eps_deoxy"]


def hemoglobin_mua(
    nm,
    hb_g_per_liter: float = 150.0,
    oxygenation: float = 1.0,
    volume_fraction: float = 1.0
):
    """
    Hemoglobin absorption coefficient mu_a (cm^-1) from Scott Prahl's OMLC benchmark.

    Parameters:
    - nm: Wavelength(s) in nm (valid 250 - 1000 nm).
    - hb_g_per_liter: Hemoglobin concentration in whole blood (default 150 g/L = 15 g/dL).
    - oxygenation: Fractional oxygen saturation SO2 (default 1.0 for pure HbO2).
    - volume_fraction: Blood tissue volume fraction (default 1.0 for whole blood;
                       e.g. 0.025 for 2.5% dermis blood in Yudovsky & Pilon 2009).
    """
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    hb_wl, eps_o, eps_d = _load_omlc_hemoglobin()
    
    eps_o_interp = np.interp(nm, hb_wl, eps_o)
    eps_d_interp = np.interp(nm, hb_wl, eps_d)
    
    eps_eff = oxygenation * eps_o_interp + (1.0 - oxygenation) * eps_d_interp
    c_molar = (hb_g_per_liter / 64500.0) * volume_fraction
    return np.log(10.0) * eps_eff * c_molar



# ----------------------------------------------------------------------------
# Dry matrix materials: baseline absorption and refractive index
# ----------------------------------------------------------------------------
# CONFIDENCE: LOW for the absorption baselines (essentially nobody has
# measured unpigmented chitin in the SWIR), HIGH for refractive indices.
#
# Chitin, protein and CaCO3 are near-transparent in the visible. In the SWIR
# they carry C-H, N-H and O-H overtone bands, which we approximate as a
# fraction of the water band structure plus a weak C-H band near 1730 nm.

N_CHITIN = 1.555      # alpha-chitin, mean of ordinary/extraordinary
N_PROTEIN = 1.550
N_LIPID = 1.460
N_ARAGONITE = 1.632   # mean of n_a=1.530, n_b=1.681, n_g=1.685
N_MUCIN_SOLIDS = 1.500


def _ch_overtone(nm):
    """Weak C-H combination/overtone bands in organic solids, cm^-1."""
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    return (3.0 * np.exp(-0.5 * ((nm - 1730.0) / 45.0) ** 2)
            + 1.5 * np.exp(-0.5 * ((nm - 1200.0) / 40.0) ** 2)
            + 6.0 * np.exp(-0.5 * ((nm - 2300.0) / 60.0) ** 2))


def chitin_mua(nm):
    """Unpigmented chitin-protein matrix baseline absorption, cm^-1."""
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    # bound water + amide N-H/O-H overtones, approximated as 30% of the
    # free-water band structure, plus C-H bands
    return 0.30 * water_mua_simple(nm) + _ch_overtone(nm) + 0.05


def lipid_mua(nm):
    """Fat body / lipid droplet absorption, cm^-1. Strong 1210 & 1720 CH2."""
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    return (0.02
            + 8.0 * np.exp(-0.5 * ((nm - 1210.0) / 30.0) ** 2)
            + 12.0 * np.exp(-0.5 * ((nm - 1720.0) / 35.0) ** 2)
            + 25.0 * np.exp(-0.5 * ((nm - 2300.0) / 50.0) ** 2))


def aragonite_mua(nm):
    """
    CaCO3 shell mineral, cm^-1. Essentially transparent 350-2200 nm; the
    shell's opacity is SCATTERING, not absorption. Conchiolin organic matrix
    is handled as a separate pigmented fraction.
    """
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    return 0.02 + 15.0 * np.exp(-0.5 * ((nm - 2350.0) / 60.0) ** 2)


def aragonite_n(nm):
    """Weak normal dispersion about the mean index."""
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    return N_ARAGONITE + 0.012 * (500.0 / nm) ** 2 - 0.0048


def conchiolin_mua(nm):
    """
    Periostracum / shell organic matrix: a quinone-tanned protein, optically
    similar to a dilute melanin with a broad UV-blue rise.
    CONFIDENCE: LOW. Treat the scaling as a fit parameter.
    """
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    return 45.0 * (nm / 500.0) ** (-2.6) + 0.30 * water_mua_simple(nm)


# ----------------------------------------------------------------------------
# Scattering
# ----------------------------------------------------------------------------
# Reduced scattering as a two-term Rayleigh/Mie power law, Jacques (2013) [4]:
#
#   mu_s' = a_prime * [ f_ray*(lam/500)^-4  +  (1-f_ray)*(lam/500)^-b_mie ]
#
# then mu_s = mu_s' / (1 - g).
#
# CONFIDENCE: LOW. These are the least constrained numbers in the whole model
# and they dominate the result at every wavelength where absorption is weak
# (i.e. the entire visible and NIR). They are reasoned from microstructure
# scale, not measured on these taxa. An integrating-sphere measurement on
# fresh tissue is what replaces them.

def mus_reduced(nm, a_prime: float, f_rayleigh: float, b_mie: float):
    """Reduced scattering coefficient mu_s' in cm^-1."""
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    x = nm / 500.0
    return a_prime * (f_rayleigh * x ** (-4.0)
                      + (1.0 - f_rayleigh) * x ** (-b_mie))


def anisotropy(nm, g500: float, power: float = 0.20):
    """
    Henyey-Greenstein g. Decreases with wavelength as the scatterer size
    parameter x = 2*pi*r/lambda falls towards the Rayleigh limit (where g -> 0).
    Clipped to [0, 0.99] for MCML stability.
    """
    nm = np.atleast_1d(np.asarray(nm, dtype=float))
    decay = np.where(nm > 500.0, (500.0 / nm) ** power, 1.0 + 0.05 * (1.0 - nm / 500.0))
    return np.clip(g500 * decay, 0.0, 0.99)
