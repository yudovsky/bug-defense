"""
Layer-stack presets.

Layers are optional by design: the stack is assembled from a common soft-body
core, and the OUTER layers decide the organism.

    core soft body                      -> larva / grub
    core + mucus + unsclerotised skin   -> slug
    core + periostracum + CaCO3 shell   -> snail
    core + sclerotised melanised cuticle-> adult beetle

Thicknesses are representative values for the ~10 mm size class. They are
geometric inputs, not fitted; change them freely.

A NOTE ON WHAT IS AND IS NOT MEASURED
-------------------------------------
Thicknesses: literature-typical, MEDIUM confidence.
Compositions: reasoned from tissue water content and pigmentation, MEDIUM.
Scattering (a_prime, f_rayleigh, b_mie, g500): LOW confidence throughout.
These dominate the solution wherever absorption is weak. Treat them as the
free parameters to fit against an integrating-sphere measurement.
"""

from __future__ import annotations

from .layers import Layer, Morphology


# ----------------------------------------------------------------------------
# Shared interior layers
# ----------------------------------------------------------------------------

def epidermis(thickness_um=22.0):
    return Layer(
        name="epidermis",
        thickness_um=thickness_um,
        composition={"water": 0.70, "protein": 0.22, "lipid": 0.08},
        a_prime=35.0, f_rayleigh=0.28, b_mie=1.3, g500=0.88,
        note="Single cell layer secreting the cuticle; mitochondria-rich, "
             "so a meaningful Rayleigh component.",
    )


def fat_body(thickness_um=450.0):
    return Layer(
        name="fat_body",
        thickness_um=thickness_um,
        composition={"water": 0.38, "lipid": 0.50, "protein": 0.12},
        a_prime=28.0, f_rayleigh=0.05, b_mie=0.8, g500=0.92,
        note="Large lipid droplets: nearly pure Mie, weak spectral slope. "
             "Carries the 1210 and 1720 nm CH2 bands.",
    )


def hemolymph(thickness_um=300.0, molluscan=False):
    if molluscan:
        comp = {"water": 0.91, "protein": 0.05, "hemocyanin": 0.04}
        note = ("Gastropod blood. Hemocyanin is copper-based: strong 346 nm "
                "band, weak 578 nm band. NOT hemoglobin.")
    else:
        comp = {"water": 0.93, "protein": 0.07}
        note = ("Insect hemolymph carries NO respiratory pigment - oxygen "
                "moves by tracheae. Do not insert hemoglobin here.")
    return Layer(
        name="hemolymph",
        thickness_um=thickness_um,
        composition=comp,
        a_prime=2.5, f_rayleigh=0.45, b_mie=1.5, g500=0.95,
        note=note,
    )


def muscle(thickness_um=1500.0):
    return Layer(
        name="muscle",
        thickness_um=thickness_um,
        composition={"water": 0.76, "protein": 0.21, "lipid": 0.03},
        a_prime=22.0, f_rayleigh=0.18, b_mie=1.0, g500=0.93,
        note="Striated fibres; strongly forward-scattering.",
    )


def gut_contents(thickness_um=2000.0, chl_mg_per_ml=0.30,
                 package_factor=0.45):
    return Layer(
        name="gut_contents",
        thickness_um=thickness_um,
        composition={"water": 0.82, "protein": 0.08, "plant_pigment": 0.10},
        a_prime=50.0, f_rayleigh=0.08, b_mie=0.8, g500=0.90,
        note="Macerated leaf tissue. Dominates 400-500 and 650-690 nm in a "
             "herbivorous larva and is routinely omitted from insect models.",
        chromophore_kwargs={
            "plant_pigment": {"chl_mg_per_ml": chl_mg_per_ml,
                              "package_factor": package_factor}
        },
    )


def viscera(thickness_um=4000.0, molluscan=True):
    comp = ({"water": 0.78, "protein": 0.16, "lipid": 0.05,
             "hemocyanin": 0.01} if molluscan
            else {"water": 0.78, "protein": 0.17, "lipid": 0.05})
    return Layer(
        name="viscera",
        thickness_um=thickness_um,
        composition=comp,
        a_prime=24.0, f_rayleigh=0.15, b_mie=1.0, g500=0.92,
        note="Lumped digestive gland / gonad / body cavity.",
    )


# ----------------------------------------------------------------------------
# Optional outer layers - these are what distinguish the organisms
# ----------------------------------------------------------------------------

def epicuticle(thickness_um=2.0):
    return Layer(
        name="epicuticle_wax",
        thickness_um=thickness_um,
        composition={"lipid": 0.85, "protein": 0.15},
        a_prime=4.0, f_rayleigh=0.10, b_mie=1.0, g500=0.80,
        optional=True,
        note="Hydrophobic wax bloom. Optically thin but sets the Fresnel "
             "boundary: n~1.47, so R0 ~ 3.6%, not the 4.8% a bare-chitin "
             "assumption gives.",
    )


def exocuticle(thickness_um=60.0, melanin=0.055, ommochrome=0.01):
    water = 0.18
    solid = 1.0 - water - melanin - ommochrome
    return Layer(
        name="exocuticle_sclerotised",
        thickness_um=thickness_um,
        composition={"water": water, "chitin": solid * 0.55,
                     "protein": solid * 0.45,
                     "melanin": melanin, "ommochrome": ommochrome},
        a_prime=130.0, f_rayleigh=0.12, b_mie=1.1, g500=0.95,
        optional=True,
        note="Quinone-tanned, melanised, dehydrated. The armour layer. "
             "Helicoidal lamellae give strong forward scattering. Set "
             "melanin -> ~0.002 for a pale larva, ~0.09 for a black beetle.",
    )


def endocuticle(thickness_um=100.0):
    return Layer(
        name="endocuticle",
        thickness_um=thickness_um,
        composition={"water": 0.42, "chitin": 0.33, "protein": 0.25},
        a_prime=70.0, f_rayleigh=0.15, b_mie=1.0, g500=0.94,
        optional=True,
        note="Untanned, hydrated lamellar chitin-protein. Much more "
             "transparent than the exocuticle above it.",
    )


def subelytral_air(thickness_um=300.0):
    return Layer(
        name="subelytral_air_gap",
        thickness_um=thickness_um,
        composition={"air": 1.0},
        a_prime=0.0, f_rayleigh=0.0, b_mie=1.0, g500=0.0,
        optional=True,
        note="THE layer that defeats dorsal beetle targeting. n=1.0 against "
             "n~1.5 cuticle gives a large Fresnel step, and with mua=mus=0 "
             "no energy is deposited. Heat cannot conduct across it either.",
    )


def mucus(thickness_um=150.0):
    return Layer(
        name="mucus",
        thickness_um=thickness_um,
        composition={"water": 0.97, "mucin": 0.03},
        a_prime=3.0, f_rayleigh=0.40, b_mie=1.5, g500=0.90,
        optional=True,
        note="Slug/snail pedal and mantle mucus. Nearly index-matched to "
             "water, so it suppresses the surface Fresnel reflection and "
             "acts as a sacrificial ablation layer in the SWIR.",
    )


def mollusc_integument(thickness_um=60.0, melanin=0.012):
    water = 0.72
    solid = 1.0 - water - melanin
    return Layer(
        name="integument",
        thickness_um=thickness_um,
        composition={"water": water, "protein": solid * 0.8,
                     "lipid": solid * 0.2, "melanin": melanin},
        a_prime=40.0, f_rayleigh=0.25, b_mie=1.2, g500=0.89,
        optional=True,
        note="Unsclerotised epithelium. No chitin armour, no air gap: "
             "optically the softest of the four targets.",
    )


def periostracum(thickness_um=20.0):
    return Layer(
        name="periostracum",
        thickness_um=thickness_um,
        composition={"conchiolin": 0.85, "water": 0.15},
        a_prime=90.0, f_rayleigh=0.20, b_mie=1.2, g500=0.92,
        optional=True,
        note="Quinone-tanned protein skin over the shell. Carries most of "
             "the shell's visible-band absorption; the mineral beneath is "
             "near-transparent.",
    )


def shell(thickness_um=800.0, organic_fraction=0.03, porosity=0.02):
    mineral = 1.0 - organic_fraction - porosity
    return Layer(
        name="shell_CaCO3",
        thickness_um=thickness_um,
        composition={"aragonite": mineral, "conchiolin": organic_fraction,
                     "air": porosity},
        a_prime=160.0, f_rayleigh=0.02, b_mie=0.45, g500=0.90,
        optional=True,
        note="Crossed-lamellar aragonite. The shell is opaque by SCATTERING, "
             "not absorption: mua stays ~0.1 cm^-1 across 400-2200 nm while "
             "mus' is 100-200 cm^-1. Features are 0.5-20 um, hence the very "
             "shallow Mie exponent. n~1.63 against n~1.0 outside is the "
             "largest index step anywhere in these models.",
    )


# ----------------------------------------------------------------------------
# Assembled presets
# ----------------------------------------------------------------------------

def beetle_elytron(include_air_gap: bool = True) -> Morphology:
    """Adult beetle, dorsal aspect through the elytron."""
    layers = [
        epicuticle(2.0),
        exocuticle(60.0, melanin=0.075, ommochrome=0.012),
        endocuticle(100.0),
    ]
    if include_air_gap:
        layers.append(subelytral_air(300.0))
    layers += [
        epidermis(20.0),
        fat_body(400.0),
        hemolymph(250.0),
        muscle(1500.0),
    ]
    return Morphology(
        name="beetle_elytron",
        layers=layers,
        n_above=1.0003, n_below=1.40,
        description="Melanised sclerotised armour over an air gap. The air "
                    "gap is the dominant architectural feature.",
    )


def beetle_ventral() -> Morphology:
    """Adult beetle, ventral/intersegmental aspect: no air gap, thin cuticle."""
    return Morphology(
        name="beetle_ventral",
        layers=[
            epicuticle(1.5),
            exocuticle(22.0, melanin=0.030, ommochrome=0.008),
            endocuticle(40.0),
            epidermis(20.0),
            fat_body(350.0),
            hemolymph(300.0),
            muscle(1500.0),
        ],
        n_above=1.0003, n_below=1.40,
        description="The soft approach on a beetle: thinner, less melanised, "
                    "and no subelytral cavity.",
    )


def larva(pale: bool = True, chl_mg_per_ml: float = 0.30) -> Morphology:
    """Soft-bodied lepidopteran/coleopteran larva with leaf-filled gut."""
    return Morphology(
        name="larva_pale" if pale else "larva_pigmented",
        layers=[
            epicuticle(1.0),
            exocuticle(8.0, melanin=0.0006 if pale else 0.030,
                       ommochrome=0.0004 if pale else 0.006),
            endocuticle(20.0),
            epidermis(22.0),
            fat_body(450.0),
            hemolymph(300.0),
            gut_contents(2000.0, chl_mg_per_ml=chl_mg_per_ml),
            muscle(600.0),
        ],
        n_above=1.0003, n_below=1.40,
        description="Thin unsclerotised cuticle, little melanin. Blue light "
                    "is not surface-absorbed here; it scatters into the body "
                    "until it meets gut chlorophyll.",
    )


def slug() -> Morphology:
    """No shell, no sclerotised exterior."""
    return Morphology(
        name="slug",
        layers=[
            mucus(150.0),
            mollusc_integument(60.0, melanin=0.015),
            Layer(name="dermal_connective", thickness_um=400.0,
                  composition={"water": 0.80, "protein": 0.17, "lipid": 0.03},
                  a_prime=30.0, f_rayleigh=0.22, b_mie=1.1, g500=0.91,
                  note="Collagen-like connective layer."),
            muscle(3000.0),
            viscera(3000.0, molluscan=True),
        ],
        n_above=1.0003, n_below=1.38,
        description="The softest target: high water content throughout, no "
                    "armour, no air gap, mucus index-matched to water.",
    )


def snail(shell_um: float = 800.0, retracted: bool = True) -> Morphology:
    """Shelled gastropod. Set retracted=True for the defensive posture."""
    layers = [periostracum(20.0), shell(shell_um)]
    if retracted:
        layers.append(Layer(
            name="mantle_cavity_air", thickness_um=250.0,
            composition={"air": 1.0},
            a_prime=0.0, f_rayleigh=0.0, b_mie=1.0, g500=0.0,
            optional=True,
            note="Air-filled mantle cavity behind a retracted body. Same "
                 "role as the beetle's subelytral gap, but thicker."))
    layers += [
        mucus(60.0),
        mollusc_integument(80.0, melanin=0.010),
        viscera(4000.0, molluscan=True),
    ]
    return Morphology(
        name="snail_retracted" if retracted else "snail_extended",
        layers=layers,
        n_above=1.0003, n_below=1.38,
        description="CaCO3 shell plus an air gap behind it when retracted. "
                    "Opaque by scattering, not absorption.",
    )


PRESETS = {
    "beetle_elytron": beetle_elytron,
    "beetle_ventral": beetle_ventral,
    "larva": larva,
    "slug": slug,
    "snail": snail,
}
