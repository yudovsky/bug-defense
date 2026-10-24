"""Absorption of the main chromophores and reduced scattering of the layers in the beetle model."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from insect_tissue_optics import basis, morphology as mo

nm = np.arange(400, 2201, 5.0)
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9aa5b1",
                     "axes.labelcolor": "#323f4b", "xtick.color": "#52606d",
                     "ytick.color": "#52606d", "axes.titlecolor": "#1f2933"})

# --- absorption: pure components at unit volume fraction ---
comps = [
    ("Water", basis.water_mua, "#2b7bba"),
    ("Melanin", basis.melanin_mua, "#7a4a21"),
    ("Ommochrome", basis.ommochrome_mua, "#d9822b"),
    ("Chitin / protein", basis.chitin_mua, "#6b8e23"),
    ("Lipid", basis.lipid_mua, "#c9a400"),
    ("Cellular baseline", basis.baseline_tissue_mua, "#7b8794"),
]
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
# Lipid: measured porcine fat (van Veen et al. 2004, OMLC) for 431-1098 nm, converted m^-1 -> cm^-1.
# The model's lipid curve (Gaussians) is shown dashed outside that range, where no measured table was used.
with open(os.path.join(HERE, "insect_tissue_optics", "omlc_fat.txt")) as fh:
    fat = np.array([[float(v) for v in ln.split()[:2]] for ln in fh if ln[:1].isdigit()])
fat_nm, fat_mua = fat[:, 0], fat[:, 1] / 100.0
for name, fn, c in comps:
    if name == "Lipid":
        m = nm > fat_nm.max()
        ax.semilogy(fat_nm, fat_mua, color=c, lw=2, label="Lipid (measured, van Veen 2004)")
        ax.semilogy(nm[m], fn(nm[m]), color=c, lw=2, ls="--", label="Lipid (model, unmeasured)")
        continue
    ax.semilogy(nm, np.clip(fn(nm), 1e-4, None), label=name, color=c, lw=2)
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("μa at 100% volume fraction (cm⁻¹)")
ax.set_ylim(1e-3, 1e3); ax.set_xlim(400, 2200)
ax.set_title("Absorption of the main chromophores", loc="left", fontweight="bold")
ax.grid(True, which="major", color="#e4e7eb", lw=0.8)
ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "beetle_absorption_chromophores.png"), facecolor="white")

# --- scattering: reduced scattering per layer ---
layers = [
    ("Epicuticle wax", mo.epicuticle(), "#c9a400"),
    ("Exocuticle", mo.exocuticle(60.0, melanin=0.075, ommochrome=0.012), "#7a4a21"),
    ("Endocuticle", mo.endocuticle(), "#b8884f"),
    ("Epidermis", mo.epidermis(), "#d9534f"),
    ("Fat body", mo.fat_body(), "#e0b000"),
    ("Hemolymph", mo.hemolymph(), "#2b7bba"),
    ("Muscle", mo.muscle(), "#a02c5a"),
]
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
for name, L, c in layers:
    ax.semilogy(nm, L.musp(nm), label=name, color=c, lw=2)
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Reduced scattering μs′ (cm⁻¹)")
ax.set_xlim(400, 2200)
ax.set_title("Reduced scattering of the model layers", loc="left", fontweight="bold")
ax.grid(True, which="major", color="#e4e7eb", lw=0.8)
ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
fig.text(0.01, 0.01, "Assumed parameters, not measured on beetles.", fontsize=8, color="#7b8794")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(HERE, "beetle_scattering_layers.png"), facecolor="white")
