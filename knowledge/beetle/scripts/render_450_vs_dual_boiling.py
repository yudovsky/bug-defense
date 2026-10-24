"""Boiling case: 450 nm 40 W single pulse vs 1940 -> 450 nm dual pulse. Dark theme."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, OUT = os.path.join(HERE, "raw_arrays"), os.path.join(HERE, "plots")
BG, PANEL, FG, MUTED = "#0d1117", "#161b22", "#f0f6fc", "#8b949e"
CA, CB = "#58a6ff", "#f59e0b"
CASES = [("matrix_450nm_40w", "450 nm, 40 W (single pulse)", CA), ("matrix_dual_1940nm_450nm", "1940 → 450 nm (dual pulse)", CB)]
kp = json.load(open(os.path.join(HERE, "kpi_cache.json")))
phase_cmap = ListedColormap(["#1e293b", "#d97706", "#06b6d4", "#000000"])
PIXMM2 = 0.015 * 0.015

fig = plt.figure(figsize=(15, 11.5), dpi=200); fig.patch.set_facecolor(BG)
fig.text(0.06, 0.955, "Boiling case: 450 nm at 40 W against the 1940 → 450 nm dual pulse", color=FG, fontsize=18, fontweight="bold")
fig.text(0.06, 0.926, "Top: tissue state at 180 ms, just before the laser turns off. Bottom: how the boiled zone, crater and ventral heating develop.",
         color=MUTED, fontsize=10.5)
data = []
for c, (name, label, col) in enumerate(CASES):
    d = np.load(os.path.join(RAW, name + ".npz")); t = d["times"] * 1000.0
    data.append((d, t, label, col, kp[name + ".npz"]["kpis"]))
    i = int(np.argmin(np.abs(t - 180.0)))
    gx = d["grid_x"] - d["grid_x"][-1] / 2.0; gz = d["grid_z"] * 1000.0
    ax = fig.add_axes([0.08 + c * 0.47, 0.47, 0.42, 0.40]); ax.set_facecolor(PANEL)
    ax.imshow(d["xz_phase"][i].T, extent=[gx[0], gx[-1], gz[-1], gz[0]], aspect="auto", cmap=phase_cmap, vmin=0, vmax=3)
    ax.set_xlim(-1.5, 1.5); ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ax.spines.values(): sp.set_color("#30363d")
    for z in (float(d["dorsal_cuticle_um"]), float(d["air_gap_end_um"]), float(d["ventral_shell_um"])):
        ax.axhline(z, color="#ffffff", lw=0.7, ls="--", alpha=0.5)
    ax.axhline(0, color="#38bdf8", lw=0.9, alpha=0.9)
    ax.set_xlabel("Radial position x (mm)", color=FG, fontsize=10)
    if c == 0: ax.set_ylabel("Depth z (µm)", color=FG, fontsize=10)
    else: ax.set_yticklabels([])
    ax.set_title(label, color=col, fontsize=14, fontweight="bold", loc="left", pad=8)
    k = kp[name + ".npz"]["kpis"]
    ax.text(0.02, 0.025, f"{k['energy_used_j']:.1f} J   boiled {k['v_boil_mm3']:.2f} mm³   ablated {k['v_abl_mm3']:.2f} mm³   depth {k['final_depth_um']:.0f} µm",
            transform=ax.transAxes, color=FG, fontsize=9, va="bottom", bbox=dict(fc=BG, ec="#30363d", alpha=0.85, boxstyle="round,pad=0.35"))

def line_panel(x0, w, title, ylabel, fn):
    ax = fig.add_axes([x0, 0.10, w, 0.25]); ax.set_facecolor(PANEL)
    for d, t, label, col, k in data:
        ax.plot(t, fn(d), color=col, lw=2.2, label=label)
    ax.axvline(200, color="#6e7681", ls=":", lw=1); ax.text(203, ax.get_ylim()[1] * 0.02, "laser off", color=MUTED, fontsize=8, va="bottom")
    ax.set_title(title, color=FG, fontsize=11.5, fontweight="bold", loc="left")
    ax.set_xlabel("Time (ms)", color=FG, fontsize=10); ax.set_ylabel(ylabel, color=FG, fontsize=10)
    ax.tick_params(colors=MUTED); ax.grid(True, color="#21262d", lw=0.8)
    for sp in ax.spines.values(): sp.set_color("#30363d")
    return ax
a1 = line_panel(0.08, 0.26, "Boiled cross-section", "Area (mm²)", lambda d: (d["xz_phase"] == 2).sum(axis=(1, 2)) * PIXMM2)
a2 = line_panel(0.40, 0.26, "Crater depth", "Depth (µm)", lambda d: d["crater_depths_mm"] * 1000.0)
a3 = line_panel(0.72, 0.26, "Peak ventral temperature", "Temperature (°C)", lambda d: d["peak_T_ventral"])
a1.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor=FG, fontsize=8.5)
fig.legend(handles=[Patch(fc="#1e293b", ec="#64748b", label="Intact tissue"), Patch(fc="#d97706", ec="#b45309", label="Charred cuticle"),
                    Patch(fc="#06b6d4", ec="#0891b2", label="Boiling tissue (≥ 100 °C)"), Patch(fc="#000000", ec="#ffffff", label="Void (ablated or air gap)")],
           loc="center", bbox_to_anchor=(0.5, 0.045), ncol=4, facecolor="#21262d", edgecolor="#30363d", labelcolor=FG, fontsize=10)
fig.text(0.06, 0.012, "Dual run: 40 W for 30 ms (1940 nm), then 20 W (450 nm). Boiled cross-section is the centre slice of the 3D run. Volumes and depth are final (1 s) values.",
         color=MUTED, fontsize=9)
fig.savefig(os.path.join(OUT, "beetle_450nm_vs_dual_1940_450_boiling.png"), facecolor=BG)
print("saved")
