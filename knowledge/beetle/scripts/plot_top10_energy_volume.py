"""Top 10 scenarios (ranked by ablated + boiled volume) of the 200-2000 nm, plume-on runs:
energy vs ablated and boiled volume, with a summary-statistics table."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BG, PANEL, FG, MUTED = "#0d1117", "#161b22", "#f0f6fc", "#8b949e"
C_ABL, C_BOIL = "#f59e0b", "#06b6d4"

d = json.load(open(os.path.join(HERE, "kpi_cache.json")))
rows = []
for k, v in d.items():
    p, n = v["kpis"], k.replace("matrix_", "").replace(".npz", "")
    if "no_plume" in n or "burst" in n or "cw_matched" in n or "1ms_2x" in n:
        continue
    w = [p.get("wavelength_nm"), p.get("wl2")] if "dual" in n else [p["wavelength_nm"]]
    if not all(x and 200 <= x <= 2000 for x in w):
        continue
    if "dual" in n:
        a, b = n.replace("dual_", "").split("_")
        label = f"{a.replace('nm','')} → {b.replace('nm','')} nm"
    else:
        label = f"{p['wavelength_nm']} nm, {int(p['power_w'])} W"
    rows.append(dict(label=label, E=p["energy_used_j"], abl=p["v_abl_mm3"], boil=p["v_boil_mm3"]))
for r in rows: r["tot"] = r["abl"] + r["boil"]
top = sorted(rows, key=lambda r: -r["tot"])[:10]

# horizontal offsets so points at the same energy do not sit on top of each other
for E in sorted({r["E"] for r in top}):
    g = sorted([r for r in top if r["E"] == E], key=lambda r: -r["boil"])
    for j, r in enumerate(g):
        r["x"] = E + (j - (len(g) - 1) / 2) * 0.32

fig = plt.figure(figsize=(12.5, 10.5), dpi=200)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.08, 0.40, 0.89, 0.50]); ax.set_facecolor(PANEL)
for r in top:
    ax.plot([r["x"], r["x"]], [r["abl"], r["boil"]], color="#30363d", lw=1.2, zorder=1)
ax.scatter([r["x"] for r in top], [r["boil"] for r in top], s=95, c=C_BOIL, edgecolor="#0d1117", zorder=3, label="Boiled volume")
ax.scatter([r["x"] for r in top], [r["abl"] for r in top], s=95, c=C_ABL, edgecolor="#0d1117", zorder=3, marker="D", label="Ablated volume")
# labels in a column to the right of each energy group, nudged apart
for E in sorted({r["E"] for r in top}):
    g = sorted([r for r in top if r["E"] == E], key=lambda r: -r["boil"])
    ys = [r["boil"] for r in g]
    for i in range(1, len(ys)):
        if ys[i - 1] - ys[i] < 0.32: ys[i] = ys[i - 1] - 0.32
    xt = max(r["x"] for r in g) + 0.45
    for r, y in zip(g, ys):
        ax.annotate(r["label"], (r["x"], r["boil"]), xytext=(xt, y), color=FG, fontsize=9, va="center",
                    arrowprops=dict(arrowstyle="-", color="#6e7681", lw=0.7, shrinkA=0, shrinkB=4), zorder=4)
ax.set_xlim(3.5, 19.5); ax.set_ylim(0, 5.6)
ax.set_xticks([4.6, 8, 16]); ax.set_xticklabels(["4.6", "8.0", "16.0"])
ax.set_xlabel("Energy delivered (J)", color=FG, fontsize=11); ax.set_ylabel("Volume (mm³)", color=FG, fontsize=11)
ax.tick_params(colors=MUTED); ax.grid(True, color="#21262d", lw=0.8)
for s in ax.spines.values(): s.set_color("#30363d")
leg = ax.legend(loc="upper left", facecolor="#21262d", edgecolor="#30363d", labelcolor=FG, fontsize=10)
fig.text(0.08, 0.955, "Top 10 scenarios: energy against ablated and boiled volume", color=FG, fontsize=17, fontweight="bold")
fig.text(0.08, 0.928, "Ranked by ablated + boiled volume. Plume on, wavelengths 200 to 2000 nm. Points at the same energy are offset sideways.",
         color=MUTED, fontsize=10.5)

# summary table
def stats(v):
    v = np.asarray(v, float)
    return [np.mean(v), np.median(v), np.std(v, ddof=1), v.min(), v.max()]
data = {
    "Energy (J)": [r["E"] for r in top],
    "Ablated volume (mm³)": [r["abl"] for r in top],
    "Boiled volume (mm³)": [r["boil"] for r in top],
    "Ablated + boiled (mm³)": [r["tot"] for r in top],
    "Volume per joule (mm³/J)": [r["tot"] / r["E"] for r in top],
}
cells = [[f"{x:.2f}" for x in stats(v)] for v in data.values()]
tax = fig.add_axes([0.30, 0.05, 0.67, 0.25]); tax.axis("off")
tbl = tax.table(cellText=cells, rowLabels=list(data), colLabels=["Mean", "Median", "Std dev", "Min", "Max"],
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.1)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#30363d"); cell.set_text_props(color=FG)
    cell.set_facecolor("#21262d" if r == 0 or c == -1 else PANEL)
    if r == 0 or c == -1: cell.set_text_props(color=FG, fontweight="bold")
    if c == -1: cell.set_text_props(ha="right")
fig.text(0.08, 0.02, "n = 10 scenarios.", color=MUTED, fontsize=9)
fig.savefig(os.path.join(HERE, "plots", "beetle_top10_energy_vs_volume.png"), facecolor=BG)
for r in top: print(r["label"], r["E"], round(r["abl"],3), round(r["boil"],3), round(r["tot"],3))
