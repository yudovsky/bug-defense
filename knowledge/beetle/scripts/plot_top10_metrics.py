"""Three top-10 scatter plots (boiled volume, ablated volume, penetration depth + speed vs energy)
and a summary image of the best scenario for each. Plume on, 200-2000 nm, no controls."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "plots")
BG, PANEL, FG, MUTED = "#0d1117", "#161b22", "#f0f6fc", "#8b949e"
C_BOIL, C_ABL, C_DEPTH = "#06b6d4", "#f59e0b", "#a78bfa"
PULSE_S = 0.2
POWER_W = 40.0   # restrict to 40 W runs (dual runs: 40 W cutter pulse then 20 W)
SUF = "_40w"

d = json.load(open(os.path.join(HERE, "kpi_cache.json")))
rows = []
for k, v in d.items():
    p, n = v["kpis"], k.replace("matrix_", "").replace(".npz", "")
    if any(t in n for t in ("no_plume", "burst", "cw_matched", "1ms_2x")): continue
    w = [p.get("wavelength_nm"), p.get("wl2")] if "dual" in n else [p["wavelength_nm"]]
    if not all(x and 200 <= x <= 2000 for x in w): continue
    if abs(p["power_w"] - POWER_W) > 1e-6: continue
    if "dual" in n:
        a, b = n.replace("dual_", "").split("_")
        label = f"{a.replace('nm','')} → {b.replace('nm','')} nm"
    else:
        label = f"{p['wavelength_nm']} nm, {int(p['power_w'])} W"
    rows.append(dict(label=label, E=p["energy_used_j"], abl=p["v_abl_mm3"], boil=p["v_boil_mm3"],
                     depth=p["final_depth_um"], speed=p["speed_mm_s"]))
for r in rows: r["tot"] = r["abl"] + r["boil"]

def top10(key):
    # ties: lower energy first, then larger total volume
    return sorted(rows, key=lambda r: (-round(r[key], 6), r["E"], -r["tot"]))[:10]

def style(ax):
    ax.set_facecolor(PANEL); ax.tick_params(colors=MUTED)
    ax.grid(True, color="#21262d", lw=0.8)
    for s in ax.spines.values(): s.set_color("#30363d")

def make(key, ylabel, title, color, fname, ylim, marker="o", twin=None, note=None):
    top = top10(key)
    for r, rk in zip(top, range(1, 11)): r["rank"] = rk
    for E in sorted({r["E"] for r in top}):
        g = sorted([r for r in top if r["E"] == E], key=lambda r: r["rank"])
        for j, r in enumerate(g): r["x"] = E + (j - (len(g) - 1) / 2) * 0.32
    fig = plt.figure(figsize=(12.5, 7.6), dpi=200); fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.08, 0.13, 0.82 if twin else 0.89, 0.72]); style(ax)
    ax.scatter([r["x"] for r in top], [r[key] for r in top], s=110, c=color, marker=marker, edgecolor=BG, zorder=3)
    span = ylim[1] - ylim[0]
    for E in sorted({r["E"] for r in top}):
        g = sorted([r for r in top if r["E"] == E], key=lambda r: -r[key])
        ys = [r[key] for r in g]
        for i in range(1, len(ys)):
            if ys[i - 1] - ys[i] < 0.055 * span: ys[i] = ys[i - 1] - 0.055 * span
        xt = max(r["x"] for r in g) + 0.45
        for r, y in zip(g, ys):
            ax.annotate(f"{r['rank']}. {r['label']}", (r["x"], r[key]), xytext=(xt, y), color=FG, fontsize=9, va="center",
                        arrowprops=dict(arrowstyle="-", color="#6e7681", lw=0.7, shrinkA=0, shrinkB=4), zorder=4)
    es = sorted({r["E"] for r in top})
    ax.set_xlim(min(es) - 1.0, max(es) + 3.4); ax.set_ylim(*ylim)
    ax.set_xticks(es); ax.set_xticklabels([f"{e:g}" for e in es])
    ax.set_xlabel("Energy delivered (J)", color=FG, fontsize=11); ax.set_ylabel(ylabel, color=FG, fontsize=11)
    if twin:
        sec = ax.secondary_yaxis("right", functions=twin[1:]); sec.set_ylabel(twin[0], color=FG, fontsize=11)
        sec.tick_params(colors=MUTED)
        for s in sec.spines.values(): s.set_color("#30363d")
    fig.text(0.08, 0.945, title, color=FG, fontsize=17, fontweight="bold")
    fig.text(0.08, 0.905, "Top 10 of the 40 W scenarios (dual runs: 40 W then 20 W). Plume on, 200 to 2000 nm. Ties broken by lower energy. Same-energy points are offset sideways.",
             color=MUTED, fontsize=9.5)
    if note: fig.text(0.08, 0.878, note, color=MUTED, fontsize=9.5)
    fig.savefig(os.path.join(OUT, fname), facecolor=BG); plt.close(fig)
    return top

t1 = make("boil", "Boiled volume (mm³)", "Boiled volume against energy", C_BOIL, f"beetle_top10_boiled_vs_energy{SUF}.png", (0, 5.8))
t2 = make("abl", "Ablated volume (mm³)", "Ablated volume against energy", C_ABL, f"beetle_top10_ablated_vs_energy{SUF}.png", (0, 1.1), marker="D")
t3 = make("depth", "Penetration depth (µm)", "Penetration depth and speed against energy", C_DEPTH, f"beetle_top10_depth_speed_vs_energy{SUF}.png",
          (0, 2800), twin=("Penetration speed (mm/s)", lambda y: y / (PULSE_S * 1000), lambda s: s * PULSE_S * 1000),
          note="Speed is depth divided by the 200 ms pulse. 2445 µm is the full body.")

# summary: best single-wavelength scenario in each plot vs the closest dual-wavelength contender
def best_dual(key):
    ds = [r for r in rows if "→" in r["label"]]
    return sorted(ds, key=lambda r: (-round(r[key], 6), r["E"], -r["tot"]))[0]
best = [("Most boiled volume", t1[0], best_dual("boil"), "boil"),
        ("Most ablated volume", t2[0], best_dual("abl"), "abl"),
        ("Deepest penetration, lowest energy", t3[0], best_dual("depth"), "depth")]
fig = plt.figure(figsize=(14.5, 7.2), dpi=200); fig.patch.set_facecolor(BG)
fig.text(0.04, 0.925, "Top 40 W scenario from each plot, and the closest dual-wavelength contender", color=FG, fontsize=18, fontweight="bold")
colors = {"boil": C_BOIL, "abl": C_ABL, "depth": C_DEPTH}
fmt = [("Energy", "E", "{:.1f} J"), ("Boiled volume", "boil", "{:.2f} mm³"), ("Ablated volume", "abl", "{:.2f} mm³"),
       ("Depth", "depth", "{:.0f} µm"), ("Speed", "speed", "{:.1f} mm/s")]
for i, (head, r, dl, key) in enumerate(best):
    x0 = 0.04 + i * 0.317
    ax = fig.add_axes([x0, 0.08, 0.30, 0.76]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc=PANEL, ec=colors[key], lw=2, transform=ax.transAxes))
    ax.text(0.05, 0.94, head.upper(), color=colors[key], fontsize=10.5, fontweight="bold", va="center")
    ax.text(0.05, 0.855, "Single", color=MUTED, fontsize=10, va="center", fontweight="bold")
    ax.text(0.22, 0.855, r["label"], color=FG, fontsize=13, va="center", fontweight="bold")
    ax.text(0.05, 0.785, "Dual", color=MUTED, fontsize=10, va="center", fontweight="bold")
    ax.text(0.22, 0.785, dl["label"], color=FG, fontsize=13, va="center", fontweight="bold")
    ax.text(0.66, 0.69, "SINGLE", color=MUTED, fontsize=9, ha="right", va="center", fontweight="bold")
    ax.text(0.93, 0.69, "DUAL", color=MUTED, fontsize=9, ha="right", va="center", fontweight="bold")
    for j, (name, k, f) in enumerate(fmt):
        y = 0.60 - j * 0.100
        hl = (k == key)
        ax.text(0.05, y, name, color=FG if hl else MUTED, fontsize=11.5, va="center", fontweight="bold" if hl else "normal")
        ax.text(0.66, y, f.format(r[k]).replace(" mm³", "").replace(" µm", "").replace(" mm/s", "").replace(" J", ""),
                color=colors[key] if hl else FG, fontsize=12, va="center", ha="right", fontweight="bold" if hl else "normal")
        ax.text(0.93, y, f.format(dl[k]).replace(" mm³", "").replace(" µm", "").replace(" mm/s", "").replace(" J", ""),
                color=colors[key] if hl else FG, fontsize=12, va="center", ha="right", fontweight="bold" if hl else "normal")
        ax.plot([0.05, 0.95], [y - 0.047, y - 0.047], color="#21262d", lw=0.8)
    unit = {"E": "J", "boil": "mm³", "abl": "mm³", "depth": "µm", "speed": "mm/s"}
    what = dict(boil="boiled volume", abl="ablated volume", depth="depth")[key]
    ax.text(0.05, 0.085, f"Dual reaches {100.0 * dl[key] / r[key]:.0f}% of the single {what},\nusing {dl['E']:.1f} J against {r['E']:.1f} J.",
            color=colors[key], fontsize=10.5, va="center", fontweight="bold", linespacing=1.5)
fig.text(0.04, 0.035, "Units: J, mm³, µm, mm/s. Single runs are 40 W for 200 ms; dual runs are 40 W for 30 ms then 20 W.\n"
         "Plume on, 200 to 2000 nm. Depth: 1940 → 1064 nm ties the dual shown at 1680 µm.", color=MUTED, fontsize=9)
fig.savefig(os.path.join(OUT, f"beetle_top3_summary{SUF}.png"), facecolor=BG)
for nm, t in (("boil", t1), ("abl", t2), ("depth", t3)):
    print(nm, [(r["label"], r["E"], round(r[nm], 3)) for r in t])
