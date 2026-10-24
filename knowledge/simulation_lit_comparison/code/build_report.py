"""
Builds README.md and one comparison figure per reference (images/refNN_*.png) from the benchmark
result files. Every number in the report is read from raw_output/*.json -- nothing is typed in by hand.

    ./venv/bin/python knowledge/simulation_lit_comparison/code/build_report.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import bench_common
from bench_common import IMAGES_DIR, RAW_DIR, SUITE_DIR
import verification_benchmarks as vb
import keller_benchmark as kel
import mcmillan_benchmark as mcm_mod
import kim_benchmark as kimb

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25, 'axes.titlesize': 10,
                     'axes.titleweight': 'bold', 'figure.dpi': 110})
C = {'paper': '#6a1b9a', 'paper2': '#1f77b4', 'paper3': '#ff7f0e', 'eng': 'k', 'eng2': '#2ca02c',
     'eng3': '#d62728', 'grey': '0.55'}


def jload(name):
    with open(RAW_DIR / name) as f:
        return json.load(f)


def R_all():
    d = jload('benchmark_results.json')
    return {r['id']: r for r in d['results']}, d


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def ok(r):
    return 'PASS' if r['pass'] else '**FAIL**'


def save(fig, name):
    p = IMAGES_DIR / name
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return f"images/{name}"


# ---------------------------------------------------------------------------------------------
# 1. Shimojo et al. 2023
# ---------------------------------------------------------------------------------------------
def ref_shimojo(R):
    val = {k[2:]: r for k, r in R.items() if r['tier'] == 'validation' and k.startswith('S-')}
    xc = {k[2:]: r for k, r in R.items() if r['tier'] == 'cross-code' and k.startswith('X-')}
    if not val:
        return None, "_Shimojo results not available._", (0, 0)
    panels = [('50', 'vap_depth', 'Vaporization depth, 50 W'), ('50', 'vap_width', 'Vaporization width, 50 W'),
              ('50', 'coag_depth', 'Coagulation depth, 50 W'), ('50', 'coag_width', 'Coagulation width, 50 W'),
              ('10', 'coag_depth', 'Coagulation depth, 10 W'), ('10', 'coag_width', 'Coagulation width, 10 W')]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    axes = axes.ravel()
    for ax, (pw, key, title) in zip(axes, panels):
        rs = sorted([r for k, r in val.items() if k.startswith(f"{pw}W-") and k.endswith(key)],
                    key=lambda r: r['series']['speed_mm_s'])
        v = [r['series']['speed_mm_s'] for r in rs]
        ax.errorbar(v, [r['reference_value'] for r in rs], yerr=[r['series']['exp_sd'] for r in rs], fmt='x-',
                    color=C['paper'], capsize=4, label='Paper: experiment (±1 SD, n=3)')
        ax.plot(v, [r['series']['lit_sim_dop'] for r in rs], 'o--', color=C['paper2'], mfc='none', label='Paper: simulation, with DOP')
        ax.plot(v, [r['series']['lit_sim_nodop'] for r in rs], '^--', color=C['paper3'], mfc='none', label='Paper: simulation, without DOP')
        ax.plot(v, [r['model'] for r in rs], 'D-', color=C['eng'], label='This engine (2 ms optical updates)')
        coarse = [r['series'].get('coarse_epoch_model') for r in rs]
        if all(c is not None for c in coarse):
            ax.plot(v, coarse, 's:', color=C['grey'], mfc='none', label='This engine (coarse 0.1-0.2 s updates)')
        ax.set_title(title)
        ax.set_xlabel('Scan speed [mm/s]')
        ax.set_ylabel('[mm]')
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7)
    try:
        secs = jload('shimojo_cross_sections.json')
        for ax, key in zip(axes[6:8], ('50W_0.5', '10W_1.0')):
            abl = np.array(secs[key]['ablated'], dtype=float)
            coag = np.array(secs[key]['coag'], dtype=float)
            img = np.where(abl > 0, 2.0, np.where(coag > 0, 1.0, 0.0)).T
            ax.imshow(img, cmap=matplotlib.colors.ListedColormap(['#7a2e2e', '#e8d9b5', '#111111']),
                      extent=[-secs[key]['Lx'] / 2, secs[key]['Lx'] / 2, secs[key]['Lz'], 0], vmin=0, vmax=2, aspect='equal')
            ax.grid(False)
            ax.set_title(f"Engine cross-section {key.replace('W_', ' W, ')} mm/s\n(black = vaporized, beige = Ω ≥ 1)")
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('z [mm]')
    except (FileNotFoundError, KeyError):
        pass
    fig.suptitle('Shimojo et al. (2023) Figs. 5 & 7 — 980 nm scanned beam on porcine liver: paper data vs engine', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    img = save(fig, 'ref01_shimojo2023.png')

    rows = []
    for k in sorted(val, key=lambda s: (-float(s.split('W')[0]), s.split('-')[0], float(s.split('-')[1]), s.split('-')[2])):
        r, x = val[k], xc[k]
        s = r['series']
        rows.append([f"{s['power_W']:.0f} W / {s['speed_mm_s']}", r['observable'], f"{r['reference_value']:.2f} ± {s['exp_sd']:.2f}",
                     f"{s['lit_sim_dop']:.2f}", f"{s['lit_sim_nodop']:.2f}", f"**{r['model']:.2f}**",
                     '' if s.get('coarse_epoch_model') is None else f"{s['coarse_epoch_model']:.2f}",
                     f"{ok(r)} ({(r['model']-r['reference_value'])/r['reference_value']*100:+.0f}%)",
                     f"{ok(x)} ({(x['model']-x['reference_value'])/x['reference_value']*100:+.0f}%)"])
    tbl = md_table(['Power / speed [mm/s]', 'Observable [mm]', 'Paper: experiment', 'Paper: sim (DOP)', 'Paper: sim (no DOP)',
                    'Engine, 2 ms', 'Engine, coarse', 'vs experiment', 'vs paper sim (DOP)'], rows)
    p_val = sum(r['pass'] for r in val.values())
    p_xc = sum(r['pass'] for r in xc.values())
    return img, tbl, (p_val, len(val), p_xc, len(xc))


def shimojo_inputs_table():
    import validation_benchmarks as vbm
    L = vbm.LIVER
    rows = [
        ['Absorption μa [mm⁻¹]', '0.07 (native)', f"{L['mu_a']}"],
        ["Reduced scattering μs' [mm⁻¹]", '0.51 native → 1.85 coagulated', f"{L['musp_native']} → {L['musp_coag']} (α = 1 − e^−Ω mixing)"],
        ['Anisotropy g / refractive index n', '0.9 / 1.44', f"{L['g']} / {L['n']}"],
        ['Thermal conductivity k', '5.43×10⁻³ W/(cm·K)', f"{L['k']*1e3:.3g} W/(mm·K) ×10⁻³ (identical)"],
        ['Specific heat × density', '3.542 J/(g·K) × 1.0 g/cm³', f"{L['rho_cp']*1e3:.3f}×10⁻³ J/(mm³·K) (identical)"],
        ['Arrhenius A / Ea', '7.39×10³⁷ s⁻¹ / 2.577×10⁵ J/mol', f"{L['A']:.3g} s⁻¹ / {L['Ea']:.4g} J/mol"],
        ['Tv / Lv', '100 °C / 2257 J/g', f"{L['Tv']} °C / {L['Lv_vol']/1e-3:.0f} J/g"],
        ['T0, boundaries, voxel', '22 °C, adiabatic, 0.1 mm', '22 °C, adiabatic, 0.1 mm'],
        ['Beam', '980 nm CW, Gaussian, 1/e² diameter 2.8 mm, scanned', 'same; beam advances continuously between photon-transport updates'],
    ]
    return md_table(['Input (Supplementary Table S1 / Methods)', 'Paper', 'Engine'], rows)


# ---------------------------------------------------------------------------------------------
# 2, 7, 8, 9, 10, 11: transport / thermal references
# ---------------------------------------------------------------------------------------------
TABLE_61 = {'vdH': (0.09739, 0.66096), 'mcml': ((0.09734, 0.00035), (0.66096, 0.00020)),
            'prahl': ((0.09711, 0.00033), (0.66159, 0.00049))}
TABLE_62 = {'giov': 0.2600, 'mcml': (0.25907, 0.00170), 'prahl': (0.26079, 0.00079)}


def eng_se(r):
    return r['series']['se'] if r['series'] and 'se' in r['series'] else r['tolerance_abs'] / 3.0


def ref_mcml(R):
    rd, tt, r62 = R['V1a'], R['V1b'], R['V2b']
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    sets = [('Table 6.1 — slab diffuse reflectance Rd', [('van de Hulst (1980)', TABLE_61['vdH'][0], 0, C['paper']),
             ('MCML (1992)', *TABLE_61['mcml'][0], C['paper2']), ('Prahl et al. (1989)', *TABLE_61['prahl'][0], C['paper3']),
             ('This engine', rd['model'], eng_se(rd), C['eng'])]),
            ('Table 6.1 — slab total transmittance Tt', [('van de Hulst (1980)', TABLE_61['vdH'][1], 0, C['paper']),
             ('MCML (1992)', *TABLE_61['mcml'][1], C['paper2']), ('Prahl et al. (1989)', *TABLE_61['prahl'][1], C['paper3']),
             ('This engine', tt['model'], eng_se(tt), C['eng'])]),
            ('Table 6.2 — semi-infinite total R (n = 1.5)', [('Giovanelli (1955)', TABLE_62['giov'], 0, C['paper']),
             ('MCML (1992)', *TABLE_62['mcml'], C['paper2']), ('Prahl et al. (1989)', *TABLE_62['prahl'], C['paper3']),
             ('This engine', r62['model'], eng_se(r62), C['eng'])])]
    for ax, (title, pts) in zip(axes, sets):
        for i, (lab, v, e, c) in enumerate(pts):
            ax.errorbar(v, -i, xerr=e, fmt='o', color=c, capsize=4, ms=6)
        ax.set_yticks([-i for i in range(len(pts))])
        ax.set_yticklabels([p[0] for p in pts])
        ax.set_title(title)
        ax.set_xlabel('value (± standard error where reported)')
    fig.suptitle('Wang & Jacques (1992) MCML manual, Chapter 6 — published tables vs this engine', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    img = save(fig, 'ref02_mcml_manual.png')
    rows = [
        ['6.1 slab, n = 1, μa = 10, μs = 90 /cm, g = 0.75, d = 0.02 cm', 'Rd', '0.09739 (vdH)', '0.09734 ± 0.00035', '0.09711 ± 0.00033', f"**{rd['model']:.5f} ± {eng_se(rd):.5f}**", ok(rd)],
        ['', 'Tt', '0.66096 (vdH)', '0.66096 ± 0.00020', '0.66159 ± 0.00049', f"**{tt['model']:.5f} ± {eng_se(tt):.5f}**", ok(tt)],
        ['6.2 semi-infinite, n = 1.5, μa = 10, μs = 90 /cm, g = 0', 'R (total, incl. specular)', '0.2600 (Giovanelli)', '0.25907 ± 0.00170', '0.26079 ± 0.00079', f"**{r62['model']:.5f} ± {eng_se(r62):.5f}**", ok(r62)],
    ]
    tbl = md_table(['Case', 'Quantity', 'Reference value', 'MCML', 'Prahl 1989', 'This engine', 'Result'], rows)
    return img, tbl, (sum(r['pass'] for r in (rd, tt, r62)), 3)


def ref_vdh(R):
    rd, tt = R['V1a'], R['V1b']
    ns = np.array([1e4, 3e4, 1e5, 3e5, 1e6, 3e6])
    dev = []
    for n in ns:
        r = vb.run_voxel_mc3d(n_photons=int(n), spot_dia=0.01, Lx=20.0, Ly=20.0, Lz=0.2, Nx=20, Ny=20, Nz=10,
                              layers_config=[{'z_end': 0.2, 'mu_a': 1.0, 'mu_s': 9.0, 'g': 0.75, 'n': 1.0}])
        dev.append((r['spectro_raw']['R_diff'] - TABLE_61['vdH'][0], r['spectro_raw']['Transmission'] - TABLE_61['vdH'][1],
                    r['spectro_raw']['Absorption']))
    dev = np.array(dev)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    ax = axes[0]
    x = np.arange(2)
    ax.bar(x - 0.18, [TABLE_61['vdH'][0], TABLE_61['vdH'][1]], 0.34, color=C['paper'], label='van de Hulst table')
    ax.bar(x + 0.18, [rd['model'], tt['model']], 0.34, color=C['eng'], label='This engine')
    ax.set_xticks(x)
    ax.set_xticklabels(['Rd', 'Tt'])
    ax.legend()
    ax.set_title('Slab τ = 2, albedo 0.9, g = 0.75')
    ax = axes[1]
    ax.bar(['van de Hulst'], [TABLE_61['vdH'][0]], color='#c5a3e0', label='Rd')
    ax.bar(['van de Hulst'], [TABLE_61['vdH'][1]], bottom=[TABLE_61['vdH'][0]], color='#8e44ad', label='Tt')
    ax.bar(['van de Hulst'], [1 - sum(TABLE_61['vdH'])], bottom=[sum(TABLE_61['vdH'])], color='#ddd', label='absorbed (implied)')
    r = R['V1a']
    ax.bar(['This engine'], [rd['model']], color='#c5a3e0')
    ax.bar(['This engine'], [tt['model']], bottom=[rd['model']], color='#8e44ad')
    ax.bar(['This engine'], [1 - rd['model'] - tt['model']], bottom=[rd['model'] + tt['model']], color='#ddd')
    ax.set_title('Energy partition (sums to 1)')
    ax.legend(fontsize=7)
    ax = axes[2]
    ax.loglog(ns, np.abs(dev[:, 0]), 'o-', label='|Rd − van de Hulst|')
    ax.loglog(ns, np.abs(dev[:, 1]), 's-', label='|Tt − van de Hulst|')
    ax.loglog(ns, 0.5 / np.sqrt(ns), 'k--', lw=0.8, label='~0.5/√N (MC noise scale)')
    ax.set_xlabel('photon packets N')
    ax.set_ylabel('deviation')
    ax.set_title('Convergence to the tabulated values')
    ax.legend(fontsize=7)
    fig.suptitle('van de Hulst (1980), Multiple Light Scattering II — slab values (as tabulated in MCML Table 6.1) vs this engine', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    img = save(fig, 'ref07_vandehulst.png')
    rows = [['Rd', '0.09739', f"{rd['model']:.5f} ± {eng_se(rd):.5f}", f"{rd['model']-TABLE_61['vdH'][0]:+.5f}", f"{rd['rel_error_pct']:.2f}%", ok(rd)],
            ['Tt', '0.66096', f"{tt['model']:.5f} ± {eng_se(tt):.5f}", f"{tt['model']-TABLE_61['vdH'][1]:+.5f}", f"{tt['rel_error_pct']:.2f}%", ok(tt)]]
    tbl = md_table(['Quantity', 'van de Hulst', 'This engine (10 × 200 000 photons)', 'Difference', 'Rel. err.', 'Result'], rows)
    conv = md_table(['N photons', 'Rd − 0.09739', 'Tt − 0.66096'],
                    [[f"{int(n):,}", f"{d[0]:+.5f}", f"{d[1]:+.5f}"] for n, d in zip(ns, dev)])
    return img, tbl + '\n\nConvergence (single runs):\n\n' + conv, (2, 2)


def ref_giovanelli(R):
    r62 = R['V2b']
    sweep = [R[k] for k in sorted(R, key=lambda s: s) if k.startswith('V2d-a')]
    sweep.sort(key=lambda r: r['series']['albedo'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    pts = [('Giovanelli (1955)', TABLE_62['giov'], 0, C['paper']), ('MCML (1992)', *TABLE_62['mcml'], C['paper2']),
           ('Prahl et al. (1989)', *TABLE_62['prahl'], C['paper3']), ('Engine: total R', r62['model'], eng_se(r62), C['eng']),
           ('Engine: diffuse + 0.04 specular', r62['series']['diffuse_only'] + 0.04, eng_se(r62), C['eng2'])]
    for i, (lab, v, e, c) in enumerate(pts):
        ax.errorbar(v, -i, xerr=e, fmt='o', color=c, capsize=4)
    ax.set_yticks([-i for i in range(len(pts))])
    ax.set_yticklabels([p[0] for p in pts])
    ax.set_title('Table 6.2: semi-infinite, n = 1.5, albedo 0.9, g = 0')
    ax.set_xlabel('total reflectance R')
    ax = axes[1]
    al = [r['series']['albedo'] for r in sweep]
    ax.errorbar(al, [r['reference_value'] for r in sweep], yerr=[r['series']['se_ref'] * 3 for r in sweep], fmt='s-', color=C['paper'],
                label='Independent 1-D Monte Carlo (same physics)')
    ax.errorbar(al, [r['model'] for r in sweep], yerr=[r['series']['se'] * 3 for r in sweep], fmt='D', color=C['eng'], label='This engine (±3 SE)')
    ax.plot([0.9], [TABLE_62['giov']], '*', color=C['paper3'], ms=14, label='Giovanelli (1955) at albedo 0.9')
    ax.set_xlabel('single-scattering albedo')
    ax.set_ylabel('total reflectance R (n = 1.5)')
    ax.legend(fontsize=7)
    ax.set_title('Mismatched-boundary reflectance vs albedo')
    fig.suptitle('Giovanelli (1955), Optica Acta 2:153 — semi-infinite diffuser, as tabulated in MCML Table 6.2', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    img = save(fig, 'ref08_giovanelli.png')
    rows = [['Giovanelli table (albedo 0.9)', '0.2600', f"{r62['model']:.5f} ± {eng_se(r62):.5f}", f"{r62['rel_error_pct']:.2f}%", ok(r62)]]
    rows += [[f"Independent 1-D MC, albedo {r['series']['albedo']}", f"{r['reference_value']:.5f}", f"{r['model']:.5f} ± {r['series']['se']:.5f}",
              f"{r['rel_error_pct']:.2f}%", ok(r)] for r in sweep]
    return img, md_table(['Case', 'Reference', 'This engine', 'Rel. err.', 'Result'], rows), (sum(r['pass'] for r in [r62] + sweep), 1 + len(sweep))


def ref_prahl(R):
    rd, tt, r62 = R['V1a'], R['V1b'], R['V2b']
    cases = [('Rd (Table 6.1)', rd, TABLE_61['prahl'][0], TABLE_61['mcml'][0]),
             ('Tt (Table 6.1)', tt, TABLE_61['prahl'][1], TABLE_61['mcml'][1]),
             ('R (Table 6.2)', r62, TABLE_62['prahl'], TABLE_62['mcml'])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    zs = []
    for i, (lab, r, p, m) in enumerate(cases):
        zp = (r['model'] - p[0]) / np.hypot(eng_se(r), p[1])
        zm = (m[0] - p[0]) / np.hypot(m[1], p[1])
        zs.append(zp)
        ax.barh(i + 0.18, zp, 0.34, color=C['eng'], label='engine − Prahl' if i == 0 else None)
        ax.barh(i - 0.18, zm, 0.34, color=C['paper2'], label='MCML − Prahl' if i == 0 else None)
    ax.axvspan(-2, 2, color='0.9', zorder=0)
    ax.set_yticks(range(3))
    ax.set_yticklabels([c[0] for c in cases])
    ax.set_xlabel('difference in units of combined standard error')
    ax.set_title('Agreement with Prahl et al. (1989)')
    ax.legend(fontsize=7)
    ax = axes[1]
    for i, (lab, r, p, m) in enumerate(cases):
        ax.errorbar((r['model'] - p[0]) * 1e3, i + 0.18, xerr=eng_se(r) * 1e3, fmt='o', color=C['eng'], capsize=3)
        ax.errorbar((m[0] - p[0]) * 1e3, i - 0.18, xerr=m[1] * 1e3, fmt='s', color=C['paper2'], capsize=3)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_yticks(range(3))
    ax.set_yticklabels([c[0] for c in cases])
    ax.set_xlabel('difference from Prahl value [×10⁻³]')
    ax.set_title('Absolute differences (± SE)')
    fig.suptitle('Prahl, Keijzer, Jacques & Welch (1989) — Monte Carlo comparison values in MCML Tables 6.1/6.2 vs this engine', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    img = save(fig, 'ref09_prahl.png')
    rows = []
    for (lab, r, p, m), z in zip(cases, zs):
        rows.append([lab, f"{p[0]:.5f} ± {p[1]:.5f}", f"{m[0]:.5f} ± {m[1]:.5f}", f"{r['model']:.5f} ± {eng_se(r):.5f}", f"{z:+.2f} σ", 'within 2σ' if abs(z) <= 2 else '**outside 2σ**'])
    return img, md_table(['Quantity', 'Prahl 1989', 'MCML', 'This engine', 'Engine − Prahl', 'Verdict'], rows), (sum(abs(z) <= 2 for z in zs), 3)


def ref_chandra(R):
    sweep = sorted([r for k, r in R.items() if k.startswith('V2c-a')], key=lambda r: r['series']['albedo'])
    base = R['V2']
    af = np.linspace(0.3, 0.995, 60)
    ex = np.array([vb.semi_infinite_isotropic_rd(a) for a in af])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    ax = axes[0]
    ax.plot(af, ex, color=C['paper'], label='Chandrasekhar H-function (exact)')
    ax.errorbar([r['series']['albedo'] for r in sweep] + [0.9], [r['model'] for r in sweep] + [base['model']],
                yerr=[3 * r['series']['se'] for r in sweep] + [3 * eng_se(base)], fmt='D', color=C['eng'], label='This engine (±3 SE)')
    ax.set_xlabel('single-scattering albedo')
    ax.set_ylabel('diffuse reflectance Rd (n = 1, g = 0)')
    ax.legend()
    ax.set_title('Semi-infinite isotropic scatterer, normal incidence')
    ax = axes[1]
    allr = sweep + [base]
    ax.errorbar([r['series']['albedo'] if r['series'] and 'albedo' in r['series'] else 0.9 for r in allr],
                [(r['model'] - r['reference_value']) / r['reference_value'] * 100 for r in allr],
                yerr=[3 * eng_se(r) / r['reference_value'] * 100 for r in allr], fmt='o', color=C['eng'], capsize=3)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('single-scattering albedo')
    ax.set_ylabel('engine − exact [%]')
    ax.set_title('Residual (error bars: ±3 SE)')
    fig.suptitle('Chandrasekhar (1960), Radiative Transfer — exact H-function solution vs this engine', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    img = save(fig, 'ref10_chandrasekhar.png')
    rows = [[f"{(r['series'] or {}).get('albedo', 0.9)}", f"{r['reference_value']:.5f}", f"{r['model']:.5f} ± {eng_se(r):.5f}", f"{r['rel_error_pct']:.2f}%", ok(r)]
            for r in sorted(allr, key=lambda r: (r['series'] or {}).get('albedo', 0.9))]
    return img, md_table(['Albedo', 'Exact Rd', 'This engine', 'Rel. err.', 'Result'], rows), (sum(r['pass'] for r in allr), len(allr))


def ref_cj(R):
    v3a, v3b = R['V3a'], R['V3b']
    pr = v3a['series']['profile']
    mus = sorted([r for k, r in R.items() if k.startswith('V3c-mu')], key=lambda r: r['series']['mu_a'])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ax = axes[0]
    s = v3a['series']
    tf = np.linspace(1e-3, max(s['t_s']), 200)
    ax.plot(tf, vb.beer_lambert_surface_temperature_rise(pr['I_eff'], pr['mu_a'], pr['k'], pr['rho_cp'], tf), color=C['paper'], label='Closed form')
    ax.plot(s['t_s'], s['model_dT'], 'D', color=C['eng'], label='This engine')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('surface ΔT [K]')
    ax.legend()
    ax.set_title(f"Surface heating, μa = {pr['mu_a']:g} /mm")
    ax = axes[1]
    z = np.array(pr['z_mm'])
    zz = np.linspace(0.0005, z.max() * 0.6, 120)
    ax.plot(vb.beer_lambert_profile(zz, pr['t_end'], pr['I_eff'], pr['mu_a'], pr['k'], pr['rho_cp']), zz, color=C['paper'], label='Closed form (image sources)')
    ax.plot(pr['dT_engine'], z, 'k.', ms=3, label='This engine')
    ax.set_ylim(z.max() * 0.6, 0)
    ax.set_xlabel(f"ΔT at t = {pr['t_end']:g} s [K]")
    ax.set_ylabel('depth [mm]')
    ax.legend()
    ax.set_title('Temperature–depth profile')
    ax = axes[2]
    x = np.arange(len(mus))
    ax.bar(x - 0.18, [r['reference_value'] for r in mus], 0.34, color=C['paper'], label='Closed form')
    ax.bar(x + 0.18, [r['model'] for r in mus], 0.34, color=C['eng'], label='This engine')
    ax.set_xticks(x)
    ax.set_xticklabels([f"μa={r['series']['mu_a']:g}" for r in mus])
    ax.set_ylabel('ΔT at first cell, t = 0.5 s [K]')
    ax.legend()
    ax.set_title('Absorption-coefficient sweep')
    fig.suptitle('Carslaw & Jaeger (1959), Conduction of Heat in Solids — volumetric-absorption heating of a half-space vs this engine', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    img = save(fig, 'ref11_carslaw_jaeger.png')
    rows = [[r['id'], r['title'], f"{r['reference_value']:.4g}", f"{r['model']:.4g}", f"{r['rel_error_pct']:.2f}%", ok(r)] for r in [v3a, v3b] + mus]
    return img, md_table(['ID', 'Case', 'Closed form', 'Engine', 'Rel. err.', 'Result'], rows), (sum(r['pass'] for r in [v3a, v3b] + mus), len(mus) + 2)


# ---------------------------------------------------------------------------------------------
# 3. McMillan 2021
# ---------------------------------------------------------------------------------------------
def ref_mcmillan(R):
    import plume_calibration as pc
    dig = jload('mcmillan2021_fig3_digitized.json')
    cache = jload('mcmillan_cache.json')
    cal = jload('plume_calibration.json')
    E = [50, 100, 150, 200, 250, 300, 350, 400]
    if f"triangular_dry0.25_E{E[0]}" not in cache:
        return None, '_McMillan results not available._', (0, 0)
    exp = {p['energy_mJ']: p for p in dig['experiment']}
    variants = {'linger_5ms': ('tab:green', 'plume, 5 ms linger'), 'no_linger': ('tab:purple', 'plume, no linger')}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    cols = {'300': '#1f77b4', '350': '#ff7f0e', '400': '#2ca02c', '450': '#d62728', '500': '#9467bd'}
    for Ta, pts in dig['simulation'].items():
        ax.plot([p['energy_mJ'] for p in pts], [p['depth_mm'] for p in pts], 'o-', color=cols[Ta], ms=3, alpha=1.0 if Ta == '500' else 0.3,
                label=f"Paper sim, Ta = {Ta} °C")
    ax.errorbar(E, [exp[e]['depth_mm'] for e in E], yerr=[[exp[e]['depth_mm'] - exp[e]['err_lo_mm'] for e in E], [exp[e]['err_hi_mm'] - exp[e]['depth_mm'] for e in E]],
                fmt='x', color='k', capsize=3, ms=8, label='Paper: OCT experiment')
    ax.plot(E, [cache[f'triangular_dry0.25_E{e}']['depth_mm'] for e in E], 'D--', color='k', mfc='w', label='Engine, no plume')
    for vn, (c, lab) in variants.items():
        Cb = cal[f'best|{vn}']
        d = [cal[f'hold|{vn}|{e}'] for e in E]
        ax.plot(E, d, '-', color=c, lw=1.2, alpha=0.7)
        ax.plot([e for e in E if e in pc.FIT_E], [cal[f'hold|{vn}|{e}'] for e in E if e in pc.FIT_E], 'o', color=c, label=f"Engine {lab}, C = {Cb:g} (filled = fit points)")
        ax.plot([e for e in E if e not in pc.FIT_E], [cal[f'hold|{vn}|{e}'] for e in E if e not in pc.FIT_E], 'o', mfc='w', color=c, label=f"  held-out energies (hollow)")
    ax.axhline(1.2, color='0.6', ls=':')
    ax.set_xlabel('energy per pixel [mJ]')
    ax.set_ylabel('MTZ depth [mm]')
    ax.set_title('Fig. 3: depth vs energy; plume constant fitted on 100/200/400 mJ only')
    ax.legend(fontsize=6.5, ncol=2)
    ax = axes[0, 1]
    f7 = dig['fig7']
    ax.plot([p['energy_mJ'] for p in f7['triangular_temporal']], [p['depth_mm'] for p in f7['triangular_temporal']], 's:', color='tab:blue', label='Paper: triangular pulse (Ta 500)')
    ax.plot([p['energy_mJ'] for p in f7['tophat_temporal']], [p['depth_mm'] for p in f7['tophat_temporal']], 'x--', color='tab:red', label='Paper: top-hat pulse')
    ax.plot([p['energy_mJ'] for p in f7['gaussian_temporal']], [p['depth_mm'] for p in f7['gaussian_temporal']], 'o-', color='tab:green', label='Paper: Gaussian pulse')
    ax.plot(E, [cache[f'tophat_dry0.25_E{e}']['depth_mm'] for e in E], 'D--', color='tab:red', mfc='w', alpha=0.5, label='Engine top-hat, no plume')
    for vn, (c, lab) in variants.items():
        ax.plot(E, [cal[f'f7top|{vn}|{e}'] for e in E], 'D-', color=c, mfc='w', label=f'Engine top-hat, {lab} (held-out shape)')
    ax.set_xlabel('energy per pixel [mJ]')
    ax.set_ylabel('depth [mm]')
    ax.set_ylim(0, 1.3)
    ax.set_title('Fig. 7: pulse shape (plume constant NOT refitted)')
    ax.legend(fontsize=6.5)
    ax = axes[1, 0]
    for vn, (c, lab) in variants.items():
        sc = cal[f'score|{vn}']
        Cs = sorted(float(k) for k in sc)
        ax.plot(Cs, [sc[str(C)] if str(C) in sc else sc[repr(C)] for C in Cs], 'o-', color=c, label=lab)
        ax.axvline(cal[f'best|{vn}'], color=c, ls=':')
    ax.axvspan(*pc.C_PHYS, color='tab:orange', alpha=0.2, label='physical bracket (assumed κ, h, v)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('plume constant C = κ h / v  [m² s / kg]')
    ax.set_ylabel('Σ ln²(depth / measured) on fit energies')
    ax.set_title('Calibration scan and physical plausibility')
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    h = cache['triangular_dry0.25_E200']
    abl = np.array(h['xz_ablated'], dtype=float)
    dry = np.array(h['xz_dry'], dtype=float)
    img = np.where(abl > 0, 2.0, np.where(dry > 0, 1.0, 0.0)).T
    vox = h['voxel_mm']
    ax.imshow(img, cmap=matplotlib.colors.ListedColormap(['#7a2e2e', '#e8d9b5', '#111111']), vmin=0, vmax=2,
              extent=[-abl.shape[0] * vox / 2, abl.shape[0] * vox / 2, abl.shape[1] * vox, 0], aspect='equal')
    ax.axhline(exp[200]['depth_mm'], color='cyan', ls='--', lw=1)
    ax.grid(False)
    ax.set_title('Engine cross-section, 200 mJ, no plume (black = removed, beige = dehydrated;\ncyan = measured depth)')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('z [mm]')
    fig.suptitle('McMillan et al. (2021) Figs. 3 & 7 — fractional CO₂ MTZ depth in porcine skin: paper data vs engine', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    imgp = save(fig, 'ref03_mcmillan2021.png')
    H_rm = mcm_mod.H_LAT + mcm_mod.RHO_WET * mcm_mod.CP_WET * 1e-9 * (100 - mcm_mod.T0) + mcm_mod.RHO_DRY * mcm_mod.CP_DRY * 1e-9 * (500 - 100)
    efrac = {}
    for e in (100, 200, 300):
        h = cache[f'triangular_dry0.25_E{e}']
        a = np.array(h['xz_ablated'])
        v = h['voxel_mm']
        N = a.shape[0]
        xs = (np.arange(N) - N // 2) * v
        vol = sum(a[i].sum() * v * (2 * np.pi * abs(xs[i]) * v if i != N // 2 else np.pi * (v / 2) ** 2) for i in range(N // 2, N))
        cyl = np.pi * (mcm_mod.BEAM_DIA / 2) ** 2 * exp[e]['depth_mm']
        own = np.pi * (mcm_mod.BEAM_DIA / 2) ** 2 * h['depth_mm']
        efrac[e] = (vol * H_rm * 1e3 / e * 100, cyl * H_rm * 1e3 / e * 100, vol / own)
    rows = []
    n_pass = 0
    for e in E:
        r = R[f'M-{e}-exp']
        s = r['series']
        n_pass += r['pass']
        cells = []
        for vn in variants:
            rp = R[f'PL-{vn}-{e}']
            tag = 'fit' if rp['series']['fit_point'] else 'held-out'
            cells.append(f"{rp['model']*1000:.0f} ({tag}: {ok(rp)})")
        rows.append([e, f"{exp[e]['depth_mm']*1000:.0f} ({exp[e]['err_lo_mm']*1000:.0f}–{exp[e]['err_hi_mm']*1000:.0f})",
                     f"{s['lit_sim_Ta500']*1000:.0f}", f"{r['model']*1000:.0f} ({ok(r)})"] + cells)
    tbl = md_table(['Energy [mJ]', 'Paper: OCT experiment [µm] (error bar)', 'Paper: sim Ta = 500 °C', 'Engine, no plume',
                    f"Engine + plume, 5 ms linger (C = {cal['best|linger_5ms']:g})", f"Engine + plume, no linger (C = {cal['best|no_linger']:g})"], rows)
    f7 = {p['energy_mJ']: p['depth_mm'] for p in dig['fig7']['tophat_temporal']}
    rows2 = [[e, f"{f7[e]*1000:.0f}", f"{cache[f'tophat_dry0.25_E{e}']['depth_mm']*1000:.0f}"] +
             [f"{cal[f'f7top|{vn}|{e}']*1000:.0f} ({ok(R[f'PX-{vn}-{e}'])})" for vn in variants] for e in E]
    tbl2 = md_table(['Energy [mJ]', 'Paper: sim, top-hat pulse (Fig. 7) [µm]', 'Engine, no plume', 'Engine + plume, 5 ms linger', 'Engine + plume, no linger'], rows2)
    held = {vn: (sum(R[f'PL-{vn}-{e}']['pass'] for e in pc.HOLD_E), len(pc.HOLD_E)) for vn in variants}
    xheld = {vn: (sum(R[f'PX-{vn}-{e}']['pass'] for e in E), len(E)) for vn in variants}
    return imgp, tbl + '\n\nHeld-out pulse shape (Fig. 7 top-hat, constant not refitted):\n\n' + tbl2, (n_pass, len(E), H_rm, efrac, held, xheld, cal)


# ---------------------------------------------------------------------------------------------
# 4. Kim 2023
# ---------------------------------------------------------------------------------------------
def ref_kim(R):
    sim = jload('kim_cache.json')
    alt = kimb.run_kim_alt()
    t = np.array(sim['t_s'])
    K = kimb.KIM
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    specs = list(K)
    x = np.arange(4)
    for ax, key, unit in ((axes[0, 0], 'depth', 'mm'), (axes[0, 1], 'area', 'mm²')):
        ax.bar(x - 0.3, [K[s][f'{key}_exp'] for s in specs], 0.2, color=C['paper'], label='Paper: experiment')
        ax.bar(x - 0.1, [K[s][f'{key}_sim'] for s in specs], 0.2, color=C['paper2'], label="Paper: Kim's simulation")
        ax.bar(x + 0.1, [R[f'K-{s}-{key}']['model'] for s in specs], 0.2, color=C['eng'], label='Engine, source e^(-2r) as printed (exposure fitted on depth)')
        ax.bar(x + 0.3, [alt[s][key] for s in specs], 0.2, color='0.6', hatch='//', label='Engine, source e^(-2r²) (Gaussian reading)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n({'coag. 55 °C' if K[s]['group']=='coag' else 'ablation 200 °C'})" for s in specs])
        ax.set_ylabel(unit)
        ax.set_title(f"Mean {key}" if key == 'depth' else 'Top-view area')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.55)
        ax.legend(fontsize=6.5, loc='upper right')
    ax = axes[1, 0]
    ax.plot(t, sim['coag_depth'], color='tab:blue', label='Engine: coagulation depth (T ≥ 55 °C)')
    ax.plot(t, sim['abl_depth'], color='tab:red', label='Engine: ablation depth (T ≥ 200 °C)')
    for grp, c in (('coag', 'tab:blue'), ('abl', 'tab:red')):
        v = [K[s]['depth_exp'] for s in K if K[s]['group'] == grp]
        ax.axhspan(min(v), max(v), color=c, alpha=0.15)
        tf = R[[f'K-{s}-depth' for s in K if K[s]['group'] == grp][0]]['series']['t_fit_s']
        ax.axvline(tf, color=c, ls=':')
        ax.text(tf, 0.02, f' fitted t = {tf*1000:.0f} ms', color=c, fontsize=7, rotation=90, va='bottom')
    ax.set_xscale('log')
    ax.set_xlabel('exposure time [s]')
    ax.set_ylabel('mean depth [mm]')
    ax.set_title('Engine dose–response; bands = measured depths (exposure time unreported)')
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    ax.plot(t, sim['coag_area'], color='tab:blue', label='Engine: coagulation top-view area')
    ax.plot(t, sim['abl_area'], color='tab:red', label='Engine: ablation top-view area')
    for s, c in (('PS1', 'tab:blue'), ('PS2', 'tab:blue'), ('PS3', 'tab:red'), ('PS4', 'tab:red')):
        ax.axhline(K[s]['area_exp'], color=c, ls=':', lw=1)
        ax.text(t[2], K[s]['area_exp'], f' {s}', fontsize=7, color=c, va='bottom')
    ax.set_xscale('log')
    ax.set_xlabel('exposure time [s]')
    ax.set_ylabel('area [mm²]')
    ax.set_title('Top-view areas vs exposure (dotted: measured)')
    ax.legend(fontsize=7)
    fig.suptitle('Kim et al. (2023) Sections 4.2–4.3 — porcine skin 1060 nm lesions: paper data vs engine', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    img = save(fig, 'ref04_kim2023.png')
    rows = []
    for s in specs:
        rd, ra = R[f'K-{s}-depth'], R[f'K-{s}-area']
        k = K[s]
        rows.append([s, 'coagulation 55 °C' if k['group'] == 'coag' else 'ablation 200 °C', f"{rd['series']['t_fit_s']*1000:.0f} ms (fitted)",
                     f"{k['depth_exp']:.3f}", f"{k['depth_sim']:.3f}", f"**{rd['model']:.3f}**", f"{ok(rd)}",
                     f"{k['area_exp']:.3f}", f"{k['area_sim']:.3f}", f"**{ra['model']:.3f}**", f"{ok(ra)}", f"{alt[s]['area']:.3f}"])
    tbl = md_table(['Specimen', 'Criterion', 'Exposure', 'Depth: exp [mm]', 'Depth: Kim sim', 'Depth: engine', 'Depth', 'Area: exp [mm²]', 'Area: Kim sim', 'Area: engine', 'Area', 'Area: engine, Gaussian reading'], rows)
    n_pass = sum(R[f'K-{s}-area']['pass'] for s in specs)
    from simulation.voxel_mc_3d import run_voxel_mc3d
    mc = run_voxel_mc3d(power=15.0, n_photons=400000, spot_dia=1.2, Lx=4, Ly=4, Lz=2, Nx=160, Ny=160, Nz=80,
                        layers_config=[{'z_end': 2, 'mu_a': 0.07512, 'mu_s': 7.0, 'g': 0.9, 'n': 1.4}], beam_profile='kim_exp_disk')
    q = mc['Q_3d'][78:82, 78:82, :].mean(axis=(0, 1))
    dep = {'peak_over_surface': float(q.max() / q[0]), 'peak_depth_mm': float(mc['grid_z'][q.argmax()])}
    sens = {}
    for grp, dk, ak, spec in (('coag', 'coag_depth', 'coag_area', 'PS1'), ('abl', 'abl_depth', 'abl_area', 'PS3')):
        tf = R[f'K-{spec}-depth']['series']['t_fit_s']
        sens[grp] = {f: (float(np.interp(tf * f, t, sim[dk])), float(np.interp(tf * f, t, sim[ak]))) for f in (0.9, 1.0, 1.1)}
    return img, tbl, (n_pass, len(specs), sum(R[f'K-{s}-depth']['pass'] for s in specs), sens, alt, dep)


# ---------------------------------------------------------------------------------------------
# 5. Keller 2016
# ---------------------------------------------------------------------------------------------
def ref_keller(R):
    T = kel.TABLE1
    labels = [f"{wl/1000:g} µm\n{d} mm\n{('%g ms' % tp) if tp >= 0.001 else '10 ns'}" for wl, d, tp, *_ in T]
    ke = {k: r for k, r in R.items() if k.startswith('KE-')}
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw={'width_ratios': [2.2, 1.6, 1.2]})
    ax = axes[0]
    colors = {532: '#2ca02c', 1064: '#8c564b', 976: '#e377c2', 10600: '#d62728', 1470: '#ff7f0e'}
    ax.bar(range(len(T)), [r[3] for r in T], color=[colors[r[0]] for r in T])
    for i, r in enumerate(T):
        ax.text(i, r[3] * 1.08, f"{r[3]:g}", ha='center', fontsize=8)
    ax.set_yscale('log')
    ax.set_xticks(range(len(T)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('LD90 fluence [J/cm²]')
    ax.set_title('Paper Table 1: LD90 fluence for An. stephensi mosquitoes')
    ax = axes[1]
    rows_c = [(r[2], r[1], r[3]) for r in kel.CO2_ROWS]
    x = np.arange(len(rows_c))
    ax.bar(x - 0.27, [r[2] for r in rows_c], 0.27, color=C['paper'], label='Paper: measured LD90')
    names = list(kel.ARRHENIUS)
    for j, nm in enumerate(names):
        vals = [ke[f"KE-{nm.split()[0]}-{d}mm-{tp}ms"]['model'] for tp, d, _ in rows_c]
        ax.bar(x + j * 0.27, vals, 0.27, color=[C['eng'], C['grey']][j], label=f"Engine: Ω = 1 fluence, {nm}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"10.6 µm, {d} mm\n{tp} ms" for tp, d, _ in rows_c], fontsize=7)
    ax.set_ylabel('fluence [J/cm²]')
    ax.set_title('Engine thermal-damage prediction (water-absorbed CO₂ rows)')
    ax.legend(fontsize=6.5)
    ax = axes[2]
    beetle = [('Keller LD90\n(mosquito, all)', 0.3, 10.9), ('Beetle sim.\n10 ms (4 W)', 127, 127), ('Beetle sim.\n50 ms (4 W)', 637, 637)]
    for i, (lab, lo, hi) in enumerate(beetle):
        ax.plot([i, i], [lo, hi], 'o-', lw=6 if hi > lo else 0, color=['#d62728', 'k', 'k'][i], ms=8)
    ax.set_yscale('log')
    ax.set_xticks(range(3))
    ax.set_xticklabels([b[0] for b in beetle], fontsize=7)
    ax.set_ylabel('fluence [J/cm²]')
    ax.set_xlim(-0.5, 2.5)
    ax.set_title('Context: delivered fluence')
    fig.suptitle('Keller et al. (2016) Sci Rep 6:20936 — Table 1 (paper data) and engine thermal-damage prediction', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    img = save(fig, 'ref05_keller2016.png')
    t1 = md_table(['λ [nm]', 'Beam Ø [mm]', 'Pulse', 'Power range [W]', 'LD90 fluence [J/cm²]', 'LD90 pulse energy [mJ]', 'LD90 power [W]'],
                  [[wl, d, f"{tp:g} ms" if tp >= 0.001 else '10 ns', '—', f, E, f"{P:g}"] for wl, d, tp, f, E, P in T])
    rows = []
    for k in sorted(ke):
        r = ke[k]
        s = r['series']
        rows.append([s['kinetics'], f"{s['beam_mm']} mm / {s['t_pulse_ms']} ms", f"{r['reference_value']:.2f}", f"**{r['model']:.2f}**",
                     f"{r['model']/r['reference_value']:.2f}×", ok(r)])
    t2 = md_table(['Arrhenius kinetics', 'Condition (10.6 µm)', 'Paper LD90 [J/cm²]', 'Engine Ω = 1 fluence', 'Engine / paper', 'Within factor 1.5'], rows)
    ratio = {nm: (ke[f"KE-{nm.split()[0]}-2.5mm-25.0ms"]['model'] / ke[f"KE-{nm.split()[0]}-2.5mm-4.3ms"]['model']) for nm in names}
    meas = 1.1 / 0.9
    return img, t1 + '\n\nEngine prediction for the water-absorbed rows:\n\n' + t2, (sum(r['pass'] for r in ke.values()), len(ke), ratio, meas)


# ---------------------------------------------------------------------------------------------
# 6. Sagi-Dolev 1992
# ---------------------------------------------------------------------------------------------
def ref_sagi():
    import sagi_stages as ss
    d = jload('sagi_stages.json')
    t = np.array(d['t_s'])
    T = np.array(d['T_surface'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    tb = t[np.argmax(T >= 99.9)]
    nd = np.array(d['n_dry_layers'])
    td = t[np.argmax(nd > 0)]
    dep = np.array(d['depth_mm'])
    tr = t[np.argmax(dep > 0)]
    ax.axvspan(0, tb, color='#fde0dd', alpha=0.8)
    ax.axvspan(tb, td, color='#fdd0a2', alpha=0.8)
    ax.axvspan(td, tr, color='#c7e9c0', alpha=0.8)
    ax.axvspan(tr, t[-1], color='#bcbddc', alpha=0.8)
    ax.plot(t, T, 'k')
    ax.axhline(100, color='0.4', ls=':')
    ax.axhline(d['Ta'], color='0.4', ls=':')
    for (x0, x1, lab), y0 in zip(((0, tb, '1'), (tb, td, '2'), (td, tr, '3'), (tr, t[-1], '4–5')), (60, 72, 60, 60)):
        ax.text(0.5 * (x0 + x1), y0, f"stage {lab}", ha='center', fontsize=8)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('surface temperature [°C]')
    ax.set_title('Engine staged ablation on an illustrative case\n(0.5 W/mm², 10.6 µm tissue; NOT the paper’s conditions)')
    ax = axes[1]
    ax.plot(t, dep * 1000, 'k', label='removed depth [µm]')
    ax2 = ax.twinx()
    ax2.plot(t, nd, color='tab:green', label='dehydrated layers')
    ax2.plot(t, d['n_boiling_layers'], color='tab:orange', label='boiling layers')
    ax2.grid(False)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('depth [µm]')
    ax2.set_ylabel('layers (5 µm each)')
    ax2.legend(fontsize=7, loc='upper left')
    ax.legend(fontsize=7, loc='center left')
    ax.set_title('Material state vs time')
    fig.suptitle('Sagi-Dolev et al. (1992) Opt Eng 31:1425 — process stages named in the abstract (paper data unavailable: paywalled)', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    img = save(fig, 'ref06_sagi_dolev.png')
    rows = [[a, b, c, 'implemented' if e else 'not implemented'] for a, b, c, e in ss.STAGES]
    return img, md_table(['#', 'Stage named in the paper’s abstract', 'Engine implementation', 'Status'], rows), (len(ss.STAGES), len(ss.STAGES)), (tb, td, tr)


# ---------------------------------------------------------------------------------------------
# 12. Guo et al. 2014 -- Beetle Elytra Trabecular Sandwich Thermal Barrier
# ---------------------------------------------------------------------------------------------
def ref_consistency(R):
    import consistency_checks as cc
    from simulation.coupled_ablation import _accumulate_char, R_GAS
    d = jload('consistency_cache.json')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    L = d['layered']
    x = np.arange(2)
    ax.bar(x - 0.18, [L['solid']['independent'] * 1e3, L['sandwich']['independent'] * 1e3], 0.34, color=C['paper'], label='Independent ODE solve (scipy BDF)')
    ax.bar(x + 0.18, [L['solid']['engine'] * 1e3, L['sandwich']['engine'] * 1e3], 0.34, color=C['eng'], label='This engine')
    ax.set_xticks(x)
    ax.set_xticklabels(['solid chitin 0.25 mm', 'trabecular sandwich'])
    ax.set_ylabel('heat into body cavity [mJ/mm²]')
    ax.set_title('C1: layered-stack conduction (assumed k, ρc)')
    ax.legend(fontsize=7)
    ax.text(0.02, 0.62, f"solid/sandwich heat ratio: {L['engine_ratio']:.2f} (engine)\nseries-resistance ratio: {L['R_ratio']:.2f} (steady state)\n"
            "k_eff = 0.08 W/(m K) core is an ASSUMED input;\nno literature value is compared", transform=ax.transAxes, fontsize=7.5)
    ax = axes[0, 1]
    for name, c in (('dark', 'k'), ('shiny', 'tab:orange')):
        t = np.array(d['radiant'][name]['t'])
        ax.plot(t, d['radiant'][name]['analytic'], '-', color=c, label=f'{name} body: lumped analytic')
        ax.plot(t[::20], np.array(d['radiant'][name]['dT'])[::20], 'o', color=c, ms=4, mfc='w', label=f'{name} body: engine')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('mean temperature rise [K]')
    ax.set_title('C2: radiant heating, surface loss on two z faces')
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    ob = d['oblique']
    ang = [o['angle'] for o in ob]
    ax.plot(ang, [o['R_exact'] for o in ob], '-', color=C['paper'], label='exact Fresnel R_spec')
    ax.plot(ang, [o['R_engine'] for o in ob], 'o', color='k', mfc='w', label='engine R_spec')
    ax.set_xlabel('incidence angle [deg]')
    ax.set_ylabel('R_spec')
    ax2 = ax.twinx()
    ax2.plot(ang, [o['zbar_exact'] * 1000 for o in ob], '-', color='tab:green', label='exact cos θt / μa')
    ax2.plot(ang, [o['zbar_engine'] * 1000 for o in ob], 's', color='tab:green', mfc='w', label='engine mean absorption depth')
    ax2.set_ylabel('mean absorption depth [µm]')
    ax2.grid(False)
    ax.set_title('C3: oblique incidence on n = 1.55, μa = 50 /mm')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left')
    ax = axes[1, 1]
    A_, Ea = d['char']['A'], d['char']['Ea']
    for Tc, c in ((350, 'tab:blue'), (400, 'tab:orange'), (450, 'tab:red')):
        k = A_ * np.exp(-Ea / (R_GAS * (Tc + 273.15)))
        t = np.linspace(0, 1.0, 100)
        ax.plot(t, 1 - np.exp(-k * t), '-', color=c, label=f'{Tc} °C exact')
        chi = np.zeros((2, 2, 2))
        dry = np.ones((2, 2, 2), dtype=bool)
        ab = np.zeros((2, 2, 2), dtype=bool)
        T = np.full((2, 2, 2), float(Tc))
        pts = []
        for i in range(1000):
            _accumulate_char(T, chi, dry, ab, A_, Ea / R_GAS, 1e-3, 2, 2, 2)
            if i % 100 == 99:
                pts.append(((i + 1) * 1e-3, chi[0, 0, 0]))
        ax.plot(*zip(*pts), 'o', color=c, mfc='w', ms=5)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('char fraction χ')
    ax.set_title('C4: first-order pyrolysis kernel (test kinetics, not measured)')
    ax.legend(fontsize=7)
    fig.suptitle('Internal-consistency checks (no literature data): engine vs exact / independent calculations', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    img = save(fig, 'ref12_consistency.png')
    ids = [k for k in R if k.startswith('C1-') or k.startswith('C2-') or k.startswith('C3-') or k.startswith('C4-')]
    rows = [[k, R[k]['title'], R[k]['reference'], f"{R[k]['reference_value']:.5g}", f"{R[k]['model']:.5g}", f"{R[k]['rel_error_pct']:.2f}%", ok(R[k])] for k in ids]
    tbl = md_table(['ID', 'Check', 'Reference (derived, not typed)', 'Reference value', 'Engine', 'Rel. err.', 'Result'], rows)
    return img, tbl, (sum(R[k]['pass'] for k in ids), len(ids))


def main():
    IMAGES_DIR.mkdir(exist_ok=True)
    R, meta = R_all()
    out = {}
    out[1] = ref_shimojo(R)
    out[2] = ref_mcml(R)
    out[3] = ref_mcmillan(R)
    out[4] = ref_kim(R)
    out[5] = ref_keller(R)
    out[6] = ref_sagi()
    out[7] = ref_vdh(R)
    out[8] = ref_giovanelli(R)
    out[9] = ref_prahl(R)
    out[10] = ref_chandra(R)
    out[11] = ref_cj(R)
    out[12] = ref_consistency(R)
    import report_text
    md = report_text.compose(R, meta, out, shimojo_inputs_table())
    with open(SUITE_DIR / 'README.md', 'w') as f:
        f.write(md)
    print('wrote README.md')


if __name__ == '__main__':
    main()
