"""
Benchmark suite master runner.

    ./venv/bin/python knowledge/simulation_lit_comparison/code/run_literature_benchmarks.py [--quick]

Runs the verification tier (analytic / canonical references) and the validation tier
(published experiments), writes raw_output/benchmark_results.json and
images/benchmark_scorecard.png. Every model number is produced by the production engine;
nothing is calibrated to the reference data, clipped, or substituted.
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import bench_common
from bench_common import IMAGES_DIR, RAW_DIR
import verification_benchmarks as vb

try:
    import validation_benchmarks as valb
except ImportError:  # validation tier is optional
    valb = None
import mcmillan_benchmark as mcm
import kim_benchmark as kim
import keller_benchmark as kel
import sagi_stages
import consistency_checks as cc
import plume_calibration as pl


def _series(results, bench_id):
    for r in results:
        if r['id'] == bench_id:
            return r
    return None


def plot_scorecard(results, out_path):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)

    # A: optical verification, model vs reference with tolerance
    ax = fig.add_subplot(gs[0, 0])
    opt = [r for r in results if r['id'].startswith(('V1', 'V2', 'C3-R'))]
    y = np.arange(len(opt))
    rel = [(r['model'] - r['reference_value']) / r['reference_value'] * 100 for r in opt]
    tol = [r['tolerance_abs'] / abs(r['reference_value']) * 100 for r in opt]
    ax.errorbar(rel, y, xerr=tol, fmt='o', color='tab:blue', ecolor='tab:gray', capsize=4,
                label='model − reference (bar = tolerance)')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['id']}: {r['title']}" for r in opt], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Relative deviation from exact reference [%]')
    ax.set_title('A. Photon transport vs exact / tabulated references')
    ax.legend(fontsize=8, loc='upper right')

    # B: transient heating
    ax = fig.add_subplot(gs[0, 1])
    r = _series(results, 'V3a')
    if r:
        s = r['series']
        t_f = np.linspace(1e-4, max(s['t_s']), 200)
        I_eff = float(r['notes'].split('I_delivered = ')[1].split(' ')[0])
        ax.plot(t_f, vb.beer_lambert_surface_temperature_rise(I_eff, 10.0, 0.53e-3, 4.18e-3, t_f),
                'k-', label='Closed-form solution')
        ax.plot(s['t_s'], s['model_dT'], 'o', color='tab:orange', label='Engine (MC + explicit FV)')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Surface temperature rise [K]')
        ax.set_title('B. V3: Beer–Lambert heating, insulated surface')
        ax.legend(fontsize=8)

    # C: ablation front
    ax = fig.add_subplot(gs[0, 2])
    colors = {'V4-10ms': 'tab:red', 'V4-2ms': 'tab:green'}
    ref_drawn = False
    for bid, c in colors.items():
        r = _series(results, bid)
        if not r:
            continue
        s = r['series']
        ax.plot(s['t_s'], np.array(s['depth_mm']) * 1000, color=c, lw=1.8,
                label=f"Engine, optical update {bid.split('-')[1]} (v = {r['model']*1000:.1f} µm/s)")
        ref_um = np.array([np.nan if v is None else v * 1000 for v in s['ref_depth_mm']])
        ax.plot(s['t_s'], ref_um, '--', color=c, lw=1.0, alpha=0.8,
                label=f"Energy-balance slope (v = {r['reference_value']*1000:.1f} µm/s)")
        ax.axvspan(*s['fit_window_s'], color='0.9', zorder=0)
        ref_drawn = True
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Ablation depth [µm]')
    ax.set_title('C. V4: steady ablation-front velocity (shaded = fit window)')
    ax.legend(fontsize=8)

    # D: validation / cross-code summary
    ax = fig.add_subplot(gs[1, 0:2])
    ax.axis('off')
    lines = []
    for tier, label in (('validation', 'vs published experiments & insect physics (Shimojo, McMillan, Kim, Keller, Guo, Ospina)'),
                        ('cross-code', "vs the authors' own simulations (Shimojo MCX, McMillan MCRT)"),
                        ('calibrated', 'depth fits with one free exposure time (Kim)')):
        rs = [r for r in results if r['tier'] == tier]
        if rs:
            lines.append(f"{label}:  {sum(r['pass'] for r in rs)}/{len(rs)} within tolerance")
    if lines:
        lines.append("Per-reference figures and tables: README.md")
    else:
        lines.append("No validation-tier benchmarks run")
    ax.text(0.02, 0.85, "D. Validation summary", fontsize=12, fontweight='bold', transform=ax.transAxes)
    for i, line in enumerate(lines):
        ax.text(0.02, 0.68 - 0.13 * i, line, fontsize=11, transform=ax.transAxes)

    # E: scorecard table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    ver = [r for r in results if r['tier'] == 'verification']
    rows = [[r['id'], f"{r['model']:.4g}", f"{r['reference_value']:.4g}",
             f"{r['rel_error_pct']:.2f}%", 'PASS' if r['pass'] else 'FAIL'] for r in ver]
    tbl = ax.table(cellText=rows, colLabels=['ID', 'Model', 'Reference', 'Rel. err', 'Result'],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.25)
    for (ri, ci), cell in tbl.get_celld().items():
        if ri > 0 and ci == 4:
            cell.set_facecolor('#c8f7c5' if rows[ri - 1][4] == 'PASS' else '#f7c5c5')
    ax.set_title(f"E. Verification scorecard: {sum(r['pass'] for r in ver)}/{len(ver)} pass")

    fig.suptitle('Voxel MC + bioheat + enthalpy-ablation engine: verification & validation benchmarks',
                 fontsize=14, fontweight='bold')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_shimojo(results, out_path):
    val = {r['id'][2:]: r for r in results if r['tier'] == 'validation'}
    xc = {r['id'][2:]: r for r in results if r['tier'] == 'cross-code'}
    if not val:
        return
    panels = [('50', 'vap_depth', 'Vaporization depth, 50 W'), ('50', 'vap_width', 'Vaporization width, 50 W'),
              ('50', 'coag_depth', 'Coagulation depth below crater, 50 W'), ('50', 'coag_width', 'Coagulation width at crater floor, 50 W'),
              ('10', 'coag_depth', 'Coagulation depth, 10 W'), ('10', 'coag_width', 'Coagulation width, 10 W')]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.ravel()
    for ax, (pw, key, title) in zip(axes, panels):
        rs = sorted([r for k, r in val.items() if k.startswith(f"{pw}W-") and k.endswith(key)],
                    key=lambda r: r['series']['speed_mm_s'])
        if not rs:
            ax.axis('off')
            continue
        v = [r['series']['speed_mm_s'] for r in rs]
        ax.errorbar(v, [r['reference_value'] for r in rs], yerr=[r['series']['exp_sd'] for r in rs],
                    fmt='x-', color='purple', capsize=4, label='Experiment (±1 SD, n=3)')
        ax.plot(v, [r['series']['lit_sim_dop'] for r in rs], 'o--', color='tab:blue', mfc='none', label='Shimojo sim, with DOP')
        ax.plot(v, [r['series']['lit_sim_nodop'] for r in rs], '^--', color='tab:orange', mfc='none', label='Shimojo sim, without DOP')
        ax.plot(v, [r['model'] for r in rs], 'D-', color='k', label='This engine (with DOP)')
        for r in rs:
            ok_e = 'E✓' if r['pass'] else 'E✗'
            ok_x = 'S✓' if xc[r['id'][2:]]['pass'] else 'S✗'
            ax.annotate(f"{ok_e} {ok_x}", (r['series']['speed_mm_s'], r['model']), textcoords='offset points',
                        xytext=(4, 6), fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Scan speed [mm/s]')
        ax.set_ylabel('[mm]')
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7, loc='upper right')
    sec_path = RAW_DIR / 'shimojo_cross_sections.json'
    if sec_path.exists():
        with open(sec_path) as f:
            secs = json.load(f)
        for ax, key in zip(axes[6:8], ('50W_0.5', '10W_1.0')):
            if key not in secs:
                ax.axis('off')
                continue
            abl = np.array(secs[key]['ablated'], dtype=float)
            coag = np.array(secs[key]['coag'], dtype=float)
            img = np.where(abl > 0, 2.0, np.where(coag > 0, 1.0, 0.0)).T
            Lx_s, Lz_s = secs[key]['Lx'], secs[key]['Lz']
            ax.imshow(img, cmap=matplotlib.colors.ListedColormap(['#7a2e2e', '#e8d9b5', '#111111']),
                      extent=[-Lx_s / 2, Lx_s / 2, Lz_s, 0], vmin=0, vmax=2, aspect='equal')
            ax.set_title(f"Engine cross-section {key.replace('W_', ' W, ')} mm/s\n(black = vaporized, beige = Ω≥1, red = native)", fontsize=9)
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('z [mm]')
    fig.suptitle('Shimojo et al. (2023) Sci Rep 13:11898 replication — 980 nm scanned Gaussian beam on ex vivo porcine liver\n'
                 'E = within 2 SD + 1 voxel of experiment; S = within 2 voxels of Shimojo DOP simulation',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-validation', action='store_true')
    parser.add_argument('--rerun-validation', action='store_true',
                        help='ignore raw_output/shimojo_case_cache.json and recompute the scans')
    args = parser.parse_args()

    IMAGES_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    results = vb.run_all_verification()
    results += cc.run_consistency(use_cache=not args.rerun_validation)[0]
    if valb is not None and not args.skip_validation:
        results += valb.run_all_validation(use_cache=not args.rerun_validation)
        results += mcm.run_mcmillan(use_cache=not args.rerun_validation)
        results += kim.run_kim(use_cache=not args.rerun_validation)[0]
        results += kel.run_keller(use_cache=not args.rerun_validation)
        results += pl.run_plume(use_cache=not args.rerun_validation)[0]
        if args.rerun_validation or not (RAW_DIR / 'sagi_stages.json').exists():
            sagi_stages.run_sagi_stages()
    elapsed = time.time() - t0

    for r in results:
        print(f"{r['id']:12s} {'PASS' if r['pass'] else 'FAIL'}  model={r['model']:.5g}  "
              f"ref={r['reference_value']:.5g} {r['units']}  rel_err={r['rel_error_pct']:.2f}%  "
              f"tol=±{r['tolerance_abs']:.3g} ({r['tolerance_basis']})")
    n_pass = sum(r['pass'] for r in results)
    print(f"\n{n_pass}/{len(results)} benchmarks pass  ({elapsed:.1f} s)")

    out_json = RAW_DIR / 'benchmark_results.json'
    with open(out_json, 'w') as f:
        json.dump({'generated': datetime.now().isoformat(timespec='seconds'),
                   'elapsed_s': round(elapsed, 1), 'n_pass': n_pass, 'n_total': len(results),
                   'results': results}, f, indent=2)
    out_png = IMAGES_DIR / 'benchmark_scorecard.png'
    plot_scorecard(results, out_png)
    print(f"[saved] {out_json}\n[saved] {out_png}")


if __name__ == '__main__':
    main()
