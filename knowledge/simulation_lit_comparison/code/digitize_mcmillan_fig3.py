"""
Digitise McMillan et al. (2021) Lasers Surg Med 53(5):731-740:
  Fig. 3 - crater depth vs pulse energy: OCT experiment with error bars, and MCRT simulations
           (top-hat beam, triangular pulse) for ablation temperatures Ta = 300..500 C;
  Fig. 7 - Ta = 500 C, Gaussian / top-hat / triangular temporal pulse profiles.

The figure is a vector matplotlib plot, so marker centres and error-bar ends are read exactly
from the PDF drawing commands (no image processing). Axes are calibrated from the tick marks.
Requires PyMuPDF. Output: ../raw_output/mcmillan2021_fig3_digitized.json (keys 'experiment', 'simulation', 'fig7')
"""
import json
from pathlib import Path
import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
PDF = HERE.parent / 'references' / 'McMillan2021_LSM_predictive_MCRT_fractional_CO2.pdf'
SERIES_COLORS = {300: (0.12157, 0.46667, 0.70588), 350: (1.0, 0.49804, 0.0549),
                 400: (0.17255, 0.62745, 0.17255), 450: (0.83922, 0.15294, 0.15686),
                 500: (0.58039, 0.40392, 0.74118)}


def digitize_fig7():
    page = pymupdf.open(str(PDF))[6]
    dr = [x for x in page.get_drawings() if x['rect'].y0 > 480]
    blk = [(x['items'][0][1], x['items'][0][2]) for x in dr if x.get('color') == (0.0, 0.0, 0.0)
           and len(x['items']) == 1 and x['items'][0][0] == 'l']
    xt = sorted({round(a.x, 1) for a, b in blk if abs(a.x - b.x) < 1e-6 and abs(a.y - 680.8) < 0.1 and abs(b.y - 683.3) < 0.1})
    yt = sorted({round(a.y, 1) for a, b in blk if abs(a.y - b.y) < 1e-6 and abs(a.x - 157.0) < 0.1 and abs(b.x - 154.6) < 0.1}, reverse=True)
    assert len(xt) == 8 and len(yt) == 9, (xt, yt)
    kx = np.polyfit(xt, np.arange(50.0, 401.0, 50.0), 1)
    ky = np.polyfit(yt, np.arange(0.05, 0.451, 0.05), 1)
    ex = lambda x: int(round(float(np.polyval(kx, x)) / 50.0) * 50)
    dy = lambda y: round(float(np.polyval(ky, y)), 4)
    red = lambda c: c is not None and c[0] > 0.9 and c[1] < 0.1 and c[2] < 0.1
    out = {'gaussian_temporal': [], 'triangular_temporal': [], 'tophat_temporal': [],
           'note': 'Ta = 500 C; markers: circle = Gaussian, square = triangular, x = top-hat temporal profile'}
    for x in dr:
        r = x['rect']
        if x.get('fill') and red(x['fill']) and len(x['items']) == 8:
            out['gaussian_temporal'].append((ex((r.x0 + r.x1) / 2), dy((r.y0 + r.y1) / 2)))
        elif x.get('fill') and red(x['fill']) and x['items'][0][0] == 're' and not x.get('color'):
            out['triangular_temporal'].append((ex((r.x0 + r.x1) / 2), dy((r.y0 + r.y1) / 2)))
    xs = [(x['items'][0][1], x['items'][0][2]) for x in dr if red(x.get('color')) and len(x['items']) == 1 and x['items'][0][0] == 'l']
    cen = {}
    for a, b in xs:
        cen.setdefault(round((a.x + b.x) / 2, 0), []).append((a.y + b.y) / 2)
    for cx, ys in cen.items():
        out['tophat_temporal'].append((ex(cx), dy(float(np.mean(ys)))))
    for k in ('gaussian_temporal', 'triangular_temporal', 'tophat_temporal'):
        out[k] = [{'energy_mJ': e, 'depth_mm': d} for e, d in sorted(set(out[k]))]
    return out


def main():
    page = pymupdf.open(str(PDF))[4]
    dr = page.get_drawings()
    black = [x for x in dr if x.get('color') == (0.0, 0.0, 0.0) and len(x['items']) == 1 and x['items'][0][0] == 'l']
    segs = [(x['items'][0][1], x['items'][0][2]) for x in black]
    xt = sorted({round(a.x, 1) for a, b in segs if abs(a.x - b.x) < 1e-6 and abs(a.y - 265.8) < 0.1 and abs(b.y - 268.6) < 0.1})
    yt = sorted({round(a.y, 1) for a, b in segs if abs(a.y - b.y) < 1e-6 and abs(a.x - 136.6) < 0.1 and abs(b.x - 133.8) < 0.1}, reverse=True)
    assert len(xt) == 8 and len(yt) == 8, (xt, yt)
    kx = np.polyfit(xt, np.arange(50.0, 401.0, 50.0), 1)       # x pt -> energy [mJ]
    ky = np.polyfit(yt, np.arange(0.1, 0.81, 0.1), 1)          # y pt -> depth [mm]
    ex = lambda x: float(np.polyval(kx, x))
    dy = lambda y: float(np.polyval(ky, y))

    out = {'source': 'McMillan et al. 2021, Fig. 3', 'units': {'energy': 'mJ/pixel', 'depth': 'mm'},
           'calibration': {'x_ticks_pt': xt, 'y_ticks_pt': yt}}
    # experiment: vertical error bars (width 1.21) and X-marker diagonals
    bars = [(a, b) for a, b in segs if abs(a.x - b.x) < 1e-6 and abs(black[0]['width'] - 0.65) < 1
            and 150 < a.y < 245 and abs((a.y - b.y)) > 1 and a.x > 140]
    exp = []
    for a, b in segs:
        if abs(a.x - b.x) < 1e-6 and a.x > 140 and 70 < min(a.y, b.y) and max(a.y, b.y) < 262 and abs(a.y - b.y) > 4:
            exp.append((a.x, min(a.y, b.y), max(a.y, b.y)))
    # drop the legend sample (marker + error bar at the top-left, y < 165 pt)
    exp = [(x, y0, y1) for x, y0, y1 in exp if not (x < 160 and y1 < 165)]
    diag = []
    for a, b in segs:
        if abs(abs(a.x - b.x) - 4.85) < 0.2 and abs(abs(a.y - b.y) - 4.85) < 0.2 and a.y > 165:
            diag.append(((a.x + b.x) / 2, (a.y + b.y) / 2))
    pts = []
    for x, y0, y1 in sorted(set(exp)):
        near = [cy for cx, cy in diag if abs(cx - x) < 0.5]
        cy = float(np.mean(near)) if near else 0.5 * (y0 + y1)
        pts.append({'energy_mJ': round(ex(x)), 'depth_mm': round(dy(cy), 4),
                    'err_lo_mm': round(dy(y1), 4), 'err_hi_mm': round(dy(y0), 4)})
    out['experiment'] = pts
    # simulations
    out['simulation'] = {}
    for Ta, col in SERIES_COLORS.items():
        sel = [x for x in dr if x.get('fill') and all(abs(c - d) < 0.01 for c, d in zip(x['fill'], col)) and len(x['items']) == 8]
        P = []
        for x in sel:
            r = x['rect']
            e = ex((r.x0 + r.x1) / 2)
            if abs(e / 25.0 - round(e / 25.0)) < 0.02 and 49 < e < 401:
                P.append((round(e), dy((r.y0 + r.y1) / 2)))
        P = sorted(set(P))
        # the legend sample reuses the 50 mJ x position at a much larger depth; keep the data point
        lo50 = min((d for e, d in P if e == 50), default=None)
        P = [(e, d) for e, d in P if not (e == 50 and d != lo50)]
        out['simulation'][str(Ta)] = [{'energy_mJ': e, 'depth_mm': round(d, 4)} for e, d in P]
    out['fig7'] = digitize_fig7()
    with open(HERE.parent / 'raw_output' / 'mcmillan2021_fig3_digitized.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('experiment:')
    for p in out['experiment']:
        print('  %4d mJ  %.3f mm  (%.3f..%.3f)' % (p['energy_mJ'], p['depth_mm'], p['err_lo_mm'], p['err_hi_mm']))
    for Ta, v in out['simulation'].items():
        print(Ta, len(v), [round(p['depth_mm'], 3) for p in v][:15])


if __name__ == '__main__':
    main()
