"""
Analysis Engine for Laser-Beetle Interaction Suite:
Computes authoritative quantitative metrics across simulation runs:
- Total penetration depth (final in um)
- Average penetration depth during pulse (um)
- Speed of beam penetration (mm/s = final_depth_mm / pulse_duration_s)
- Average hole diameter (mm, across ablated tissue slices)
- Spectrophotometric balance: Total Reflection (R), Absorption (A), Transmission (T)
- Core temperature: Spatial average at center (|x| <= 0.30 mm, z = 780-1035 um, excluding void) and Maximum
- Ventral temperature: Spatial average at center (|x| <= 0.30 mm, z = 2355-2445 um, excluding void) and Maximum
- Material phase volumes: Total ablated volume (mm^3) and Total boiling volume (mm^3)
"""

import os
import glob
import numpy as np

# Spectrophotometric baseline computed via 3D Voxel MC on intact 13-layer beetle stack
SPECTRO_MAP = {
    320:   {'R': 6.23,  'A': 92.96, 'T': 0.00, 'Lat': 0.81},
    450:   {'R': 13.54, 'A': 77.92, 'T': 0.13, 'Lat': 8.41},
    532:   {'R': 19.61, 'A': 61.33, 'T': 0.60, 'Lat': 18.46},
    1064:  {'R': 23.68, 'A': 11.51, 'T': 4.58, 'Lat': 60.23},
    1470:  {'R': 9.63,  'A': 86.86, 'T': 0.11, 'Lat': 3.40},
    1940:  {'R': 5.00,  'A': 94.86, 'T': 0.00, 'Lat': 0.13},
    10600: {'R': 4.28,  'A': 95.72, 'T': 0.00, 'Lat': 0.00},
}

def analyze_npz(npz_path: str, pulse_duration_s: float = 0.200):
    d = np.load(npz_path)
    grid_x = d['grid_x'] - (d['grid_x'][-1] / 2.0)
    grid_z = d['grid_z'] * 1000.0  # um
    times = d['times'] * 1000.0   # ms
    xz_T = d['xz_T']
    xz_phase = d['xz_phase']
    depths = d['crater_depths_mm'] * 1000.0 # um
    DX = float(d.get('DX', 0.015))
    DZ = float(d.get('DZ', 0.015))

    n_epochs = len(times)
    idx_pulse = min(int(round(pulse_duration_s * 1000.0)) - 1, n_epochs - 1)
    
    final_depth_um = float(depths[idx_pulse])
    avg_depth_um = float(np.mean(depths[:idx_pulse+1]))
    speed_mm_s = float((final_depth_um / 1000.0) / pulse_duration_s)

    # Hole diameter across ablated solid tissue slices (excluding air gap 165 to 465 um)
    ph_end = xz_phase[idx_pulse]
    T_end = xz_T[idx_pulse]

    iz_tissue = (grid_z >= 0) & (grid_z <= final_depth_um)
    iz_solid = iz_tissue & ~((grid_z >= 165.0) & (grid_z <= 465.0))

    diameters = []
    for iz in range(len(grid_z)):
        if iz_solid[iz]:
            abl_count = np.sum(ph_end[:, iz] == 3)
            if abl_count > 0:
                diameters.append(abl_count * DX)
    avg_hole_dia_mm = float(np.mean(diameters)) if len(diameters) > 0 else 0.0

    # Core & Ventral regions
    ix_center = np.abs(grid_x) <= 0.30
    iz_core = (grid_z >= 780.0) & (grid_z <= 1035.0)
    iz_ven = (grid_z >= 2355.0) & (grid_z <= 2445.0)

    c_mask = np.outer(ix_center, iz_core)
    core_valid = c_mask & (ph_end != 3)
    core_avg = float(np.mean(T_end[core_valid])) if np.any(core_valid) else 20.0
    core_max = float(np.max(T_end[c_mask])) if np.any(c_mask) else 20.0

    v_mask = np.outer(ix_center, iz_ven)
    ven_valid = v_mask & (ph_end != 3)
    ven_avg = float(np.mean(T_end[ven_valid])) if np.any(ven_valid) else 20.0
    ven_max = float(np.max(T_end[v_mask])) if np.any(v_mask) else 20.0

    # Volumes of ablated and boiled material
    v_abl_mm3 = 0.0
    v_boil_mm3 = 0.0
    for iz in range(len(grid_z)):
        if grid_z[iz] >= 0 and not (165.0 <= grid_z[iz] <= 465.0):
            n_abl = np.sum(ph_end[:, iz] == 3)
            n_boil = np.sum(ph_end[:, iz] == 2)
            r_abl = (n_abl * DX) / 2.0
            r_tot = ((n_abl + n_boil) * DX) / 2.0
            v_abl_mm3 += np.pi * (r_abl**2) * DZ
            v_boil_mm3 += np.pi * (max(0.0, r_tot**2 - r_abl**2)) * DZ

    # Wavelength info
    wl = int(round(float(d.get('wavelength_nm', 10600.0)))) if 'wavelength_nm' in d else None
    power = float(d.get('power_w', 40.0)) if 'power_w' in d else None

    # Check if dual wavelength
    if 'laser_states' in d and wl is None:
        sp = {'R': 0.0, 'A': 0.0, 'T': 0.0}
    elif wl in SPECTRO_MAP:
        sp = SPECTRO_MAP[wl]
    else:
        sp = {'R': 4.0, 'A': 95.0, 'T': 0.0}

    return {
        'path': npz_path,
        'wavelength_nm': wl,
        'power_w': power,
        'final_depth_um': final_depth_um,
        'avg_depth_um': avg_depth_um,
        'speed_mm_s': speed_mm_s,
        'avg_hole_dia_mm': avg_hole_dia_mm,
        'R_pct': sp.get('R', 0.0),
        'A_pct': sp.get('A', 0.0),
        'T_pct': sp.get('T', 0.0),
        'core_avg_T': core_avg,
        'core_max_T': core_max,
        'ven_avg_T': ven_avg,
        'ven_max_T': ven_max,
        'v_abl_mm3': float(v_abl_mm3),
        'v_boil_mm3': float(v_boil_mm3),
        'energy_used_j': 4.60 if ('laser_states' in d or 'dual' in npz_path) else (float(power * pulse_duration_s) if power is not None else 8.00),
    }

if __name__ == '__main__':
    raw_dir = 'knowledge/beetle/raw_arrays'
    files = sorted(glob.glob(os.path.join(raw_dir, 'matrix_*_40w.npz')))
    print(f"{'Run':<24s} | {'Final Depth':<11s} | {'Avg Depth':<10s} | {'Speed':<9s} | {'Hole Dia':<9s} | {'R [%]':<6s} | {'A [%]':<6s} | {'T [%]':<6s} | {'Core Avg/Max':<14s} | {'Ven Avg/Max':<14s} | {'V_abl':<8s} | {'V_boil':<8s}")
    print("-" * 155)
    for f in files:
        if 'dual' in f: continue
        res = analyze_npz(f)
        name = os.path.basename(f).replace('.npz', '')
        print(f"{name:<24s} | {res['final_depth_um']:7.1f} µm | {res['avg_depth_um']:6.1f} µm | {res['speed_mm_s']:5.2f}mm/s | {res['avg_hole_dia_mm']:5.3f} mm | {res['R_pct']:5.2f}% | {res['A_pct']:5.2f}% | {res['T_pct']:5.2f}% | {res['core_avg_T']:5.1f}/{res['core_max_T']:5.1f}C | {res['ven_avg_T']:5.1f}/{res['ven_max_T']:5.1f}C | {res['v_abl_mm3']:6.4f}mm3 | {res['v_boil_mm3']:6.4f}mm3")
