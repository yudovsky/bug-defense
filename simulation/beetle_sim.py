"""
Interactive driver for the coupled 3D beetle ablation simulation.

Builds the layered beetle from an editable layer list, runs CoupledAblationEngine with the same
closures as knowledge/beetle/scripts/run_highres_knowledge_suite.py, and yields the x-z mid-plane
slices (absorbed power, temperature, plume, tissue phase) for the four-panel animation.
The defaults reproduce the published 532 nm, 40 W, 22.2 degree run.
"""

import base64
import math
import time

import numpy as np

from simulation.coupled_ablation import CoupledAblationEngine
from simulation.run_multispectral_matrix import CHAR_STATE, H_LAT_WATER, compute_phase_map
from simulation.tissue_stack import eval_single_layer

LX = LY = 3.0                  # mm lateral domain
AIR_TOP_UM = 315.0             # ambient air above the dorsal surface
T_AMBIENT = 20.0
T_BOIL = 100.0
T_ABL_CUTICLE = 350.0
H_EFF = 15.0e-6                # W/(mm^2 K) surface loss
DT_EPOCH = 0.001               # 1 ms optical/thermal coupling step
DT_SUB = 5.0e-5                # 50 us thermal sub-step

K_AIR = 0.026e-3
RHO_CP_AIR = 1.206e-6

PLUME_CLOSURE = {
    'model': 'diffusion',
    'D_xy': 1.0,
    'D_z': 2.0,
    'v_drift': 0.0,
    'tau_linger_s': float('inf'),
    'blast_expansion': False,
    'blast_sigma_xy_mm': 0.080,
    'blast_sigma_z_mm': 0.090,
    'blast_z_bias_mm': 0.0,
}

# Voxel size [um] and photons per step for each resolution preset.
RESOLUTIONS = {
    'preview': {'voxel_um': 30.0, 'photons': 50_000, 'label': 'Preview (30 um voxels)'},
    'standard': {'voxel_um': 20.0, 'photons': 100_000, 'label': 'Standard (20 um voxels)'},
    'full': {'voxel_um': 15.0, 'photons': 150_000, 'label': 'Full (15 um voxels, as published)'},
}

# Thermal units: k [W/(mm K)], rho_cp [J/(mm^3 K)]. 'boils' layers hold at 100 C while water
# boils off; 'dry' layers are cuticle that chars and ablates at 350 C.
DEFAULT_LAYERS = [
    {'name': 'epicuticle_wax', 'label': 'Epicuticle wax', 'thickness_um': 15.0,
     'composition': {'lipid': 0.85, 'protein': 0.15},
     'k': 0.22e-3, 'rho_cp': 1.98e-3, 'kind': 'dry'},
    {'name': 'exocuticle_sclerotised', 'label': 'Dorsal exocuticle', 'thickness_um': 60.0,
     'composition': {'chitin': 0.403, 'protein': 0.33, 'water': 0.18, 'melanin': 0.075, 'ommochrome': 0.012},
     'k': 0.28e-3, 'rho_cp': 2.40e-3, 'kind': 'dry'},
    {'name': 'endocuticle', 'label': 'Dorsal endocuticle', 'thickness_um': 90.0,
     'composition': {'water': 0.42, 'chitin': 0.33, 'protein': 0.25},
     'k': 0.35e-3, 'rho_cp': 2.50e-3, 'kind': 'dry'},
    {'name': 'subelytral_air_gap', 'label': 'Subelytral air gap', 'thickness_um': 300.0,
     'composition': {'air': 1.0},
     'k': K_AIR, 'rho_cp': RHO_CP_AIR, 'kind': 'air'},
    {'name': 'epidermis', 'label': 'Epidermis', 'thickness_um': 15.0,
     'composition': {'water': 0.70, 'protein': 0.22, 'lipid': 0.08},
     'k': 0.48e-3, 'rho_cp': 3.50e-3, 'kind': 'boils'},
    {'name': 'fat_body', 'label': 'Fat body', 'thickness_um': 300.0,
     'composition': {'lipid': 0.50, 'water': 0.38, 'protein': 0.12},
     'k': 0.32e-3, 'rho_cp': 2.80e-3, 'kind': 'boils'},
    {'name': 'hemolymph', 'label': 'Hemolymph', 'thickness_um': 255.0,
     'composition': {'water': 0.93, 'protein': 0.07},
     'k': 0.58e-3, 'rho_cp': 3.89e-3, 'kind': 'boils'},
    {'name': 'muscle', 'label': 'Muscle', 'thickness_um': 1020.0,
     'composition': {'water': 0.76, 'protein': 0.21, 'lipid': 0.03},
     'k': 0.50e-3, 'rho_cp': 3.65e-3, 'kind': 'boils'},
    {'name': 'fat_body', 'label': 'Ventral fat body', 'thickness_um': 300.0,
     'composition': {'lipid': 0.50, 'water': 0.38, 'protein': 0.12},
     'k': 0.32e-3, 'rho_cp': 2.80e-3, 'kind': 'boils'},
    {'name': 'endocuticle', 'label': 'Ventral endocuticle', 'thickness_um': 45.0,
     'composition': {'water': 0.42, 'chitin': 0.33, 'protein': 0.25},
     'k': 0.35e-3, 'rho_cp': 2.50e-3, 'kind': 'dry'},
    {'name': 'exocuticle_sclerotised', 'label': 'Ventral exocuticle + wax', 'thickness_um': 45.0,
     'composition': {'chitin': 0.43, 'protein': 0.352, 'water': 0.18, 'melanin': 0.03, 'ommochrome': 0.008},
     'k': 0.28e-3, 'rho_cp': 2.40e-3, 'kind': 'dry'},
]

DEFAULT_LASER = {
    'wavelength_nm': 532.0,
    'power_w': 40.0,
    'spot_dia_mm': 0.5,
    'theta_deg': float(math.degrees(math.atan(1.0 / 2.445))),   # 22.2 deg: crosses the body mid-plane
    'pulse_ms': 200.0,
    'duration_ms': 1000.0,
    'plume': True,
    'resolution': 'full',
}


def default_setup():
    return {
        'layers': [dict(l, composition=dict(l['composition'])) for l in DEFAULT_LAYERS],
        'laser': dict(DEFAULT_LASER),
        'resolutions': {k: v['label'] for k, v in RESOLUTIONS.items()},
    }


def evaluate_layers(layers, wavelength_nm):
    """Optical properties of each layer at the given wavelength, plus its depth range [um]."""
    out = []
    z = 0.0
    for lay in layers:
        ev = eval_single_layer(lay, float(wavelength_nm))
        th = float(lay['thickness_um'])
        out.append({
            'mu_a': ev['mu_a'], 'mu_s': ev['mu_s'], 'mu_s_prime': ev['mu_s_prime'],
            'g': ev['g'], 'n': ev['n'], 'penetration_depth_mm': ev['penetration_depth_mm'],
            'z_start_um': round(z, 1), 'z_end_um': round(z + th, 1),
        })
        z += th
    return out


def _encode(arr, dtype):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode('ascii')


def _stride(n, limit=120):
    return max(1, int(math.ceil(n / limit)))


def build_model(layers, wavelength_nm, voxel_um):
    """Voxelises the layer stack into a 3D grid; returns arrays and per-layer slice ranges."""
    d = voxel_um / 1000.0
    nx = ny = int(round(LX / d))
    nz_air = int(round(AIR_TOP_UM / voxel_um))
    slices = [max(1, int(round(float(l['thickness_um']) / voxel_um))) for l in layers]
    nz = nz_air + sum(slices)
    shape = (nx, ny, nz)

    mu_a = np.zeros(shape)
    mu_s = np.zeros(shape)
    g = np.zeros(shape)
    n = np.ones(shape)
    k_grid = np.full(shape, K_AIR)
    rho_cp = np.full(shape, RHO_CP_AIR)
    h_lat = np.full(shape, 1e12)
    dry = np.zeros(shape, dtype=bool)
    ablated0 = np.zeros(shape, dtype=bool)
    ablated0[:, :, :nz_air] = True
    is_air = np.zeros(shape, dtype=bool)
    is_air[:, :, :nz_air] = True

    ranges = []
    k0 = nz_air
    for lay, ns in zip(layers, slices):
        k1 = k0 + ns
        ev = eval_single_layer(lay, float(wavelength_nm))
        kind = lay.get('kind', 'boils')
        mu_a[:, :, k0:k1] = ev['mu_a']
        mu_s[:, :, k0:k1] = ev['mu_s']
        g[:, :, k0:k1] = ev['g']
        n[:, :, k0:k1] = ev['n']
        k_grid[:, :, k0:k1] = float(lay['k'])
        rho_cp[:, :, k0:k1] = float(lay['rho_cp'])
        h_lat[:, :, k0:k1] = {'boils': H_LAT_WATER, 'dry': 0.0, 'air': 1e12}[kind]
        dry[:, :, k0:k1] = kind == 'dry'
        is_air[:, :, k0:k1] = kind == 'air'
        ranges.append((k0, k1, kind, lay.get('name', '')))
        k0 = k1

    return {
        'shape': shape, 'nz_air': nz_air, 'dz': d, 'ranges': ranges, 'is_air': is_air,
        'optical': {'mu_a': mu_a, 'mu_s': mu_s, 'g': g, 'n': n},
        'k_grid': k_grid, 'rho_cp_grid': rho_cp, 'h_lat_grid': h_lat,
        'dry_mask': dry, 'initial_ablated': ablated0,
        'dry_state': {'mu_a': mu_a.copy(), 'mu_s': mu_s.copy(), 'g': g.copy(), 'n': n.copy(),
                      'k': k_grid.copy(), 'rho_cp': rho_cp.copy(), 't_remove': T_ABL_CUTICLE},
    }


def _regions(ranges, nz_air):
    """Slice ranges used for the panel annotations and the inset temperature traces."""
    kinds = [r[2] for r in ranges]
    first_air = next((i for i, kd in enumerate(kinds) if kd == 'air'), None)
    last_soft = max((i for i, kd in enumerate(kinds) if kd == 'boils'), default=None)
    hemo = next((i for i, r in enumerate(ranges) if r[3] == 'hemolymph'), None)
    soft = [i for i, kd in enumerate(kinds) if kd == 'boils']
    core = hemo if hemo is not None else (soft[len(soft) // 2] if soft else len(ranges) // 2)
    ventral = (ranges[last_soft][1], ranges[-1][1]) if last_soft is not None and last_soft < len(ranges) - 1 \
        else (ranges[-1][0], ranges[-1][1])
    return {
        'air_gap': (ranges[first_air][0], ranges[first_air][1]) if first_air is not None else None,
        'core': (ranges[core][0], ranges[core][1]),
        'ventral': ventral,
        'surface': nz_air,
    }


def run(layers, laser, frame_ms_active=2.0, frame_ms_cooling=10.0):
    """Generator of events: one 'setup', then 'frame' events, then 'done'."""
    res = RESOLUTIONS.get(laser.get('resolution', 'full'), RESOLUTIONS['full'])
    voxel_um = res['voxel_um']
    wl = float(laser['wavelength_nm'])
    power = float(laser['power_w'])
    spot = float(laser['spot_dia_mm'])
    theta = float(laser['theta_deg'])
    pulse_s = float(laser['pulse_ms']) / 1000.0
    total_s = float(laser['duration_ms']) / 1000.0
    photons = int(laser.get('photons') or res['photons'])

    geo = build_model(layers, wl, voxel_um)
    nx, ny, nz = geo['shape']
    d = geo['dz']
    lz = nz * d
    nz_air = geo['nz_air']
    l_beetle = (nz - nz_air) * d
    reg = _regions(geo['ranges'], nz_air)

    # Aim so the beam axis crosses the lateral centre half-way through the body
    tan_t = math.tan(math.radians(theta))
    x_dorsal = LX / 2.0 - tan_t * l_beetle / 2.0
    beam_center = (x_dorsal - nz_air * d * tan_t, LY / 2.0)

    engine = CoupledAblationEngine(
        optical_props=geo['optical'], k_grid=geo['k_grid'], rho_cp_grid=geo['rho_cp_grid'],
        h_lat_grid=geo['h_lat_grid'], Lx=LX, Ly=LY, Lz=lz, power=power, spot_dia=spot,
        t_abl=T_BOIL, t_ambient=T_AMBIENT, dt_epoch=DT_EPOCH, dt_sub=DT_SUB, n_photons=photons,
        h_eff=H_EFF, beam_center=beam_center, beam_profile='gaussian', theta_inc_deg=theta,
        phi_inc_deg=0.0, dry_state=geo['dry_state'], char_state=CHAR_STATE,
        plume_closure=PLUME_CLOSURE if laser.get('plume', True) else None,
        initial_dry_mask=geo['dry_mask'], initial_ablated_mask=geo['initial_ablated'],
        air_gap_range=reg['air_gap'], wavelength_nm=wl, mc_engine='auto', thermal_engine='auto',
    )

    sx, sz = _stride(nx), _stride(nz)
    mid_y = ny // 2
    grid_x = (np.arange(nx) + 0.5) * d - LX / 2.0
    grid_z = ((np.arange(nz) + 0.5) - nz_air) * d * 1000.0      # um, 0 = dorsal surface
    tissue = ~geo['is_air'][::sx, mid_y, ::sz]
    n_tissue = max(1, int(np.sum(tissue)))
    um = lambda k: round((k - nz_air) * d * 1000.0, 1)

    yield {'event': 'setup', 'data': {
        'nx': len(grid_x[::sx]), 'nz': len(grid_z[::sz]),
        'x_mm': [round(v, 4) for v in grid_x[::sx]], 'z_um': [round(v, 2) for v in grid_z[::sz]],
        'grid': f'{nx}x{ny}x{nz}', 'voxel_um': voxel_um, 'photons': photons,
        'lines_um': {
            'surface': 0.0,
            'air_gap_start': um(reg['air_gap'][0]) if reg['air_gap'] else None,
            'air_gap_end': um(reg['air_gap'][1]) if reg['air_gap'] else None,
            'ventral_shell': um(reg['ventral'][0]),
        },
        'layers': [{'name': r[3], 'kind': r[2], 'z_start_um': um(r[0]), 'z_end_um': um(r[1])} for r in geo['ranges']],
        'laser': {'wavelength_nm': wl, 'power_w': power, 'spot_dia_mm': spot, 'theta_deg': theta,
                  'pulse_ms': pulse_s * 1000.0, 'duration_ms': total_s * 1000.0, 'plume': bool(laser.get('plume', True))},
    }}

    n_epochs = int(round(total_s / DT_EPOCH))
    e_inc = e_abs = e_refl = e_trans = 0.0
    last_frame_t = -1e9
    t0 = time.time()
    for ep in range(n_epochs):
        on = ep * DT_EPOCH < pulse_s
        engine.power = power if on else 0.0
        info = engine.step_epoch()
        if on:
            sp = engine.last_mc.get('spectro_raw', {}) if engine.last_mc else {}
            e = power * DT_EPOCH
            e_inc += e
            e_abs += e * float(info['absorbed_frac'])
            e_refl += e * float(sp.get('R_spec', 0.0) + sp.get('R_diff', 0.0))
            e_trans += e * float(sp.get('Transmission', 0.0))

        t_ms = engine.t * 1000.0
        step_ms = frame_ms_active if on else frame_ms_cooling
        if t_ms - last_frame_t + 1e-6 < step_ms and ep != n_epochs - 1:
            continue
        last_frame_t = t_ms

        T = engine.T[::sx, mid_y, ::sz]
        Q = engine.Q[::sx, mid_y, ::sz]
        phase = compute_phase_map(engine.T, engine.chi, engine.dry, engine.ablated, engine.H_acc,
                                  is_air_gap=geo['is_air'])[::sx, mid_y, ::sz]
        plume = engine.plume.mu_a_plume[::sx, mid_y, ::sz] if engine.plume is not None else np.zeros_like(T)

        c0, c1 = reg['core']
        v0, v1 = reg['ventral']
        core = engine.T[:, mid_y, c0:c1][~engine.ablated[:, mid_y, c0:c1]]
        ven = engine.T[:, mid_y, v0:v1][~engine.ablated[:, mid_y, v0:v1]]
        abl = engine.ablated[:, mid_y, nz_air:]
        depth_um = (int(np.where(abl)[1].max()) + 1) * d * 1000.0 if np.any(abl) else 0.0
        elapsed = time.time() - t0

        yield {'event': 'frame', 'data': {
            't_ms': round(t_ms, 2), 'laser_on': bool(on), 'progress': round((ep + 1) / n_epochs, 4),
            'eta_s': round(elapsed / (ep + 1) * (n_epochs - ep - 1), 1),
            'Q': _encode(np.maximum(Q, 1e-4).T, '<f4'), 'T': _encode(T.T, '<f4'),
            'plume': _encode(plume.T, '<f4'), 'phase': _encode(phase.T, 'u1'),
            'transmission_pct': round(100.0 * float(info['plume_transmission']), 2),
            'core_peak': round(float(core.max()) if core.size else T_AMBIENT, 2),
            'core_mean': round(float(core.mean()) if core.size else T_AMBIENT, 2),
            'ventral_peak': round(float(ven.max()) if ven.size else T_AMBIENT, 2),
            'ventral_mean': round(float(ven.mean()) if ven.size else T_AMBIENT, 2),
            'ablated_pct': round(100.0 * float(np.sum((phase == 3) & tissue)) / n_tissue, 2),
            'boiled_pct': round(100.0 * float(np.sum((phase == 2) & tissue)) / n_tissue, 2),
            'depth_um': round(depth_um, 1),
            'energy_j': {'incident': round(e_inc, 3), 'absorbed': round(e_abs, 3),
                         'reflected': round(e_refl, 3), 'transmitted': round(e_trans, 3)},
        }}

    yield {'event': 'done', 'data': {'elapsed_s': round(time.time() - t0, 1)}}
