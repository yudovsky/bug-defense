"""
Dual-Wavelength Sequential Laser Ablation Matrix Runner:
Runs sequential two-color exposures and plume ablation comparisons on the isotropic 15 um beetle mesh:
1. Phase 1 (0 to 30 ms): Wavelength 1 @ 40 W (Dorsal shell cutter prepulse)
2. Phase 2 (30 to 200 ms, 170 ms duration): Wavelength 2 @ 20 W (Volumetric body heater)
3. Phase 3 (200 to 1000 ms, 800 ms duration): Laser OFF (0 W cooling & subelytral cavity plume venting)

Targeted Cutter-to-Heater Matrix (9 permutations):
- Cutters: 10,600 nm (Far-IR CO2), 1940 nm (Thulium water peak), 320 nm (UV Excimer)
- Heaters: 532 nm (Green Nd:YAG), 1064 nm (NIR Nd:YAG), 450 nm (Blue diode)

Instructive Plume Comparison (Item 5):
- 10,600 nm @ 40 W with plume dynamics vs without plume dynamics (plume_closure enabled=False).
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import time
import argparse
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, PowerNorm, ListedColormap
from matplotlib.patches import Patch
from PIL import Image

try:
    import objc
except ImportError:
    objc = None

sys.path.insert(0, REPO_ROOT)

from simulation.coupled_ablation import CoupledAblationEngine
from simulation.run_multispectral_matrix import compute_phase_map
from simulation.tissue_stack import eval_single_layer
from simulation.tissue_stack import BASE_LAYERS

KNOWLEDGE_DIR = os.path.join(REPO_ROOT, "knowledge/beetle")
RAW_DIR = os.path.join(KNOWLEDGE_DIR, "raw_arrays")
PLOTS_DIR = os.path.join(KNOWLEDGE_DIR, "plots")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Grid parameters: Isotropic 15 um discretization
DX = 0.015
DY = 0.015
DZ = 0.015
LX = 3.0
LY = 3.0
NX = int(round(LX / DX)) # 200
NY = int(round(LY / DY)) # 200

NZ_AIR_TOP = 21                       # 315 um top ambient air
NZ_BEETLE = 163                       # 2445 um beetle body
NZ = NZ_AIR_TOP + NZ_BEETLE           # 184 slices total (7,360,000 voxels)
L_AIR_TOP = NZ_AIR_TOP * DZ           # 0.315 mm
L_BEETLE = NZ_BEETLE * DZ             # 2.445 mm
LZ = NZ * DZ                          # 2.760 mm

# Tilted beam geometry
Z_DORSAL_PHYS = L_AIR_TOP                                # 0.315 mm
Z_VENTRAL_PHYS = LZ                                      # 2.760 mm
X_DORSAL_PHYS = 1.000                                    # mm (centered x = -0.500 mm)
X_VENTRAL_PHYS = 2.000                                   # mm (centered x = +0.500 mm)
TAN_THETA = (X_VENTRAL_PHYS - X_DORSAL_PHYS) / (Z_VENTRAL_PHYS - Z_DORSAL_PHYS)
THETA_INC_DEG = float(np.degrees(np.arctan(TAN_THETA)))  # ~22.24 deg
PHI_INC_DEG = 0.0
X_TOP_PHYS = X_DORSAL_PHYS - Z_DORSAL_PHYS * TAN_THETA   # ~0.87117 mm
Y_TOP_PHYS = LY / 2.0                                    # 1.500 mm
BEAM_CENTER = (X_TOP_PHYS, Y_TOP_PHYS)

SPOT_DIA = 0.50               # mm
T_TOTAL = 1.000               # 1.000 s total
T_PHASE1 = 0.030              # 30 ms for Phase 1
T_PHASE2 = 0.200              # 200 ms end of Phase 2
DT_EPOCH = 0.001              # 1 ms per epoch (1000 epochs)
DT_SUB = 50e-6                # 50 us (20 substeps/epoch; CFL stable)
N_PHOTONS = 150_000           # 150k photons per active MC step

T_AMBIENT = 20.0
T_BOIL = 100.0
T_ABL_CUTICLE = 350.0
H_EFF = 15.0e-6
H_LAT_WATER = 2.26e6 * 1.0e-6 # J/mm^3

CHAR_STATE = {
    'A': 1.2e4,
    'Ea': 5.8e4,
    'mu_a': 15.0,
    'mu_s': 1.0,
    'g': 0.90,
    't_remove': T_ABL_CUTICLE,
    'chi_remove': 0.95
}

def build_beetle_model(wl_nm: float):
    shape = (NX, NY, NZ)
    mu_a = np.zeros(shape, dtype=np.float64)
    mu_s = np.zeros(shape, dtype=np.float64)
    g = np.zeros(shape, dtype=np.float64)
    n = np.ones(shape, dtype=np.float64)
    k_grid = np.full(shape, 0.026e-3, dtype=np.float64)
    rho_cp_grid = np.full(shape, 1.206e-6, dtype=np.float64)
    h_lat_grid = np.full(shape, 1e12, dtype=np.float64)
    dry_mask = np.zeros(shape, dtype=bool)
    initial_ablated = np.zeros(shape, dtype=bool)

    initial_ablated[:, :, :NZ_AIR_TOP] = True

    evaluated_layers = [eval_single_layer(lay, float(wl_nm)) for lay in BASE_LAYERS]

    beetle_layers = [
        (0, 1, 0, 0.22e-3, 1.98e-3, 0.0, True),              # epicuticle wax (15 um)
        (1, 5, 1, 0.28e-3, 2.40e-3, 0.0, True),              # dorsal exocuticle (60 um)
        (5, 11, 2, 0.35e-3, 2.50e-3, 0.0, True),             # dorsal endocuticle (90 um)
        (11, 31, 3, 0.026e-3, 1.206e-6, 1e12, False),        # subelytral air gap (300 um)
        (31, 32, 4, 0.48e-3, 3.50e-3, H_LAT_WATER, False),   # epidermis (15 um)
        (32, 52, 5, 0.32e-3, 2.80e-3, H_LAT_WATER, False),   # fat body (300 um)
        (52, 69, 6, 0.58e-3, 3.89e-3, H_LAT_WATER, False),   # hemolymph (255 um)
        (69, 137, 7, 0.50e-3, 3.65e-3, H_LAT_WATER, False),  # muscle (1020 um)
        (137, 157, 8, 0.32e-3, 2.80e-3, H_LAT_WATER, False), # ventral fat (300 um)
        (157, 160, 10, 0.35e-3, 2.50e-3, 0.0, True),         # ventral endocuticle (45 um)
        (160, 163, 11, 0.28e-3, 2.40e-3, 0.0, True),         # ventral exocuticle + wax (45 um)
    ]

    for k0, k1, lay_idx, kval, rhocpval, hlatval, is_dry in beetle_layers:
        ev = evaluated_layers[lay_idx]
        gk0 = NZ_AIR_TOP + k0
        gk1 = NZ_AIR_TOP + k1
        mu_a[:, :, gk0:gk1] = float(ev['mu_a'])
        mu_s[:, :, gk0:gk1] = float(ev['mu_s'])
        g[:, :, gk0:gk1] = float(ev['g'])
        n[:, :, gk0:gk1] = float(ev['n'])
        k_grid[:, :, gk0:gk1] = kval
        rho_cp_grid[:, :, gk0:gk1] = rhocpval
        h_lat_grid[:, :, gk0:gk1] = hlatval
        dry_mask[:, :, gk0:gk1] = is_dry

    optical = {'mu_a': mu_a, 'mu_s': mu_s, 'g': g, 'n': n}
    dry_state = {
        'mu_a': mu_a.copy(), 'mu_s': mu_s.copy(), 'g': g.copy(), 'n': n.copy(),
        'k': k_grid.copy(), 'rho_cp': rho_cp_grid.copy(), 't_remove': T_ABL_CUTICLE
    }
    return {
        'optical': optical,
        'k_grid': k_grid,
        'rho_cp_grid': rho_cp_grid,
        'h_lat_grid': h_lat_grid,
        'dry_mask': dry_mask,
        'initial_ablated': initial_ablated,
        'dry_state': dry_state,
    }

def run_simulation(wl1: int, wl2: int, p1: float = 40.0, p2: float = 20.0, plume_enabled: bool = True, force: bool = False):
    tag = f"dual_{wl1}nm_{wl2}nm" if wl2 is not None else f"{wl1}nm_{int(p1)}w_no_plume"
    out_file = os.path.join(RAW_DIR, f"matrix_{tag}.npz")
    out_gif = os.path.join(PLOTS_DIR, f"beetle_{tag}.gif")

    if not force and os.path.exists(out_file) and os.path.exists(out_gif):
        print(f"[{tag}] Output already exists. Skipping simulation.")
        return out_file, out_gif

    print(f"\n{'='*75}")
    if wl2 is not None:
        print(f"  STARTING DUAL RUN: {wl1} nm ({int(T_PHASE1*1000)}ms @ {p1}W) -> {wl2} nm ({int((T_PHASE2-T_PHASE1)*1000)}ms @ {p2}W)")
    else:
        print(f"  STARTING NO-PLUME COMPARISON: {wl1} nm ({int(T_PHASE2*1000)}ms @ {p1}W) [Plume Disabled]")
    print(f"  Mesh: Isotropic 15 um (200x200x184 voxels = 7.36M elements)")
    print(f"  Tilted Beam: θ = {THETA_INC_DEG:.1f}°, dorsal x = -0.50 mm -> ventral x = +0.50 mm")
    print(f"{'='*75}")

    air_gap_start = NZ_AIR_TOP + 11   # slice 32 (480 um)
    air_gap_end = NZ_AIR_TOP + 31    # slice 52 (780 um)

    geo_1 = build_beetle_model(wl1)
    geo_2 = build_beetle_model(wl2) if wl2 is not None else None

    plume_cfg = {
        'enabled': plume_enabled,
        'model': 'diffusion',
        'D_xy': 1.0,
        'D_z': 2.0,
        'v_drift': 0.0,
        'tau_linger_s': float('inf'),
        'blast_expansion': False,
    }

    engine = CoupledAblationEngine(
        optical_props=geo_1['optical'],
        k_grid=geo_1['k_grid'],
        rho_cp_grid=geo_1['rho_cp_grid'],
        h_lat_grid=geo_1['h_lat_grid'],
        Lx=LX, Ly=LY, Lz=LZ,
        power=p1,
        spot_dia=SPOT_DIA,
        t_abl=T_BOIL,
        t_ambient=T_AMBIENT,
        dt_epoch=DT_EPOCH,
        dt_sub=DT_SUB,
        n_photons=N_PHOTONS,
        h_eff=H_EFF,
        beam_center=BEAM_CENTER,
        beam_profile='gaussian',
        theta_inc_deg=THETA_INC_DEG,
        phi_inc_deg=PHI_INC_DEG,
        dry_state=geo_1['dry_state'],
        char_state=CHAR_STATE,
        plume_closure=plume_cfg,
        initial_dry_mask=geo_1['dry_mask'],
        initial_ablated_mask=geo_1['initial_ablated'],
        air_gap_range=(air_gap_start, air_gap_end),
        wavelength_nm=float(wl1),
        mc_engine='gpu',
        thermal_engine='gpu',
    )

    n_epochs = int(round(T_TOTAL / DT_EPOCH))
    mid_y = NY // 2

    is_air_gap = np.zeros((NX, NY, NZ), dtype=bool)
    is_air_gap[:, :, :NZ_AIR_TOP] = True
    is_air_gap[:, :, air_gap_start:air_gap_end] = True

    ix_beam = np.clip(np.round((X_TOP_PHYS + TAN_THETA * (np.arange(NZ) * DZ)) / DX).astype(int), 0, NX - 1)

    times = []
    laser_states = []
    plume_trans = []
    centerline_T = np.zeros((n_epochs, NZ), dtype=np.float32)
    centerline_Q = np.zeros((n_epochs, NZ), dtype=np.float32)
    centerline_phase = np.zeros((n_epochs, NZ), dtype=np.uint8)

    xz_T = np.zeros((n_epochs, NX, NZ), dtype=np.float32)
    xz_Q = np.zeros((n_epochs, NX, NZ), dtype=np.float32)
    xz_phase = np.zeros((n_epochs, NX, NZ), dtype=np.uint8)
    xz_plume = np.zeros((n_epochs, NX, NZ), dtype=np.float32)

    peak_T_dorsal = []
    peak_T_core = []
    peak_T_ventral = []
    crater_depths_mm = []

    t_perf = None
    t_boil = None
    t_start = time.time()

    for ep in range(n_epochs):
        t_now_approx = ep * DT_EPOCH

        if wl2 is not None:
            # Dual wavelength sequencing
            if t_now_approx < T_PHASE1:
                state_label = f"{wl1} nm @ {int(p1)} W"
            elif t_now_approx < T_PHASE2:
                state_label = f"{wl2} nm @ {int(p2)} W"
                if engine.wavelength_nm != float(wl2):
                    print(f"\n  >>> [WAVELENGTH SWITCH] t = {t_now_approx*1000:.1f} ms: Switching to λ = {wl2} nm, P = {p2} W!")
                    intact = ~engine.ablated
                    engine.optical['mu_a'][intact] = geo_2['optical']['mu_a'][intact]
                    engine.optical['mu_s'][intact] = geo_2['optical']['mu_s'][intact]
                    engine.optical['g'][intact] = geo_2['optical']['g'][intact]
                    engine.optical['n'][intact] = geo_2['optical']['n'][intact]

                    if engine.dry_state is not None:
                        engine.dry_state['mu_a'][intact] = geo_2['dry_state']['mu_a'][intact]
                        engine.dry_state['mu_s'][intact] = geo_2['dry_state']['mu_s'][intact]
                        engine.dry_state['g'][intact] = geo_2['dry_state']['g'][intact]
                        engine.dry_state['n'][intact] = geo_2['dry_state']['n'][intact]

                    engine._wet = (engine.optical['mu_a'].copy(), engine.k_grid.copy(), engine.rho_cp_grid.copy())
                    engine._ds_arrays = None
                    engine._init_ds_cs_arrays()

                    if engine.plume is not None:
                        engine.plume.set_wavelength(wl2)
                        vmask = engine.compute_void_mask()
                        engine.plume.sync_extinction(vmask)

                    engine.wavelength_nm = float(wl2)
                    engine.power = p2
            else:
                state_label = "Laser OFF [Cooling]"
                if engine.power > 0.0:
                    engine.power = 0.0
        else:
            # Single wavelength comparison (no plume)
            if t_now_approx < T_PHASE2:
                state_label = f"{wl1} nm @ {int(p1)} W [No Plume]"
            else:
                state_label = "Laser OFF [Cooling]"
                if engine.power > 0.0:
                    engine.power = 0.0

        ep_t0 = time.time()
        info = engine.step_epoch()
        t_now = engine.t
        times.append(t_now)
        laser_states.append(state_label)

        T_now = engine.T
        Q_now = engine.Q
        chi_now = engine.chi
        dry_now = engine.dry
        abl_now = engine.ablated
        H_now = engine.H_acc

        phase_3d = compute_phase_map(T_now, chi_now, dry_now, abl_now, H_now, is_air_gap=is_air_gap)

        centerline_T[ep, :] = T_now[ix_beam, mid_y, np.arange(NZ)].astype(np.float32)
        centerline_Q[ep, :] = Q_now[ix_beam, mid_y, np.arange(NZ)].astype(np.float32)
        centerline_phase[ep, :] = phase_3d[ix_beam, mid_y, np.arange(NZ)].astype(np.uint8)

        xz_T[ep, :, :] = T_now[:, mid_y, :].astype(np.float32)
        xz_Q[ep, :, :] = Q_now[:, mid_y, :].astype(np.float32)
        xz_phase[ep, :, :] = phase_3d[:, mid_y, :].astype(np.uint8)
        if engine.plume is not None:
            xz_plume[ep, :, :] = engine.plume.mu_a_plume[:, mid_y, :].astype(np.float32)

        abl_slice = engine.ablated[:, mid_y, NZ_AIR_TOP:]
        if np.any(abl_slice):
            max_k = int(np.where(abl_slice)[1].max()) + 1
            depth_mm = max_k * DZ
        else:
            depth_mm = 0.0
        crater_depths_mm.append(depth_mm)
        plume_trans.append(info.get('plume_transmission', 1.0))

        T_dor = float(np.max(T_now[:, mid_y, NZ_AIR_TOP:air_gap_start]))
        T_core = float(np.max(T_now[:, mid_y, NZ_AIR_TOP + 52 : NZ_AIR_TOP + 69]))
        T_ven = float(np.max(T_now[:, mid_y, NZ_AIR_TOP + 157 : NZ_AIR_TOP + 163]))

        peak_T_dorsal.append(T_dor)
        peak_T_core.append(T_core)
        peak_T_ventral.append(T_ven)

        if t_perf is None and np.any(engine.ablated[:, mid_y, air_gap_start - 1]):
            t_perf = t_now
            print(f"  >>> [PERFORATION] Dorsal shell breached at t = {t_perf*1000:.1f} ms!")
        if t_boil is None and T_core >= 99.5:
            t_boil = t_now
            print(f"  >>> [CORE BOIL] Hemolymph reached boiling (100°C) at t = {t_boil*1000:.1f} ms!")

        ep_dt = time.time() - ep_t0
        if ep % 100 == 0 or ep == n_epochs - 1 or ep == 29 or ep == 199:
            mode_tag = state_label
            print(f"  Ep {ep+1:4d}/{n_epochs} | t={t_now*1000:6.1f} ms [{mode_tag:^24s}] | Depth={depth_mm*1000:4.0f} um | T_dor={T_dor:5.1f}C | T_core={T_core:5.1f}C | Trans={info.get('plume_transmission', 1.0)*100:5.1f}% | ({ep_dt:.2f}s/ep)")

    # Save simulation results
    print(f"\n  [SAVING RAW ARRAYS] -> {out_file}...")
    np.savez_compressed(
        out_file,
        times=np.array(times, dtype=np.float32),
        laser_states=np.array(laser_states),
        grid_x=np.linspace(0, LX, NX, dtype=np.float32),
        grid_z=np.linspace(-L_AIR_TOP, LZ - L_AIR_TOP, NZ, dtype=np.float32),
        centerline_T=centerline_T,
        centerline_Q=centerline_Q,
        centerline_phase=centerline_phase,
        xz_T=xz_T,
        xz_Q=xz_Q,
        xz_phase=xz_phase,
        xz_plume=xz_plume,
        plume_trans=np.array(plume_trans, dtype=np.float32),
        peak_T_dorsal=np.array(peak_T_dorsal, dtype=np.float32),
        peak_T_core=np.array(peak_T_core, dtype=np.float32),
        peak_T_ventral=np.array(peak_T_ventral, dtype=np.float32),
        crater_depths_mm=np.array(crater_depths_mm, dtype=np.float32),
        t_perforation_s=t_perf if t_perf is not None else -1.0,
        t_boil_s=t_boil if t_boil is not None else -1.0,
        wavelength_nm=float(wl1),
        secondary_wavelength_nm=float(wl2) if wl2 is not None else -1.0,
        power_w=p1,
        secondary_power_w=p2 if wl2 is not None else 0.0,
        theta_inc_deg=THETA_INC_DEG,
        phi_inc_deg=PHI_INC_DEG,
        dorsal_cuticle_um=165.0,
        air_gap_end_um=465.0,
        ventral_shell_um=2355.0,
        NX=NX, NY=NY, NZ=NZ,
        DX=DX, DY=DY, DZ=DZ,
        dt_epoch=DT_EPOCH,
    )
    sim_dur = time.time() - t_start
    print(f"  [SAVED] {out_file} ({sim_dur:.1f}s)")

    # Clean up memory
    del engine, geo_1, geo_2, xz_T, xz_Q, xz_phase, xz_plume, centerline_T, centerline_Q, centerline_phase
    gc.collect()

    # Render animation GIF
    try:
        from render_native_highres_animation import render_native_animation
        render_native_animation(out_file, out_gif)
    except Exception as e:
        print(f"  [WARN] Fallback to inline renderer due to: {e}")
        render_dual_animation(out_file, out_gif)
    return out_file, out_gif

def render_dual_animation(npz_file: str, out_gif: str):
    print(f"\n[Native Renderer] Loading simulation data from: {npz_file}")
    data = np.load(npz_file)
    times = data['times'] * 1000.0        # ms
    states = data['laser_states']
    grid_x = data['grid_x'] - (data['grid_x'][-1] / 2.0)  # centered radial mm [-1.5, +1.5]
    grid_z = data['grid_z'] * 1000.0      # depth um
    xz_T = data['xz_T']                  # shape (n_epochs, nx, nz)
    xz_Q = data['xz_Q']                  # shape (n_epochs, nx, nz)
    xz_plume = data['xz_plume']          # shape (n_epochs, nx, nz)
    xz_phase = data['xz_phase']          # shape (n_epochs, nx, nz)
    plume_trans = data['plume_trans'] * 100.0 # %
    n_epochs = len(times)

    # Compute Core and Ventral temperatures averaged at center of animal, excluding ablated voxels
    ix_center = np.abs(grid_x) <= 0.30
    iz_core = (grid_z >= 780.0) & (grid_z <= 1035.0)
    iz_ven = (grid_z >= 2355.0) & (grid_z <= 2445.0)

    core_T_hist = np.zeros(n_epochs, dtype=np.float32)
    ven_T_hist = np.zeros(n_epochs, dtype=np.float32)

    for ep_i in range(n_epochs):
        ph_i = xz_phase[ep_i]
        T_i = xz_T[ep_i]

        core_mask = np.outer(ix_center, iz_core) & (ph_i != 3)
        core_T_hist[ep_i] = float(np.mean(T_i[core_mask])) if np.any(core_mask) else 20.0

        ven_mask = np.outer(ix_center, iz_ven) & (ph_i != 3)
        ven_T_hist[ep_i] = float(np.mean(T_i[ven_mask])) if np.any(ven_mask) else 20.0

    nx, nz = len(grid_x), len(grid_z)

    # Select frames
    phase1_idx = [i for i in range(n_epochs) if times[i] <= 30.0 and i % 2 == 0]
    phase2_idx = [i for i in range(n_epochs) if 30.0 < times[i] <= 200.0 and i % 4 == 0]
    phase3_idx = [i for i in range(n_epochs) if times[i] > 200.0 and i % 20 == 0]
    if (n_epochs - 1) not in phase3_idx:
        phase3_idx.append(n_epochs - 1)
    frame_indices = phase1_idx + phase2_idx + phase3_idx

    print(f"Selected {len(frame_indices)} frames ({len(phase1_idx)} P1, {len(phase2_idx)} P2, {len(phase3_idx)} P3).")

    extent = [grid_x[0], grid_x[-1], grid_z[-1], grid_z[0]]

    t_cmap = plt.cm.inferno.copy()
    t_cmap.set_bad('#000000')
    t_norm = Normalize(vmin=20.0, vmax=350.0)
    q_norm = LogNorm(vmin=1e-1, vmax=1e4)

    plume_cmap = plt.cm.magma.copy()
    plume_max_val = max(0.1, float(np.max(xz_plume)))
    plume_norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=plume_max_val)

    phase_colors = ['#1e293b', '#d97706', '#06b6d4', '#000000']
    phase_cmap = ListedColormap(phase_colors)
    legend_elements = [
        Patch(facecolor='#1e293b', edgecolor='#64748b', label='Intact Solid / Soft Tissue (T < 100°C)'),
        Patch(facecolor='#d97706', edgecolor='#b45309', label='Pyrolyzed / Charred Cuticle (χ ≥ 0.10)'),
        Patch(facecolor='#06b6d4', edgecolor='#0891b2', label='Boiling Hemolymph / Viscera (T ≥ 100°C)'),
        Patch(facecolor='#000000', edgecolor='#ffffff', label='Ablated Void / Subelytral Air Cavity (n=1.0, µ=0)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor('#0d1117')
    fig.subplots_adjust(top=0.915, bottom=0.08, left=0.07, right=0.93, hspace=0.22, wspace=0.18)
    dorsal_cuticle_end = 165.0
    air_gap_end = 465.0
    ventral_shell_start = 2355.0

    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        if grid_z[0] < -10.0:
            ax.axhline(0, color='#38bdf8', linestyle='-', linewidth=0.9, alpha=0.8)
        ax.axhline(dorsal_cuticle_end, color='#ffffff', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(air_gap_end, color='#ffffff', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(ventral_shell_start, color='#ffffff', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.tick_params(colors='#c9d1d9', labelsize=9)
        ax.set_xlim(-1.5, 1.5)

    im1 = axes[0, 0].imshow(np.maximum(xz_Q[0].T, 1e-4), extent=extent, aspect='auto', cmap='plasma', norm=q_norm)
    im2 = axes[0, 1].imshow(xz_T[0].T, extent=extent, aspect='auto', cmap=t_cmap, norm=t_norm)
    im3 = axes[1, 0].imshow(xz_plume[0].T, extent=extent, aspect='auto', cmap=plume_cmap, norm=plume_norm)
    im4 = axes[1, 1].imshow(xz_phase[0].T, extent=extent, aspect='auto', cmap=phase_cmap, vmin=0, vmax=3)

    t1 = axes[0, 0].set_title('', color='#f0f6fc', fontsize=10.5, fontweight='bold', pad=6)
    t2 = axes[0, 1].set_title('', color='#f0f6fc', fontsize=10.5, fontweight='bold', pad=6)
    t3 = axes[1, 0].set_title('', color='#f0f6fc', fontsize=10.5, fontweight='bold', pad=6)
    t4 = axes[1, 1].set_title('', color='#58a6ff', fontsize=10.5, fontweight='bold', pad=6)

    axes[0, 0].set_ylabel("Depth z [µm]", color='#f0f6fc', fontsize=10)
    axes[1, 0].set_ylabel("Depth z [µm]", color='#f0f6fc', fontsize=10)
    axes[1, 0].set_xlabel("Radial x [mm]", color='#f0f6fc', fontsize=10)
    axes[1, 1].set_xlabel("Radial x [mm]", color='#f0f6fc', fontsize=10)

    cb1 = fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cb1.set_label("Q [W/mm³]", color='#f0f6fc', fontsize=9)
    cb1.ax.tick_params(colors='#c9d1d9', labelsize=8)

    cb2 = fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cb2.set_label("T [°C]", color='#f0f6fc', fontsize=9)
    cb2.ax.tick_params(colors='#c9d1d9', labelsize=8)

    cb3 = fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cb3.set_label("µ_plume [mm⁻¹]", color='#f0f6fc', fontsize=9)
    cb3.ax.tick_params(colors='#c9d1d9', labelsize=8)

    axes[1, 1].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.22),
                      ncol=2, facecolor='#161b22', edgecolor='#30363d', labelcolor='#f0f6fc', fontsize=8)

    # Insets in Panel 2 and Panel 3 - moved to bottom-left
    ax_ins_t = axes[0, 1].inset_axes([0.05, 0.08, 0.37, 0.28])
    ax_ins = axes[1, 0].inset_axes([0.05, 0.08, 0.37, 0.28])

    suptitle_obj = fig.suptitle('', color='#ffffff', fontsize=13, fontweight='bold', y=0.965)

    frames = []
    durations = []
    contours = []
    t_max_val = float(times[-1])
    max_core_disp = max(135.0, float(np.max(core_T_hist)) + 15.0)

    # Tissue mask for ablated and boiled percentage normalization (excluding subelytral air gap)
    iz_tissue = (grid_z >= 0) & ~((grid_z >= 165.0) & (grid_z <= 465.0))
    total_tissue_elements = float(np.sum(iz_tissue) * len(grid_x))

    # Parse wavelengths for cumulative energy tracking
    base_name = os.path.basename(npz_file)
    spectro_ref = {
        320:   {'R': 0.0623, 'A': 0.9296, 'T': 0.0000},
        450:   {'R': 0.1354, 'A': 0.7792, 'T': 0.0013},
        532:   {'R': 0.1961, 'A': 0.6133, 'T': 0.0060},
        1064:  {'R': 0.2368, 'A': 0.1151, 'T': 0.0458},
        1470:  {'R': 0.0963, 'A': 0.8686, 'T': 0.0011},
        1940:  {'R': 0.0500, 'A': 0.9486, 'T': 0.0000},
        10600: {'R': 0.0428, 'A': 0.9572, 'T': 0.0000},
    }
    wl1_val, wl2_val = 10600, None
    if 'dual_' in base_name:
        try:
            pts = base_name.replace('.npz', '').split('_')
            wl1_val = int(pts[1].replace('dual', '').replace('nm', '')) if 'dual' in pts[1] else int(pts[2].replace('nm', ''))
            wl2_val = int(pts[3].replace('nm', '')) if len(pts) > 3 else int(pts[2].replace('nm', ''))
        except Exception:
            wl1_val, wl2_val = 10600, 532
    sp1 = spectro_ref.get(wl1_val, {'R': 0.0428, 'A': 0.9572, 'T': 0.0})
    sp2 = spectro_ref.get(wl2_val, {'R': 0.1961, 'A': 0.6133, 'T': 0.006}) if wl2_val else sp1

    print("Rendering frames via persistent canvas...")
    t_render_start = time.time()

    for count, ep in enumerate(frame_indices):
        t_now = float(times[ep])
        st_now = str(states[ep])
        trans_val = float(plume_trans[ep])
        is_active = (t_now <= 200.0)

        # Cumulative energy tracking in Joules
        if wl2_val is not None:
            if t_now <= 30.0:
                e1 = 40.0 * (t_now / 1000.0)
                e_dep = e1 * sp1['A']
                e_refl = e1 * sp1['R']
                e_trans = e1 * sp1['T']
            else:
                e1 = 40.0 * 0.030
                e2 = 20.0 * (min(t_now, 200.0) - 30.0) / 1000.0
                e_dep = e1 * sp1['A'] + e2 * sp2['A']
                e_refl = e1 * sp1['R'] + e2 * sp2['R']
                e_trans = e1 * sp1['T'] + e2 * sp2['T']
        else:
            e1 = 40.0 * (min(t_now, 200.0) / 1000.0)
            e_dep = e1 * sp1['A']
            e_refl = e1 * sp1['R']
            e_trans = e1 * sp1['T']

        suptitle_obj.set_text(
            f"Laser Interaction & Cavity Plume | t = {t_now:.1f} ms [{st_now}]"
        )

        im1.set_data(np.maximum(xz_Q[ep].T, 1e-4))
        phase_slice = xz_phase[ep].T
        T_disp = xz_T[ep].T.copy()
        T_disp[phase_slice == 3] = np.nan
        im2.set_data(T_disp)
        im3.set_data(xz_plume[ep].T)
        im4.set_data(phase_slice)

        # Phase percentages
        abl_count = float(np.sum(phase_slice[iz_tissue, :] == 3))
        boil_count = float(np.sum(phase_slice[iz_tissue, :] == 2))
        abl_pct = 100.0 * abl_count / max(1.0, total_tissue_elements)
        boil_pct = 100.0 * boil_count / max(1.0, total_tissue_elements)

        laser_state = "Laser: ON" if (is_active and np.nanmax(xz_Q[ep]) > 0.05) else "Laser: OFF"
        t1.set_text(f"1) Q(x,z) [{st_now}] | Dep: {e_dep:.2f}J, Refl: {e_refl:.2f}J, Trans: {e_trans:.2f}J")
        t2.set_text(f"2) Temperature T(x,z) [°C] | Core Avg: {core_T_hist[ep]:.1f}°C | Ventral Avg: {ven_T_hist[ep]:.1f}°C")
        t3.set_text(f"3) Native 2D Plume Vapor Cloud µ_plume(x,z) | Trans: {trans_val:.1f}%")
        t4.set_text(f"4) Phase Map [t={t_now:.1f}ms] | Ablated: {abl_pct:.1f}% | Cum. Boiled: {boil_pct:.1f}%")

        for c in contours:
            try:
                c.remove()
            except Exception:
                for coll in getattr(c, 'collections', []):
                    coll.remove()
        contours.clear()

        if not np.all(np.isnan(T_disp)):
            if np.nanmax(T_disp) >= 99.5:
                c1 = axes[0, 1].contour(grid_x, grid_z, T_disp, levels=[99.5], colors=['#00f0ff'], linewidths=1.2, linestyles='--')
                contours.append(c1)
            if np.nanmax(T_disp) >= 345.0:
                c2 = axes[0, 1].contour(grid_x, grid_z, T_disp, levels=[345.0], colors=['#ffffff'], linewidths=1.2, linestyles='-')
                contours.append(c2)

        # Dynamic update: Temperature inset
        ax_ins_t.cla()
        ax_ins_t.set_facecolor('#000000')
        ax_ins_t.tick_params(colors='#8b949e', labelsize=7)
        for spine in ax_ins_t.spines.values():
            spine.set_color('#30363d')
        ax_ins_t.grid(True, color='#21262d', linestyle=':', alpha=0.6)
        ax_ins_t.set_xlim(0, t_max_val)
        ax_ins_t.set_ylim(15.0, max_core_disp)
        ax_ins_t.set_title('Core & Ventral Avg T [°C]', color='#f0f6fc', fontsize=8, pad=2)

        idx_hist = [i for i in range(len(times)) if times[i] <= t_now]
        if len(idx_hist) > 0:
            ax_ins_t.plot(times[idx_hist], core_T_hist[idx_hist], color='#00f0ff', linewidth=1.6, label='Core Avg')
            ax_ins_t.plot(times[idx_hist], ven_T_hist[idx_hist], color='#f43f5e', linewidth=1.6, label='Ventral Avg')
        ax_ins_t.scatter([t_now], [core_T_hist[ep]], color='#00f0ff', s=18, zorder=5)
        ax_ins_t.scatter([t_now], [ven_T_hist[ep]], color='#f43f5e', s=18, zorder=5)
        ax_ins_t.legend(loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#f0f6fc', fontsize=6.5, framealpha=0.85, borderpad=0.2, handlelength=1.0)

        # Dynamic update: Transmission inset
        ax_ins.cla()
        ax_ins.set_facecolor('#000000')
        ax_ins.tick_params(colors='#8b949e', labelsize=7)
        for spine in ax_ins.spines.values():
            spine.set_color('#30363d')
        ax_ins.grid(True, color='#21262d', linestyle=':', alpha=0.6)
        ax_ins.set_xlim(0, t_max_val)
        ax_ins.set_ylim(min(50, float(np.min(plume_trans)) - 5), 105)
        ax_ins.set_title('Transmission [%]', color='#f0f6fc', fontsize=8, pad=2)

        if len(idx_hist) > 0:
            ax_ins.plot(times[idx_hist], plume_trans[idx_hist], color='#38bdf8', linewidth=1.8)
        ax_ins.scatter([t_now], [trans_val], color='#f59e0b', s=25, zorder=5)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())

        img_rgb = Image.fromarray(buf).convert('RGB')
        img_p = img_rgb.convert('P', palette=Image.Palette.ADAPTIVE)
        frames.append(img_p)
        durations.append(70 if is_active else 50)

        if count % 20 == 0 or count == len(frame_indices) - 1:
            print(f"  -> Rendered frame {count+1}/{len(frame_indices)} (t = {t_now:.1f} ms)")

    plt.close(fig)
    durations[-1] = 1500

    print(f"Compiling {len(frames)} frames into GIF: {out_gif}...")
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True
    )
    render_dur = time.time() - t_render_start
    total_playback_s = sum(durations) / 1000.0
    print(f"[SUCCESS] Animation compiled: {out_gif} in {render_dur:.1f}s (Playback: {total_playback_s:.2f}s)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dual-Wavelength & Plume Ablation Suite")
    parser.add_argument("--wl1", type=int, default=10600, help="First wavelength (nm)")
    parser.add_argument("--wl2", type=int, default=532, help="Second wavelength (nm)")
    parser.add_argument("--p1", type=float, default=40.0, help="Phase 1 Power (W)")
    parser.add_argument("--p2", type=float, default=20.0, help="Phase 2 Power (W)")
    parser.add_argument("--no-plume", action="store_true", help="Disable plume dynamics")
    parser.add_argument("--force", action="store_true", help="Force re-simulation even if output exists")
    parser.add_argument("--render-only", action="store_true", help="Only render GIF from existing NPZ")
    args = parser.parse_args()

    if args.no_plume:
        # Run single wavelength without plume
        out_f = os.path.join(RAW_DIR, f"matrix_{args.wl1}nm_{int(args.p1)}w_no_plume.npz")
        out_g = os.path.join(PLOTS_DIR, f"beetle_{args.wl1}nm_{int(args.p1)}w_no_plume.gif")
        if args.render_only:
            render_dual_animation(out_f, out_g)
        else:
            run_simulation(args.wl1, None, p1=args.p1, p2=0.0, plume_enabled=False, force=args.force)
    else:
        out_f = os.path.join(RAW_DIR, f"matrix_dual_{args.wl1}nm_{args.wl2}nm.npz")
        out_g = os.path.join(PLOTS_DIR, f"beetle_dual_{args.wl1}nm_{args.wl2}nm.gif")
        if args.render_only:
            render_dual_animation(out_f, out_g)
        else:
            run_simulation(args.wl1, args.wl2, p1=args.p1, p2=args.p2, plume_enabled=True, force=args.force)
