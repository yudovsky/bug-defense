"""
Multi-Spectral Multi-Power Matrix Simulation Runner for Armored Beetle.
Simulates 12 parametric combinations:
  Powers: P = [40, 80, 160, 320] Watts
  Wavelengths: lambda = [450, 1064, 10600] nm

Physical Parameters:
  - 1e6 photons per Voxel Monte Carlo run
  - dt_epoch = 0.020 s (20 ms, 4x of 5 ms)
  - 12 epochs (total exposure = 240 ms)
  - dt_sub = 25 us (800 sub-steps per epoch)
  - Plume closure: C_plume = 0.5 m^2/kg, tau_linger = 5 ms
  - Charring kinetics: A = 1e7 1/s, Ea = 65 kJ/mol
  - Stefan enthalpy boiling plateau at 100°C (H_lat = 2.26 J/mm^3)
  - Cuticle pyrolysis / ablation at 350°C
  - Full tissue phase tracking:
      0: Intact Tissue
      1: Pyrolyzed / Charred Cuticle (chi > 0.1)
      2: Boiling Hemolymph / Viscera (T >= 100°C)
      3: Ablated Void Cavity
"""

import os
import sys
import time
import json
import numpy as np

from simulation.coupled_ablation import CoupledAblationEngine, K_AIR, RHO_CP_AIR
from simulation.tissue_stack import eval_single_layer
from simulation.tissue_stack import BASE_LAYERS

OUTPUT_DIR = "simulation_results"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw_arrays")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Grid Specification
NX, NY, NZ = 100, 100, 122
LX, LY, LZ = 3.0, 3.0, 2.44  # mm
DX = LX / NX                 # 0.030 mm = 30 um
DY = LY / NY                 # 0.030 mm = 30 um
DZ = LZ / NZ                 # 0.020 mm = 20 um

SPOT_DIA = 0.50              # mm
BEAM_CENTER = (LX / 2.0, LY / 2.0)
T_EXPOSURE = 0.240           # 240 ms
DT_EPOCH = 0.020             # 20 ms per epoch (4x of 5 ms)
DT_SUB = 2.5e-5              # 25 us
N_PHOTONS = 1_000_000        # 1e6 photons per MC run

T_AMBIENT = 20.0
T_BOIL = 100.0
T_ABL_CUTICLE = 350.0
H_LAT_WATER = 2.26
H_EFF = 15.0e-6

PLUME_CLOSURE = {
    'model': 'diffusion',
    'D_xy': 1.0,               # Radial diffusion coefficient (mm^2/s)
    'D_z': 4.0,                # Axial chimney diffusion coefficient (mm^2/s)
    'v_drift': 0.0,            # Pure diffusion
    'tau_linger_s': float('inf'),  # Open exterior boundary clearance
    'blast_expansion': False,
    'blast_sigma_xy_mm': 0.080,
    'blast_sigma_z_mm': 0.090,
    'blast_z_bias_mm': 0.0,
}

CHAR_STATE = {
    'A': 1.0e7,
    'Ea': 65000.0,
    'mu_a': 15.0,            # High absorption in charred state
    'mu_s': 20.0,
    'k': 0.35e-3,
    'rho_cp': 2.0e-3,
    't_remove': T_ABL_CUTICLE,
    'chi_remove': 0.95
}


def build_beetle_model_for_wavelength(wl_nm):
    """
    Constructs 3D Cartesian property arrays for the 13-layer beetle model at given wavelength.
    """
    shape = (NX, NY, NZ)
    mu_a = np.zeros(shape, dtype=np.float64)
    mu_s = np.zeros(shape, dtype=np.float64)
    g = np.zeros(shape, dtype=np.float64)
    n = np.ones(shape, dtype=np.float64)
    
    k_grid = np.zeros(shape, dtype=np.float64)
    rho_cp_grid = np.zeros(shape, dtype=np.float64)
    h_lat_grid = np.zeros(shape, dtype=np.float64)
    
    dry_mask = np.zeros(shape, dtype=bool)
    layer_id_grid = np.zeros(shape, dtype=np.int32)
    
    # Evaluate optical properties for all base layers at wl_nm
    evaluated_layers = []
    for lay in BASE_LAYERS:
        ev = eval_single_layer(lay, float(wl_nm))
        evaluated_layers.append(ev)
        
    # Layer voxel mapping:
    # 0: epicuticle_wax (k=0..1) -> 20 um
    # 1: exocuticle_sclerotised (k=1..3) -> 40 um (total 60 um)
    # 2: endocuticle (k=3..8) -> 100 um (total 160 um)
    # 3: subelytral_air_gap (k=8..23) -> 300 um (total 460 um)
    # 4: epidermis (k=23..24) -> 20 um
    # 5: fat_body (k=24..39) -> 300 um
    # 6: hemolymph (k=39..52) -> 260 um
    # 7: muscle (k=52..102) -> 1000 um
    # 8: fat_body (k=102..117) -> 300 um
    # 9: epidermis (k=117..118) -> 20 um
    # 10: endocuticle (k=118..120) -> 40 um
    # 11: exocuticle_sclerotised (k=120..122) -> 40 um
    
    layer_ranges = [
        (0, 1, 0, 0.22e-3, 1.98e-3, 0.0, True),          # wax
        (1, 3, 1, 0.28e-3, 2.40e-3, 0.0, True),          # dorsal exo
        (3, 8, 2, 0.35e-3, 2.50e-3, 0.0, True),          # dorsal endo
        (8, 23, 3, 0.026e-3, 1.206e-6, 1e12, False),     # air gap (non-ablating)
        (23, 24, 4, 0.48e-3, 3.50e-3, H_LAT_WATER, False),# epidermis
        (24, 39, 5, 0.32e-3, 2.80e-3, H_LAT_WATER, False),# fat body
        (39, 52, 6, 0.58e-3, 3.89e-3, H_LAT_WATER, False),# hemolymph
        (52, 102, 7, 0.50e-3, 3.65e-3, H_LAT_WATER, False),# muscle
        (102, 117, 8, 0.32e-3, 2.80e-3, H_LAT_WATER, False),# ventral fat
        (117, 118, 9, 0.48e-3, 3.50e-3, H_LAT_WATER, False),# ventral epi
        (118, 120, 10, 0.35e-3, 2.50e-3, 0.0, True),     # ventral endo
        (120, 122, 11, 0.28e-3, 2.40e-3, 0.0, True),     # ventral exo plate
    ]
    
    for k0, k1, lay_idx, kval, rhocpval, hlatval, is_dry in layer_ranges:
        ev = evaluated_layers[lay_idx]
        mu_a[:, :, k0:k1] = float(ev['mu_a'])
        mu_s[:, :, k0:k1] = float(ev['mu_s'])
        g[:, :, k0:k1] = float(ev['g'])
        n[:, :, k0:k1] = float(ev['n'])
        k_grid[:, :, k0:k1] = kval
        rho_cp_grid[:, :, k0:k1] = rhocpval
        h_lat_grid[:, :, k0:k1] = hlatval
        dry_mask[:, :, k0:k1] = is_dry
        layer_id_grid[:, :, k0:k1] = lay_idx + 1

    optical = {'mu_a': mu_a, 'mu_s': mu_s, 'g': g, 'n': n}
    
    dry_state = {
        'mu_a': mu_a.copy(),
        'mu_s': mu_s.copy(),
        'g': g.copy(),
        'n': n.copy(),
        'k': k_grid.copy(),
        'rho_cp': rho_cp_grid.copy(),
        't_remove': T_ABL_CUTICLE
    }
    
    return {
        'optical': optical,
        'k_grid': k_grid,
        'rho_cp_grid': rho_cp_grid,
        'h_lat_grid': h_lat_grid,
        'dry_mask': dry_mask,
        'dry_state': dry_state,
        'layer_id_grid': layer_id_grid
    }


def compute_phase_map(T, chi, dry, ablated, H_acc, is_air_gap=None):
    """
    Computes discrete tissue phase classification:
      0: Intact solid/visceral tissue (ambient/warm)
      1: Pyrolyzed / Charred Cuticle (chi >= 0.10)
      2: Boiling Viscera / Hemolymph (T >= 99.5°C or H_acc > 0)
      3: Ablated Void Cavity
    """
    phase = np.zeros_like(T, dtype=np.uint8)
    
    # Boiling hemolymph & soft viscera (air gap cannot boil)
    if is_air_gap is not None:
        boiling = ((T >= 99.5) | (H_acc > 0.0)) & (~is_air_gap)
        charred = dry & (chi >= 0.10) & (~is_air_gap)
    else:
        boiling = (T >= 99.5) | (H_acc > 0.0)
        charred = dry & (chi >= 0.10)
    
    phase[boiling] = 2
    phase[charred] = 1
    phase[ablated] = 3
    if is_air_gap is not None:
        phase[is_air_gap] = 3
    
    return phase


def run_single_simulation(wl_nm, power_w, t_exposure=None, dt_epoch=None, n_photons=None, t_laser_on=None, enable_plume=True, output_filename=None):
    t_exp = float(t_exposure) if t_exposure is not None else T_EXPOSURE
    dt_ep = float(dt_epoch) if dt_epoch is not None else DT_EPOCH
    n_phot = int(n_photons) if n_photons is not None else N_PHOTONS
    t_pulse = float(t_laser_on) if t_laser_on is not None else t_exp

    plume_str = "ENABLED" if enable_plume else "DISABLED"
    print(f"\n" + "="*80)
    print(f"RUNNING: Wavelength = {wl_nm} nm | Laser Power = {power_w} W | Plume: {plume_str}")
    print(f"Photons = {n_phot:,} | dt_epoch = {dt_ep*1000:.0f} ms | Duration = {t_exp*1000:.0f} ms | Laser On = {t_pulse*1000:.0f} ms")
    print("="*80)
    
    geo = build_beetle_model_for_wavelength(wl_nm)
    
    engine = CoupledAblationEngine(
        optical_props=geo['optical'],
        k_grid=geo['k_grid'],
        rho_cp_grid=geo['rho_cp_grid'],
        h_lat_grid=geo['h_lat_grid'],
        Lx=LX, Ly=LY, Lz=LZ,
        power=power_w,
        spot_dia=SPOT_DIA,
        t_abl=T_BOIL,
        t_ambient=T_AMBIENT,
        dt_epoch=dt_ep,
        dt_sub=DT_SUB,
        n_photons=n_phot,
        h_eff=H_EFF,
        beam_center=BEAM_CENTER,
        beam_profile='gaussian',
        dry_state=geo['dry_state'],
        char_state=CHAR_STATE,
        plume_closure=PLUME_CLOSURE if enable_plume else None,
        initial_dry_mask=geo['dry_mask'],
        air_gap_range=(8, 23),
        wavelength_nm=wl_nm,
    )
    
    n_epochs = int(round(t_exp / dt_ep))
    mid_x = NX // 2
    mid_y = NY // 2
    
    is_air_gap = np.zeros((NX, NY, NZ), dtype=bool)
    is_air_gap[:, :, 8:23] = True

    times = []
    plume_trans = []
    centerline_T = np.zeros((n_epochs, NZ), dtype=np.float32)
    centerline_Q = np.zeros((n_epochs, NZ), dtype=np.float32)
    centerline_phase = np.zeros((n_epochs, NZ), dtype=np.uint8)
    
    xz_T = np.zeros((n_epochs, NX, NZ), dtype=np.float32)
    xz_Q = np.zeros((n_epochs, NX, NZ), dtype=np.float32)
    xz_phase = np.zeros((n_epochs, NX, NZ), dtype=np.uint8)
    xz_plume = np.zeros((n_epochs, NX, NZ), dtype=np.float32)
    
    peak_T_dorsal = []
    peak_T_ventral = []
    peak_T_core = []
    crater_depths_mm = []
    
    t_perf = None
    t_boil = None
    t_ventral_blister = None
    
    t0 = time.time()
    for ep in range(n_epochs):
        ep_t0 = time.time()
        if (ep * dt_ep) >= t_pulse:
            engine.power = 0.0
        info = engine.step_epoch()
        t_now = engine.t
        times.append(t_now)
        
        T_now = engine.T
        Q_now = engine.Q
        chi_now = engine.chi
        dry_now = engine.dry
        abl_now = engine.ablated
        H_now = engine.H_acc
        
        phase_3d = compute_phase_map(T_now, chi_now, dry_now, abl_now, H_now, is_air_gap=is_air_gap)
        
        centerline_T[ep, :] = T_now[mid_x, mid_y, :].astype(np.float32)
        centerline_Q[ep, :] = Q_now[mid_x, mid_y, :].astype(np.float32)
        centerline_phase[ep, :] = phase_3d[mid_x, mid_y, :].astype(np.uint8)
        
        xz_T[ep, :, :] = T_now[:, mid_y, :].astype(np.float32)
        xz_Q[ep, :, :] = Q_now[:, mid_y, :].astype(np.float32)
        xz_phase[ep, :, :] = phase_3d[:, mid_y, :].astype(np.uint8)
        if engine.plume is not None:
            xz_plume[ep, :, :] = engine.plume.mu_a_plume[:, mid_y, :].astype(np.float32)
        
        depth_mm = engine.centerline_depth_mm(mid_x, mid_y)
        crater_depths_mm.append(depth_mm)
        plume_trans.append(info['plume_transmission'])
        
        T_dor = float(np.max(T_now[:, mid_y, :8]))
        T_core = float(np.max(T_now[:, mid_y, 39:52]))
        T_ven = float(np.max(T_now[:, mid_y, 118:122]))
        
        peak_T_dorsal.append(T_dor)
        peak_T_core.append(T_core)
        peak_T_ventral.append(T_ven)
        
        if t_perf is None and engine.ablated[mid_x, mid_y, 7]:
            t_perf = t_now
            print(f"  >>> [PERFORATION] Dorsal shell breached at t = {t_perf*1000:.0f} ms!")
        if t_boil is None and T_core >= 99.5:
            t_boil = t_now
            print(f"  >>> [CORE BOIL] Hemolymph reached boiling (100°C) at t = {t_boil*1000:.0f} ms!")
        if t_ventral_blister is None and T_ven >= 99.5:
            t_ventral_blister = t_now
            print(f"  >>> [VENTRAL BLISTER] Ventral plate reached {T_ven:.1f}°C at t = {t_ventral_blister*1000:.0f} ms!")
            
        ep_dt = time.time() - ep_t0
        print(f"  Ep {ep+1:2d}/{n_epochs} | t={t_now*1000:5.0f} ms | Depth={depth_mm*1000:4.0f} um | "
              f"T_dor={T_dor:5.1f}C | T_core={T_core:5.1f}C | T_ven={T_ven:5.1f}C | "
              f"Trans={info['plume_transmission']*100:4.1f}% | ({ep_dt:.2f} s)")
              
    elapsed = time.time() - t0
    print(f"Run completed in {elapsed:.1f} s ({elapsed/n_epochs:.2f} s/epoch).")
    
    # Save npz
    os.makedirs(RAW_DIR, exist_ok=True)
    if output_filename is not None:
        out_file = os.path.join(RAW_DIR, output_filename)
    else:
        out_file = os.path.join(RAW_DIR, f"matrix_{wl_nm}nm_{power_w:.0f}w.npz")
    np.savez_compressed(
        out_file,
        wavelength=wl_nm,
        power=power_w,
        times=np.array(times),
        centerline_T=centerline_T,
        centerline_Q=centerline_Q,
        centerline_phase=centerline_phase,
        xz_T=xz_T,
        xz_Q=xz_Q,
        xz_phase=xz_phase,
        xz_plume=xz_plume,
        crater_depths_mm=np.array(crater_depths_mm),
        plume_trans=np.array(plume_trans),
        peak_T_dorsal=np.array(peak_T_dorsal),
        peak_T_core=np.array(peak_T_core),
        peak_T_ventral=np.array(peak_T_ventral),
        t_perf=t_perf if t_perf is not None else -1.0,
        t_boil=t_boil if t_boil is not None else -1.0,
        t_ventral_blister=t_ventral_blister if t_ventral_blister is not None else -1.0,
        grid_x=np.linspace(DX/2.0, LX - DX/2.0, NX),
        grid_y=np.linspace(DY/2.0, LY - DY/2.0, NY),
        grid_z=np.linspace(DZ/2.0, LZ - DZ/2.0, NZ),
    )
    print(f"Saved: {out_file}")
    
    return {
        'wavelength': wl_nm,
        'power': power_w,
        't_perf': t_perf,
        't_boil': t_boil,
        't_ventral_blister': t_ventral_blister,
        'final_T_dorsal': peak_T_dorsal[-1],
        'final_T_core': peak_T_core[-1],
        'final_T_ventral': peak_T_ventral[-1],
        'final_depth_um': crater_depths_mm[-1] * 1000.0,
        'out_file': out_file
    }


def run_full_matrix():
    wavelengths = [450, 1064, 10600]
    powers = [40.0, 80.0, 160.0, 320.0]
    
    results = []
    total_runs = len(wavelengths) * len(powers)
    run_idx = 0
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    progress_file = os.path.join(OUTPUT_DIR, "matrix_progress.json")
    
    print("=" * 85)
    print(f"STARTING MULTI-SPECTRAL MULTI-POWER MATRIX: {total_runs} RUNS")
    print(f"Wavelengths: {wavelengths} nm | Powers: {powers} W")
    print("=" * 85)
    
    start_time = time.time()
    for wl in wavelengths:
        for p in powers:
            run_idx += 1
            print(f"\n>>> MATRIX PROGRESS: RUN {run_idx}/{total_runs} ({(run_idx-1)/total_runs*100:.1f}%) <<<")
            
            res = run_single_simulation(wl, p)
            results.append(res)
            
            # Update live progress json for cadenced status reporting
            elapsed = time.time() - start_time
            avg_per_run = elapsed / run_idx
            eta = avg_per_run * (total_runs - run_idx)
            
            with open(progress_file, 'w') as f:
                json.dump({
                    'current_run': run_idx,
                    'total_runs': total_runs,
                    'current_wl': wl,
                    'current_power': p,
                    'elapsed_s': elapsed,
                    'eta_s': eta,
                    'results': results
                }, f, indent=2)
                
    total_elapsed = time.time() - start_time
    print("\n" + "="*85)
    print(f"ALL {total_runs} MATRIX RUNS COMPLETED in {total_elapsed/60:.2f} minutes.")
    print("="*85)
    
    # Save master summary json
    summary_path = os.path.join(OUTPUT_DIR, "multispectral_matrix_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Master summary saved to {summary_path}")
    return results


if __name__ == "__main__":
    run_full_matrix()
