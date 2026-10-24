"""
Sagi-Dolev et al. (1992) Opt Eng 31(7):1425-1431 -- model-structure comparison.

The paper is paywalled; only its abstract was verified, so NO numerical comparison is possible. The
abstract lists the stages of the model: (1) heating to the boiling temperature of water, (2) boiling
and evaporation of the water content, (3) heating of the dehydrated tissue to its ablation
temperature, (4) removal of the relevant material, (5) adapting the new geometrical boundaries.
The engine's two-stage mode implements the same sequence; this module runs it on an illustrative
quasi-1-D case (conditions are NOT from the paper) and records when each stage occurs.
"""
import json
import numpy as np

import bench_common  # noqa: F401
from bench_common import RAW_DIR
import mcmillan_benchmark as mb
from simulation.coupled_ablation import CoupledAblationEngine

STAGES = [
    ("1", "Heating to the boiling temperature of water", "explicit bioheat solver, T rises to T_abl = 100 C", True),
    ("2", "Boiling and evaporation of the water content", "latent-heat plateau: superheat banked as enthalpy, T held at 100 C", True),
    ("3", "Heating of the dehydrated tissue to its ablation temperature", "dry-state voxel (dry rho c, k, mu_a) heats freely to t_remove", True),
    ("4", "Removal of the relevant material", "voxel converted to void (air) when T >= t_remove", True),
    ("5", "Adapting the new geometrical boundaries", "optical field recomputed in the new geometry every epoch (MC) and thermal contact to air updated", True),
]


def run_sagi_stages(I=0.5, Ta=300.0, t_end=1.2, dz=0.005, Nz=100, dt_epoch=0.002, n_photons=60000):
    Nx = Ny = 4
    Lx = 0.2
    w0 = 10.0 * (Lx / 2.0)
    shp = (Nx, Ny, Nz)
    eng = CoupledAblationEngine(
        optical_props={'mu_a': np.full(shp, mb.MU_A_WET), 'mu_s': np.zeros(shp), 'g': np.zeros(shp), 'n': np.ones(shp)},
        k_grid=np.full(shp, mb.K_WET * 1e-3), rho_cp_grid=np.full(shp, mb.RHO_WET * mb.CP_WET * 1e-9),
        h_lat_grid=np.full(shp, mb.H_LAT), Lx=Lx, Ly=Lx, Lz=Nz * dz,
        power=I * np.pi * w0 ** 2 / 2.0, spot_dia=2 * w0, t_abl=100.0, t_ambient=25.0,
        dt_epoch=dt_epoch, dt_sub=4e-5, n_photons=n_photons, h_eff=0.0,
        dry_state=dict(mu_a=0.25 * mb.MU_A_WET, mu_s=0.0, g=0.0, n=1.0, k=mb.K_DRY * 1e-3,
                       rho_cp=mb.RHO_DRY * mb.CP_DRY * 1e-9, t_remove=Ta))
    rec = {k: [] for k in ('t_s', 'T_surface', 'depth_mm', 'n_boiling_layers', 'n_dry_layers')}
    for _ in range(int(round(t_end / dt_epoch))):
        eng.step_epoch()
        col_abl = eng.ablated[:, :, :].all(axis=(0, 1))
        k_surf = int(np.argmin(col_abl)) if not col_abl.all() else Nz - 1
        rec['t_s'].append(eng.t)
        rec['T_surface'].append(float(eng.T[:, :, k_surf].mean()))
        rec['depth_mm'].append(float(col_abl.sum() * dz))
        rec['n_boiling_layers'].append(int(((eng.H_acc > 0) & ~eng.dry & ~eng.ablated).any(axis=(0, 1)).sum()))
        rec['n_dry_layers'].append(int(eng.dry.any(axis=(0, 1)).sum()))
    rec.update({'I_W_mm2': I, 'Ta': Ta, 'T_boil': 100.0, 'dz': dz, 'stages': STAGES})
    with open(RAW_DIR / 'sagi_stages.json', 'w') as f:
        json.dump(rec, f)
    return rec


if __name__ == "__main__":
    r = run_sagi_stages()
    t = np.array(r['t_s']); T = np.array(r['T_surface']); d = np.array(r['depth_mm'])
    tb = t[np.argmax(T >= 99.9)]; td = t[np.argmax(np.array(r['n_dry_layers']) > 0)]; tr = t[np.argmax(d > 0)]
    print(f"boiling starts {tb*1000:.0f} ms, first dry layer {td*1000:.0f} ms, first removal {tr*1000:.0f} ms, depth at end {d[-1]*1000:.0f} um")
