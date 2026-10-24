"""
Native High-Resolution Animation Renderer (Zero Up-Sampling):
- Strictly Native Spatial Grid: 200x244 elements (dx=15 um, dz=10 um) — NO spatial interpolation
- Strictly Native Temporal Frames: 1 ms physical simulation steps — NO temporal sub-frame interpolation
- Pure Physical Boundary: Cavity void and subelytral air gap rendered as solid jet-black (#000000)
- Native 2D Plume Field: Direct from the high-dispersion simulation with real-time transmission inset
- 2x2 Panel Layout: Optical Deposition | Temperature | 2D Plume Extinction | 4-Phase Map
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, PowerNorm, ListedColormap
from matplotlib.patches import Patch
from PIL import Image

sys.path.insert(0, REPO_ROOT)

KNOWLEDGE_DIR = os.path.join(REPO_ROOT, "knowledge/beetle")
RAW_DIR = os.path.join(KNOWLEDGE_DIR, "raw_arrays")
PLOTS_DIR = os.path.join(KNOWLEDGE_DIR, "plots")

def render_native_animation(
    npz_file: str = None,
    out_gif: str = None,
    active_stride: int = 2,     # Sample every 2 ms during active drilling (100 crisp native frames)
    cooling_stride: int = 2,    # Sample every 2 ms during cooling (smooth plume venting/dissipation)
    active_frame_ms: int = 70,  # ~14 fps playback during active drilling
    cooling_frame_ms: int = 50, # ~20 fps playback during cooling
    hold_final_ms: int = 1500   # 1.5s hold on final state
):
    if npz_file is None:
        npz_file = os.path.join(RAW_DIR, "matrix_320nm_20w_1ms_2x.npz")
    if out_gif is None:
        out_gif = os.path.join(PLOTS_DIR, "beetle_native_320nm_20w_1ms_2x.gif")

    print(f"\n[Native Renderer] Loading raw simulation data from: {npz_file}")
    data = np.load(npz_file)
    times = data['times'] * 1000.0        # ms
    grid_x = data['grid_x'] - (data['grid_x'][-1] / 2.0)  # centered radial mm
    grid_z = data['grid_z'] * 1000.0      # depth um
    xz_T = data['xz_T']                  # shape (n_epochs, nx, nz)
    xz_Q = data['xz_Q']                  # shape (n_epochs, nx, nz)
    xz_plume = data['xz_plume']          # shape (n_epochs, nx, nz)
    xz_phase = data['xz_phase']          # shape (n_epochs, nx, nz)
    plume_trans = data['plume_trans'] * 100.0 # %
    n_epochs = len(times)

    nx, nz = len(grid_x), len(grid_z)
    dx_val = (grid_x[1] - grid_x[0]) if nx > 1 else 0.015

    # Extend radial domain to [-1.5, 1.5] mm if needed
    target_extent_x = (-1.5, 1.5)
    if grid_x[0] > target_extent_x[0] + 0.5 * dx_val:
        n_left = int(round((grid_x[0] - target_extent_x[0]) / dx_val))
        n_right = int(round((target_extent_x[1] - grid_x[-1]) / dx_val))
        if n_left > 0 or n_right > 0:
            print(f"Extending radial domain from [{grid_x[0]:.2f}, {grid_x[-1]:.2f}] to [{target_extent_x[0]:.2f}, {target_extent_x[1]:.2f}] mm (+{n_left} left, +{n_right} right voxels)...")
            grid_x = np.linspace(target_extent_x[0], target_extent_x[1], n_left + len(grid_x) + n_right)
            xz_T = np.concatenate([np.tile(xz_T[:, 0:1, :], (1, n_left, 1)), xz_T, np.tile(xz_T[:, -1:, :], (1, n_right, 1))], axis=1)
            xz_Q = np.concatenate([np.tile(xz_Q[:, 0:1, :], (1, n_left, 1)), xz_Q, np.tile(xz_Q[:, -1:, :], (1, n_right, 1))], axis=1)
            xz_plume = np.concatenate([np.tile(xz_plume[:, 0:1, :], (1, n_left, 1)), xz_plume, np.tile(xz_plume[:, -1:, :], (1, n_right, 1))], axis=1)
            xz_phase = np.concatenate([np.tile(xz_phase[:, 0:1, :], (1, n_left, 1)), xz_phase, np.tile(xz_phase[:, -1:, :], (1, n_right, 1))], axis=1)
            nx = len(grid_x)

    print(f"Native grid dimensions: {nx} radial x {nz} depth (dx={dx_val*1000:.0f} um, dz={grid_z[-1]/nz:.0f} um, radial span: [{grid_x[0]:.2f}, {grid_x[-1]:.2f}] mm)")
    print("Zero up-sampling: All fields rendered directly at native resolution.")

    # Select native simulation frames
    active_indices = [i for i in range(n_epochs) if times[i] <= 200.0 and i % active_stride == 0]
    cooling_indices = [i for i in range(n_epochs) if times[i] > 200.0 and i % cooling_stride == 0]
    if (n_epochs - 1) not in cooling_indices:
        cooling_indices.append(n_epochs - 1)
    frame_indices = active_indices + cooling_indices

    print(f"Selected {len(frame_indices)} native simulation frames ({len(active_indices)} active, {len(cooling_indices)} cooling).")

    extent = [grid_x[0], grid_x[-1], grid_z[-1], grid_z[0]]

    # Colormaps & normalizations
    t_cmap = plt.cm.inferno.copy()
    t_cmap.set_bad('#000000') # Solid jet-black for void
    t_norm = Normalize(vmin=20.0, vmax=350.0)
    q_norm = LogNorm(vmin=1e-1, vmax=1e4)

    plume_cmap = plt.cm.magma.copy()
    plume_max_val = max(0.1, float(np.max(xz_plume)))
    # PowerNorm gamma=0.45 enhances visibility of low-density plume (0.01 - 0.08 mm^-1) while preserving 0 as jet-black
    plume_norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=plume_max_val)

    phase_colors = ['#1e293b', '#d97706', '#06b6d4', '#000000']
    phase_cmap = ListedColormap(phase_colors)
    legend_elements = [
        Patch(facecolor='#1e293b', edgecolor='#64748b', label='Intact Solid / Soft Tissue (T < 100°C)'),
        Patch(facecolor='#d97706', edgecolor='#b45309', label='Pyrolyzed / Charred Cuticle (χ ≥ 0.10)'),
        Patch(facecolor='#06b6d4', edgecolor='#0891b2', label='Boiling Hemolymph / Viscera (T ≥ 100°C)'),
        Patch(facecolor='#000000', edgecolor='#ffffff', label='Ablated Void / Subelytral Air Cavity (n=1.0, µ=0)'),
    ]

    # Setup persistent canvas once (bypasses thousands of figure recreations)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor('#0d1117')
    fig.subplots_adjust(top=0.915, bottom=0.08, left=0.07, right=0.93, hspace=0.22, wspace=0.18)
    dorsal_cuticle_end = float(data['dorsal_cuticle_um']) if 'dorsal_cuticle_um' in data else 150.0
    air_gap_end = float(data['air_gap_end_um']) if 'air_gap_end_um' in data else 450.0
    ventral_shell_start = float(data['ventral_shell_um']) if 'ventral_shell_um' in data else 2300.0

    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        if grid_z[0] < -10.0:
            ax.axhline(0, color='#38bdf8', linestyle='-', linewidth=0.9, alpha=0.8) # Dorsal surface (z = 0)
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

    cb1 = fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.02)
    cb1.set_label("Q [W/mm³]", color='#f0f6fc', fontsize=9)
    cb1.ax.tick_params(colors='#c9d1d9', labelsize=8)

    cb2 = fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.02)
    cb2.set_label("T [°C]", color='#f0f6fc', fontsize=9)
    cb2.ax.tick_params(colors='#c9d1d9', labelsize=8)

    cb3 = fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.02)
    cb3.set_label("µ_plume [mm⁻¹]", color='#f0f6fc', fontsize=9)
    cb3.ax.tick_params(colors='#c9d1d9', labelsize=8)

    axes[1, 1].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.22),
                       ncol=2, facecolor='#21262d', edgecolor='#30363d', labelcolor='#f0f6fc', fontsize=8.5)

    wl_nm_val = int(data['wavelength_nm']) if 'wavelength_nm' in data else 320
    power_w_val = float(data['power_w']) if 'power_w' in data else 20.0
    th_val = float(data['theta_inc_deg']) if ('theta_inc_deg' in data and data['theta_inc_deg'].ndim == 0 and not np.isnan(float(data['theta_inc_deg']))) else 0.0
    tilt_str = f" | Tilted θ = {th_val:.1f}° (x = -0.5 to +0.5 mm)" if th_val > 1.0 else ""
    fig.suptitle(f"Native Simulation (Zero Up-Sampling): λ = {wl_nm_val} nm | P = {power_w_val:.0f} W{tilt_str} (dt = 1 ms | Grid = {nx}x{nz})",
                 color='#58a6ff', fontsize=13, fontweight='bold', y=0.965)

    t_max_val = max(300.0, float(np.max(times)))

    # Temperature inset in Panel 2 (Core & Ventral T vs time) - moved to bottom-left
    ax_ins_t = axes[0, 1].inset_axes([0.05, 0.08, 0.36, 0.28])
    ax_ins_t.set_facecolor('#000000')
    ax_ins_t.tick_params(colors='#8b949e', labelsize=7)
    for spine in ax_ins_t.spines.values():
        spine.set_color('#30363d')
    ax_ins_t.grid(True, color='#21262d', linestyle=':', alpha=0.6)
    ax_ins_t.set_xlim(0, t_max_val)
    ax_ins_t.set_ylim(15.0, 365.0)
    ax_ins_t.set_title('Core & Ventral T [°C]', color='#f0f6fc', fontsize=8, pad=2)

    # Transmission inset in Panel 3 - moved to bottom-left
    ax_ins = axes[1, 0].inset_axes([0.05, 0.08, 0.36, 0.28])
    ax_ins.set_facecolor('#000000')
    ax_ins.tick_params(colors='#8b949e', labelsize=7)
    for spine in ax_ins.spines.values():
        spine.set_color('#30363d')
    ax_ins.grid(True, color='#21262d', linestyle=':', alpha=0.6)
    ax_ins.set_xlim(0, t_max_val)
    ax_ins.set_ylim(min(50, float(np.min(plume_trans)) - 5), 105)
    ax_ins.set_title('Transmission [%]', color='#f0f6fc', fontsize=8, pad=2)

    fig.canvas.draw()

    frames = []
    durations = []
    contours = []

    z_scale = 1000.0 if np.max(grid_z) < 10.0 else 1.0
    gz_um = grid_z * z_scale
    iz_core = (gz_um >= 780.0) & (gz_um <= 1040.0)
    iz_ven = (gz_um >= 2360.0) & (gz_um <= 2440.0)

    # Total tissue domain for phase volume normalization (excluding air gap)
    iz_tissue = (gz_um >= 0.0) & ~((gz_um >= 165.0) & (gz_um <= 465.0))
    total_tissue_elements = float(np.sum(iz_tissue) * len(grid_x))

    # Spectrophotometric constants for cumulative energy tracking
    spectro_ref = {
        320:   {'R': 0.0623, 'A': 0.9296, 'T': 0.0000},
        450:   {'R': 0.1354, 'A': 0.7792, 'T': 0.0013},
        532:   {'R': 0.1961, 'A': 0.6133, 'T': 0.0060},
        1064:  {'R': 0.2368, 'A': 0.1151, 'T': 0.0458},
        1470:  {'R': 0.0963, 'A': 0.8686, 'T': 0.0011},
        1940:  {'R': 0.0500, 'A': 0.9486, 'T': 0.0000},
        10600: {'R': 0.0428, 'A': 0.9572, 'T': 0.0000},
    }
    wl_int = int(round(wl_nm_val))
    sp_props = spectro_ref.get(wl_int, {'R': 0.05, 'A': 0.95, 'T': 0.00})

    if 'peak_T_core' in data and len(data['peak_T_core']) == n_epochs:
        core_T_hist = np.asarray(data['peak_T_core'], dtype=np.float32)
    else:
        core_T_hist = np.array([float(np.nanmax(xz_T[i, :, iz_core])) for i in range(n_epochs)], dtype=np.float32)

    # Core volume mean across all voxels in the viscera layer (excluding void/nans)
    core_mean_hist = np.array([float(np.nanmean(xz_T[i, :, iz_core])) for i in range(n_epochs)], dtype=np.float32)

    if 'peak_T_ventral' in data and len(data['peak_T_ventral']) == n_epochs:
        ven_T_hist = np.asarray(data['peak_T_ventral'], dtype=np.float32)
    else:
        ven_T_hist = np.array([float(np.nanmax(xz_T[i, :, iz_ven])) for i in range(n_epochs)], dtype=np.float32)

    # Ventral volume mean across the ventral layer
    ven_mean_hist = np.array([float(np.nanmean(xz_T[i, :, iz_ven])) for i in range(n_epochs)], dtype=np.float32)

    print("Rendering native simulation frames via persistent canvas blitting...")
    for count, ep in enumerate(frame_indices):
        t_now = times[ep]
        is_active = (t_now <= 200.0)
        trans_val = plume_trans[ep]

        # Cumulative energy tracking up to t_now
        t_active_s = min(t_now, 200.0) / 1000.0
        e_inc = power_w_val * t_active_s
        e_dep = e_inc * sp_props['A']
        e_refl = e_inc * sp_props['R']
        e_trans = e_inc * sp_props['T']

        # Native slices (shape: nz, nx)
        T_slice = xz_T[ep, :, :].T
        Q_slice = xz_Q[ep, :, :].T
        plume_slice = xz_plume[ep, :, :].T
        phase_slice = xz_phase[ep, :, :].T

        # Phase volume percentages
        abl_count = float(np.sum(phase_slice[iz_tissue, :] == 3))
        boil_count = float(np.sum(phase_slice[iz_tissue, :] == 2))
        abl_pct = 100.0 * abl_count / max(1.0, total_tissue_elements)
        boil_pct = 100.0 * boil_count / max(1.0, total_tissue_elements)

        void_mask = (phase_slice == 3)
        T_disp = T_slice.copy()
        T_disp[void_mask] = np.nan

        Q_disp = np.maximum(Q_slice.copy(), 1e-4)

        im1.set_data(Q_disp)
        im2.set_data(T_disp)
        im3.set_data(plume_slice)
        im4.set_data(phase_slice)

        laser_state = "Laser: ON" if (is_active and np.nanmax(Q_disp) > 0.05) else "Laser: OFF"
        t1.set_text(f"1) Q(x,z) [t={t_now:.1f}ms | {laser_state}] | Dep: {e_dep:.2f}J, Refl: {e_refl:.2f}J, Trans: {e_trans:.2f}J")

        T_core_now = float(np.nanmax(T_disp[iz_core, :])) if np.any(iz_core) and not np.all(np.isnan(T_disp[iz_core, :])) else 20.0
        T_core_mean_now = float(np.nanmean(T_disp[iz_core, :])) if np.any(iz_core) and not np.all(np.isnan(T_disp[iz_core, :])) else 20.0
        T_ven_now = float(np.nanmax(T_disp[iz_ven, :])) if np.any(iz_ven) and not np.all(np.isnan(T_disp[iz_ven, :])) else 20.0
        T_ven_mean_now = float(np.nanmean(T_disp[iz_ven, :])) if np.any(iz_ven) and not np.all(np.isnan(T_disp[iz_ven, :])) else 20.0
        t2.set_text(f"2) Temperature T(x,z) [°C] | Core: Peak {T_core_now:.1f}°C (Mean {T_core_mean_now:.1f}°C) | Ventral: Peak {T_ven_now:.1f}°C (Mean {T_ven_mean_now:.1f}°C)")
        t3.set_text(f"3) Native 2D Plume Vapor Cloud µ_plume(x,z) | Trans: {trans_val:.1f}%")
        t4.set_text(f"4) Phase Map [t={t_now:.1f}ms] | Ablated: {abl_pct:.1f}% | Cum. Boiled: {boil_pct:.1f}%")

        for c in contours:
            try:
                for coll in c.collections:
                    coll.remove()
            except:
                try: c.remove()
                except: pass
        contours.clear()

        if not np.all(np.isnan(T_disp)):
            if np.nanmax(T_disp) >= 99.5:
                c1 = axes[0, 1].contour(grid_x, grid_z, T_disp, levels=[99.5], colors=['#00f0ff'], linewidths=1.2, linestyles='--')
                contours.append(c1)
            if np.nanmax(T_disp) >= 345.0:
                c2 = axes[0, 1].contour(grid_x, grid_z, T_disp, levels=[345.0], colors=['#ffffff'], linewidths=1.2, linestyles='-')
                contours.append(c2)

        # Dynamic update: Temperature inset (Panel 2)
        ax_ins_t.cla()
        ax_ins_t.set_facecolor('#000000')
        ax_ins_t.tick_params(colors='#8b949e', labelsize=6.5)
        for spine in ax_ins_t.spines.values():
            spine.set_color('#30363d')
        ax_ins_t.grid(True, color='#21262d', linestyle=':', alpha=0.6)
        ax_ins_t.set_xlim(0, t_max_val)
        ax_ins_t.set_ylim(15.0, 365.0)
        ax_ins_t.set_title('Core & Ventral T [°C]', color='#f0f6fc', fontsize=7.5, pad=2)

        idx_hist = [i for i in range(len(times)) if times[i] <= t_now]
        if len(idx_hist) > 0:
            ax_ins_t.plot(times[idx_hist], core_T_hist[idx_hist], color='#00f0ff', linestyle='--', linewidth=1.4, label='Core (Peak)')
            ax_ins_t.plot(times[idx_hist], core_mean_hist[idx_hist], color='#38bdf8', linestyle='-', linewidth=1.6, label='Core (Mean)')
            ax_ins_t.plot(times[idx_hist], ven_T_hist[idx_hist], color='#f43f5e', linestyle='--', linewidth=1.2, label='Ventral (Peak)')
            ax_ins_t.plot(times[idx_hist], ven_mean_hist[idx_hist], color='#fb7185', linestyle='-', linewidth=1.4, label='Ventral (Mean)')
        ax_ins_t.scatter([t_now], [core_T_hist[ep]], color='#00f0ff', s=14, zorder=5)
        ax_ins_t.scatter([t_now], [core_mean_hist[ep]], color='#38bdf8', s=14, zorder=5)
        ax_ins_t.scatter([t_now], [ven_T_hist[ep]], color='#f43f5e', s=12, zorder=5)
        ax_ins_t.scatter([t_now], [ven_mean_hist[ep]], color='#fb7185', s=12, zorder=5)
        ax_ins_t.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d', labelcolor='#f0f6fc', fontsize=5.8, framealpha=0.85, borderpad=0.2, handlelength=1.2)

        # Dynamic update: Transmission inset (Panel 3)
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
        durations.append(active_frame_ms if is_active else cooling_frame_ms)

        if count % 20 == 0 or count == len(frame_indices) - 1:
            print(f"  -> Rendered native frame {count+1}/{len(frame_indices)} (t = {t_now:.1f} ms)")

    plt.close(fig)
    durations[-1] = hold_final_ms

    print(f"Compiling {len(frames)} native frames into GIF: {out_gif}...")
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True
    )
    total_playback_s = sum(durations) / 1000.0
    print(f"[SUCCESS] Native animation compiled: {out_gif}")
    print(f"Playback duration: {total_playback_s:.2f} seconds ({len(frames)} native frames)")

    return out_gif

if __name__ == "__main__":
    npz_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    act_stride = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    cool_stride = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    render_native_animation(npz_file=npz_arg, out_gif=out_arg, active_stride=act_stride, cooling_stride=cool_stride)
