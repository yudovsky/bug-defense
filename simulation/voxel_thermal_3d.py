"""
3D Cartesian Voxel Bioheat Thermal Diffusion Solver Module.
Solves the 3D transient heat diffusion equation on heterogeneous voxel grids:
    rho_cp(x, y, z) * dT/dt = div(k(x, y, z) * grad(T)) + Q(x, y, z, t) - Losses

Features:
- Harmonic mean thermal conductivity k_{face} at material interfaces
- Convective & linearized radiative boundary conditions at exterior boundaries (h_eff)
- Fast Numba-parallel (prange) 3D 7-point stencil kernel
- Explicit-CFL conditioning of high-diffusivity (air/void) cells derived from the actual dt
- Supports active laser pulse duration followed by passive thermal cooling
- Returns 3D temperature tensor over time T(x, y, z, t) and centerline histories.
"""

import time
import numpy as np
from numba import njit, prange

try:
    from simulation.voxel_thermal_metal import is_metal_available as _is_metal_available, run_thermal3d_substeps_metal, MetalThermalContext
except ImportError:
    def _is_metal_available() -> bool:
        return False
    run_thermal3d_substeps_metal = None
    MetalThermalContext = None

CFL_SAFETY_FACTOR = 0.40


@njit(parallel=True, fastmath=True)
def _thermal3d_step_explicit(
    T,
    T_new,
    Q,
    k_grid,
    rho_cp_grid,
    Nx, Ny, Nz,
    dx, dy, dz,
    dt,
    h_eff,
    t_ambient,
    is_active
):
    """
    Single explicit time-step of 3D heat diffusion with harmonic mean k.
    """
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    idz2 = 1.0 / (dz * dz)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                T_c = T[i, j, k]
                k_c = k_grid[i, j, k]
                cap = rho_cp_grid[i, j, k]
                
                # --- X direction conduction ---
                if i > 0:
                    k_w = k_grid[i-1, j, k]
                    k_face_w = 2.0 * k_c * k_w / (k_c + k_w + 1e-12)
                    flux_w = k_face_w * (T[i-1, j, k] - T_c) * idx2
                else:
                    flux_w = 0.0 # Insulated / symmetric
                    
                if i < Nx - 1:
                    k_e = k_grid[i+1, j, k]
                    k_face_e = 2.0 * k_c * k_e / (k_c + k_e + 1e-12)
                    flux_e = k_face_e * (T[i+1, j, k] - T_c) * idx2
                else:
                    flux_e = 0.0
                    
                # --- Y direction conduction ---
                if j > 0:
                    k_s = k_grid[i, j-1, k]
                    k_face_s = 2.0 * k_c * k_s / (k_c + k_s + 1e-12)
                    flux_s = k_face_s * (T[i, j-1, k] - T_c) * idy2
                else:
                    flux_s = 0.0
                    
                if j < Ny - 1:
                    k_n = k_grid[i, j+1, k]
                    k_face_n = 2.0 * k_c * k_n / (k_c + k_n + 1e-12)
                    flux_n = k_face_n * (T[i, j+1, k] - T_c) * idy2
                else:
                    flux_n = 0.0
                    
                # --- Z direction conduction ---
                # Top boundary (k=0): Convection to ambient
                if k > 0:
                    k_top = k_grid[i, j, k-1]
                    k_face_top = 2.0 * k_c * k_top / (k_c + k_top + 1e-12)
                    flux_top = k_face_top * (T[i, j, k-1] - T_c) * idz2
                else:
                    # Convective loss at top surface: half-cell conduction in series with h_eff,
                    # per unit cell volume (area / volume = 1 / dz). Reduces to h_eff / dz for small Biot number.
                    flux_top = (t_ambient - T_c) / (dz * (0.5 * dz / max(k_c, 1e-30) + 1.0 / max(h_eff, 1e-30))) if h_eff > 0.0 else 0.0
                    
                # Bottom boundary (k=Nz-1): Convection to ambient
                if k < Nz - 1:
                    k_bot = k_grid[i, j, k+1]
                    k_face_bot = 2.0 * k_c * k_bot / (k_c + k_bot + 1e-12)
                    flux_bot = k_face_bot * (T[i, j, k+1] - T_c) * idz2
                else:
                    flux_bot = (t_ambient - T_c) / (dz * (0.5 * dz / max(k_c, 1e-30) + 1.0 / max(h_eff, 1e-30))) if h_eff > 0.0 else 0.0
                    
                div_q = flux_w + flux_e + flux_s + flux_n + flux_top + flux_bot
                
                source = Q[i, j, k] if is_active else 0.0
                
                # Temperature increment
                T_new[i, j, k] = T_c + dt * (div_q + source) / cap


@njit(parallel=True, fastmath=True)
def _apply_latent_heat_plateau(T, H_acc, rho_cp_grid, ablated, t_abl, Nx, Ny, Nz, dry, t_void):
    """
    Enthalpy-method (Stefan) phase change at the ablation plateau T_abl, applied in place
    after each explicit step.

    Intact cells: any superheat above T_abl is banked into H_acc [J/mm^3] as latent
    vaporization enthalpy and T is held at T_abl; if a partially vaporized cell cools
    below T_abl, banked enthalpy is released before its temperature drops.
    Dehydrated ("dry") cells are left untouched (they heat freely above T_abl).
    Ablated cavity cells are clamped to <= t_void (vapor is ejected).
    """
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                t = T[i, j, k]
                if ablated[i, j, k]:
                    if t > t_void:
                        T[i, j, k] = t_void
                elif dry[i, j, k]:
                    continue
                elif t > t_abl:
                    H_acc[i, j, k] += (t - t_abl) * rho_cp_grid[i, j, k]
                    T[i, j, k] = t_abl
                elif H_acc[i, j, k] > 0.0:
                    cap = rho_cp_grid[i, j, k]
                    release = min(H_acc[i, j, k], (t_abl - t) * cap)
                    H_acc[i, j, k] -= release
                    T[i, j, k] = t + release / cap


class VoxelThermalSolver3D:
    def __init__(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        grid_z: np.ndarray,
        k_grid: np.ndarray,
        rho_cp_grid: np.ndarray,
        h_conv: float = 15.0, # W/(m^2*K)
        emissivity: float = 0.95,
        t_ambient: float = 20.0
    ):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.Nx = len(grid_x)
        self.Ny = len(grid_y)
        self.Nz = len(grid_z)
        
        self.dx = float(grid_x[1] - grid_x[0]) if self.Nx > 1 else 0.05
        self.dy = float(grid_y[1] - grid_y[0]) if self.Ny > 1 else 0.05
        self.dz = float(grid_z[1] - grid_z[0]) if self.Nz > 1 else 0.04
        
        # Unit conversion for thermal conductivity: W/(m*K) -> W/(mm*K) = 1e-3
        # If user passes W/(m*K), convert to W/(mm*K)
        if np.max(k_grid) > 0.05:
            self.k_grid = k_grid * 1e-3
        else:
            self.k_grid = k_grid.copy()
            
        # Volumetric heat capacity: J/(m^3*K) -> J/(mm^3*K) = 1e-9
        if np.max(rho_cp_grid) > 100.0:
            self.rho_cp_grid = rho_cp_grid * 1e-9
        else:
            self.rho_cp_grid = rho_cp_grid.copy()
            
        # Effective heat transfer coefficient at boundary:
        # h_rad = 4 * sigma * eps * T_amb^3
        # h_conv [W/(m^2*K)] -> W/(mm^2*K) = 1e-6
        sigma_sb = 5.670374e-8
        t_amb_k = t_ambient + 273.15
        h_rad = 4.0 * sigma_sb * emissivity * (t_amb_k**3)
        self.h_eff = (h_conv + h_rad) * 1e-6 # W/(mm^2*K)
        self.t_ambient = float(t_ambient)

    def cfl_conditioned_rho_cp(self, dt: float) -> np.ndarray:
        """
        Return a copy of rho_cp with high-diffusivity cells (air/void) raised so that
        alpha = k / rho_cp satisfies the explicit 3D stability limit for this dt.
        """
        inv_d2 = 1.0 / (self.dx ** 2) + 1.0 / (self.dy ** 2) + 1.0 / (self.dz ** 2)
        alpha_max_stable = CFL_SAFETY_FACTOR / (dt * inv_d2)
        rho_cp = self.rho_cp_grid.copy()
        alpha_curr = self.k_grid / np.maximum(rho_cp, 1e-12)
        unstable = alpha_curr > alpha_max_stable
        if np.any(unstable):
            rho_cp[unstable] = self.k_grid[unstable] / alpha_max_stable
        return rho_cp

    def solve(
        self,
        Q_3d: np.ndarray,
        pulse_duration: float = 0.20,
        total_time: float = 1.00,
        dt: float = 0.001, # 1.0 ms
        snapshot_times: list = None,
        engine: str = 'auto',
    ):
        """
        Execute 3D transient thermal solve with snapshot recording.
        Supports native Apple Metal GPU acceleration.
        """
        n_steps = int(round(total_time / dt))
        rho_cp_eff = self.cfl_conditioned_rho_cp(dt)
        T_curr = np.full((self.Nx, self.Ny, self.Nz), self.t_ambient, dtype=np.float64)
        T_next = np.empty_like(T_curr)
        
        snapshots = {}
        if snapshot_times is None:
            snapshot_times = [0.0, pulse_duration, total_time]
            
        snap_targets = sorted(list(set([round(float(t), 4) for t in snapshot_times])))
        
        mid_x = self.Nx // 2
        mid_y = self.Ny // 2
        mid_z = self.Nz // 2
        
        center_history = []
        surface_history = []
        time_history = []

        engine_clean = str(engine).strip().lower()
        use_gpu = False
        if engine_clean in ('auto', 'gpu') and _is_metal_available() and MetalThermalContext is not None:
            use_gpu = True
        elif engine_clean == 'gpu':
            raise RuntimeError("Metal GPU acceleration requested (engine='gpu'), but Metal is not available.")

        t0 = time.time()
        if use_gpu:
            gpu_ctx = MetalThermalContext(
                self.Nx, self.Ny, self.Nz,
                self.dx, self.dy, self.dz,
                dt, self.k_grid, rho_cp_eff,
                self.h_eff, self.t_ambient
            )
            gpu_ctx.set_T(T_curr)
            gpu_ctx.set_Q(Q_3d)

            for step in range(n_steps + 1):
                cur_t = round(step * dt, 4)
                T_view = gpu_ctx.get_T_view()
                center_history.append(float(T_view[mid_x, mid_y, mid_z]))
                surface_history.append(float(T_view[mid_x, mid_y, 0]))
                time_history.append(cur_t)

                if cur_t in snap_targets:
                    snapshots[cur_t] = gpu_ctx.get_T()

                if step < n_steps:
                    is_active = (cur_t < pulse_duration)
                    gpu_ctx.advance(1, int(is_active))
        else:
            for step in range(n_steps + 1):
                cur_t = round(step * dt, 4)
                
                # Record center and surface temperatures
                center_history.append(float(T_curr[mid_x, mid_y, mid_z]))
                surface_history.append(float(T_curr[mid_x, mid_y, 0]))
                time_history.append(cur_t)
                
                if cur_t in snap_targets:
                    snapshots[cur_t] = T_curr.copy()
                    
                if step < n_steps:
                    is_active = (cur_t < pulse_duration)
                    _thermal3d_step_explicit(
                        T_curr,
                        T_next,
                        Q_3d,
                        self.k_grid,
                        rho_cp_eff,
                        self.Nx, self.Ny, self.Nz,
                        self.dx, self.dy, self.dz,
                        dt,
                        self.h_eff,
                        self.t_ambient,
                        is_active
                    )
                    # Swap buffers
                    T_curr, T_next = T_next, T_curr
                    
        solve_time = time.time() - t0
        
        return {
            'snapshots': snapshots,
            'time_history': np.array(time_history),
            'center_history': np.array(center_history),
            'surface_history': np.array(surface_history),
            'solve_time': round(solve_time, 3),
            'peak_surface_T': max(surface_history),
            'peak_center_T': max(center_history),
            'engine': 'gpu' if use_gpu else 'cpu',
        }
