"""
Generic coupled 3D Voxel Monte Carlo + bioheat + enthalpy-method ablation driver.

One epoch = (1) optical transport through the current voxel geometry, (2) explicit
bioheat sub-stepping with the latent-heat plateau at T_abl applied after every sub-step,
(3) ablation of voxels whose banked latent enthalpy reached their requirement, which are
converted to void (air) optically and thermally. Geometry is updated only at epoch ends.

This is the single code path used by the beetle dynamic-ablation study and by the
literature/analytic benchmark suite.
"""

import time
import numpy as np
from numba import njit, prange

from simulation.voxel_mc_3d import run_voxel_mc3d
from simulation.voxel_thermal_3d import _thermal3d_step_explicit, _apply_latent_heat_plateau
from simulation.plume import PlumeAbsorptionField, MultispectralPlumeField, _diffuse_plume_step_numba

try:
    from simulation.metal_ablation_solver import is_metal_available as is_metal_ablation_available, MetalAblationContext
except ImportError:
    def is_metal_ablation_available() -> bool:
        return False
    MetalAblationContext = None

K_AIR = 0.026e-3       # W/(mm*K)
RHO_CP_AIR = 1.206e-6  # J/(mm^3*K)
R_GAS = 8.314          # J/(mol*K)


@njit(parallel=True, fastmath=True)
def _accumulate_arrhenius(T, Omega, ablated, A, Ea_over_R, dt, Nx, Ny, Nz):
    """Omega += dt * A * exp(-Ea / (R * T_K)) for intact voxels (Arrhenius damage integral)."""
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if not ablated[i, j, k]:
                    Omega[i, j, k] += dt * A * np.exp(-Ea_over_R / (T[i, j, k] + 273.15))


@njit(parallel=True, fastmath=True)
def _accumulate_char(T, chi, dry, ablated, A, Ea_over_R, dt, Nx, Ny, Nz):
    """
    First-order pyrolysis of dry organic matrix to char:
        dchi/dt = (1 - chi) * A * exp(-Ea / (R * T_K))
    Exact analytical integration across dt:
        chi_new = 1.0 - (1.0 - chi_old) * exp(-rate * dt)
    Active only in the dry (dehydrated) state, for non-ablated voxels.
    """
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if dry[i, j, k] and not ablated[i, j, k]:
                    tk = T[i, j, k] + 273.15
                    rate = A * np.exp(-Ea_over_R / tk)
                    chi[i, j, k] = 1.0 - (1.0 - chi[i, j, k]) * np.exp(-rate * dt)


@njit(parallel=True, fastmath=True)
def _update_single_stage_ablation_numba(H_acc, h_lat, T, t_abl, ablated, to_remove):
    """Parallel single-stage threshold ablation removal check."""
    N = H_acc.size
    h_acc_flat = H_acc.ravel()
    h_lat_flat = h_lat.ravel()
    T_flat = T.ravel()
    abl_flat = ablated.ravel()
    rem_flat = to_remove.ravel()
    for i in prange(N):
        if abl_flat[i]:
            rem_flat[i] = False
        else:
            rem_flat[i] = (h_acc_flat[i] >= h_lat_flat[i]) and (T_flat[i] >= t_abl)


@njit(parallel=True, fastmath=True)
def _update_phase_properties_numba(
    H_acc, h_lat, T, chi, dry, ablated,
    mu_a, mu_s, g, n, k_grid, rho_cp_grid,
    wet_mu_a, wet_k, wet_rho_cp,
    ds_mu_a, ds_mu_s, ds_g, ds_n, ds_k, ds_rho_cp, ds_t_rem,
    has_cs, has_cs_mu_a, cs_mu_a, cs_mu_s, cs_k, cs_rho_cp, cs_t_rem, cs_chi_rem,
    has_cs_mu_s, has_cs_k, has_cs_rho_cp,
    alpha_max_stable, to_remove
):
    """
    Fused parallel phase transition and property update kernel across all voxels.
    Combines boiling enthalpy interpolation, wet-to-dry state transition, carbonization
    char property interpolation, CFL stability enforcement, and crater ablation detection.
    """
    N = H_acc.size
    h_acc_flat = H_acc.ravel()
    h_lat_flat = h_lat.ravel()
    T_flat = T.ravel()
    chi_flat = chi.ravel()
    dry_flat = dry.ravel()
    ablated_flat = ablated.ravel()
    mu_a_flat = mu_a.ravel()
    mu_s_flat = mu_s.ravel()
    g_flat = g.ravel()
    n_flat = n.ravel()
    k_flat = k_grid.ravel()
    rho_cp_flat = rho_cp_grid.ravel()
    wet_mu_a_flat = wet_mu_a.ravel()
    wet_k_flat = wet_k.ravel()
    wet_rho_cp_flat = wet_rho_cp.ravel()
    to_remove_flat = to_remove.ravel()

    for i in prange(N):
        if ablated_flat[i]:
            to_remove_flat[i] = False
            continue

        is_dry = dry_flat[i]
        h_acc_val = h_acc_flat[i]
        h_lat_val = max(h_lat_flat[i], 1e-30)

        # 1. Boiling / wet-to-dry transition
        if not is_dry:
            if h_acc_val > 0.0:
                frac = min(max(h_acc_val / h_lat_val, 0.0), 1.0)
                dry_mu_a = ds_mu_a[0] if ds_mu_a.size == 1 else ds_mu_a[i]
                dry_k = ds_k[0] if ds_k.size == 1 else ds_k[i]
                dry_rho_cp = ds_rho_cp[0] if ds_rho_cp.size == 1 else ds_rho_cp[i]

                mu_a_flat[i] = wet_mu_a_flat[i] + (dry_mu_a - wet_mu_a_flat[i]) * frac
                k_flat[i] = wet_k_flat[i] + (dry_k - wet_k_flat[i]) * frac
                rho_cp_flat[i] = wet_rho_cp_flat[i] + (dry_rho_cp - wet_rho_cp_flat[i]) * frac

                if h_acc_val >= h_lat_flat[i]:
                    is_dry = True
                    dry_flat[i] = True
                    h_acc_flat[i] = 0.0
                    mu_a_flat[i] = dry_mu_a
                    mu_s_flat[i] = ds_mu_s[0] if ds_mu_s.size == 1 else ds_mu_s[i]
                    g_flat[i] = ds_g[0] if ds_g.size == 1 else ds_g[i]
                    n_flat[i] = ds_n[0] if ds_n.size == 1 else ds_n[i]
                    k_flat[i] = dry_k
                    rho_cp_flat[i] = dry_rho_cp

        # 2. Dry / Char state & Ablation removal check
        if is_dry:
            dry_mu_a = ds_mu_a[0] if ds_mu_a.size == 1 else ds_mu_a[i]
            dry_k = ds_k[0] if ds_k.size == 1 else ds_k[i]
            dry_rho_cp = ds_rho_cp[0] if ds_rho_cp.size == 1 else ds_rho_cp[i]

            if has_cs:
                chi_val = chi_flat[i]
                if has_cs_mu_a:
                    c_mu_a = cs_mu_a[0] if cs_mu_a.size == 1 else cs_mu_a[i]
                    mu_a_flat[i] = dry_mu_a + chi_val * (c_mu_a - dry_mu_a)

                if has_cs_mu_s:
                    dry_mu_s = ds_mu_s[0] if ds_mu_s.size == 1 else ds_mu_s[i]
                    c_mu_s = cs_mu_s[0] if cs_mu_s.size == 1 else cs_mu_s[i]
                    mu_s_flat[i] = dry_mu_s + chi_val * (c_mu_s - dry_mu_s)

                if has_cs_k:
                    c_k = cs_k[0] if cs_k.size == 1 else cs_k[i]
                    k_flat[i] = dry_k + chi_val * (c_k - dry_k)

                if has_cs_rho_cp:
                    c_rho = cs_rho_cp[0] if cs_rho_cp.size == 1 else cs_rho_cp[i]
                    rho_flat = dry_rho_cp + chi_val * (c_rho - dry_rho_cp)
                else:
                    rho_flat = rho_cp_flat[i]

                min_rhocp = k_flat[i] / alpha_max_stable
                if rho_flat < min_rhocp:
                    rho_flat = min_rhocp
                rho_cp_flat[i] = rho_flat

                rem_t = cs_t_rem[0] if cs_t_rem.size == 1 else cs_t_rem[i]
                to_remove_flat[i] = (T_flat[i] >= rem_t) or (chi_val >= cs_chi_rem)
            else:
                rem_t = ds_t_rem[0] if ds_t_rem.size == 1 else ds_t_rem[i]
                to_remove_flat[i] = T_flat[i] >= rem_t
        else:
            to_remove_flat[i] = False


class CoupledAblationEngine:
    def __init__(
        self,
        optical_props: dict,
        k_grid: np.ndarray,
        rho_cp_grid: np.ndarray,
        h_lat_grid: np.ndarray,
        Lx: float, Ly: float, Lz: float,
        power: float,
        spot_dia: float,
        t_abl: float,
        t_ambient: float = 20.0,
        dt_epoch: float = 0.01,
        dt_sub: float = 5e-5,
        n_photons: int = 100000,
        h_eff: float = 0.0,
        r_max: float = 0.0,
        beam_center=None,
        stability_factor: float = 0.35,
        k_void: float = K_AIR,
        rho_cp_void: float = RHO_CP_AIR,
        arrhenius=None,
        beam_profile: str = 'gaussian',
        dry_state=None,
        char_state=None,
        plume_closure=None,
        theta_inc_deg: float = 0.0,
        phi_inc_deg: float = 0.0,
        initial_dry_mask=None,
        initial_ablated_mask=None,
        air_gap_range=None,
        wavelength_nm: float = 10600.0,
        mc_engine: str = 'auto',
        thermal_engine: str = 'auto',
    ):
        """
        optical_props: dict of 3D arrays 'mu_a', 'mu_s', 'g', 'n' [mm^-1, -, -]
        k_grid [W/(mm K)], rho_cp_grid [J/(mm^3 K)], h_lat_grid [J/mm^3] (latent enthalpy to ablate)
        h_eff [W/(mm^2 K)] convective/radiative loss at the z=0 and z=Lz planes.
        beam_center: optional (x, y) [mm] beam axis; may be changed between epochs (scanning).
        power may be set to 0 between epochs for laser-off cooling (optical step skipped).
        arrhenius: optional (A [1/s], Ea [J/mol]); integrates the damage parameter self.Omega
        on every thermal sub-step.
        dry_state: optional two-stage phase change (McMillan et al. 2021 style). When a voxel has
        banked its latent enthalpy (water boiled off) it becomes a dehydrated voxel with the optical
        / thermal properties in dry_state = dict(mu_a, mu_s, g, n, k, rho_cp, t_remove) and heats
        freely; it is removed (void) once it reaches t_remove. While boiling, mu_a, k and rho_cp are
        interpolated linearly between the wet and dry values with the banked-enthalpy fraction.
        """
        self.optical = {key: np.array(optical_props[key], dtype=np.float64, copy=True)
                        for key in ('mu_a', 'mu_s', 'g', 'n')}
        self.Nx, self.Ny, self.Nz = self.optical['mu_a'].shape
        self.Lx, self.Ly, self.Lz = float(Lx), float(Ly), float(Lz)
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz
        self.voxel_vol = self.dx * self.dy * self.dz

        self.power = float(power)
        self.spot_dia = float(spot_dia)
        self.t_abl = float(t_abl)
        self.t_ambient = float(t_ambient)
        self.dt_epoch = float(dt_epoch)
        self.dt_sub = float(dt_sub)
        self.n_substeps = max(1, int(round(self.dt_epoch / self.dt_sub)))
        self.n_photons = int(n_photons)
        self.h_eff = float(h_eff)
        self.r_max = float(r_max)
        self.beam_center = beam_center
        self.beam_profile = beam_profile
        self.theta_inc_deg = float(theta_inc_deg)
        self.phi_inc_deg = float(phi_inc_deg)
        self.mc_engine = mc_engine

        self.k_grid = np.array(k_grid, dtype=np.float64, copy=True)
        self.rho_cp_grid = np.array(rho_cp_grid, dtype=np.float64, copy=True)
        self.h_lat = np.array(h_lat_grid, dtype=np.float64, copy=True)

        # Explicit-scheme stability: raise rho_cp only where diffusivity exceeds the limit
        inv_d2 = 1.0 / self.dx ** 2 + 1.0 / self.dy ** 2 + 1.0 / self.dz ** 2
        self.alpha_max_stable = stability_factor / (self.dt_sub * inv_d2)
        alpha = self.k_grid / np.maximum(self.rho_cp_grid, 1e-12)
        unstable = alpha > self.alpha_max_stable
        self.cfl_conditioned_mask = unstable.copy()
        if np.any(unstable):
            self.rho_cp_grid[unstable] = self.k_grid[unstable] / self.alpha_max_stable
        self.k_void = float(k_void)
        self.rho_cp_void = max(float(rho_cp_void), self.k_void / self.alpha_max_stable)

        shape = (self.Nx, self.Ny, self.Nz)
        self.T = np.full(shape, self.t_ambient, dtype=np.float64)
        self._T_next = np.empty_like(self.T)
        self.H_acc = np.zeros(shape, dtype=np.float64)
        self.ablation_time = np.full(shape, -1.0, dtype=np.float64)
        if initial_ablated_mask is not None:
            self.ablated = np.array(initial_ablated_mask, dtype=np.bool_, copy=True)
            self.ablation_time[self.ablated] = 0.0
            self.optical['mu_a'][self.ablated] = 0.0
            self.optical['mu_s'][self.ablated] = 0.0
            self.optical['g'][self.ablated] = 0.0
            self.optical['n'][self.ablated] = 1.0
            self.k_grid[self.ablated] = self.k_void
            self.rho_cp_grid[self.ablated] = self.rho_cp_void
            self.T[self.ablated] = self.t_ambient
        else:
            self.ablated = np.zeros(shape, dtype=np.bool_)
        self.Q = np.zeros(shape, dtype=np.float64)
        self.arrhenius = None if arrhenius is None else (float(arrhenius[0]), float(arrhenius[1]) / R_GAS)
        self.Omega = np.zeros(shape, dtype=np.float64)
        self.T_peak = np.full(shape, self.t_ambient, dtype=np.float64)  # max temperature at epoch ends
        self.dry_state = dry_state
        if initial_dry_mask is not None:
            self.dry = np.array(initial_dry_mask, dtype=np.bool_, copy=True)
        else:
            self.dry = np.zeros(shape, dtype=np.bool_)
        self._wet = None
        if dry_state is not None:
            if isinstance(dry_state.get('rho_cp'), np.ndarray):
                dry_state['rho_cp'] = np.maximum(dry_state['rho_cp'], self.k_grid / self.alpha_max_stable)
            self._wet = (self.optical['mu_a'].copy(), self.k_grid.copy(), self.rho_cp_grid.copy())
        self.t_void = float(np.max(dry_state['t_remove'])) if dry_state is not None else self.t_abl

        # Carbonization state (pyrolysis of dry organic matrix into char)
        self.char_state = char_state
        self.chi = np.zeros(shape, dtype=np.float64)
        if char_state is not None:
            self._arrhenius_char = (float(char_state['A']), float(char_state['Ea']) / R_GAS)
            if 't_remove' in char_state:
                self.t_void = float(np.max(char_state['t_remove']))
        else:
            self._arrhenius_char = None

        self.air_gap_range = air_gap_range
        self.wavelength_nm = float(wavelength_nm)

        # 3D Spatiotemporal Multispectral Plume field
        self.plume_closure = plume_closure
        self.plume = None
        if plume_closure is not None:
            if isinstance(plume_closure, (PlumeAbsorptionField, MultispectralPlumeField)):
                self.plume = plume_closure
            elif isinstance(plume_closure, dict) and plume_closure.get('enabled', True):
                wl = float(plume_closure.get('wavelength_nm', self.wavelength_nm))
                tau_s = float(plume_closure.get('tau_linger_s', 0.00875))
                sigma_xy = float(plume_closure.get('sigma_xy_mm', plume_closure.get('sigma_mm', 0.060)))
                sigma_z = float(plume_closure.get('sigma_z_mm', plume_closure.get('sigma_mm', 0.040)))
                z_drift = float(plume_closure.get('z_drift_mm', 0.030))
                delta_mu = plume_closure.get('delta_mu_per_voxel', plume_closure.get('delta_mu'))
                mu_max = plume_closure.get('mu_a_max', plume_closure.get('mu_max'))
                p_model = str(plume_closure.get('model', 'diffusion')).lower()
                D_plume = float(plume_closure.get('D_plume', plume_closure.get('D', 2.0)))
                D_xy = float(plume_closure['D_xy']) if 'D_xy' in plume_closure else D_plume
                D_z = float(plume_closure['D_z']) if 'D_z' in plume_closure else D_plume
                v_drift = float(plume_closure.get('v_drift', plume_closure.get('vz', 80.0)))
                blast_exp = bool(plume_closure.get('blast_expansion', True))
                blast_s_xy = float(plume_closure.get('blast_sigma_xy_mm', 0.080))
                blast_s_z = float(plume_closure.get('blast_sigma_z_mm', 0.090))
                blast_bias = float(plume_closure.get('blast_z_bias_mm', -0.050))
                self.plume = MultispectralPlumeField(
                    shape=shape,
                    dx_mm=self.dx,
                    dy_mm=self.dy,
                    dz_mm=self.dz,
                    wavelength_nm=wl,
                    tau_linger_s=tau_s,
                    sigma_xy_mm=sigma_xy,
                    sigma_z_mm=sigma_z,
                    z_drift_mm=z_drift,
                    delta_mu_per_voxel=delta_mu,
                    mu_a_max=mu_max,
                    model=p_model,
                    D_plume=D_plume,
                    D_xy=D_xy,
                    D_z=D_z,
                    v_drift=v_drift,
                    blast_expansion=blast_exp,
                    blast_sigma_xy_mm=blast_s_xy,
                    blast_sigma_z_mm=blast_s_z,
                    blast_z_bias_mm=blast_bias,
                )

        if self.plume is not None and getattr(self.plume, 'model', 'separable') == 'diffusion':
            self._P_next = np.empty_like(self.plume.raw_field)
        else:
            self._P_next = None

        self.thermal_engine = str(thermal_engine).strip().lower()
        self.use_gpu_thermal = False
        self.metal_ablation_ctx = None
        if self.thermal_engine in ('auto', 'gpu') and is_metal_ablation_available() and MetalAblationContext is not None:
            try:
                plume_params = None
                if self.plume is not None and getattr(self.plume, 'model', 'separable') == 'diffusion':
                    plume_params = {
                        'D_xy': self.plume.D_xy,
                        'D_z': self.plume.D_z,
                        'vz': self.plume.v_drift,
                        'tau_linger': self.plume.tau_linger,
                    }
                self.metal_ablation_ctx = MetalAblationContext(
                    self.Nx, self.Ny, self.Nz,
                    self.dx, self.dy, self.dz,
                    self.dt_sub, self.h_eff, self.t_ambient,
                    self.t_abl, self.t_void,
                    self.arrhenius, self._arrhenius_char,
                    plume_params=plume_params,
                )
                self.use_gpu_thermal = True
            except Exception as e:
                if self.thermal_engine == 'gpu':
                    raise e
                self.use_gpu_thermal = False
        elif self.thermal_engine == 'gpu':
            raise RuntimeError("Metal GPU thermal ablation requested (thermal_engine='gpu'), but Metal is not available.")

        self._to_remove = np.empty((self.Nx, self.Ny, self.Nz), dtype=np.bool_)
        self._dummy_f64 = np.zeros(1, dtype=np.float64)
        self._ds_arrays = None
        self._cs_arrays = None
        self._cs_chi_remove = 1.0
        self._init_ds_cs_arrays()

        self.t = 0.0
        self.epoch = 0
        self.last_mc = None

    def _init_ds_cs_arrays(self):
        """Pre-cache dry_state and char_state parameters as contiguous float64 arrays for Numba."""
        if self.dry_state is not None:
            self._ds_arrays = {}
            for k in ('mu_a', 'mu_s', 'g', 'n', 'k', 'rho_cp', 't_remove'):
                if k in self.dry_state:
                    v = self.dry_state[k]
                elif k == 't_remove':
                    v = self.t_abl
                elif k in self.optical:
                    v = self.optical[k]
                else:
                    v = 0.0
                self._ds_arrays[k] = np.array([v], dtype=np.float64) if np.ndim(v) == 0 else np.ascontiguousarray(v.ravel(), dtype=np.float64)
        else:
            self._ds_arrays = None

        if self.char_state is not None:
            cs = self.char_state
            self._cs_arrays = {}
            for k in ('mu_a', 'mu_s', 'k', 'rho_cp'):
                if k in cs:
                    v = cs[k]
                    self._cs_arrays[k] = np.array([v], dtype=np.float64) if np.ndim(v) == 0 else np.ascontiguousarray(v.ravel(), dtype=np.float64)
            t_rem = cs.get('t_remove', self.dry_state.get('t_remove', self.t_abl) if self.dry_state else self.t_abl)
            self._cs_arrays['t_remove'] = np.array([t_rem], dtype=np.float64) if np.ndim(t_rem) == 0 else np.ascontiguousarray(t_rem.ravel(), dtype=np.float64)
            self._cs_chi_remove = float(cs.get('chi_remove', 1.0))
        else:
            self._cs_arrays = None
            self._cs_chi_remove = 1.0

    def step_epoch(self) -> dict:
        """Advance one coupled epoch; returns per-epoch diagnostics."""
        t_opt0 = time.time()
        delivered_power = self.power
        if delivered_power > 0.0:
            if self.plume is not None and np.any(self.plume.mu_ext_plume > 0.0):
                opt_props = {k: v.copy() for k, v in self.optical.items()}
                opt_props['mu_a'] = self.optical['mu_a'] + self.plume.mu_a_plume
                opt_props['mu_s'] = self.optical['mu_s'] + self.plume.mu_s_plume
                plume_in_void = (self.plume.mu_ext_plume > 0.001) & self.ablated
                opt_props['g'][plume_in_void] = 0.80
            else:
                opt_props = self.optical

            mc = run_voxel_mc3d(
                power=delivered_power,
                n_photons=self.n_photons,
                spot_dia=self.spot_dia,
                Lx=self.Lx, Ly=self.Ly, Lz=self.Lz,
                Nx=self.Nx, Ny=self.Ny, Nz=self.Nz,
                r_max=self.r_max,
                custom_3d_props=opt_props,
                launch_axis='z',
                beam_center=self.beam_center,
                beam_profile=self.beam_profile,
                theta_inc_deg=self.theta_inc_deg,
                phi_inc_deg=self.phi_inc_deg,
                engine=self.mc_engine,
            )
            self.last_mc = mc
            self.Q_opt = mc['Q_3d']
            self.Phi = mc['Phi_3d']
            self.Q = self.Q_opt.copy()

            # Decouple plume optical attenuation from thermal explosion in void gas:
            # Photons absorbed or scattered by plume attenuate the beam without depositing heat into void gas (Q_thermal_void = 0)
            Q_thermal = self.Q_opt.copy()
            if np.any(self.ablated):
                Q_thermal[self.ablated] = 0.0

            absorbed = float(np.sum(mc['A_3d']))
            closure = float(mc['energy_closure'])
        else:
            self.Q = np.zeros_like(self.T)
            self.Q_opt = self.Q
            self.Phi = np.zeros_like(self.T)
            Q_thermal = self.Q
            absorbed, closure = 0.0, 1.0
        t_opt = time.time() - t_opt0

        t_th0 = time.time()
        has_diff_plume = (self.plume is not None and getattr(self.plume, 'model', 'separable') == 'diffusion')
        void_mask = self.compute_void_mask() if has_diff_plume else self.ablated
        if self.use_gpu_thermal and self.metal_ablation_ctx is not None:
            plume_in = self.plume.raw_field if has_diff_plume else None
            self.metal_ablation_ctx.set_state(
                self.T, self.H_acc, self.Omega, self.chi,
                self.dry, self.ablated, Q_thermal,
                self.k_grid, self.rho_cp_grid,
                plume=plume_in,
                plume_void=void_mask
            )
            self.metal_ablation_ctx.advance_substeps(self.n_substeps, is_active=1)
            self.T, self.H_acc, self.Omega, self.chi = self.metal_ablation_ctx.get_state()
            if plume_in is not None:
                self.metal_ablation_ctx.get_plume(out_arr=self.plume.raw_field)
        else:
            T, T_next = self.T, self._T_next
            if has_diff_plume:
                P, P_next = self.plume.raw_field, self._P_next
                decay = float(np.exp(-self.dt_sub / self.plume.tau_linger)) if (self.plume.tau_linger > 0.0 and not np.isinf(self.plume.tau_linger)) else 1.0
                D_xy_p = self.plume.D_xy
                D_z_p = self.plume.D_z
                vz_p = self.plume.v_drift

            for _ in range(self.n_substeps):
                _thermal3d_step_explicit(
                    T, T_next, Q_thermal, self.k_grid, self.rho_cp_grid,
                    self.Nx, self.Ny, self.Nz, self.dx, self.dy, self.dz,
                    self.dt_sub, self.h_eff, self.t_ambient, True
                )
                _apply_latent_heat_plateau(
                    T_next, self.H_acc, self.rho_cp_grid, self.ablated, self.t_abl,
                    self.Nx, self.Ny, self.Nz, self.dry, self.t_void
                )
                if self.arrhenius is not None:
                    _accumulate_arrhenius(T_next, self.Omega, self.ablated, self.arrhenius[0],
                                          self.arrhenius[1], self.dt_sub, self.Nx, self.Ny, self.Nz)
                if self._arrhenius_char is not None:
                    _accumulate_char(T_next, self.chi, self.dry, self.ablated,
                                     self._arrhenius_char[0], self._arrhenius_char[1],
                                     self.dt_sub, self.Nx, self.Ny, self.Nz)
                if has_diff_plume:
                    _diffuse_plume_step_numba(
                        P, P_next, void_mask,
                        self.Nx, self.Ny, self.Nz,
                        self.dx, self.dy, self.dz,
                        self.dt_sub, D_xy_p, D_z_p, vz_p, decay
                    )
                    P, P_next = P_next, P

                T, T_next = T_next, T
            self.T, self._T_next = T, T_next
            if has_diff_plume:
                self.plume.raw_field[:] = P
        np.maximum(self.T_peak, self.T, out=self.T_peak)
        t_th = time.time() - t_th0

        self.t += self.dt_epoch
        self.epoch += 1

        if self.dry_state is None:
            _update_single_stage_ablation_numba(
                self.H_acc, self.h_lat, self.T, self.t_abl, self.ablated, self._to_remove
            )
            n_new = self._remove(self._to_remove)
        else:
            if self._ds_arrays is None:
                self._init_ds_cs_arrays()
            has_cs = self.char_state is not None
            has_cs_mu_a = has_cs and ('mu_a' in self._cs_arrays)
            has_cs_mu_s = has_cs and ('mu_s' in self._cs_arrays)
            has_cs_k = has_cs and ('k' in self._cs_arrays)
            has_cs_rho = has_cs and ('rho_cp' in self._cs_arrays)

            _update_phase_properties_numba(
                self.H_acc, self.h_lat, self.T, self.chi, self.dry, self.ablated,
                self.optical['mu_a'], self.optical['mu_s'], self.optical['g'], self.optical['n'],
                self.k_grid, self.rho_cp_grid,
                self._wet[0], self._wet[1], self._wet[2],
                self._ds_arrays['mu_a'], self._ds_arrays['mu_s'], self._ds_arrays['g'], self._ds_arrays['n'],
                self._ds_arrays['k'], self._ds_arrays['rho_cp'], self._ds_arrays['t_remove'],
                has_cs, has_cs_mu_a,
                self._cs_arrays['mu_a'] if has_cs_mu_a else self._dummy_f64,
                self._cs_arrays['mu_s'] if has_cs_mu_s else self._dummy_f64,
                self._cs_arrays['k'] if has_cs_k else self._dummy_f64,
                self._cs_arrays['rho_cp'] if has_cs_rho else self._dummy_f64,
                self._cs_arrays['t_remove'] if has_cs else self._dummy_f64,
                self._cs_chi_remove,
                has_cs_mu_s, has_cs_k, has_cs_rho,
                self.alpha_max_stable, self._to_remove
            )

            n_new = self._remove(self._to_remove)

        # 3D Plume field update for subsequent epoch
        newly_mask = self._to_remove
        if self.plume is not None:
            void_mask = self.compute_void_mask()
            if getattr(self.plume, 'model', 'separable') == 'diffusion':
                self.plume.sync_extinction(void_mask)
            else:
                self.plume.update(self.dt_epoch, newly_mask, void_mask)

            mid_x, mid_y = self.Nx // 2, self.Ny // 2
            plume_tau = float(np.sum(self.plume.mu_a_plume[mid_x, mid_y, :]) * self.dz)
            plume_transmission = float(np.exp(-plume_tau))
            plume_peak = float(np.max(self.plume.mu_a_plume))
            plume_mean = float(np.mean(self.plume.mu_a_plume[self.ablated])) if np.any(self.ablated) else 0.0
        else:
            plume_tau = 0.0
            plume_transmission = 1.0
            plume_peak = 0.0
            plume_mean = 0.0

        return {
            'epoch': self.epoch,
            'time_s': self.t,
            'n_new': n_new,
            'absorbed_frac': absorbed,
            'mc_energy_closure': closure,
            't_optical_s': t_opt,
            't_thermal_s': t_th,
            'plume_transmission': plume_transmission,
            'plume_tau': plume_tau,
            'plume_peak_mu_a': plume_peak,
            'plume_mean_mu_a': plume_mean,
        }

    def compute_void_mask(self) -> np.ndarray:
        """Computes open void domain for plume diffusion, including breached subelytral cavity."""
        void_mask = self.ablated.copy()
        if self.air_gap_range is not None:
            k0, k1 = self.air_gap_range
            if k0 > 0 and np.any(self.ablated[:, :, k0 - 1]):
                void_mask[:, :, k0:k1] = True
        return void_mask

    def _remove(self, newly) -> int:
        """Convert voxels in mask `newly` to void (air) optically and thermally."""
        n_new = int(np.sum(newly))
        if n_new > 0:
            self.ablated[newly] = True
            self.dry[newly] = False
            self.chi[newly] = 0.0
            self.ablation_time[newly] = self.t
            self.optical['mu_a'][newly] = 0.0
            self.optical['mu_s'][newly] = 0.0
            self.optical['g'][newly] = 0.0
            self.optical['n'][newly] = 1.0
            self.k_grid[newly] = self.k_void
            self.rho_cp_grid[newly] = self.rho_cp_void
            self.T[newly] = self.t_void
            self.H_acc[newly] = 0.0

            # If air gap range is defined, connect perforated columns through the air gap
            if self.air_gap_range is not None:
                k0, k1 = self.air_gap_range
                if k0 > 0:
                    perforated_cols = self.ablated[:, :, k0 - 1]
                    if np.any(perforated_cols):
                        for ix, iy in zip(*np.where(perforated_cols)):
                            if not self.ablated[ix, iy, k0]:
                                self.ablated[ix, iy, k0:k1] = True
                                self.ablation_time[ix, iy, k0:k1] = self.t

            # Point-source explosive plume injection at JUST the exploding voxel locations
            if self.plume is not None:
                void_mask = self.compute_void_mask()
                self.plume.inject(newly, void_mask=void_mask)
        return n_new

    def centerline_depth_mm(self, ix: int = None, iy: int = None, k_surface: int = 0) -> float:
        """Depth of the deepest ablated voxel bottom face along a column (default: grid centre, below k_surface)."""
        ix = self.Nx // 2 if ix is None else ix
        iy = self.Ny // 2 if iy is None else iy
        idx = np.where(self.ablated[ix, iy, k_surface:])[0]
        return float((idx.max() + 1) * self.dz) if idx.size else 0.0
