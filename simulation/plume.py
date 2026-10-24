"""
3D Multispectral Plume Optics & Extinction Field Module.

Implements wavelength-dependent spatiotemporal optical extinction for laser-induced ablation plumes:
- Ejecta mass density with anisotropic 3D Gaussian dispersion kernel & upward recoil bias (-z_drift)
- Rapid temporal exponential clearance with millisecond aerosol clearance time tau_linger (8.75 ms)
- Multispectral mass extinction cross-sections: CO2 (gas/steam), SWIR, NIR, Blue (Mie/soot)
- Smooth asymptotic saturation dynamic: mu_ext = mu_max * tanh(raw_field / mu_max)
- Strict confinement to void/crater/ambient open space
"""

import numpy as np


from numba import njit, prange


@njit(parallel=True, fastmath=True)
def _diffuse_plume_step_numba(
    C: np.ndarray,
    C_new: np.ndarray,
    ablated: np.ndarray,
    Nx: int,
    Ny: int,
    Nz: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    D_xy: float,
    D_z: float,
    vz: float,
    decay: float,
):
    """
    Explicit 3D finite-difference anisotropic diffusion & drift for plume in the void cavity.
    Enforces:
    - Domain Rule 1: Plume strictly confined to void/cavity (solid voxels = 0.0)
    - Domain Rule 2: Open boundary clearing (plume = 0.0 at outer grid walls)
    - Impermeable Neumann reflection at intact solid tissue crater walls
    """
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    idz2 = 1.0 / (dz * dz)

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # 1. Solid intact tissue has zero plume
                if not ablated[i, j, k]:
                    C_new[i, j, k] = 0.0
                    continue

                # 2. Outer domain boundaries: clears to ambient open space
                if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1 or k == 0 or k == Nz - 1:
                    C_new[i, j, k] = 0.0
                    continue

                C_c = C[i, j, k]

                # X-flux (faces w and e; zero-flux into solid tissue)
                C_w = C[i - 1, j, k] if ablated[i - 1, j, k] else C_c
                C_e = C[i + 1, j, k] if ablated[i + 1, j, k] else C_c

                # Y-flux (faces s and n)
                C_s = C[i, j - 1, k] if ablated[i, j - 1, k] else C_c
                C_n = C[i, j + 1, k] if ablated[i, j + 1, k] else C_c

                # Z-flux (faces b and t; upward recoil moves towards surface k=0)
                C_b = C[i, j, k - 1] if ablated[i, j, k - 1] else C_c
                C_t = C[i, j, k + 1] if ablated[i, j, k + 1] else C_c

                diff = (D_xy * ((C_w - 2.0 * C_c + C_e) * idx2 +
                                (C_s - 2.0 * C_c + C_n) * idy2) +
                        D_z  * ((C_b - 2.0 * C_c + C_t) * idz2))

                adv = -vz * (C_c - C_t) / dz if vz > 0.0 else 0.0

                c_val = (C_c + dt * (diff + adv)) * decay
                C_new[i, j, k] = max(c_val, 0.0)


@njit(fastmath=True)
def _convolve_separable_plume_numba(
    raw_field: np.ndarray,
    impulse: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    delta_mu: float,
    x0_out: int,
    y0_out: int,
    z0_out: int,
):
    """
    Tap-based separable 3D convolution of an ablation impulse bounding box with anisotropic Gaussian recoil kernel.
    Deposits delta_mu * (impulse * (kx (x) ky (x) kz)) directly into raw_field with zero branching in inner loops.
    """
    Lx, Ly, Lz = impulse.shape
    Nx, Ny, Nz = raw_field.shape
    len_kx = len(kx)
    len_ky = len(ky)
    len_kz = len(kz)

    # Pass 1: 1D convolution along x -> intermediate buffer of shape (Lx + len_kx - 1, Ly, Lz)
    OutX = Lx + len_kx - 1
    tmp1 = np.zeros((OutX, Ly, Lz), dtype=np.float32)
    for kxi in range(len_kx):
        w = kx[kxi]
        for ix in range(Lx):
            ox = ix + kxi
            for y in range(Ly):
                for z in range(Lz):
                    tmp1[ox, y, z] += impulse[ix, y, z] * w

    # Pass 2: 1D convolution along y -> intermediate buffer of shape (OutX, Ly + len_ky - 1, Lz)
    OutY = Ly + len_ky - 1
    tmp2 = np.zeros((OutX, OutY, Lz), dtype=np.float32)
    for kyj in range(len_ky):
        w = ky[kyj]
        for ox in range(OutX):
            for iy in range(Ly):
                oy = iy + kyj
                for z in range(Lz):
                    tmp2[ox, oy, z] += tmp1[ox, iy, z] * w

    # Pass 3: 1D convolution along z and in-place accumulate directly into raw_field
    for kzk in range(len_kz):
        w = kz[kzk] * delta_mu
        for ox in range(OutX):
            gx = x0_out + ox
            if 0 <= gx < Nx:
                for oy in range(OutY):
                    gy = y0_out + oy
                    if 0 <= gy < Ny:
                        for iz in range(Lz):
                            gz = z0_out + iz + kzk
                            if 0 <= gz < Nz:
                                raw_field[gx, gy, gz] += tmp2[ox, oy, iz] * w


class MultispectralPlumeField:
    """
    Multispectral spatiotemporal optical extinction field for laser-induced ablation plumes.

    Supports:
    1. Direct 2-Channel 3D Diffusion (`model='diffusion'`):
       - Point-source explosive injection at JUST the exploding voxel locations
       - 3D diffusion and upward advection strictly inside the void cavity
       - Dirichlet clearance at open domain boundaries
    2. Separable Kernel Dispersion (`model='separable'`):
       - 3D separable Gaussian convolution for backward compatibility
    """

    # Mass extinction cross-section scaling factors (1.5x baseline absorption + wavelength-dependent Mie scattering)
    EXTINCTION_MAP = {
        10600: {'mu_a_max': 0.15,  'mu_s_max': 0.05, 'delta_mu': 0.075}, # CO2: steam/gas dominant
        1940:  {'mu_a_max': 0.375, 'mu_s_max': 0.15, 'delta_mu': 0.18},  # Thulium: water vapor & droplets
        1470:  {'mu_a_max': 0.45,  'mu_s_max': 0.20, 'delta_mu': 0.22},  # SWIR: aerosol + water
        1064:  {'mu_a_max': 0.675, 'mu_s_max': 0.60, 'delta_mu': 0.33},  # NIR: soot particulate scatter
        532:   {'mu_a_max': 1.275, 'mu_s_max': 0.80, 'delta_mu': 0.63},  # Green: particulate + Rayleigh
        450:   {'mu_a_max': 1.80,  'mu_s_max': 1.00, 'delta_mu': 0.90},  # Blue: Mie/Rayleigh soot
        320:   {'mu_a_max': 2.40,  'mu_s_max': 1.20, 'delta_mu': 1.20},  # UV-A: Rayleigh soot
    }

    def __init__(
        self,
        shape: tuple,
        dx_mm: float = 0.030,
        dy_mm: float = 0.030,
        dz_mm: float = 0.020,
        wavelength_nm: float = None,
        tau_linger_s: float = 0.00875,
        sigma_xy_mm: float = 0.060,
        sigma_z_mm: float = 0.040,
        z_drift_mm: float = 0.030,
        delta_mu_per_voxel: float = None,
        mu_a_max: float = None,
        sigma_mm: float = None,
        model: str = 'separable',
        D_plume: float = 2.0,
        D_xy: float = None,
        D_z: float = None,
        v_drift: float = 80.0,
        blast_expansion: bool = True,
        blast_sigma_xy_mm: float = 0.080,
        blast_sigma_z_mm: float = 0.090,
        blast_z_bias_mm: float = -0.050,
    ):
        self.shape = shape
        self.dx = float(dx_mm)
        self.dy = float(dy_mm)
        self.dz = float(dz_mm)
        self.tau_linger = float(tau_linger_s)
        self.wavelength = float(wavelength_nm) if wavelength_nm is not None else 10600.0
        self.model = str(model).lower()
        self.D_plume = float(D_plume)
        self.D_xy = float(D_xy) if D_xy is not None else self.D_plume
        self.D_z = float(D_z) if D_z is not None else self.D_plume
        self.v_drift = float(v_drift)
        self.blast_expansion = bool(blast_expansion)
        self.blast_sigma_xy = float(blast_sigma_xy_mm)
        self.blast_sigma_z = float(blast_sigma_z_mm)
        self.blast_z_bias = float(blast_z_bias_mm)

        if sigma_mm is not None:
            self.sigma_xy = float(sigma_mm)
            self.sigma_z = float(sigma_mm)
            self.z_drift = float(z_drift_mm) if z_drift_mm != 0.030 else 0.0
        else:
            self.sigma_xy = float(sigma_xy_mm)
            self.sigma_z = float(sigma_z_mm)
            self.z_drift = float(z_drift_mm)

        # Pull wavelength-specific extinction scaling, allowing explicit overrides
        wl_key = int(round(self.wavelength))
        cfg = self.EXTINCTION_MAP.get(wl_key, {'mu_a_max': 0.30, 'mu_s_max': 0.15, 'delta_mu': 0.15})

        self.mu_a_max = float(cfg.get('mu_a_max', cfg.get('mu_max', 0.30)))
        self.mu_s_max = float(cfg.get('mu_s_max', 0.25 * self.mu_a_max))
        self.mu_max = self.mu_a_max + self.mu_s_max

        if mu_a_max is not None:
            self.mu_a_max = float(mu_a_max)
            self.mu_max = self.mu_a_max + self.mu_s_max

        if delta_mu_per_voxel is not None:
            self.delta_mu = float(delta_mu_per_voxel)
        else:
            self.delta_mu = float(cfg.get('delta_mu', 0.15))

        # Preallocate mass / raw extinction and saturated output fields
        self.raw_field = np.zeros(shape, dtype=np.float32)
        self.mu_ext_plume = np.zeros(shape, dtype=np.float32)
        self._tmp_field = np.zeros(shape, dtype=np.float32)

        # Precompute localized sub-kernel footprint (3-sigma bounds + drift)
        self.rx = max(1, int(np.ceil(3.0 * self.sigma_xy / self.dx)))
        self.ry = max(1, int(np.ceil(3.0 * self.sigma_xy / self.dy)))
        self.rz = max(1, int(np.ceil((3.0 * self.sigma_z + self.z_drift) / self.dz)))

        # Precompute 1D separable components
        self.kx_1d = np.exp(-((np.arange(-self.rx, self.rx + 1, dtype=np.float32) * self.dx)**2) / (2.0 * self.sigma_xy**2)).astype(np.float32)
        self.ky_1d = np.exp(-((np.arange(-self.ry, self.ry + 1, dtype=np.float32) * self.dy)**2) / (2.0 * self.sigma_xy**2)).astype(np.float32)
        self.kz_1d = np.exp(-(((np.arange(-self.rz, self.rz + 1, dtype=np.float32) * self.dz) + self.z_drift)**2) / (2.0 * self.sigma_z**2)).astype(np.float32)

        # Retain 3D kernel for full backward compatibility
        self.kernel = (self.kx_1d[:, None, None] * self.ky_1d[None, :, None] * self.kz_1d[None, None, :]).astype(np.float32)
        k_max = float(np.max(self.kernel))
        if k_max > 0.0:
            self.kernel /= k_max
            self.kz_1d /= k_max

        # Precompute explosive gas blast kernel (depletes source voxels and disperses into cavity)
        self.rx_b = max(1, int(np.ceil(2.5 * self.blast_sigma_xy / self.dx)))
        self.ry_b = max(1, int(np.ceil(2.5 * self.blast_sigma_xy / self.dy)))
        self.rz_b = max(1, int(np.ceil((2.5 * self.blast_sigma_z + abs(self.blast_z_bias)) / self.dz)))

        self.kx_b = np.exp(-((np.arange(-self.rx_b, self.rx_b + 1, dtype=np.float32) * self.dx)**2) / (2.0 * self.blast_sigma_xy**2)).astype(np.float32)
        self.ky_b = np.exp(-((np.arange(-self.ry_b, self.ry_b + 1, dtype=np.float32) * self.dy)**2) / (2.0 * self.blast_sigma_xy**2)).astype(np.float32)
        self.kz_b = np.exp(-(((np.arange(-self.rz_b, self.rz_b + 1, dtype=np.float32) * self.dz) - self.blast_z_bias)**2) / (2.0 * self.blast_sigma_z**2)).astype(np.float32)

        k_max_b = float(np.max(self.kx_b) * np.max(self.ky_b) * np.max(self.kz_b))
        if k_max_b > 0.0:
            scale_b = float(0.35 / np.cbrt(k_max_b))
            self.kx_b *= scale_b
            self.ky_b *= scale_b
            self.kz_b *= scale_b

    def set_wavelength(self, wavelength_nm: float):
        """Update wavelength-dependent plume optical properties."""
        self.wavelength = float(wavelength_nm)
        wl_key = int(round(self.wavelength))
        cfg = self.EXTINCTION_MAP.get(wl_key, {'mu_a_max': 0.30, 'mu_s_max': 0.15, 'delta_mu': 0.15})
        self.mu_a_max = float(cfg.get('mu_a_max', cfg.get('mu_max', 0.30)))
        self.mu_s_max = float(cfg.get('mu_s_max', 0.25 * self.mu_a_max))
        self.mu_max = self.mu_a_max + self.mu_s_max
        self.delta_mu = float(cfg.get('delta_mu', 0.15))

    @property
    def mu_a_plume(self) -> np.ndarray:
        """Wavelength-dependent absorption coefficient field of the plume."""
        ratio = self.mu_a_max / max(1e-6, self.mu_max)
        return self.mu_ext_plume * ratio

    @property
    def mu_s_plume(self) -> np.ndarray:
        """Wavelength-dependent scattering coefficient field of the plume."""
        ratio = self.mu_s_max / max(1e-6, self.mu_max)
        return self.mu_ext_plume * ratio

    def inject(self, newly_mask: np.ndarray, void_mask: np.ndarray = None):
        """
        Explosive plume injection:
        - When blast_expansion is True (default in diffusion mode):
          Vaporized solid mass immediately expands into the void cavity across blast radius
          (R_blast ~ 0.12 - 0.15 mm with upward recoil bias -z).
          Source voxels at the crater wall immediately deplete and disperse into the cavity.
        - When blast_expansion is False:
          Single-voxel point injection on the exploding voxels.
        - Strictly confined to void space (never penetrates solid tissue).
        """
        if newly_mask is None or not np.any(newly_mask):
            return

        if self.blast_expansion:
            wx = np.where(np.any(newly_mask, axis=(1, 2)))[0]
            wy = np.where(np.any(newly_mask, axis=(0, 2)))[0]
            wz = np.where(np.any(newly_mask, axis=(0, 1)))[0]
            x_min, x_max = wx[0], wx[-1] + 1
            y_min, y_max = wy[0], wy[-1] + 1
            z_min, z_max = wz[0], wz[-1] + 1
            impulse = newly_mask[x_min:x_max, y_min:y_max, z_min:z_max].astype(np.float32)

            _convolve_separable_plume_numba(
                self.raw_field,
                impulse,
                self.kx_b,
                self.ky_b,
                self.kz_b,
                self.delta_mu,
                int(x_min - self.rx_b),
                int(y_min - self.ry_b),
                int(z_min - self.rz_b),
            )
            if void_mask is not None:
                self.raw_field[~void_mask] = 0.0
        else:
            self.raw_field[newly_mask] += self.delta_mu
            if void_mask is not None:
                self.raw_field[~void_mask] = 0.0

    def diffuse_substep(self, void_mask: np.ndarray, dt_sub: float):
        """
        Executes one explicit 3D diffusion sub-step in the void domain.
        """
        decay = float(np.exp(-dt_sub / self.tau_linger)) if (self.tau_linger > 0.0 and not np.isinf(self.tau_linger)) else 1.0
        _diffuse_plume_step_numba(
            self.raw_field,
            self._tmp_field,
            void_mask,
            self.shape[0], self.shape[1], self.shape[2],
            self.dx, self.dy, self.dz,
            dt_sub,
            self.D_xy,
            self.D_z,
            self.v_drift,
            decay,
        )
        self.raw_field[:] = self._tmp_field

    def sync_extinction(self, void_mask: np.ndarray = None):
        """
        Confines plume strictly to void space, applies smooth asymptotic saturation,
        and updates mu_ext_plume.
        """
        if void_mask is not None:
            self.raw_field[~void_mask] = 0.0
        if self.mu_max is not None and self.mu_max > 0.0:
            np.tanh(self.raw_field / self.mu_max, out=self.mu_ext_plume)
            self.mu_ext_plume *= self.mu_max
        else:
            self.mu_ext_plume[:] = self.raw_field
        if void_mask is not None:
            self.mu_ext_plume[~void_mask] = 0.0
        return self.mu_ext_plume

    def update(self, dt_s: float, newly_ablated_mask: np.ndarray, void_mask: np.ndarray) -> np.ndarray:
        """
        Updates the plume extinction field.
        In 'diffusion' mode: point injection + 3D void diffusion + boundary clearance.
        In 'separable' mode: clearance decay + fast separable kernel deposition.
        """
        # Synchronize raw_field if mu_ext_plume was manually populated from outside
        if not np.any(self.raw_field) and np.any(self.mu_ext_plume):
            self.raw_field[:] = self.mu_ext_plume

        if self.model == 'diffusion':
            # Point source injection at just the exploding voxel locations
            self.inject(newly_ablated_mask)

            # Sub-stepped 3D void diffusion
            if dt_s > 0.0:
                dt_sub = 5e-5
                n_sub = max(1, int(round(dt_s / dt_sub)))
                actual_dt_sub = dt_s / n_sub
                for _ in range(n_sub):
                    self.diffuse_substep(void_mask, actual_dt_sub)

            return self.sync_extinction(void_mask)

        # Legacy separable kernel path
        # 1. Exponential clearance decay
        if self.tau_linger > 0.0 and dt_s > 0.0:
            self.raw_field *= float(np.exp(-dt_s / self.tau_linger))

        # 2. Vectorized / Parallel Separable Kernel Splatting
        any_x = np.any(newly_ablated_mask, axis=(1, 2))
        if np.any(any_x):
            any_y = np.any(newly_ablated_mask, axis=(0, 2))
            any_z = np.any(newly_ablated_mask, axis=(0, 1))
            wx = np.where(any_x)[0]
            wy = np.where(any_y)[0]
            wz = np.where(any_z)[0]
            x_min, x_max = wx[0], wx[-1] + 1
            y_min, y_max = wy[0], wy[-1] + 1
            z_min, z_max = wz[0], wz[-1] + 1
            impulse = newly_ablated_mask[x_min:x_max, y_min:y_max, z_min:z_max].astype(np.float32)

            _convolve_separable_plume_numba(
                self.raw_field,
                impulse,
                self.kx_1d,
                self.ky_1d,
                self.kz_1d,
                self.delta_mu,
                int(x_min - self.rx),
                int(y_min - self.ry),
                int(z_min - self.rz),
            )

        # 3. Confine plume strictly to cavity and ambient void
        self.raw_field *= void_mask

        # 4. Smooth asymptotic saturation dynamic
        if self.mu_max is not None and self.mu_max > 0.0:
            np.tanh(self.raw_field / self.mu_max, out=self.mu_ext_plume)
            self.mu_ext_plume *= self.mu_max
        else:
            self.mu_ext_plume[:] = self.raw_field

        return self.mu_ext_plume

    def reset(self):
        """Clear the plume absorption/extinction field."""
        self.raw_field.fill(0.0)
        self.mu_ext_plume.fill(0.0)
        self._tmp_field.fill(0.0)


# Backward-compatible alias
PlumeAbsorptionField = MultispectralPlumeField
