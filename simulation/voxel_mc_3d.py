"""
3D Cartesian Voxel Monte Carlo (MC3D) Engine Module.
Simulates stochastic photon transport through arbitrary 3D voxelized media:
- Full 3D Cartesian grid (x, y, z) with voxel-resolved optical properties (mu_a, mu_s, g, n)
- Supports arbitrary angles of incidence (theta_inc, phi_inc) for oblique laser illumination
- Supports arbitrary surface/internal cavities, void crater cells, and non-planar geometry
- Optical-depth voxel marching: exact free paths across heterogeneous voxel boundaries and
  exact (single-chord) ballistic transport through void/ablated cells
- Angle-resolved Fresnel reflection / total internal reflection and Snell refraction at every
  refractive-index step (entrance face, internal interfaces, crater walls, exit through the top)
- Numba-parallel (prange) photon propagation core with Henyey-Greenstein scattering
- Outputs 3D absorbed energy density A(x, y, z), volumetric heat source Q(x, y, z) [W/mm^3],
  fluence rate Phi(x, y, z) [1/mm^2], and spectrophotometric energy conservation (R, A, T).
"""

import time
import warnings
import numpy as np
from numba import njit, prange, get_num_threads

try:
    from simulation.voxel_mc_metal import is_metal_available as _is_metal_available, run_voxel_mc_metal
except ImportError:
    def _is_metal_available() -> bool:
        return False
    run_voxel_mc_metal = None

MC3D_SCORING_MEMORY_BUDGET_BYTES = 512 * 1024 ** 2
MC3D_ENERGY_CLOSURE_TOL = 0.01
_MAX_EVENTS_PER_PHOTON = 10_000_000


@njit(fastmath=True, inline='always')
def _fresnel_unpolarized(n1, n2, cos_i):
    """Unpolarized Fresnel reflectance for a ray going from index n1 into n2."""
    if abs(n1 - n2) < 1e-9:
        return 0.0
    cos_i = min(1.0, abs(cos_i))
    if cos_i > 0.999999:
        r = (n1 - n2) / (n1 + n2)
        return r * r
    sin_t2 = (n1 / n2) * (n1 / n2) * (1.0 - cos_i * cos_i)
    if sin_t2 >= 1.0:
        return 1.0  # total internal reflection
    cos_t = np.sqrt(1.0 - sin_t2)
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)
    return 0.5 * (rs * rs + rp * rp)


@njit(fastmath=True, inline='always')
def _refract_axis(ux, uy, uz, axis, eta):
    """Snell refraction across an axis-aligned face (normal along `axis`), eta = n1/n2."""
    if axis == 0:
        sgn = 1.0 if ux > 0.0 else -1.0
        uy *= eta
        uz *= eta
        ux = sgn * np.sqrt(max(0.0, 1.0 - uy * uy - uz * uz))
    elif axis == 1:
        sgn = 1.0 if uy > 0.0 else -1.0
        ux *= eta
        uz *= eta
        uy = sgn * np.sqrt(max(0.0, 1.0 - ux * ux - uz * uz))
    else:
        sgn = 1.0 if uz > 0.0 else -1.0
        ux *= eta
        uy *= eta
        uz = sgn * np.sqrt(max(0.0, 1.0 - ux * ux - uy * uy))
    return ux, uy, uz


@njit(fastmath=True, inline='always')
def _dist_to_voxel_exit(x, y, z, ux, uy, uz, ix, iy, iz, dx, dy, dz):
    """Exact ray/box distance to the exit face of voxel (ix, iy, iz): (s, axis, step_dir)."""
    big = 1e30
    if ux > 1e-15:
        sx = ((ix + 1) * dx - x) / ux
    elif ux < -1e-15:
        sx = (ix * dx - x) / ux
    else:
        sx = big
    if uy > 1e-15:
        sy = ((iy + 1) * dy - y) / uy
    elif uy < -1e-15:
        sy = (iy * dy - y) / uy
    else:
        sy = big
    if uz > 1e-15:
        sz = ((iz + 1) * dz - z) / uz
    elif uz < -1e-15:
        sz = (iz * dz - z) / uz
    else:
        sz = big
    if sx <= sy and sx <= sz:
        return max(sx, 0.0), 0, (1 if ux > 0.0 else -1)
    elif sy <= sz:
        return max(sy, 0.0), 1, (1 if uy > 0.0 else -1)
    return max(sz, 0.0), 2, (1 if uz > 0.0 else -1)


@njit(fastmath=True, inline='always')
def _sample_beam_offsets(sigma_axis, beam_radius, beam_profile):
    """
    Transverse launch offsets (u1, u2) [mm].
    profile 0: Gaussian, per-axis sigma = sigma_axis (1/e^2 radius w0 = 2 sigma)
    profile 1: uniform top-hat disk of radius beam_radius
    profile 2: disk of radius beam_radius with exp(-2 r/R) areal weighting (Kim et al. 2023 source, literal reading)
    profile 3: disk of radius beam_radius with exp(-2 (r/R)^2) areal weighting (Gaussian truncated at its 1/e^2 radius)
    """
    if beam_profile == 1:
        r = beam_radius * np.sqrt(np.random.random())
        a = 2.0 * np.pi * np.random.random()
        return r * np.cos(a), r * np.sin(a)
    if beam_profile == 2 or beam_profile == 3:
        while True:
            rn = np.sqrt(np.random.random())
            if np.random.random() < (np.exp(-2.0 * rn) if beam_profile == 2 else np.exp(-2.0 * rn * rn)):
                a = 2.0 * np.pi * np.random.random()
                return beam_radius * rn * np.cos(a), beam_radius * rn * np.sin(a)
    return np.random.normal(0.0, sigma_axis), np.random.normal(0.0, sigma_axis)


@njit(fastmath=True)
def _trace_one_photon(
    sigma_axis, beam_radius, beam_profile,
    init_ux, init_uy, init_uz,
    cos_ph, sin_ph, cos_th_safe,
    x_center, y_center, z_center,
    Nx, Ny, Nz,
    dx, dy, dz,
    Lx, Ly, Lz,
    mu_a_grid, mu_s_grid, g_grid, n_grid,
    n_ambient,
    r_max_sq,
    launch_axis,
    A_grid, L_grid, tally, lateral_escape_z
):
    """
    Trace a single photon packet. tally = [specular, diffuse, transmitted, lateral].

    Transport uses optical-depth marching: a dimensionless optical depth
    tau = -ln(xi) is sampled once per free flight and consumed voxel by voxel
    (tau -= mu_t * chord), so free paths are exact in heterogeneous media and
    void voxels (mu_t = 0) are crossed in a single exact chord. Every internal
    face with a refractive-index step applies angle-resolved Fresnel
    reflection / TIR and Snell refraction; the entrance face (depth index 0)
    does the same against n_ambient. The far depth face and lateral faces are
    treated as index-matched (absorbing) truncation boundaries.
    """
    u1, u2 = _sample_beam_offsets(sigma_axis, beam_radius, beam_profile)

    if launch_axis == 0:
        depth_axis = 2
        x = x_center + (u1 / cos_th_safe) * cos_ph - u2 * sin_ph
        y = y_center + (u1 / cos_th_safe) * sin_ph + u2 * cos_ph
        z = 0.0
        if x < 0.0 or x >= Lx or y < 0.0 or y >= Ly or (
                r_max_sq > 0.0 and (x - x_center) ** 2 + (y - y_center) ** 2 >= r_max_sq):
            tally[3] += 1.0
            lateral_escape_z[0] += 1.0
            return
        ix = min(int(x / dx), Nx - 1)
        iy = min(int(y / dy), Ny - 1)
        iz = 0
    else:
        depth_axis = 0
        x = 0.0
        y = y_center + (u1 / cos_th_safe) * cos_ph - u2 * sin_ph
        z = z_center + (u1 / cos_th_safe) * sin_ph + u2 * cos_ph
        if y < 0.0 or y >= Ly or z < 0.0 or z >= Lz or (
                r_max_sq > 0.0 and (y - y_center) ** 2 + (z - z_center) ** 2 >= r_max_sq):
            tally[3] += 1.0
            lateral_escape_z[0] += 1.0
            return
        ix = 0
        iy = min(int(y / dy), Ny - 1)
        iz = min(int(z / dz), Nz - 1)

    ux = init_ux
    uy = init_uy
    uz = init_uz

    # Entrance face: deterministic specular split + Snell refraction into the first voxel
    n_in = n_grid[ix, iy, iz]
    cos_in = uz if depth_axis == 2 else ux
    r_entry = _fresnel_unpolarized(n_ambient, n_in, cos_in)
    tally[0] += r_entry
    weight = 1.0 - r_entry
    if weight <= 0.0:
        return
    if abs(n_in - n_ambient) > 1e-9:
        ux, uy, uz = _refract_axis(ux, uy, uz, depth_axis, n_ambient / n_in)

    tau = -np.log(max(np.random.random(), 1e-12))

    for _ in range(_MAX_EVENTS_PER_PHOTON):
        mu_a = mu_a_grid[ix, iy, iz]
        mu_t = mu_a + mu_s_grid[ix, iy, iz]
        s_exit, axis, sdir = _dist_to_voxel_exit(x, y, z, ux, uy, uz, ix, iy, iz, dx, dy, dz)

        if mu_t > 1e-12 and mu_t * s_exit >= tau:
            # ---- Interaction inside the current voxel ----
            s_col = tau / mu_t
            L_grid[ix, iy, iz] += weight * s_col
            x += s_col * ux
            y += s_col * uy
            z += s_col * uz

            if r_max_sq > 0.0:
                if launch_axis == 0:
                    rr = (x - x_center) ** 2 + (y - y_center) ** 2
                else:
                    rr = (y - y_center) ** 2 + (z - z_center) ** 2
                if rr >= r_max_sq:
                    tally[3] += weight
                    lateral_escape_z[iz if launch_axis == 0 else ix] += weight
                    return

            dw = weight * (mu_a / mu_t)
            weight -= dw
            A_grid[ix, iy, iz] += dw

            if weight < 1e-4:
                if np.random.random() < 0.1:
                    weight *= 10.0
                else:
                    return

            # Henyey-Greenstein anisotropic scattering deflection
            g_loc = g_grid[ix, iy, iz]
            xi_g = np.random.random()
            if np.abs(g_loc) < 1e-4:
                cos_theta = 2.0 * xi_g - 1.0
            else:
                temp = (1.0 - g_loc * g_loc) / (1.0 - g_loc + 2.0 * g_loc * xi_g)
                cos_theta = (1.0 + g_loc * g_loc - temp * temp) / (2.0 * g_loc)
                cos_theta = max(-1.0, min(1.0, cos_theta))

            sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
            psi = 2.0 * np.pi * np.random.random()
            cos_psi = np.cos(psi)
            sin_psi = np.sin(psi)

            if np.abs(uz) > 0.99999:
                sign_z = 1.0 if uz > 0.0 else -1.0
                ux = sin_theta * cos_psi
                uy = sin_theta * sin_psi
                uz = cos_theta * sign_z
            else:
                denom = np.sqrt(max(1.0 - uz * uz, 1e-12))
                n_ux = sin_theta * (ux * uz * cos_psi - uy * sin_psi) / denom + ux * cos_theta
                n_uy = sin_theta * (uy * uz * cos_psi + ux * sin_psi) / denom + uy * cos_theta
                n_uz = -sin_theta * cos_psi * denom + uz * cos_theta
                ux = n_ux
                uy = n_uy
                uz = n_uz

            norm_u = np.sqrt(ux * ux + uy * uy + uz * uz)
            if norm_u > 1e-12:
                ux /= norm_u
                uy /= norm_u
                uz /= norm_u

            tau = -np.log(max(np.random.random(), 1e-12))
            continue

        # ---- Traverse the full voxel chord and land exactly on the exit face ----
        tau -= mu_t * s_exit
        L_grid[ix, iy, iz] += weight * s_exit
        x += s_exit * ux
        y += s_exit * uy
        z += s_exit * uz
        jx = ix
        jy = iy
        jz = iz
        if axis == 0:
            jx += sdir
            x = (ix + 1) * dx if sdir > 0 else ix * dx
        elif axis == 1:
            jy += sdir
            y = (iy + 1) * dy if sdir > 0 else iy * dy
        else:
            jz += sdir
            z = (iz + 1) * dz if sdir > 0 else iz * dz

        if axis == depth_axis:
            jd = jz if depth_axis == 2 else jx
            n_depth = Nz if depth_axis == 2 else Nx
            if jd < 0:
                # Leaving through the entrance face: Fresnel / TIR against ambient
                cos_out = uz if depth_axis == 2 else ux
                r_out = _fresnel_unpolarized(n_grid[ix, iy, iz], n_ambient, cos_out)
                if r_out > 0.0 and np.random.random() < r_out:
                    if depth_axis == 2:
                        uz = -uz
                    else:
                        ux = -ux
                    continue
                tally[1] += weight
                return
            if jd >= n_depth:
                tally[2] += weight
                return
        elif jx < 0 or jx >= Nx or jy < 0 or jy >= Ny or jz < 0 or jz >= Nz:
            tally[3] += weight
            lateral_escape_z[iz if launch_axis == 0 else ix] += weight
            return

        if r_max_sq > 0.0:
            if launch_axis == 0:
                rr = (x - x_center) ** 2 + (y - y_center) ** 2
            else:
                rr = (y - y_center) ** 2 + (z - z_center) ** 2
            if rr >= r_max_sq:
                tally[3] += weight
                lateral_escape_z[iz if launch_axis == 0 else ix] += weight
                return

        # Internal refractive-index step: Fresnel reflection / TIR, else Snell refraction
        n1 = n_grid[ix, iy, iz]
        n2 = n_grid[jx, jy, jz]
        if abs(n1 - n2) > 1e-9:
            cos_face = ux if axis == 0 else (uy if axis == 1 else uz)
            r_int = _fresnel_unpolarized(n1, n2, cos_face)
            if np.random.random() < r_int:
                if axis == 0:
                    ux = -ux
                elif axis == 1:
                    uy = -uy
                else:
                    uz = -uz
                continue
            ux, uy, uz = _refract_axis(ux, uy, uz, axis, n1 / n2)

        ix = jx
        iy = jy
        iz = jz


@njit(parallel=True, fastmath=True)
def _mc3d_trace_photons(
    n_photons,
    n_chunks,
    sigma_axis,
    theta_inc_rad,
    phi_inc_rad,
    x_center,
    y_center,
    z_center,
    Nx, Ny, Nz,
    dx, dy, dz,
    Lx, Ly, Lz,
    mu_a_grid,
    mu_s_grid,
    g_grid,
    n_grid,
    n_ambient=1.0,
    r_max=0.0,
    launch_axis=0,
    beam_radius=0.0,
    beam_profile=0
):
    """
    Numba-parallel 3D Cartesian Monte Carlo core.
    Photons are split into `n_chunks` independent batches traced with prange; each
    batch scores into private A/L/tally arrays (no write races), reduced by the caller.
    Supports launch along +z (launch_axis=0) or along +x (launch_axis=1).

    sigma_axis is the per-axis standard deviation of the Gaussian beam (w0 / 2 for
    a beam with 1/e^2 intensity radius w0).

    Returns per-chunk arrays: A [n_chunks, Nx, Ny, Nz], L [n_chunks, Nx, Ny, Nz],
    tally [n_chunks, 4] = (specular, diffuse, transmitted, lateral) and
    lateral_escape_z [n_chunks, N_depth].
    """
    N_depth_dim = Nz if launch_axis == 0 else Nx
    A_loc = np.zeros((n_chunks, Nx, Ny, Nz), dtype=np.float64)
    L_loc = np.zeros((n_chunks, Nx, Ny, Nz), dtype=np.float64)
    tally = np.zeros((n_chunks, 4), dtype=np.float64)
    lat_z = np.zeros((n_chunks, N_depth_dim), dtype=np.float64)

    sin_th = np.sin(theta_inc_rad)
    cos_th = np.cos(theta_inc_rad)
    sin_ph = np.sin(phi_inc_rad)
    cos_ph = np.cos(phi_inc_rad)
    cos_th_safe = max(cos_th, 0.05)

    if launch_axis == 0:
        init_ux = sin_th * cos_ph
        init_uy = sin_th * sin_ph
        init_uz = cos_th
    else:
        init_ux = cos_th
        init_uy = sin_th * cos_ph
        init_uz = sin_th * sin_ph

    r_max_sq = r_max * r_max if r_max > 0.0 else 0.0

    for c in prange(n_chunks):
        p_start = (c * n_photons) // n_chunks
        p_end = ((c + 1) * n_photons) // n_chunks
        A_c = A_loc[c]
        L_c = L_loc[c]
        t_c = tally[c]
        z_c = lat_z[c]
        for _p in range(p_start, p_end):
            _trace_one_photon(
                sigma_axis, beam_radius, beam_profile,
                init_ux, init_uy, init_uz,
                cos_ph, sin_ph, cos_th_safe,
                x_center, y_center, z_center,
                Nx, Ny, Nz,
                dx, dy, dz,
                Lx, Ly, Lz,
                mu_a_grid, mu_s_grid, g_grid, n_grid,
                n_ambient,
                r_max_sq,
                launch_axis,
                A_c, L_c, t_c, z_c
            )

    return A_loc, L_loc, tally, lat_z


def run_voxel_mc3d(
    power: float = 1.0,
    pulse_duration: float = 0.20,
    n_photons: int = 200000,
    spot_dia: float = 0.50,
    theta_inc_deg: float = 0.0,
    phi_inc_deg: float = 0.0,
    Lx: float = 4.0,
    Ly: float = 4.0,
    Lz: float = 2.42,
    Nx: int = 80,
    Ny: int = 80,
    Nz: int = 60,
    r_max: float = 0.0,
    layers_config: list = None,
    custom_3d_props: dict = None,
    launch_axis: str = 'z',
    beam_center=None,
    beam_profile: str = 'gaussian',
    engine: str = 'auto',
):
    """
    High-level orchestrator for 3D Voxel Monte Carlo photon transport.
    
    Parameters:
    -----------
    power : float [W]
        Total incident laser beam power.
    pulse_duration : float [s]
        Duration of the active laser pulse.
    n_photons : int
        Number of Monte Carlo photon packets.
    spot_dia : float [mm]
        Incident beam 1/e^2 Gaussian spot diameter.
    theta_inc_deg : float [degrees]
        Angle of incidence (0 deg = normal, 90 deg = grazing).
    phi_inc_deg : float [degrees]
        Azimuthal incidence angle (0 = along surface primary axis).
    Lx, Ly, Lz : float [mm]
        Physical dimensions of the 3D domain.
    Nx, Ny, Nz : int
        Grid voxel resolution along x, y, z.
    r_max : float [mm]
        Lateral cylindrical cutoff radius (0.0 = domain boundaries only).
    layers_config : list of dicts
        Optional planar layer stack: [{'z_end': float, 'mu_a': float, 'mu_s': float, 'g': float, 'n': float}, ...]
    custom_3d_props : dict
        Optional 3D arrays: {'mu_a': array[Nx, Ny, Nz], 'mu_s': ..., 'g': ..., 'n': ...}
    launch_axis : str
        'z' (default): Top-to-bottom injection at z=0 along +z, layers along z.
        'x': Left-to-right injection at x=0 along +x, layers along x.
    beam_center : tuple (a, b) [mm], optional
        Beam axis position on the entrance face: (x, y) for launch_axis='z', (y, z) for 'x'.
        Defaults to the centre of the entrance face.
    beam_profile : 'gaussian' | 'tophat' | 'kim_exp_disk' | 'kim_gauss_disk'
        Gaussian: 1/e^2 radius spot_dia/2. Top-hat: uniform disk of diameter spot_dia.
        kim_exp_disk: disk of diameter spot_dia with exp(-2 r/R) weighting (Kim et al. 2023, as printed);
        kim_gauss_disk: the same disk with exp(-2 (r/R)^2) weighting (the Gaussian reading of that formula).
    engine : 'auto' | 'gpu' | 'cpu'
        'auto' (default): Automatically dispatches to Apple Metal GPU if available and
        configuration is supported (launch_axis='z', beam_profile='gaussian'); falls back
        transparently to CPU Numba otherwise.
        'gpu': Forces Apple Metal GPU execution (raises RuntimeError if unavailable or unsupported).
        'cpu': Forces multi-threaded CPU Numba execution.
        
    Returns:
    --------
    results : dict containing:
        - 'grid_x', 'grid_y', 'grid_z': coordinate vectors
        - 'A_3d': 3D absorbed photon weight array
        - 'Q_3d': 3D volumetric absorbed power density [W/mm^3]
        - 'Phi_3d': 3D optical fluence rate [mm^-2]
        - 'spectro': {'R_spec', 'R_diff', 'R_total', 'Absorption', 'Transmission', 'Lateral_loss'}
        - 'launch_axis': 'z' or 'x'
    """
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    
    grid_x = np.linspace(dx/2.0, Lx - dx/2.0, Nx)
    grid_y = np.linspace(dy/2.0, Ly - dy/2.0, Ny)
    grid_z = np.linspace(dz/2.0, Lz - dz/2.0, Nz)
    
    launch_axis_clean = launch_axis.strip().lower()
    launch_axis_int = 1 if launch_axis_clean == 'x' else 0
    
    # Initialize 3D property grids
    if custom_3d_props is not None:
        mu_a_grid = custom_3d_props['mu_a'].astype(np.float64)
        mu_s_grid = custom_3d_props['mu_s'].astype(np.float64)
        g_grid = custom_3d_props['g'].astype(np.float64)
        n_grid = custom_3d_props['n'].astype(np.float64)
    else:
        mu_a_grid = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        mu_s_grid = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        g_grid = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        n_grid = np.ones((Nx, Ny, Nz), dtype=np.float64)
        
        # Populate from planar layers_config
        if layers_config is not None:
            if launch_axis_int == 0:
                # Top-to-bottom: depth coordinate along z
                z_start = 0.0
                for lay in layers_config:
                    z_end = lay['z_end']
                    iz_mask = (grid_z >= z_start) & (grid_z < z_end)
                    if np.any(iz_mask):
                        mu_a_grid[:, :, iz_mask] = lay['mu_a']
                        mu_s_grid[:, :, iz_mask] = lay['mu_s']
                        g_grid[:, :, iz_mask] = lay['g']
                        n_grid[:, :, iz_mask] = lay['n']
                    z_start = z_end
            else:
                # Left-to-right: depth coordinate along x
                x_start = 0.0
                for lay in layers_config:
                    x_end = lay['z_end']
                    ix_mask = (grid_x >= x_start) & (grid_x < x_end)
                    if np.any(ix_mask):
                        mu_a_grid[ix_mask, :, :] = lay['mu_a']
                        mu_s_grid[ix_mask, :, :] = lay['mu_s']
                        g_grid[ix_mask, :, :] = lay['g']
                        n_grid[ix_mask, :, :] = lay['n']
                    x_start = x_end
            
    # Entrance-face Fresnel reflection and Snell refraction are applied per photon inside
    # the kernel, against the actual voxel each photon enters (so ablated craters open at
    # the surface correctly carry no entrance reflection; their floors/walls do).
    n_ambient = 1.0

    # Engine selection and GPU dispatch
    engine_clean = str(engine).strip().lower()
    can_use_gpu = (
        launch_axis_clean == 'z'
        and beam_profile == 'gaussian'
        and _is_metal_available()
    )

    if engine_clean == 'gpu':
        if not _is_metal_available():
            raise RuntimeError("Metal GPU acceleration requested (engine='gpu'), but Apple Metal is not available.")
        if launch_axis_clean != 'z':
            raise ValueError(f"Metal GPU MC currently requires launch_axis='z', but got launch_axis='{launch_axis}'.")
        if beam_profile != 'gaussian':
            raise ValueError(f"Metal GPU MC currently requires beam_profile='gaussian', but got beam_profile='{beam_profile}'.")
        use_gpu = True
    elif engine_clean == 'auto':
        use_gpu = can_use_gpu
    elif engine_clean == 'cpu':
        use_gpu = False
    else:
        raise ValueError(f"Unknown engine='{engine}'. Valid options: 'auto', 'gpu', 'cpu'.")

    if use_gpu:
        props = {
            'mu_a': mu_a_grid,
            'mu_s': mu_s_grid,
            'g': g_grid,
            'n': n_grid,
        }
        return run_voxel_mc_metal(
            power=power,
            pulse_duration=pulse_duration,
            n_photons=n_photons,
            spot_dia=spot_dia,
            theta_inc_deg=theta_inc_deg,
            phi_inc_deg=phi_inc_deg,
            Lx=Lx, Ly=Ly, Lz=Lz,
            Nx=Nx, Ny=Ny, Nz=Nz,
            r_max=r_max,
            custom_3d_props=props,
            beam_center=beam_center,
            n_ambient=n_ambient,
        )
    if launch_axis_int == 0:
        x_center = Lx / 2.0
        y_center = Ly / 2.0
        z_center = 0.0
        if beam_center is not None:
            x_center, y_center = float(beam_center[0]), float(beam_center[1])
    else:
        x_center = 0.0
        y_center = Ly / 2.0
        z_center = Lz / 2.0
        if beam_center is not None:
            y_center, z_center = float(beam_center[0]), float(beam_center[1])

    theta_rad = np.radians(theta_inc_deg)
    phi_rad = np.radians(phi_inc_deg)

    # Gaussian beam I(r) ~ exp(-2 r^2 / w0^2) with w0 = spot_dia / 2 (1/e^2 radius):
    # per-axis standard deviation is w0 / 2.
    w0 = spot_dia / 2.0
    sigma_axis = w0 / 2.0
    beam_profile_int = {'gaussian': 0, 'tophat': 1, 'kim_exp_disk': 2, 'kim_gauss_disk': 3}[beam_profile]

    # Per-thread private scoring grids, capped by a memory budget.
    n_vox = Nx * Ny * Nz
    bytes_per_chunk = 2 * 8 * n_vox
    max_chunks_mem = max(1, int(MC3D_SCORING_MEMORY_BUDGET_BYTES // max(bytes_per_chunk, 1)))
    n_chunks = max(1, min(get_num_threads(), max_chunks_mem, max(1, int(n_photons) // 5000)))

    mu_a_grid = np.ascontiguousarray(mu_a_grid)
    mu_s_grid = np.ascontiguousarray(mu_s_grid)
    g_grid = np.ascontiguousarray(g_grid)
    n_grid = np.ascontiguousarray(n_grid)

    t0 = time.time()
    A_loc, L_loc, tally_loc, lat_esc_loc = _mc3d_trace_photons(
        int(n_photons),
        int(n_chunks),
        float(sigma_axis),
        float(theta_rad),
        float(phi_rad),
        float(x_center),
        float(y_center),
        float(z_center),
        Nx, Ny, Nz,
        float(dx), float(dy), float(dz),
        float(Lx), float(Ly), float(Lz),
        mu_a_grid,
        mu_s_grid,
        g_grid,
        n_grid,
        float(n_ambient),
        float(r_max),
        launch_axis_int,
        float(w0),
        int(beam_profile_int)
    )
    elapsed = time.time() - t0

    A_raw = A_loc.sum(axis=0)
    L_raw = L_loc.sum(axis=0)
    spec_w, diff_w, trans_w, lat_w = (float(v) for v in tally_loc.sum(axis=0))
    lat_esc_z = lat_esc_loc.sum(axis=0)

    total_abs_w = float(np.sum(A_raw))
    tot_photons = float(n_photons)

    R_spec = spec_w / tot_photons
    R_diff = diff_w / tot_photons
    R_tot = R_spec + R_diff
    A_tot = total_abs_w / tot_photons
    T_tot = trans_w / tot_photons
    Lat_tot = lat_w / tot_photons

    # Russian roulette is unbiased, so R + A + T + lateral must close to 1 within MC noise.
    energy_closure = R_tot + A_tot + T_tot + Lat_tot
    if abs(energy_closure - 1.0) > MC3D_ENERGY_CLOSURE_TOL:
        warnings.warn(
            f"Voxel MC energy closure {energy_closure:.5f} deviates from 1 by more than "
            f"{MC3D_ENERGY_CLOSURE_TOL:.3f} (R={R_tot:.4f}, A={A_tot:.4f}, T={T_tot:.4f}, "
            f"Lat={Lat_tot:.4f})",
            RuntimeWarning,
        )

    # Normalization to physical volumetric power density Q [W/mm^3]
    voxel_vol = dx * dy * dz
    Q_3d = (A_raw / tot_photons) * (power / voxel_vol)

    # Optical fluence rate Phi [1/mm^2] from the exact track-length estimator,
    # valid uniformly in tissue and void/air voxels.
    Phi_3d = (L_raw / tot_photons) / voxel_vol

    return {
        'grid_x': grid_x,
        'grid_y': grid_y,
        'grid_z': grid_z,
        'dx': dx, 'dy': dy, 'dz': dz,
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
        'A_3d': A_raw / tot_photons,
        'lateral_escape_z': lat_esc_z / tot_photons,
        'Q_3d': Q_3d,
        'Phi_3d': Phi_3d,
        'spectro': {
            'R_spec': round(R_spec * 100.0, 2),
            'R_diff': round(R_diff * 100.0, 2),
            'R_tot': round(R_tot * 100.0, 2),
            'Absorption': round(A_tot * 100.0, 2),
            'Transmission': round(T_tot * 100.0, 2),
            'Lateral_loss': round(Lat_tot * 100.0, 2)
        },
        'energy_closure': float(energy_closure),
        'spectro_raw': {
            'R_spec': R_spec, 'R_diff': R_diff, 'Absorption': A_tot,
            'Transmission': T_tot, 'Lateral_loss': Lat_tot
        },
        'launch_axis': launch_axis_clean,
        'elapsed_seconds': round(elapsed, 3),
        'photons_per_sec': int(n_photons / max(elapsed, 1e-4)),
        'engine': 'cpu',
    }
