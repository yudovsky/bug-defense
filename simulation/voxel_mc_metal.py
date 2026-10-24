"""
3D Cartesian Voxel Monte Carlo Engine - Apple Metal GPU Accelerated Module.
Provides high-throughput stochastic photon transport on Apple Silicon M-series GPUs.

Features:
- Native Metal Shading Language (MSL) compute kernel running directly on Apple Silicon GPU
- Unified Memory Architecture (UMA) zero-copy buffer sharing between CPU and GPU
- Full support for multilayered media, refractive index steps, and angle-resolved Fresnel/Snell
- Non-normal incident Gaussian laser beam (arbitrary theta_inc, phi_inc, spot diameter)
- Atomic hardware float accumulation for 3D absorbed energy and path length
- Drop-in interface parity with simulation.voxel_mc_3d.run_voxel_mc3d
"""

import time
import ctypes
import numpy as np

_METAL_AVAILABLE = None
_METAL_DEVICE = None
_METAL_PIPELINE = None


def is_metal_available() -> bool:
    """Check if Apple Metal GPU is available on the current system."""
    global _METAL_AVAILABLE, _METAL_DEVICE
    if _METAL_AVAILABLE is not None:
        return _METAL_AVAILABLE
    try:
        import Metal
        dev = Metal.MTLCreateSystemDefaultDevice()
        if dev is not None:
            _METAL_AVAILABLE = True
            _METAL_DEVICE = dev
            return True
    except Exception:
        pass
    _METAL_AVAILABLE = False
    return False


_METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

struct RngState {
    uint64_t state;
    uint64_t inc;
};

inline uint32_t pcg32(thread RngState* rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + (rng->inc | 1ULL);
    uint32_t xorshifted = ((old >> 18u) ^ old) >> 27u;
    uint32_t rot = old >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

inline float rand_float(thread RngState* rng) {
    return (float)pcg32(rng) / 4294967296.0f;
}

inline float2 rand_normal2(thread RngState* rng) {
    float u1 = max(rand_float(rng), 1e-10f);
    float u2 = rand_float(rng);
    float r = sqrt(-2.0f * log(u1));
    float theta = 2.0f * 3.14159265358979323846f * u2;
    return float2(r * cos(theta), r * sin(theta));
}

inline float fresnel_unpolarized(float n1, float n2, float cos_i) {
    if (abs(n1 - n2) < 1e-6f) return 0.0f;
    cos_i = clamp(abs(cos_i), 0.0f, 1.0f);
    if (cos_i > 0.999999f) {
        float r = (n1 - n2) / (n1 + n2);
        return r * r;
    }
    float sin_t2 = (n1 / n2) * (n1 / n2) * (1.0f - cos_i * cos_i);
    if (sin_t2 >= 1.0f) return 1.0f; // Total Internal Reflection
    float cos_t = sqrt(max(0.0f, 1.0f - sin_t2));
    float rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    float rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i);
    return 0.5f * (rs * rs + rp * rp);
}

inline float3 refract_axis(float3 u, int axis, float eta) {
    float3 out_u = u;
    if (axis == 0) {
        float sgn = (u.x > 0.0f) ? 1.0f : -1.0f;
        out_u.y *= eta;
        out_u.z *= eta;
        out_u.x = sgn * sqrt(max(0.0f, 1.0f - out_u.y * out_u.y - out_u.z * out_u.z));
    } else if (axis == 1) {
        float sgn = (u.y > 0.0f) ? 1.0f : -1.0f;
        out_u.x *= eta;
        out_u.z *= eta;
        out_u.y = sgn * sqrt(max(0.0f, 1.0f - out_u.x * out_u.x - out_u.z * out_u.z));
    } else {
        float sgn = (u.z > 0.0f) ? 1.0f : -1.0f;
        out_u.x *= eta;
        out_u.y *= eta;
        out_u.z = sgn * sqrt(max(0.0f, 1.0f - out_u.x * out_u.x - out_u.y * out_u.y));
    }
    return out_u;
}

inline float dist_to_exit(float3 pos, float3 u, int3 ivox, float3 dxyz, thread int& out_axis, thread int& out_sdir) {
    float big = 1e30f;
    float sx = (u.x > 1e-15f) ? (((float)(ivox.x + 1) * dxyz.x - pos.x) / u.x) :
               ((u.x < -1e-15f) ? (((float)ivox.x * dxyz.x - pos.x) / u.x) : big);
    float sy = (u.y > 1e-15f) ? (((float)(ivox.y + 1) * dxyz.y - pos.y) / u.y) :
               ((u.y < -1e-15f) ? (((float)ivox.y * dxyz.y - pos.y) / u.y) : big);
    float sz = (u.z > 1e-15f) ? (((float)(ivox.z + 1) * dxyz.z - pos.z) / u.z) :
               ((u.z < -1e-15f) ? (((float)ivox.z * dxyz.z - pos.z) / u.z) : big);

    if (sx <= sy && sx <= sz) {
        out_axis = 0;
        out_sdir = (u.x > 0.0f) ? 1 : -1;
        return max(sx, 0.0f);
    } else if (sy <= sz) {
        out_axis = 1;
        out_sdir = (u.y > 0.0f) ? 1 : -1;
        return max(sy, 0.0f);
    } else {
        out_axis = 2;
        out_sdir = (u.z > 0.0f) ? 1 : -1;
        return max(sz, 0.0f);
    }
}

inline float3 scatter_hg(float3 u, float g, thread RngState* rng) {
    float xi_g = rand_float(rng);
    float cos_theta;
    if (abs(g) < 1e-4f) {
        cos_theta = 2.0f * xi_g - 1.0f;
    } else {
        float temp = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi_g);
        cos_theta = (1.0f + g * g - temp * temp) / (2.0f * g);
        cos_theta = clamp(cos_theta, -1.0f, 1.0f);
    }
    float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
    float psi = 2.0f * 3.14159265358979323846f * rand_float(rng);
    float cos_psi = cos(psi);
    float sin_psi = sin(psi);

    float3 new_u;
    if (abs(u.z) > 0.99999f) {
        float sign_z = (u.z > 0.0f) ? 1.0f : -1.0f;
        new_u = float3(sin_theta * cos_psi, sin_theta * sin_psi, cos_theta * sign_z);
    } else {
        float denom = sqrt(max(1.0f - u.z * u.z, 1e-12f));
        new_u.x = sin_theta * (u.x * u.z * cos_psi - u.y * sin_psi) / denom + u.x * cos_theta;
        new_u.y = sin_theta * (u.y * u.z * cos_psi + u.x * sin_psi) / denom + u.y * cos_theta;
        new_u.z = -sin_theta * cos_psi * denom + u.z * cos_theta;
    }
    return normalize(new_u);
}

struct LaunchParams {
    uint32_t Nx, Ny, Nz;
    float dx, dy, dz;
    float Lx, Ly, Lz;
    float init_ux, init_uy, init_uz;
    float cos_ph, sin_ph, cos_th_safe;
    float x_center, y_center;
    float sigma_axis;
    float n_ambient;
    float r_max_sq;
    uint32_t base_seed;
    uint32_t total_photons;
    uint32_t max_steps;
};

kernel void trace_photons(
    constant LaunchParams& p [[buffer(0)]],
    device const float* mu_a_grid [[buffer(1)]],
    device const float* mu_s_grid [[buffer(2)]],
    device const float* g_grid [[buffer(3)]],
    device const float* n_grid [[buffer(4)]],
    device atomic_float* A_grid [[buffer(5)]],
    device atomic_float* L_grid [[buffer(6)]],
    device atomic_float* tallies [[buffer(7)]], // 0: spec, 1: diff, 2: trans, 3: lat
    uint id [[thread_position_in_grid]]
) {
    if (id >= p.total_photons) return;

    RngState rng;
    rng.state = ((uint64_t)p.base_seed * 6364136223846793005ULL) ^ ((uint64_t)id * 1442695040888963407ULL);
    rng.inc = ((uint64_t)id << 1u) | 1ULL;
    pcg32(&rng);
    pcg32(&rng);

    float2 u_samp = rand_normal2(&rng) * p.sigma_axis;
    float x = p.x_center + (u_samp.x / p.cos_th_safe) * p.cos_ph - u_samp.y * p.sin_ph;
    float y = p.y_center + (u_samp.x / p.cos_th_safe) * p.sin_ph + u_samp.y * p.cos_ph;
    float z = 0.0f;

    if (x < 0.0f || x >= p.Lx || y < 0.0f || y >= p.Ly ||
        (p.r_max_sq > 0.0f && ((x - p.x_center)*(x - p.x_center) + (y - p.y_center)*(y - p.y_center)) >= p.r_max_sq)) {
        atomic_fetch_add_explicit(&tallies[3], 1.0f, memory_order_relaxed);
        return;
    }

    int ix = clamp((int)(x / p.dx), 0, (int)(p.Nx - 1));
    int iy = clamp((int)(y / p.dy), 0, (int)(p.Ny - 1));
    int iz = 0;

    float3 u = float3(p.init_ux, p.init_uy, p.init_uz);

    uint stride_x = p.Ny * p.Nz;
    uint stride_y = p.Nz;
    uint idx = ix * stride_x + iy * stride_y + iz;

    float n_in = n_grid[idx];
    float cos_in = u.z;
    float r_entry = fresnel_unpolarized(p.n_ambient, n_in, cos_in);
    atomic_fetch_add_explicit(&tallies[0], r_entry, memory_order_relaxed);
    float weight = 1.0f - r_entry;
    if (weight <= 0.0f) return;

    if (abs(n_in - p.n_ambient) > 1e-6f) {
        u = refract_axis(u, 2, p.n_ambient / n_in);
    }

    float3 dxyz = float3(p.dx, p.dy, p.dz);
    float3 pos = float3(x, y, z);
    int3 ivox = int3(ix, iy, iz);

    float tau = -log(max(rand_float(&rng), 1e-12f));

    for (uint step = 0; step < p.max_steps; ++step) {
        idx = ivox.x * stride_x + ivox.y * stride_y + ivox.z;
        float mu_a = mu_a_grid[idx];
        float mu_s = mu_s_grid[idx];
        float mu_t = mu_a + mu_s;

        int exit_axis = 0;
        int exit_sdir = 0;
        float s_exit = dist_to_exit(pos, u, ivox, dxyz, exit_axis, exit_sdir);

        if (mu_t > 1e-12f && mu_t * s_exit >= tau) {
            // Collision occurs inside current voxel
            float s_col = tau / mu_t;
            atomic_fetch_add_explicit(&L_grid[idx], weight * s_col, memory_order_relaxed);
            pos += s_col * u;

            if (p.r_max_sq > 0.0f) {
                float rr = (pos.x - p.x_center)*(pos.x - p.x_center) + (pos.y - p.y_center)*(pos.y - p.y_center);
                if (rr >= p.r_max_sq) {
                    atomic_fetch_add_explicit(&tallies[3], weight, memory_order_relaxed);
                    return;
                }
            }

            float dw = weight * (mu_a / mu_t);
            weight -= dw;
            atomic_fetch_add_explicit(&A_grid[idx], dw, memory_order_relaxed);

            // Russian roulette
            if (weight < 1e-4f) {
                if (rand_float(&rng) < 0.1f) {
                    weight *= 10.0f;
                } else {
                    return;
                }
            }

            // Anisotropic Henyey-Greenstein scattering
            float g_loc = g_grid[idx];
            u = scatter_hg(u, g_loc, &rng);
            tau = -log(max(rand_float(&rng), 1e-12f));
            continue;
        }

        // Advance to voxel boundary face
        tau -= mu_t * s_exit;
        atomic_fetch_add_explicit(&L_grid[idx], weight * s_exit, memory_order_relaxed);
        pos += s_exit * u;

        int3 jvox = ivox;
        if (exit_axis == 0) {
            jvox.x += exit_sdir;
            pos.x = (exit_sdir > 0) ? ((float)(ivox.x + 1) * dxyz.x) : ((float)ivox.x * dxyz.x);
        } else if (exit_axis == 1) {
            jvox.y += exit_sdir;
            pos.y = (exit_sdir > 0) ? ((float)(ivox.y + 1) * dxyz.y) : ((float)ivox.y * dxyz.y);
        } else {
            jvox.z += exit_sdir;
            pos.z = (exit_sdir > 0) ? ((float)(ivox.z + 1) * dxyz.z) : ((float)ivox.z * dxyz.z);
        }

        // Domain boundary conditions
        if (exit_axis == 2) {
            if (jvox.z < 0) {
                // Exit through top entrance face: Fresnel reflection against ambient
                float r_out = fresnel_unpolarized(n_grid[idx], p.n_ambient, abs(u.z));
                if (r_out > 0.0f && rand_float(&rng) < r_out) {
                    u.z = -u.z;
                    continue;
                }
                atomic_fetch_add_explicit(&tallies[1], weight, memory_order_relaxed);
                return;
            }
            if (jvox.z >= (int)p.Nz) {
                // Exit through bottom face
                atomic_fetch_add_explicit(&tallies[2], weight, memory_order_relaxed);
                return;
            }
        } else if (jvox.x < 0 || jvox.x >= (int)p.Nx || jvox.y < 0 || jvox.y >= (int)p.Ny) {
            // Lateral exit
            atomic_fetch_add_explicit(&tallies[3], weight, memory_order_relaxed);
            return;
        }

        // Internal voxel interface step
        uint jidx = jvox.x * stride_x + jvox.y * stride_y + jvox.z;
        float n_curr = n_grid[idx];
        float n_next = n_grid[jidx];
        if (abs(n_curr - n_next) > 1e-4f) {
            float cos_face = (exit_axis == 0) ? abs(u.x) : ((exit_axis == 1) ? abs(u.y) : abs(u.z));
            float r_int = fresnel_unpolarized(n_curr, n_next, cos_face);
            if (r_int > 0.0f && rand_float(&rng) < r_int) {
                if (exit_axis == 0) u.x = -u.x;
                else if (exit_axis == 1) u.y = -u.y;
                else u.z = -u.z;
                continue;
            }
            u = refract_axis(u, exit_axis, n_curr / n_next);
        }

        ivox = jvox;
    }
}
"""


class _LaunchParamsStruct(ctypes.Structure):
    _fields_ = [
        ("Nx", ctypes.c_uint32),
        ("Ny", ctypes.c_uint32),
        ("Nz", ctypes.c_uint32),
        ("dx", ctypes.c_float),
        ("dy", ctypes.c_float),
        ("dz", ctypes.c_float),
        ("Lx", ctypes.c_float),
        ("Ly", ctypes.c_float),
        ("Lz", ctypes.c_float),
        ("init_ux", ctypes.c_float),
        ("init_uy", ctypes.c_float),
        ("init_uz", ctypes.c_float),
        ("cos_ph", ctypes.c_float),
        ("sin_ph", ctypes.c_float),
        ("cos_th_safe", ctypes.c_float),
        ("x_center", ctypes.c_float),
        ("y_center", ctypes.c_float),
        ("sigma_axis", ctypes.c_float),
        ("n_ambient", ctypes.c_float),
        ("r_max_sq", ctypes.c_float),
        ("base_seed", ctypes.c_uint32),
        ("total_photons", ctypes.c_uint32),
        ("max_steps", ctypes.c_uint32),
    ]


def _get_metal_pipeline():
    global _METAL_DEVICE, _METAL_PIPELINE
    if _METAL_PIPELINE is not None:
        return _METAL_DEVICE, _METAL_PIPELINE

    import Metal
    _METAL_DEVICE = Metal.MTLCreateSystemDefaultDevice()
    if _METAL_DEVICE is None:
        raise RuntimeError("No Apple Metal device found.")

    opts = Metal.MTLCompileOptions.new()
    library, err = _METAL_DEVICE.newLibraryWithSource_options_error_(_METAL_SOURCE, opts, None)
    if err:
        raise RuntimeError(f"Metal compilation error: {err}")

    func = library.newFunctionWithName_("trace_photons")
    pipeline, err = _METAL_DEVICE.newComputePipelineStateWithFunction_error_(func, None)
    if err:
        raise RuntimeError(f"Compute pipeline error: {err}")

    _METAL_PIPELINE = pipeline
    return _METAL_DEVICE, _METAL_PIPELINE


def run_voxel_mc_metal(
    power: float = 1.0,
    pulse_duration: float = 0.20,
    n_photons: int = 100_000,
    Lx: float = 2.0,
    Ly: float = 2.0,
    Lz: float = 1.0,
    Nx: int = 40,
    Ny: int = 40,
    Nz: int = 50,
    custom_3d_props: dict = None,
    layers_config: list = None,
    theta_inc_deg: float = 0.0,
    phi_inc_deg: float = 0.0,
    spot_dia: float = 0.50,
    beam_center: tuple = None,
    n_ambient: float = 1.0,
    r_max: float = 0.0,
    base_seed: int = None,
    max_steps: int = 50_000,
) -> dict:
    """
    Execute 3D Cartesian Voxel Monte Carlo on Apple Metal GPU.

    Returns dict matching run_voxel_mc3d schema:
      A_3d, Q_3d, Phi_3d, spectro, spectro_raw, energy_closure, elapsed_seconds, photons_per_sec.
    """
    import Metal

    device, pipeline = _get_metal_pipeline()

    dx = float(Lx / Nx)
    dy = float(Ly / Ny)
    dz = float(Lz / Nz)

    x_center = float(Lx / 2.0) if beam_center is None else float(beam_center[0])
    y_center = float(Ly / 2.0) if beam_center is None else float(beam_center[1])

    theta_inc = float(np.radians(theta_inc_deg))
    phi_inc = float(np.radians(phi_inc_deg))
    cos_th = float(np.cos(theta_inc))
    sin_th = float(np.sin(theta_inc))
    cos_ph = float(np.cos(phi_inc))
    sin_ph = float(np.sin(phi_inc))
    cos_th_safe = max(cos_th, 1e-6)

    init_ux = float(sin_th * cos_ph)
    init_uy = float(sin_th * sin_ph)
    init_uz = float(cos_th)

    # 1/e^2 beam radius w0 = spot_dia / 2; Gaussian sigma = w0 / 2 = spot_dia / 4
    sigma_axis = float((spot_dia / 2.0) / 2.0)
    seed = int(base_seed) if base_seed is not None else int(np.random.randint(1, 2**31 - 1))

    # Prepare Params struct
    p = _LaunchParamsStruct()
    p.Nx, p.Ny, p.Nz = Nx, Ny, Nz
    p.dx, p.dy, p.dz = dx, dy, dz
    p.Lx, p.Ly, p.Lz = float(Lx), float(Ly), float(Lz)
    p.init_ux = init_ux
    p.init_uy = init_uy
    p.init_uz = init_uz
    p.cos_ph = cos_ph
    p.sin_ph = sin_ph
    p.cos_th_safe = cos_th_safe
    p.x_center = x_center
    p.y_center = y_center
    p.sigma_axis = sigma_axis
    p.n_ambient = float(n_ambient)
    p.r_max_sq = float(r_max * r_max) if r_max > 0.0 else 0.0
    p.base_seed = seed
    p.total_photons = int(n_photons)
    p.max_steps = int(max_steps)

    # Prepare optical grids
    if custom_3d_props is None:
        mu_a_grid = np.zeros((Nx, Ny, Nz), dtype=np.float32)
        mu_s_grid = np.zeros((Nx, Ny, Nz), dtype=np.float32)
        g_grid = np.zeros((Nx, Ny, Nz), dtype=np.float32)
        n_grid = np.ones((Nx, Ny, Nz), dtype=np.float32)
        if layers_config is not None:
            grid_z = np.linspace(dz / 2.0, Lz - dz / 2.0, Nz)
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
        custom_3d_props = {
            'mu_a': mu_a_grid,
            'mu_s': mu_s_grid,
            'g': g_grid,
            'n': n_grid,
        }

    mu_a = np.ascontiguousarray(custom_3d_props['mu_a'], dtype=np.float32)
    mu_s = np.ascontiguousarray(custom_3d_props['mu_s'], dtype=np.float32)
    g = np.ascontiguousarray(custom_3d_props['g'], dtype=np.float32)
    n = np.ascontiguousarray(custom_3d_props['n'], dtype=np.float32)

    n_voxels = Nx * Ny * Nz
    A_grid = np.zeros(n_voxels, dtype=np.float32)
    L_grid = np.zeros(n_voxels, dtype=np.float32)
    tallies = np.zeros(4, dtype=np.float32)

    p_bytes = bytes(p)
    p_buf = device.newBufferWithBytes_length_options_(p_bytes, len(p_bytes), Metal.MTLResourceStorageModeShared)
    mu_a_buf = device.newBufferWithBytes_length_options_(mu_a.tobytes(), mu_a.nbytes, Metal.MTLResourceStorageModeShared)
    mu_s_buf = device.newBufferWithBytes_length_options_(mu_s.tobytes(), mu_s.nbytes, Metal.MTLResourceStorageModeShared)
    g_buf = device.newBufferWithBytes_length_options_(g.tobytes(), g.nbytes, Metal.MTLResourceStorageModeShared)
    n_buf = device.newBufferWithBytes_length_options_(n.tobytes(), n.nbytes, Metal.MTLResourceStorageModeShared)

    A_buf = device.newBufferWithBytes_length_options_(A_grid.tobytes(), A_grid.nbytes, Metal.MTLResourceStorageModeShared)
    L_buf = device.newBufferWithBytes_length_options_(L_grid.tobytes(), L_grid.nbytes, Metal.MTLResourceStorageModeShared)
    tal_buf = device.newBufferWithBytes_length_options_(tallies.tobytes(), tallies.nbytes, Metal.MTLResourceStorageModeShared)

    cmd_queue = device.newCommandQueue()
    cmd_buf = cmd_queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)

    encoder.setBuffer_offset_atIndex_(p_buf, 0, 0)
    encoder.setBuffer_offset_atIndex_(mu_a_buf, 0, 1)
    encoder.setBuffer_offset_atIndex_(mu_s_buf, 0, 2)
    encoder.setBuffer_offset_atIndex_(g_buf, 0, 3)
    encoder.setBuffer_offset_atIndex_(n_buf, 0, 4)
    encoder.setBuffer_offset_atIndex_(A_buf, 0, 5)
    encoder.setBuffer_offset_atIndex_(L_buf, 0, 6)
    encoder.setBuffer_offset_atIndex_(tal_buf, 0, 7)

    grid_size = Metal.MTLSize(width=n_photons, height=1, depth=1)
    tg_size = Metal.MTLSize(width=min(pipeline.maxTotalThreadsPerThreadgroup(), 256), height=1, depth=1)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, tg_size)
    encoder.endEncoding()

    t0 = time.time()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()
    elapsed = max(time.time() - t0, 1e-6)

    # Read back results (Zero-copy unified memory views)
    A_raw = np.frombuffer(A_buf.contents().as_buffer(A_grid.nbytes), dtype=np.float32).reshape(Nx, Ny, Nz)
    L_raw = np.frombuffer(L_buf.contents().as_buffer(L_grid.nbytes), dtype=np.float32).reshape(Nx, Ny, Nz)
    t_out = np.frombuffer(tal_buf.contents().as_buffer(tallies.nbytes), dtype=np.float32)

    R_spec = float(t_out[0]) / n_photons
    R_diff = float(t_out[1]) / n_photons
    T_tot = float(t_out[2]) / n_photons
    Lat_tot = float(t_out[3]) / n_photons

    A_norm = A_raw / n_photons
    A_tot = float(np.sum(A_norm))
    R_tot = R_spec + R_diff
    energy_closure = R_tot + A_tot + T_tot + Lat_tot

    voxel_vol = dx * dy * dz
    Q_3d = A_norm * (power / voxel_vol)
    Phi_3d = (L_raw / n_photons) / voxel_vol

    grid_x = np.linspace(dx / 2.0, Lx - dx / 2.0, Nx)
    grid_y = np.linspace(dy / 2.0, Ly - dy / 2.0, Ny)
    grid_z = np.linspace(dz / 2.0, Lz - dz / 2.0, Nz)

    return {
        'grid_x': grid_x,
        'grid_y': grid_y,
        'grid_z': grid_z,
        'dx': dx, 'dy': dy, 'dz': dz,
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
        'A_3d': A_norm.astype(np.float64),
        'lateral_escape_z': np.zeros(Nz, dtype=np.float64),
        'Q_3d': Q_3d.astype(np.float64),
        'Phi_3d': Phi_3d.astype(np.float64),
        'spectro': {
            'R_spec': round(R_spec * 100.0, 2),
            'R_diff': round(R_diff * 100.0, 2),
            'R_tot': round(R_tot * 100.0, 2),
            'Absorption': round(A_tot * 100.0, 2),
            'Transmission': round(T_tot * 100.0, 2),
            'Lateral_loss': round(Lat_tot * 100.0, 2),
        },
        'spectro_raw': {
            'R_spec': R_spec,
            'R_diff': R_diff,
            'Absorption': A_tot,
            'Transmission': T_tot,
            'Lateral_loss': Lat_tot,
        },
        'energy_closure': float(energy_closure),
        'launch_axis': 'z',
        'elapsed_seconds': round(elapsed, 4),
        'photons_per_sec': int(n_photons / elapsed),
        'gpu_device_name': str(device.name()),
        'engine': 'gpu',
    }
