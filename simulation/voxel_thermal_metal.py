"""
Apple Metal GPU 3D Cartesian Bioheat Thermal Diffusion Engine.

Solves the transient 3D bioheat diffusion PDE on heterogeneous voxel grids:
    rho_cp(x, y, z) * dT/dt = div(k(x, y, z) * grad(T)) + Q(x, y, z, t) - Losses

Features:
- Native Metal Shading Language compute kernel targeting Apple Silicon (M-series GPUs).
- Harmonic interface thermal conductivity k_face at material boundaries.
- Convective & linearized radiative boundary conditions at top and bottom surfaces.
- Multi-substep asynchronous command buffer execution with zero-copy unified memory.
"""

import ctypes
import time
import numpy as np

_METAL_AVAILABLE = None
_METAL_DEVICE = None
_METAL_PIPELINE = None
_METAL_QUEUE = None


def is_metal_available() -> bool:
    """Check if Apple Metal GPU is available on the current system."""
    global _METAL_AVAILABLE, _METAL_DEVICE, _METAL_QUEUE
    if _METAL_AVAILABLE is not None:
        return _METAL_AVAILABLE
    try:
        import Metal
        dev = Metal.MTLCreateSystemDefaultDevice()
        if dev is not None:
            _METAL_AVAILABLE = True
            _METAL_DEVICE = dev
            _METAL_QUEUE = dev.newCommandQueue()
            return True
    except Exception:
        pass
    _METAL_AVAILABLE = False
    return False


_THERMAL_METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

struct ThermalParams {
    uint32_t Nx, Ny, Nz;
    float dx, dy, dz;
    float dt;
    float h_eff;
    float t_ambient;
    int32_t is_active;
};

kernel void thermal3d_step(
    constant ThermalParams& p [[buffer(0)]],
    device const float* T [[buffer(1)]],
    device float* T_new [[buffer(2)]],
    device const float* Q [[buffer(3)]],
    device const float* k_grid [[buffer(4)]],
    device const float* rho_cp_grid [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= p.Nx || gid.y >= p.Ny || gid.z >= p.Nz) return;

    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;

    uint stride_x = p.Ny * p.Nz;
    uint stride_y = p.Nz;
    uint idx = i * stride_x + j * stride_y + k;

    float T_c = T[idx];
    float k_c = k_grid[idx];
    float cap = max(rho_cp_grid[idx], 1e-12f);

    float idx2 = 1.0f / (p.dx * p.dx);
    float idy2 = 1.0f / (p.dy * p.dy);
    float idz2 = 1.0f / (p.dz * p.dz);

    // X flux
    float flux_w = 0.0f;
    if (i > 0) {
        float k_w = k_grid[(i - 1) * stride_x + j * stride_y + k];
        float k_face = 2.0f * k_c * k_w / (k_c + k_w + 1e-12f);
        flux_w = k_face * (T[(i - 1) * stride_x + j * stride_y + k] - T_c) * idx2;
    }
    float flux_e = 0.0f;
    if (i + 1 < p.Nx) {
        float k_e = k_grid[(i + 1) * stride_x + j * stride_y + k];
        float k_face = 2.0f * k_c * k_e / (k_c + k_e + 1e-12f);
        flux_e = k_face * (T[(i + 1) * stride_x + j * stride_y + k] - T_c) * idx2;
    }

    // Y flux
    float flux_s = 0.0f;
    if (j > 0) {
        float k_s = k_grid[i * stride_x + (j - 1) * stride_y + k];
        float k_face = 2.0f * k_c * k_s / (k_c + k_s + 1e-12f);
        flux_s = k_face * (T[i * stride_x + (j - 1) * stride_y + k] - T_c) * idy2;
    }
    float flux_n = 0.0f;
    if (j + 1 < p.Ny) {
        float k_n = k_grid[i * stride_x + (j + 1) * stride_y + k];
        float k_face = 2.0f * k_c * k_n / (k_c + k_n + 1e-12f);
        flux_n = k_face * (T[i * stride_x + (j + 1) * stride_y + k] - T_c) * idy2;
    }

    // Z flux & boundary losses
    float flux_b = 0.0f;
    if (k > 0) {
        float k_b = k_grid[i * stride_x + j * stride_y + (k - 1)];
        float k_face = 2.0f * k_c * k_b / (k_c + k_b + 1e-12f);
        flux_b = k_face * (T[i * stride_x + j * stride_y + (k - 1)] - T_c) * idz2;
    } else {
        flux_b = (p.h_eff > 0.0f) ? ((p.t_ambient - T_c) / (p.dz * (0.5f * p.dz / max(k_c, 1e-30f) + 1.0f / max(p.h_eff, 1e-30f)))) : 0.0f;
    }

    float flux_t = 0.0f;
    if (k + 1 < p.Nz) {
        float k_t = k_grid[i * stride_x + j * stride_y + (k + 1)];
        float k_face = 2.0f * k_c * k_t / (k_c + k_t + 1e-12f);
        flux_t = k_face * (T[i * stride_x + j * stride_y + (k + 1)] - T_c) * idz2;
    } else {
        flux_t = (p.h_eff > 0.0f) ? ((p.t_ambient - T_c) / (p.dz * (0.5f * p.dz / max(k_c, 1e-30f) + 1.0f / max(p.h_eff, 1e-30f)))) : 0.0f;
    }

    float q_val = (p.is_active != 0) ? Q[idx] : 0.0f;
    float div_flux = flux_w + flux_e + flux_s + flux_n + flux_b + flux_t;

    T_new[idx] = T_c + p.dt * (div_flux + q_val) / cap;
}
"""


class _ThermalParamsStruct(ctypes.Structure):
    _fields_ = [
        ('Nx', ctypes.c_uint32),
        ('Ny', ctypes.c_uint32),
        ('Nz', ctypes.c_uint32),
        ('dx', ctypes.c_float),
        ('dy', ctypes.c_float),
        ('dz', ctypes.c_float),
        ('dt', ctypes.c_float),
        ('h_eff', ctypes.c_float),
        ('t_ambient', ctypes.c_float),
        ('is_active', ctypes.c_int32)
    ]


def _get_thermal_pipeline():
    global _METAL_PIPELINE
    if _METAL_PIPELINE is not None:
        return _METAL_DEVICE, _METAL_QUEUE, _METAL_PIPELINE

    if not is_metal_available():
        raise RuntimeError("Apple Metal GPU device is not available.")

    import Metal
    opts = Metal.MTLCompileOptions.alloc().init()
    lib, err = _METAL_DEVICE.newLibraryWithSource_options_error_(_THERMAL_METAL_SOURCE, opts, None)
    if err:
        raise RuntimeError(f"Metal library compilation error: {err}")

    func = lib.newFunctionWithName_('thermal3d_step')
    if func is None:
        raise RuntimeError("Function 'thermal3d_step' not found in Metal library.")

    pipeline, err = _METAL_DEVICE.newComputePipelineStateWithFunction_error_(func, None)
    if err:
        raise RuntimeError(f"Compute pipeline creation error: {err}")

    _METAL_PIPELINE = pipeline
    return _METAL_DEVICE, _METAL_QUEUE, _METAL_PIPELINE


def run_thermal3d_substeps_metal(
    T: np.ndarray,
    Q: np.ndarray,
    k_grid: np.ndarray,
    rho_cp_grid: np.ndarray,
    Nx: int,
    Ny: int,
    Nz: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    n_substeps: int,
    h_eff: float = 0.0,
    t_ambient: float = 20.0,
    is_active: int = 1
) -> np.ndarray:
    """
    Execute n_substeps of 3D transient heat diffusion on Apple Metal GPU.

    Parameters:
    -----------
    T : array of shape (Nx, Ny, Nz)
        Initial temperature field [°C].
    Q : array of shape (Nx, Ny, Nz)
        Volumetric heat source [W/mm^3].
    k_grid : array of shape (Nx, Ny, Nz)
        Thermal conductivity [W/(mm K)].
    rho_cp_grid : array of shape (Nx, Ny, Nz)
        Volumetric heat capacity [J/(mm^3 K)].
    Nx, Ny, Nz : int
        Grid dimensions.
    dx, dy, dz : float [mm]
        Voxel grid spatial spacing.
    dt : float [s]
        Substep time step.
    n_substeps : int
        Number of substeps to advance.
    h_eff : float [W/(mm^2 K)]
        Effective surface loss coefficient.
    t_ambient : float [°C]
        Ambient temperature.
    is_active : int
        1 if laser heat source Q is active, 0 for passive cooling.

    Returns:
    --------
    T_final : array of shape (Nx, Ny, Nz), float64
        Updated temperature field.
    """
    import Metal

    device, queue, pipeline = _get_thermal_pipeline()

    params = _ThermalParamsStruct(
        Nx, Ny, Nz,
        float(dx), float(dy), float(dz),
        float(dt),
        float(h_eff),
        float(t_ambient),
        int(is_active)
    )
    p_bytes = bytes(params)

    T_f32 = np.ascontiguousarray(T, dtype=np.float32)
    Q_f32 = np.ascontiguousarray(Q, dtype=np.float32)
    k_f32 = np.ascontiguousarray(k_grid, dtype=np.float32)
    rho_f32 = np.ascontiguousarray(rho_cp_grid, dtype=np.float32)

    b_params = device.newBufferWithBytes_length_options_(p_bytes, len(p_bytes), Metal.MTLResourceStorageModeShared)
    b_T_A = device.newBufferWithBytes_length_options_(T_f32.tobytes(), T_f32.nbytes, Metal.MTLResourceStorageModeShared)
    b_T_B = device.newBufferWithBytes_length_options_(T_f32.tobytes(), T_f32.nbytes, Metal.MTLResourceStorageModeShared)
    b_Q = device.newBufferWithBytes_length_options_(Q_f32.tobytes(), Q_f32.nbytes, Metal.MTLResourceStorageModeShared)
    b_k = device.newBufferWithBytes_length_options_(k_f32.tobytes(), k_f32.nbytes, Metal.MTLResourceStorageModeShared)
    b_rhocp = device.newBufferWithBytes_length_options_(rho_f32.tobytes(), rho_f32.nbytes, Metal.MTLResourceStorageModeShared)

    grid = Metal.MTLSize(width=Nx, height=Ny, depth=Nz)
    tg = Metal.MTLSize(width=min(pipeline.maxTotalThreadsPerThreadgroup(), 8), height=8, depth=min(Nz, 4))

    cur_in, cur_out = b_T_A, b_T_B
    chunk_size = 250
    steps_remaining = n_substeps

    while steps_remaining > 0:
        steps_this_chunk = min(steps_remaining, chunk_size)
        cmd_buf = queue.commandBuffer()
        enc = cmd_buf.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        enc.setBuffer_offset_atIndex_(b_params, 0, 0)
        enc.setBuffer_offset_atIndex_(b_Q, 0, 3)
        enc.setBuffer_offset_atIndex_(b_k, 0, 4)
        enc.setBuffer_offset_atIndex_(b_rhocp, 0, 5)

        for _ in range(steps_this_chunk):
            enc.setBuffer_offset_atIndex_(cur_in, 0, 1)
            enc.setBuffer_offset_atIndex_(cur_out, 0, 2)
            enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
            enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
            cur_in, cur_out = cur_out, cur_in

        enc.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        steps_remaining -= steps_this_chunk

    T_final = np.frombuffer(cur_in.contents().as_buffer(T_f32.nbytes), dtype=np.float32).reshape(Nx, Ny, Nz)
    return T_final.astype(np.float64)


class MetalThermalContext:
    """
    Persistent Metal GPU context for 3D transient thermal diffusion.
    Preallocates unified memory shared buffers for temperature, source, conductivity,
    and capacity grids, enabling zero-reallocation multi-substep execution.
    """
    def __init__(
        self,
        Nx: int,
        Ny: int,
        Nz: int,
        dx: float,
        dy: float,
        dz: float,
        dt: float,
        k_grid: np.ndarray,
        rho_cp_grid: np.ndarray,
        h_eff: float = 0.0,
        t_ambient: float = 20.0,
    ):
        import Metal
        self.device, self.queue, self.pipeline = _get_thermal_pipeline()
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = float(dx), float(dy), float(dz)
        self.dt = float(dt)
        self.h_eff = float(h_eff)
        self.t_ambient = float(t_ambient)
        self.n_voxels = Nx * Ny * Nz
        self.nbytes = self.n_voxels * 4

        p_active = _ThermalParamsStruct(Nx, Ny, Nz, self.dx, self.dy, self.dz, self.dt, self.h_eff, self.t_ambient, 1)
        p_passive = _ThermalParamsStruct(Nx, Ny, Nz, self.dx, self.dy, self.dz, self.dt, self.h_eff, self.t_ambient, 0)
        self.b_params_active = self.device.newBufferWithBytes_length_options_(bytes(p_active), len(bytes(p_active)), Metal.MTLResourceStorageModeShared)
        self.b_params_passive = self.device.newBufferWithBytes_length_options_(bytes(p_passive), len(bytes(p_passive)), Metal.MTLResourceStorageModeShared)

        k_f32 = np.ascontiguousarray(k_grid, dtype=np.float32)
        rho_f32 = np.ascontiguousarray(rho_cp_grid, dtype=np.float32)
        self.b_k = self.device.newBufferWithBytes_length_options_(k_f32.tobytes(), k_f32.nbytes, Metal.MTLResourceStorageModeShared)
        self.b_rhocp = self.device.newBufferWithBytes_length_options_(rho_f32.tobytes(), rho_f32.nbytes, Metal.MTLResourceStorageModeShared)

        self.b_T_A = self.device.newBufferWithLength_options_(self.nbytes, Metal.MTLResourceStorageModeShared)
        self.b_T_B = self.device.newBufferWithLength_options_(self.nbytes, Metal.MTLResourceStorageModeShared)
        self.b_Q = self.device.newBufferWithLength_options_(self.nbytes, Metal.MTLResourceStorageModeShared)

        self.cur_in = self.b_T_A
        self.cur_out = self.b_T_B

        self.grid = Metal.MTLSize(width=Nx, height=Ny, depth=Nz)
        self.tg = Metal.MTLSize(width=min(self.pipeline.maxTotalThreadsPerThreadgroup(), 8), height=8, depth=min(Nz, 4))

    def set_T(self, T: np.ndarray):
        T_f32 = np.ascontiguousarray(T, dtype=np.float32)
        dst = np.frombuffer(self.cur_in.contents().as_buffer(self.nbytes), dtype=np.float32)
        dst[:] = T_f32.ravel()

    def set_Q(self, Q: np.ndarray):
        Q_f32 = np.ascontiguousarray(Q, dtype=np.float32)
        dst = np.frombuffer(self.b_Q.contents().as_buffer(self.nbytes), dtype=np.float32)
        dst[:] = Q_f32.ravel()

    def advance(self, n_substeps: int, is_active: int = 1):
        if n_substeps <= 0:
            return
        import Metal
        b_p = self.b_params_active if is_active else self.b_params_passive
        chunk_size = 250
        remaining = n_substeps

        while remaining > 0:
            count = min(remaining, chunk_size)
            cmd_buf = self.queue.commandBuffer()
            enc = cmd_buf.computeCommandEncoder()
            enc.setComputePipelineState_(self.pipeline)
            enc.setBuffer_offset_atIndex_(b_p, 0, 0)
            enc.setBuffer_offset_atIndex_(self.b_Q, 0, 3)
            enc.setBuffer_offset_atIndex_(self.b_k, 0, 4)
            enc.setBuffer_offset_atIndex_(self.b_rhocp, 0, 5)

            for _ in range(count):
                enc.setBuffer_offset_atIndex_(self.cur_in, 0, 1)
                enc.setBuffer_offset_atIndex_(self.cur_out, 0, 2)
                enc.dispatchThreads_threadsPerThreadgroup_(self.grid, self.tg)
                enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
                self.cur_in, self.cur_out = self.cur_out, self.cur_in

            enc.endEncoding()
            cmd_buf.commit()
            cmd_buf.waitUntilCompleted()
            remaining -= count

    def get_T(self) -> np.ndarray:
        arr = np.frombuffer(self.cur_in.contents().as_buffer(self.nbytes), dtype=np.float32).copy()
        return arr.reshape(self.Nx, self.Ny, self.Nz).astype(np.float64)

    def get_T_view(self) -> np.ndarray:
        return np.frombuffer(self.cur_in.contents().as_buffer(self.nbytes), dtype=np.float32).reshape(self.Nx, self.Ny, self.Nz)
