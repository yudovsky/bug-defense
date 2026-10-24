"""
Apple Metal GPU Fused Multi-Phase Dynamic Bioheat & Ablation Engine.

Fuses:
1. 3D Cartesian finite-difference heat diffusion with harmonic interface conductivities.
2. Convective and linearized radiative boundary losses at exterior boundaries.
3. Stefan latent-heat plateau enthalpy banking (H_acc) and boiling plateau clipping.
4. Henriques-Arrhenius tissue thermal damage integration:
       dOmega/dt = A * exp(-Ea / (R * T_K))
5. First-order cuticle pyrolysis and charring kinetics:
       dchi/dt = (1 - chi) * A_char * exp(-Ea_char / (R * T_K))

Executes multi-substep blocks directly on Apple Silicon GPU using unified memory
and buffer execution barriers, eliminating per-substep Python and memory roundtrips.
"""

import ctypes
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


_FUSED_ABLATION_METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

struct FusedAblationParams {
    uint32_t Nx, Ny, Nz;
    float dx, dy, dz;
    float dt;
    float h_eff;
    float t_ambient;
    float t_abl;
    float t_void;
    int32_t is_active;
    int32_t has_arrhenius;
    float arr_A;
    float arr_Ea_over_R;
    int32_t has_char;
    float char_A;
    float char_Ea_over_R;
    int32_t has_plume;
    float plume_D_xy;
    float plume_D_z;
    float plume_vz;
    float plume_decay;
};

kernel void fused_ablation3d_step(
    constant FusedAblationParams& p   [[buffer(0)]],
    device const float* T             [[buffer(1)]],
    device float* T_new               [[buffer(2)]],
    device float* H_acc               [[buffer(3)]],
    device float* Omega               [[buffer(4)]],
    device float* chi                 [[buffer(5)]],
    device const uint8_t* dry         [[buffer(6)]],
    device const uint8_t* ablated     [[buffer(7)]],
    device const float* Q             [[buffer(8)]],
    device const float* k_grid        [[buffer(9)]],
    device const float* rho_cp_grid   [[buffer(10)]],
    device const float* plume_in      [[buffer(11)]],
    device float* plume_out           [[buffer(12)]],
    device const uint8_t* plume_void  [[buffer(13)]],
    uint3 gid                         [[thread_position_in_grid]]
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
        float k_b = k_grid[idx - 1];
        float k_face = 2.0f * k_c * k_b / (k_c + k_b + 1e-12f);
        flux_b = k_face * (T[idx - 1] - T_c) * idz2;
    } else {
        flux_b = -p.h_eff * (T_c - p.t_ambient) / p.dz;
    }

    float flux_t = 0.0f;
    if (k + 1 < p.Nz) {
        float k_t = k_grid[idx + 1];
        float k_face = 2.0f * k_c * k_t / (k_c + k_t + 1e-12f);
        flux_t = k_face * (T[idx + 1] - T_c) * idz2;
    } else {
        flux_t = -p.h_eff * (T_c - p.t_ambient) / p.dz;
    }

    float div_flux = (flux_w + flux_e) + (flux_s + flux_n) + (flux_b + flux_t);
    float q_vol = (p.is_active != 0) ? Q[idx] : 0.0f;
    float T_trial = T_c + (p.dt / cap) * (div_flux + q_vol);

    bool is_abl = (ablated[idx] != 0);
    bool is_dry_cell = (dry[idx] != 0);

    // 2. Stefan latent-heat plateau enthalpy banking
    float h_acc = H_acc[idx];
    if (is_abl) {
        if (T_trial > p.t_void) {
            T_trial = p.t_void;
        }
    } else if (!is_dry_cell) {
        if (T_trial > p.t_abl) {
            h_acc += (T_trial - p.t_abl) * cap;
            T_trial = p.t_abl;
        } else if (h_acc > 0.0f) {
            float release = min(h_acc, (p.t_abl - T_trial) * cap);
            h_acc -= release;
            T_trial += release / cap;
        }
    }
    H_acc[idx] = h_acc;

    // 3. Henriques-Arrhenius tissue thermal damage integral
    if (!is_abl && p.has_arrhenius != 0) {
        float tk = T_trial + 273.15f;
        Omega[idx] += p.dt * p.arr_A * exp(-p.arr_Ea_over_R / tk);
    }

    // 4. First-order cuticle pyrolysis charring
    if (is_dry_cell && !is_abl && p.has_char != 0) {
        float tk = T_trial + 273.15f;
        float rate = p.char_A * exp(-p.char_Ea_over_R / tk);
        chi[idx] = 1.0f - (1.0f - chi[idx]) * exp(-rate * p.dt);
    }

    T_new[idx] = T_trial;

    // 5. Multispectral Plume 3D Diffusion & Open Boundary Clearing
    if (p.has_plume != 0) {
        bool is_pvoid = (plume_void[idx] != 0);
        if (!is_pvoid) {
            // Plume only exists in void/crater cavity; solid tissue has zero plume
            plume_out[idx] = 0.0f;
        } else if (i == 0 || i + 1 == p.Nx || j == 0 || j + 1 == p.Ny || k == 0 || k + 1 == p.Nz) {
            // Open exterior boundaries: plume diffuses out and clears to ambient space
            plume_out[idx] = 0.0f;
        } else {
            float C_c = plume_in[idx];

            // X-direction flux (zero-flux at impermeable solid tissue crater walls)
            bool w_void = (plume_void[(i - 1) * stride_x + j * stride_y + k] != 0);
            float C_w = w_void ? plume_in[(i - 1) * stride_x + j * stride_y + k] : C_c;

            bool e_void = (plume_void[(i + 1) * stride_x + j * stride_y + k] != 0);
            float C_e = e_void ? plume_in[(i + 1) * stride_x + j * stride_y + k] : C_c;

            // Y-direction flux
            bool s_void = (plume_void[i * stride_x + (j - 1) * stride_y + k] != 0);
            float C_s = s_void ? plume_in[i * stride_x + (j - 1) * stride_y + k] : C_c;

            bool n_void = (plume_void[i * stride_x + (j + 1) * stride_y + k] != 0);
            float C_n = n_void ? plume_in[i * stride_x + (j + 1) * stride_y + k] : C_c;

            // Z-direction flux (laser enters at k=0; upward recoil drift moves towards k=0)
            bool b_void = (plume_void[idx - 1] != 0);
            float C_b = b_void ? plume_in[idx - 1] : C_c;

            bool t_void = (plume_void[idx + 1] != 0);
            float C_t = t_void ? plume_in[idx + 1] : C_c;

            float diff = (p.plume_D_xy * ((C_w - 2.0f * C_c + C_e) * idx2 +
                                          (C_s - 2.0f * C_c + C_n) * idy2) +
                          p.plume_D_z  * ((C_b - 2.0f * C_c + C_t) * idz2));

            float adv = 0.0f;
            if (p.plume_vz > 0.0f) {
                adv = -p.plume_vz * (C_c - C_t) / p.dz;
            }

            float d_rate = p.plume_D_xy * (idx2 + idy2) + p.plume_D_z * idz2;
            float dt_plume = (d_rate > 1e-6f) ? min(p.dt, 0.45f / (2.0f * d_rate)) : p.dt;
            float C_new = (C_c + dt_plume * (diff + adv)) * p.plume_decay;
            plume_out[idx] = max(C_new, 0.0f);
        }
    }
}
"""


class _FusedAblationParamsStruct(ctypes.Structure):
    _fields_ = [
        ('Nx', ctypes.c_uint32), ('Ny', ctypes.c_uint32), ('Nz', ctypes.c_uint32),
        ('dx', ctypes.c_float), ('dy', ctypes.c_float), ('dz', ctypes.c_float),
        ('dt', ctypes.c_float), ('h_eff', ctypes.c_float), ('t_ambient', ctypes.c_float),
        ('t_abl', ctypes.c_float), ('t_void', ctypes.c_float),
        ('is_active', ctypes.c_int32), ('has_arrhenius', ctypes.c_int32),
        ('arr_A', ctypes.c_float), ('arr_Ea_over_R', ctypes.c_float),
        ('has_char', ctypes.c_int32), ('char_A', ctypes.c_float), ('char_Ea_over_R', ctypes.c_float),
        ('has_plume', ctypes.c_int32), ('plume_D_xy', ctypes.c_float), ('plume_D_z', ctypes.c_float),
        ('plume_vz', ctypes.c_float), ('plume_decay', ctypes.c_float),
    ]


def _get_fused_ablation_pipeline():
    global _METAL_DEVICE, _METAL_QUEUE, _METAL_PIPELINE
    if _METAL_PIPELINE is not None:
        return _METAL_DEVICE, _METAL_QUEUE, _METAL_PIPELINE

    if not is_metal_available():
        raise RuntimeError("Apple Metal GPU is not available on this platform.")

    import Metal
    opts = Metal.MTLCompileOptions.alloc().init()
    lib, err = _METAL_DEVICE.newLibraryWithSource_options_error_(_FUSED_ABLATION_METAL_SOURCE, opts, None)
    if err:
        raise RuntimeError(f"Metal library compilation error: {err}")

    func = lib.newFunctionWithName_('fused_ablation3d_step')
    if func is None:
        raise RuntimeError("Function 'fused_ablation3d_step' not found in Metal library.")

    pipeline, err = _METAL_DEVICE.newComputePipelineStateWithFunction_error_(func, None)
    if err:
        raise RuntimeError(f"Compute pipeline creation error: {err}")

    _METAL_PIPELINE = pipeline
    return _METAL_DEVICE, _METAL_QUEUE, _METAL_PIPELINE


class MetalAblationContext:
    """
    Persistent Metal GPU context managing coupled multi-phase ablation state.
    Preallocates zero-copy unified memory buffers for all state variables.
    """
    def __init__(
        self,
        Nx: int, Ny: int, Nz: int,
        dx: float, dy: float, dz: float,
        dt_sub: float,
        h_eff: float,
        t_ambient: float,
        t_abl: float,
        t_void: float,
        arrhenius=None,
        char_params=None,
        plume_params=None,
    ):
        import Metal
        self.device, self.queue, self.pipeline = _get_fused_ablation_pipeline()
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = float(dx), float(dy), float(dz)
        self.dt = float(dt_sub)
        self.h_eff = float(h_eff)
        self.t_ambient = float(t_ambient)
        self.t_abl = float(t_abl)
        self.t_void = float(t_void)

        self.has_arrhenius = 1 if arrhenius is not None else 0
        self.arr_A = float(arrhenius[0]) if arrhenius is not None else 0.0
        self.arr_Ea_over_R = float(arrhenius[1]) if arrhenius is not None else 0.0

        self.has_char = 1 if char_params is not None else 0
        self.char_A = float(char_params[0]) if char_params is not None else 0.0
        self.char_Ea_over_R = float(char_params[1]) if char_params is not None else 0.0

        if plume_params is not None:
            self.has_plume = 1
            d_fallback = float(plume_params.get('D', plume_params.get('D_plume', 0.5)))
            self.plume_D_xy = float(plume_params.get('D_xy', d_fallback))
            self.plume_D_z = float(plume_params.get('D_z', d_fallback))
            self.plume_vz = float(plume_params.get('vz', plume_params.get('v_drift', 15.0)))
            tau = float(plume_params.get('tau_linger', 0.0))
            self.plume_decay = float(np.exp(-self.dt / tau)) if (tau > 0.0 and not np.isinf(tau)) else 1.0
        else:
            self.has_plume = 0
            self.plume_D_xy = 0.0
            self.plume_D_z = 0.0
            self.plume_vz = 0.0
            self.plume_decay = 1.0

        self.n_voxels = Nx * Ny * Nz
        self.nbytes_f32 = self.n_voxels * 4
        self.nbytes_u8 = self.n_voxels

        # Preallocate GPU buffers in zero-copy shared memory
        self.b_T_A = self.device.newBufferWithLength_options_(self.nbytes_f32, Metal.MTLResourceStorageModeShared)
        self.b_T_B = self.device.newBufferWithLength_options_(self.nbytes_f32, Metal.MTLResourceStorageModeShared)
        self.b_H_acc = self.device.newBufferWithLength_options_(self.nbytes_f32, Metal.MTLResourceStorageModeShared)
        self.b_Omega = self.device.newBufferWithLength_options_(self.nbytes_f32, Metal.MTLResourceStorageModeShared)
        self.b_chi = self.device.newBufferWithLength_options_(self.nbytes_f32, Metal.MTLResourceStorageModeShared)
        self.b_dry = self.device.newBufferWithLength_options_(self.nbytes_u8, Metal.MTLResourceStorageModeShared)
        self.b_abl = self.device.newBufferWithLength_options_(self.nbytes_u8, Metal.MTLResourceStorageModeShared)
        self.b_Q = self.device.newBufferWithLength_options_(self.nbytes_f32, Metal.MTLResourceStorageModeShared)
        self.b_k = self.device.newBufferWithLength_options_(self.nbytes_f32, Metal.MTLResourceStorageModeShared)
        self.b_rho = self.device.newBufferWithLength_options_(self.nbytes_f32, Metal.MTLResourceStorageModeShared)
        self.b_plume_A = self.device.newBufferWithLength_options_(self.nbytes_f32, Metal.MTLResourceStorageModeShared)
        self.b_plume_B = self.device.newBufferWithLength_options_(self.nbytes_f32, Metal.MTLResourceStorageModeShared)
        self.b_plume_void = self.device.newBufferWithLength_options_(self.nbytes_u8, Metal.MTLResourceStorageModeShared)

        self.cur_in = self.b_T_A
        self.cur_out = self.b_T_B
        self.cur_P_in = self.b_plume_A
        self.cur_P_out = self.b_plume_B

        # Parameter structs for active heating vs passive cooling
        p_act = _FusedAblationParamsStruct(
            Nx, Ny, Nz, self.dx, self.dy, self.dz, self.dt,
            self.h_eff, self.t_ambient, self.t_abl, self.t_void,
            1, self.has_arrhenius, self.arr_A, self.arr_Ea_over_R,
            self.has_char, self.char_A, self.char_Ea_over_R,
            self.has_plume, self.plume_D_xy, self.plume_D_z, self.plume_vz, self.plume_decay
        )
        p_pass = _FusedAblationParamsStruct(
            Nx, Ny, Nz, self.dx, self.dy, self.dz, self.dt,
            self.h_eff, self.t_ambient, self.t_abl, self.t_void,
            0, self.has_arrhenius, self.arr_A, self.arr_Ea_over_R,
            self.has_char, self.char_A, self.char_Ea_over_R,
            self.has_plume, self.plume_D_xy, self.plume_D_z, self.plume_vz, self.plume_decay
        )
        self.b_params_active = self.device.newBufferWithBytes_length_options_(bytes(p_act), len(bytes(p_act)), Metal.MTLResourceStorageModeShared)
        self.b_params_passive = self.device.newBufferWithBytes_length_options_(bytes(p_pass), len(bytes(p_pass)), Metal.MTLResourceStorageModeShared)

        self.grid = Metal.MTLSize(width=Nx, height=Ny, depth=Nz)
        self.tg = Metal.MTLSize(width=min(self.pipeline.maxTotalThreadsPerThreadgroup(), 8), height=8, depth=min(Nz, 4))

    def set_state(self, T, H_acc, Omega, chi, dry, ablated, Q, k_grid, rho_cp_grid, plume=None, plume_void=None):
        """Upload full simulation state into unified memory buffers."""
        np.frombuffer(self.cur_in.contents().as_buffer(self.nbytes_f32), dtype=np.float32)[:] = np.ascontiguousarray(T, dtype=np.float32).ravel()
        np.frombuffer(self.b_H_acc.contents().as_buffer(self.nbytes_f32), dtype=np.float32)[:] = np.ascontiguousarray(H_acc, dtype=np.float32).ravel()
        np.frombuffer(self.b_Omega.contents().as_buffer(self.nbytes_f32), dtype=np.float32)[:] = np.ascontiguousarray(Omega, dtype=np.float32).ravel()
        np.frombuffer(self.b_chi.contents().as_buffer(self.nbytes_f32), dtype=np.float32)[:] = np.ascontiguousarray(chi, dtype=np.float32).ravel()
        np.frombuffer(self.b_dry.contents().as_buffer(self.nbytes_u8), dtype=np.uint8)[:] = np.ascontiguousarray(dry, dtype=np.uint8).ravel()
        np.frombuffer(self.b_abl.contents().as_buffer(self.nbytes_u8), dtype=np.uint8)[:] = np.ascontiguousarray(ablated, dtype=np.uint8).ravel()
        np.frombuffer(self.b_Q.contents().as_buffer(self.nbytes_f32), dtype=np.float32)[:] = np.ascontiguousarray(Q, dtype=np.float32).ravel()
        np.frombuffer(self.b_k.contents().as_buffer(self.nbytes_f32), dtype=np.float32)[:] = np.ascontiguousarray(k_grid, dtype=np.float32).ravel()
        np.frombuffer(self.b_rho.contents().as_buffer(self.nbytes_f32), dtype=np.float32)[:] = np.ascontiguousarray(rho_cp_grid, dtype=np.float32).ravel()
        if plume is not None:
            np.frombuffer(self.cur_P_in.contents().as_buffer(self.nbytes_f32), dtype=np.float32)[:] = np.ascontiguousarray(plume, dtype=np.float32).ravel()
        if plume_void is not None:
            np.frombuffer(self.b_plume_void.contents().as_buffer(self.nbytes_u8), dtype=np.uint8)[:] = np.ascontiguousarray(plume_void, dtype=np.uint8).ravel()
        else:
            np.frombuffer(self.b_plume_void.contents().as_buffer(self.nbytes_u8), dtype=np.uint8)[:] = np.ascontiguousarray(ablated, dtype=np.uint8).ravel()

    def advance_substeps(self, n_substeps: int, is_active: int = 1):
        """Execute multi-substep block with memory barriers and chunked command buffers."""
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
            enc.setBuffer_offset_atIndex_(self.b_H_acc, 0, 3)
            enc.setBuffer_offset_atIndex_(self.b_Omega, 0, 4)
            enc.setBuffer_offset_atIndex_(self.b_chi, 0, 5)
            enc.setBuffer_offset_atIndex_(self.b_dry, 0, 6)
            enc.setBuffer_offset_atIndex_(self.b_abl, 0, 7)
            enc.setBuffer_offset_atIndex_(self.b_Q, 0, 8)
            enc.setBuffer_offset_atIndex_(self.b_k, 0, 9)
            enc.setBuffer_offset_atIndex_(self.b_rho, 0, 10)
            enc.setBuffer_offset_atIndex_(self.b_plume_void, 0, 13)

            for _ in range(count):
                enc.setBuffer_offset_atIndex_(self.cur_in, 0, 1)
                enc.setBuffer_offset_atIndex_(self.cur_out, 0, 2)
                enc.setBuffer_offset_atIndex_(self.cur_P_in, 0, 11)
                enc.setBuffer_offset_atIndex_(self.cur_P_out, 0, 12)
                enc.dispatchThreads_threadsPerThreadgroup_(self.grid, self.tg)
                enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
                self.cur_in, self.cur_out = self.cur_out, self.cur_in
                self.cur_P_in, self.cur_P_out = self.cur_P_out, self.cur_P_in

            enc.endEncoding()
            cmd_buf.commit()
            cmd_buf.waitUntilCompleted()
            remaining -= count

    def get_state(self):
        """Extract updated state tensors from zero-copy shared memory."""
        shape = (self.Nx, self.Ny, self.Nz)
        T = np.frombuffer(self.cur_in.contents().as_buffer(self.nbytes_f32), dtype=np.float32).reshape(shape).astype(np.float64).copy()
        H_acc = np.frombuffer(self.b_H_acc.contents().as_buffer(self.nbytes_f32), dtype=np.float32).reshape(shape).astype(np.float64).copy()
        Omega = np.frombuffer(self.b_Omega.contents().as_buffer(self.nbytes_f32), dtype=np.float32).reshape(shape).astype(np.float64).copy()
        chi = np.frombuffer(self.b_chi.contents().as_buffer(self.nbytes_f32), dtype=np.float32).reshape(shape).astype(np.float64).copy()
        return T, H_acc, Omega, chi

    def get_plume(self, out_arr=None):
        """Extract updated plume density tensor from zero-copy shared memory."""
        shape = (self.Nx, self.Ny, self.Nz)
        buf = np.frombuffer(self.cur_P_in.contents().as_buffer(self.nbytes_f32), dtype=np.float32).reshape(shape)
        if out_arr is not None:
            out_arr[:] = buf
            return out_arr
        return buf.copy()
