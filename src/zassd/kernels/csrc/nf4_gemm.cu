// M-invariant NF4 (bitsandbytes layout) decode GEMM: y[MP,N] = x[MP,K] @ dequant(W)[N,K]^T (+bias).
// One warp handles R output rows and all MP activation rows: every weight is read and dequantised once.
// Per (m,n): lane l accumulates, in increasing step order and increasing element order, fmaf(x, w, acc) over its own
// 8 K-positions of every 256-element step, then a fixed xor-butterfly over the 32 lanes. That arithmetic does not depend on
// MP, R, or the other rows, so row results are bitwise independent of the block they are computed in.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

template <typename T, int MP, int R>
__global__ void nf4_gemm_kernel(const T* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ am8,
                                const float* __restrict__ code, const float* __restrict__ am2,
                                const float* __restrict__ code2, float offset, const float* __restrict__ bias,
                                T* __restrict__ y, int N, int K) {
  __shared__ float s_code[16][32];  // lane-private copy of the NF4 codebook: conflict-free lookups
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < 16 * 32; i += blockDim.x) s_code[i >> 5][i & 31] = code[i >> 5];
  __syncthreads();
  const int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * R;
  float acc[R][MP];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < MP; ++m) acc[r][m] = 0.f;
  const int nsteps = (K + 1023) / 1024;  // each lane owns 32 consecutive K positions of every 1024-element step
  const int kb_row = K / 2;               // bytes per weight row
  uint4 wcur[R];
  auto load_w = [&](int step, uint4* dst) {
    const int k0 = step * 1024 + lane * 32;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = n0 + r;
      dst[r] = make_uint4(0, 0, 0, 0);
      if (n < N && k0 < K) dst[r] = __ldcs(reinterpret_cast<const uint4*>(w + (size_t)n * kb_row + k0 / 2));
    }
  };
  load_w(0, wcur);
  for (int s = 0; s < nsteps; ++s) {
    uint4 wnext[R];
    if (s + 1 < nsteps) load_w(s + 1, wnext);  // prefetch the next step while computing this one
    const int k0 = s * 1024 + lane * 32;
    const bool valid = k0 < K;
    float scale[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = n0 + r;
      scale[r] = 0.f;
      if (n < N && valid) {
        const long sidx = (long)n * (K / 64) + (k0 >> 6);
        scale[r] = code2[am8[sidx]] * am2[sidx >> 8] + offset;
      }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {  // 8 K-positions per chunk
      float xf[MP][8];
#pragma unroll
      for (int m = 0; m < MP; ++m) {
        if (valid) {
          uint4 xv = *reinterpret_cast<const uint4*>(x + (size_t)m * K + k0 + 8 * c);
          const T* xp = reinterpret_cast<const T*>(&xv);
#pragma unroll
          for (int j = 0; j < 8; ++j) xf[m][j] = to_f<T>(xp[j]);
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j) xf[m][j] = 0.f;
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint32_t wv = c == 0 ? wcur[r].x : c == 1 ? wcur[r].y : c == 2 ? wcur[r].z : wcur[r].w;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t byte = (wv >> (8 * j)) & 0xFF;
#ifdef DIAG_NO_ROUND
          const float w0 = s_code[byte >> 4][lane] * scale[r];
          const float w1 = s_code[byte & 15][lane] * scale[r];
#elif defined(DIAG_NO_LUT)
          const float w0 = to_f<T>(from_f<T>((float)(byte >> 4) * scale[r]));
          const float w1 = to_f<T>(from_f<T>((float)(byte & 15) * scale[r]));
#else
          const float w0 = to_f<T>(from_f<T>(s_code[byte >> 4][lane] * scale[r]));
          const float w1 = to_f<T>(from_f<T>(s_code[byte & 15][lane] * scale[r]));
#endif
#pragma unroll
          for (int m = 0; m < MP; ++m) {
            acc[r][m] = fmaf(xf[m][2 * j], w0, acc[r][m]);
            acc[r][m] = fmaf(xf[m][2 * j + 1], w1, acc[r][m]);
          }
        }
      }
    }
    if (s + 1 < nsteps) {
#pragma unroll
      for (int r = 0; r < R; ++r) wcur[r] = wnext[r];
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int n = n0 + r;
#pragma unroll
    for (int m = 0; m < MP; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
      if (lane == 0 && n < N) {
        if (bias != nullptr) v += bias[n];
        y[(size_t)m * N + n] = from_f<T>(v);
      }
    }
  }
}

template <typename T, int MP, int R>
void launch(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& am8, const torch::Tensor& code,
            const torch::Tensor& am2, const torch::Tensor& code2, double offset, const torch::Tensor& bias,
            torch::Tensor& y, int N, int K, int warps) {
  const int threads = warps * 32;
  const int rows_per_block = warps * R;
  const int blocks = (N + rows_per_block - 1) / rows_per_block;
  nf4_gemm_kernel<T, MP, R><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const T*>(x.data_ptr()), w.data_ptr<uint8_t>(), am8.data_ptr<uint8_t>(), code.data_ptr<float>(),
      am2.data_ptr<float>(), code2.data_ptr<float>(), (float)offset,
      bias.defined() && bias.numel() ? bias.data_ptr<float>() : nullptr,
      reinterpret_cast<T*>(y.data_ptr()), N, K);
}

torch::Tensor nf4_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor am8, torch::Tensor code, torch::Tensor am2,
                       torch::Tensor code2, double offset, torch::Tensor bias, int64_t mp, int64_t rows, int64_t warps) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(0) == mp);
  const int K = x.size(1);
  const int N = am8.numel() * 64 / K;
  auto y = torch::empty({mp, N}, x.options());
  const bool bf = x.scalar_type() == at::kBFloat16;
#define DISPATCH(MPV, RV)                                                                                  \
  if (mp == MPV && rows == RV) {                                                                            \
    if (bf) launch<__nv_bfloat16, MPV, RV>(x, w, am8, code, am2, code2, offset, bias, y, N, K, (int)warps); \
    else launch<__half, MPV, RV>(x, w, am8, code, am2, code2, offset, bias, y, N, K, (int)warps);           \
    return y;                                                                                                \
  }
  DISPATCH(1, 1) DISPATCH(1, 2) DISPATCH(1, 4) DISPATCH(1, 8) DISPATCH(2, 1) DISPATCH(2, 2) DISPATCH(2, 4) DISPATCH(2, 8) DISPATCH(4, 1) DISPATCH(4, 2) DISPATCH(4, 4) DISPATCH(4, 8) DISPATCH(8, 1) DISPATCH(8, 2) DISPATCH(8, 4) DISPATCH(8, 8)
  TORCH_CHECK(false, "unsupported (mp, rows)");
}
