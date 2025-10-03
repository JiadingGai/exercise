# lora_dsm_tma_demo_fixed2.py
# LoRA shrink->expand with:
#  1) Naive (HBM round-trip)
#  2) DSM cluster + cp.async pipeline (double-buffered), CORRECT block-wide async copies
#
# Requires: CUDA 12+, Hopper (sm_90) for DSM kernel.
import os
import torch
from torch.utils.cpp_extension import load_inline

# Compile for Hopper unless you override
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

# ---- cluster sizing ----
C = 3                          # number of consumer CTAs
CLUSTER_SIZE = C + 1           # 1 producer + C consumers
ARCH_FLAG = "-arch=sm_90"      # use "-arch=sm_90a" if your nvcc supports it

cpp_src = r"""
#include <torch/extension.h>
#include <vector>

void naive_shrink_expand_launcher(
  at::Tensor x, at::Tensor A, at::Tensor B,
  at::Tensor shard_starts, at::Tensor shard_sizes,
  at::Tensor R, at::Tensor Out);

void dsm_shrink_expand_pipelined_launcher(
  at::Tensor x, at::Tensor A, at::Tensor B,
  at::Tensor shard_starts, at::Tensor shard_sizes,
  at::Tensor R, at::Tensor Out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("naive_shrink_expand",  &naive_shrink_expand_launcher,
        "LoRA shrink->expand naive (HBM)");
  m.def("dsm_shrink_expand_pipelined", &dsm_shrink_expand_pipelined_launcher,
        "LoRA shrink->expand with DSM + cp.async pipeline (H100)");
}
"""

cuda_src = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
namespace cg = cooperative_groups;

#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 4
#endif

// Tunables for consumer pipeline:
#ifndef KTILE
#define KTILE 16      // rows (r dimension) per stage
#endif
#ifndef NTILE
#define NTILE 32      // columns (per-shard) per stage
#endif

// ================== NAIVE PATH ====================

// K1: Y = X @ A   (global-memory output)
__global__ void shrink_naive_kernel(
    const float* __restrict__ X, const float* __restrict__ A,
    float* __restrict__ Y, int B, int d, int r) {
  for (int b = blockIdx.x; b < B; b += gridDim.x) {
    const float* xrow = X + b * d;
    for (int k = threadIdx.x; k < r; k += blockDim.x) {
      float acc = 0.f;
      const float* Acol = A + k;            // A[i*r + k]
      #pragma unroll 1
      for (int i = 0; i < d; ++i) {
        acc += xrow[i] * Acol[i * r];
      }
      Y[b * r + k] = acc;
    }
  }
}

// K2: one CTA per shard j: Z_j = Y @ B_j
__global__ void expand_naive_kernel(
    const float* __restrict__ Y,
    const float* __restrict__ B,
    const int* __restrict__ shard_starts,
    const int* __restrict__ shard_sizes,
    const float* __restrict__ R,
    float* __restrict__ Out,
    int Bsz, int r, int d) {

  int j = blockIdx.x;  // shard id
  int start = shard_starts[j];
  int width = shard_sizes[j];

  for (int b = 0; b < Bsz; ++b) {
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
      float acc = 0.f;
      int col = start + c;
      const float* Bcol = B + col;           // B[k*d + col]
      #pragma unroll 1
      for (int k = 0; k < r; ++k) {
        acc += Y[b * r + k] * Bcol[k * d];
      }
      Out[b * d + col] = R[b * d + col] + acc;
    }
  }
}

// ================== DSM CLUSTER + cp.async PIPELINE (H100) ====================
//
// 1 producer (rank 0) + C consumers (1..C) in one cluster.
// Producer computes Y[b,:] into its SMEM; consumers map it via DSM.
// Consumers process shard columns in tiles of NTILE, and k in tiles of KTILE,
// using a double-buffered cuda::pipeline to overlap cp.async loads of B-tiles with compute.
//
// IMPORTANT: Each call to cuda::memcpy_async(block, ...) must be executed by
// ALL threads in the block (cooperative copy). We fix that here.
//
// Grid for this demo: exactly one cluster (gridDim.x == CLUSTER_SIZE).
//
extern "C"
__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
void dsm_shrink_expand_pipelined_kernel(
    const float* __restrict__ X,     // [B, d]
    const float* __restrict__ A,     // [d, r]
    const float* __restrict__ B,     // [r, d] row-major
    const int*   __restrict__ shard_starts, // [C]
    const int*   __restrict__ shard_sizes,  // [C]
    const float* __restrict__ R,     // [B, d]
    float*       __restrict__ Out,   // [B, d]
    int Bsz, int d, int r) {

#if __CUDA_ARCH__ < 900
  return; // requires Hopper
#endif

  cg::cluster_group cluster = cg::this_cluster();
  cg::thread_block block = cg::this_thread_block();
  const int rank = cluster.block_rank();       // 0..CLUSTER_SIZE-1

  // ---------------- Shared memory layout ----------------
  extern __shared__ __align__(16) unsigned char smem_raw[];
  float* Yrow0 = reinterpret_cast<float*>(smem_raw);               // [r]
  float* smem_after = Yrow0 + r;
  // Double-buffer for B tiles (for consumers only). Layout: [2][KTILE * NTILE]
  float* Btiles = smem_after;

  // Per-CTA pipeline state (block scope)
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;

  // Map producer SMEM for consumers:
  float* Yrow_remote =
      (rank == 0) ? Yrow0
                  : static_cast<float*>(cluster.map_shared_rank((void*)Yrow0, /*producer=*/0));

  // Each consumer's shard meta
  int shard_start = 0, shard_width = 0;
  if (rank > 0) {
    int j = rank - 1;
    shard_start = shard_starts[j];
    shard_width = shard_sizes[j];
  }

  // ---------------- Batch loop ----------------
  for (int b = 0; b < Bsz; ++b) {

    // ---- Producer computes Y[b,:] into SMEM ----
    if (rank == 0) {
      const float* xrow = X + b * d;
      for (int k = threadIdx.x; k < r; k += blockDim.x) {
        float acc = 0.f;
        const float* Acol = A + k;     // A[i*r + k]
        #pragma unroll 1
        for (int i = 0; i < d; ++i) {
          acc += xrow[i] * Acol[i * r];
        }
        Yrow0[k] = acc;
      }
    }

    // Handoff: publish Y to consumers
    cluster.sync();

    // ---- Consumers: pipelined GEMM using Y[b,:] and B_j ----
    if (rank > 0) {
      auto pipe = cuda::make_pipeline(block, &pipe_state);

      // Process this shard's columns in tiles of NTILE
      for (int c0 = 0; c0 < shard_width; c0 += NTILE) {
        const int cols = min(NTILE, shard_width - c0);

        int stage = 0;

        // Preload stage 0 (rows 0..KTILE-1)
        pipe.producer_acquire();
        for (int tk = 0; tk < KTILE; ++tk) {
          const int k = tk;
          if (k < r) {
            const float* gsrc = B + k * d + (shard_start + c0);
            float* sdst = Btiles + stage * (KTILE * NTILE) + tk * NTILE;
            cuda::memcpy_async(block, sdst, gsrc, cols * sizeof(float), pipe);
          }
        }
        pipe.producer_commit();

        for (int k0 = 0; k0 < r; k0 += KTILE) {
          const int read_stage = stage;

          // Prefetch next stage (rows k0+KTILE .. k0+2*KTILE-1)
          const int next_k0 = k0 + KTILE;
          if (next_k0 < r) {
            int next_stage = stage ^ 1;
            pipe.producer_acquire();
            for (int tk = 0; tk < KTILE; ++tk) {
              const int k = next_k0 + tk;
              if (k < r) {
                const float* gsrc = B + k * d + (shard_start + c0);
                float* sdst = Btiles + next_stage * (KTILE * NTILE) + tk * NTILE;
                cuda::memcpy_async(block, sdst, gsrc, cols * sizeof(float), pipe);
              }
            }
            pipe.producer_commit();
          }

          // Wait for current stage to arrive
          pipe.consumer_wait();

          // Compute on arrived tile
          const float* sB = Btiles + read_stage * (KTILE * NTILE);
          for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            float acc = 0.f;
            #pragma unroll 1
            for (int kk = 0; kk < KTILE; ++kk) {
              const int k = k0 + kk;
              if (k < r) {
                float yk = Yrow_remote[k];
                float bkc = sB[kk * NTILE + c]; // row-major within the tile
                acc += yk * bkc;
              }
            }
            int col = shard_start + c0 + c;
            float* outp = &Out[b * d + col];
            if (k0 == 0) {
              *outp = R[b * d + col];   // write base once at first K tile
            }
            *outp += acc;               // accumulate LoRA contribution
          }

          pipe.consumer_release();
          stage ^= 1;
        } // k0
      } // c0
    } // rank>0

    // Allow producer to reuse Yrow buffer
    cluster.sync();
  } // b
}

// -------------------- HOST LAUNCHERS ----------------------

void naive_shrink_expand_launcher(
  at::Tensor x, at::Tensor A, at::Tensor B,
  at::Tensor shard_starts, at::Tensor shard_sizes,
  at::Tensor R, at::Tensor Out) {

  TORCH_CHECK(x.is_cuda() && A.is_cuda() && B.is_cuda() && R.is_cuda() && Out.is_cuda(),
              "All tensors must be CUDA");
  TORCH_CHECK(x.dtype() == at::kFloat && A.dtype() == at::kFloat && B.dtype() == at::kFloat &&
              R.dtype() == at::kFloat && Out.dtype() == at::kFloat, "Use float32 for this demo");

  x = x.contiguous(); A = A.contiguous(); B = B.contiguous(); R = R.contiguous(); Out = Out.contiguous();

  int Bsz = (int)x.size(0);
  int d   = (int)x.size(1);
  int r   = (int)A.size(1);
  TORCH_CHECK(A.size(0) == d, "A must be [d, r]");
  TORCH_CHECK(B.size(0) == r && B.size(1) == d, "B must be [r, d]");

  // temp Y on HBM
  auto Y = at::empty({Bsz, r}, x.options());

  dim3 grid1((unsigned)max(1, min(Bsz, 80)));
  dim3 blk1(256);
  shrink_naive_kernel<<<grid1, blk1>>>(
      x.data_ptr<float>(), A.data_ptr<float>(), Y.data_ptr<float>(), Bsz, d, r);

  int C = (int)shard_starts.size(0);
  dim3 grid2((unsigned)C);
  dim3 blk2(256);
  expand_naive_kernel<<<grid2, blk2>>>(
      Y.data_ptr<float>(), B.data_ptr<float>(),
      shard_starts.data_ptr<int>(), shard_sizes.data_ptr<int>(),
      R.data_ptr<float>(), Out.data_ptr<float>(),
      Bsz, r, d);
}

void dsm_shrink_expand_pipelined_launcher(
  at::Tensor x, at::Tensor A, at::Tensor B,
  at::Tensor shard_starts, at::Tensor shard_sizes,
  at::Tensor R, at::Tensor Out) {

  TORCH_CHECK(x.is_cuda() && A.is_cuda() && B.is_cuda() && R.is_cuda() && Out.is_cuda(),
              "All tensors must be CUDA");
  TORCH_CHECK(x.dtype() == at::kFloat && A.dtype() == at::kFloat && B.dtype() == at::kFloat &&
              R.dtype() == at::kFloat && Out.dtype() == at::kFloat, "Use float32 for this demo");

  x = x.contiguous(); A = A.contiguous(); B = B.contiguous(); R = R.contiguous(); Out = Out.contiguous();

  int Bsz = (int)x.size(0);
  int d   = (int)x.size(1);
  int r   = (int)A.size(1);
  TORCH_CHECK(A.size(0) == d, "A must be [d, r]");
  TORCH_CHECK(B.size(0) == r && B.size(1) == d, "B must be [r, d]");

#if CUDART_VERSION >= 12000
  cudaFuncSetAttribute(
      (const void*)dsm_shrink_expand_pipelined_kernel,
      cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
#endif

  // One cluster in this demo
  dim3 grid(CLUSTER_SIZE, 1, 1);
  dim3 blk(256, 1, 1);

  // Shared memory size:
  //   Y row: r floats
  //   B tiles: 2 * (KTILE * NTILE) floats
  size_t dyn_smem = sizeof(float) * ( (size_t)r + 2ull * (size_t)KTILE * (size_t)NTILE );

  dsm_shrink_expand_pipelined_kernel<<<grid, blk, dyn_smem>>>(
      x.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
      shard_starts.data_ptr<int>(), shard_sizes.data_ptr<int>(),
      R.data_ptr<float>(), Out.data_ptr<float>(),
      Bsz, d, r);
}
"""

# Build
mod = load_inline(
    name="lora_dsm_tma_ext_fixed2",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[ARCH_FLAG, "-O3", "-std=c++17", f"-DCLUSTER_SIZE={CLUSTER_SIZE}"],
    verbose=True,
)

# ----------------------- quick demo ------------------------
if __name__ == "__main__":
    B, d, r = 8, 256, 16
    C = (CLUSTER_SIZE - 1)
    assert C > 0, "Need at least 1 consumer"

    # Split d columns across C shards
    shard_sizes = [d // C + (1 if i < d % C else 0) for i in range(C)]
    starts = [0]
    for i in range(1, C):
        starts.append(starts[-1] + shard_sizes[i-1])

    device = "cuda"
    X = torch.randn(B, d, device=device, dtype=torch.float32)
    A = torch.randn(d, r, device=device, dtype=torch.float32)
    Bmat = torch.randn(r, d, device=device, dtype=torch.float32)
    R = torch.randn(B, d, device=device, dtype=torch.float32)
    Out_naive = torch.empty(B, d, device=device, dtype=torch.float32)
    Out_pipe  = torch.empty(B, d, device=device, dtype=torch.float32)
    starts_t = torch.tensor(starts, device=device, dtype=torch.int32)
    sizes_t  = torch.tensor(shard_sizes, device=device, dtype=torch.int32)

    # run both
    torch.cuda.nvtx.range_push("naive_shrink_expand")
    mod.naive_shrink_expand(X, A, Bmat, starts_t, sizes_t, R, Out_naive)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("dsm_shrink_expand_pipelined")
    mod.dsm_shrink_expand_pipelined(X, A, Bmat, starts_t, sizes_t, R, Out_pipe)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    print("max |naive - pipelined| =", (Out_naive - Out_pipe).abs().max().item())

