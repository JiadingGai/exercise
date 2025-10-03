# lora_dsm_demo.py
import torch
from torch.utils.cpp_extension import load_inline

# ---- choose consumers and cluster size ----
C = 3                          # number of consumer CTAs
CLUSTER_SIZE = C + 1           # 1 producer (rank 0) + C consumers
ARCH_FLAG = "-arch=sm_90a"     # H100 target

cpp_src = r"""
#include <torch/extension.h>
#include <vector>

void naive_shrink_expand_launcher(
  at::Tensor x, at::Tensor A, at::Tensor B,
  at::Tensor shard_starts, at::Tensor shard_sizes,
  at::Tensor R, at::Tensor Out);

void dsm_shrink_expand_launcher(
  at::Tensor x, at::Tensor A, at::Tensor B,
  at::Tensor shard_starts, at::Tensor shard_sizes,
  at::Tensor R, at::Tensor Out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("naive_shrink_expand",  &naive_shrink_expand_launcher,
        "LoRA shrink->expand naive (HBM)");
  m.def("dsm_shrink_expand",    &dsm_shrink_expand_launcher,
        "LoRA shrink->expand with DSM cluster (H100)");
}
"""

cuda_src = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 4  // default; overridden by -DCLUSTER_SIZE
#endif

// ----------------- small helpers -----------------
static inline void checkCuda(cudaError_t e) {
  if (e != cudaSuccess) {
    printf("CUDA runtime error %d\n", (int)e);
  }
}

template <typename T>
__device__ inline T ld_g(const T* p) { return __ldg(p); }

// ================== NAIVE PATH ====================

// K1: Y = X @ A   (global-memory output)
__global__ void shrink_naive_kernel(
    const float* __restrict__ X, const float* __restrict__ A,
    float* __restrict__ Y, int B, int d, int r) {
  // simple, correctness-first loop nest
  for (int b = blockIdx.x; b < B; b += gridDim.x) {
    for (int k = threadIdx.x; k < r; k += blockDim.x) {
      float acc = 0.f;
      const float* xrow = X + b * d;
      const float* Acol = A + k;            // A[i*r + k]
      #pragma unroll 1
      for (int i = 0; i < d; ++i) {
        acc += xrow[i] * Acol[i * r];
      }
      Y[b * r + k] = acc;
    }
  }
}

// K2: for shard j (grid.x = #shards, one CTA per shard): Z_j = Y @ B_j
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

// ================== DSM CLUSTER (H100) ====================
//
// 1 producer (rank 0) + C consumers (1..C) in a single cluster.
// Producer computes a Y row into its __shared__ buffer.
// Consumers map that buffer via DSM (cluster.map_shared_rank) and consume it.
// We use blocking cluster.sync() to keep the example minimal.
// For real perf, you'd tile and overlap with TMA+mbarrier.
//
// Compile target must be sm_90a; gridDim.x must equal CLUSTER_SIZE for this demo.
//
extern "C"
__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
void dsm_shrink_expand_kernel(
    const float* __restrict__ X,     // [B, d]
    const float* __restrict__ A,     // [d, r]
    const float* __restrict__ B,     // [r, d]
    const int*   __restrict__ shard_starts, // [C]
    const int*   __restrict__ shard_sizes,  // [C]
    const float* __restrict__ R,     // [B, d]
    float*       __restrict__ Out,   // [B, d]
    int Bsz, int d, int r) {

#if __CUDA_ARCH__ < 900
  return; // requires Hopper
#endif

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();       // 0..CLUSTER_SIZE-1
  const int C    = CLUSTER_SIZE - 1;
  extern __shared__ float smem[];              // producer's Y row
  float* Yrow_local = smem;                    // size r

  // Consumers map producer's shared buffer
  float* Yrow_remote =
      (rank == 0) ? Yrow_local
                  : static_cast<float*>(cluster.map_shared_rank((void*)Yrow_local, /*producer=*/0));

  for (int b = 0; b < Bsz; ++b) {
    // ---- Producer computes Y[b,:] into its SMEM ----
    if (rank == 0) {
      const float* xrow = X + b * d;
      for (int k = threadIdx.x; k < r; k += blockDim.x) {
        float acc = 0.f;
        const float* Acol = A + k;   // A[i*r + k]
        #pragma unroll 1
        for (int i = 0; i < d; ++i) {
          acc += xrow[i] * Acol[i * r];
        }
        Yrow_local[k] = acc;
      }
    }

    // Handoff: ensure producer wrote Yrow before consumers read it
    cluster.sync();

    // ---- Consumers compute their shard using remote Y[b,:] ----
    if (rank > 0) {
      int j = rank - 1; // shard id
      int start = shard_starts[j];
      int width = shard_sizes[j];

      for (int c = threadIdx.x; c < width; c += blockDim.x) {
        float acc = 0.f;
        int col = start + c;
        const float* Bcol = B + col;    // B[k*d + col]
        #pragma unroll 1
        for (int k = 0; k < r; ++k) {
          acc += Yrow_remote[k] * Bcol[k * d];
        }
        Out[b * d + col] = R[b * d + col] + acc;
      }
    }

    // Ensure all consumers finished before producer overwrites Yrow
    cluster.sync();
  }
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
  checkCuda(cudaGetLastError());

  int C = (int)shard_starts.size(0);
  dim3 grid2((unsigned)C);
  dim3 blk2(256);
  expand_naive_kernel<<<grid2, blk2>>>(
      Y.data_ptr<float>(), B.data_ptr<float>(),
      shard_starts.data_ptr<int>(), shard_sizes.data_ptr<int>(),
      R.data_ptr<float>(), Out.data_ptr<float>(),
      Bsz, r, d);
  checkCuda(cudaGetLastError());
}

void dsm_shrink_expand_launcher(
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
  // allow non-portable cluster sizes (e.g., 16 on H100)
  cudaFuncSetAttribute(
      (const void*)dsm_shrink_expand_kernel,
      cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
#endif

  dim3 grid(CLUSTER_SIZE, 1, 1);  // one cluster in this demo
  dim3 blk(256, 1, 1);
  size_t dyn_smem = sizeof(float) * (size_t)r; // one Y row

  dsm_shrink_expand_kernel<<<grid, blk, dyn_smem>>>(
      x.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
      shard_starts.data_ptr<int>(), shard_sizes.data_ptr<int>(),
      R.data_ptr<float>(), Out.data_ptr<float>(),
      Bsz, d, r);
  checkCuda(cudaGetLastError());
}
"""

# build extension (note: CLUSTER_SIZE passed via -D; no f-strings in sources)
#mod = load_inline(
#    name="lora_dsm_ext",
#    cpp_sources=[cpp_src],
#    cuda_sources=[cuda_src],
#    extra_cflags=["-O3", "-std=c++17"],
#    # -rdc=true is generally safe here; sm_90a required for DSM kernel
#    extra_cuda_cflags=[ARCH_FLAG, "-O3", "-std=c++17", "-rdc=true", f"-DCLUSTER_SIZE={CLUSTER_SIZE}"],
#    verbose=True,
#)
mod = load_inline(
    name="lora_dsm_ext",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    extra_cflags=["-O3", "-std=c++17"],
    # REMOVE "-rdc=true" here:
    extra_cuda_cflags=[ARCH_FLAG, "-O3", "-std=c++17", f"-DCLUSTER_SIZE={CLUSTER_SIZE}"],
    verbose=True,
)


# ----------------------- quick demo ------------------------
if __name__ == "__main__":
    B, d, r = 8, 64, 8
    C = (CLUSTER_SIZE - 1)
    assert C > 0, "Need at least 1 consumer"

    # Evenly split d columns across C shards
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
    Out_dsm = torch.empty(B, d, device=device, dtype=torch.float32)
    starts_t = torch.tensor(starts, device=device, dtype=torch.int32)
    sizes_t  = torch.tensor(shard_sizes, device=device, dtype=torch.int32)

    # run both
    mod.naive_shrink_expand(X, A, Bmat, starts_t, sizes_t, R, Out_naive)
    torch.cuda.synchronize()
    mod.dsm_shrink_expand(X, A, Bmat, starts_t, sizes_t, R, Out_dsm)
    torch.cuda.synchronize()

    print("max |naive - dsm| =", (Out_naive - Out_dsm).abs().max().item())

