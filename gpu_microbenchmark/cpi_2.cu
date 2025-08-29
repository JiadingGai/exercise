// cpi_ldx_unrollspec.cu
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CHECK_CUDA(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  std::exit(1);} } while(0)

struct Args {
  const char* mode = "micro";   // "micro" or "full"
  const char* op   = "lds";     // "lds" or "ldg"
  const char* cache= "ca";      // ldg cache: "ca" or "cg"
  int   iters     = 90000;      // loop iterations
  int   unroll    = 8;          // 1,2,4,8 (compile-time specialization)
  int   threads   = 32;         // micro: 32; full: 256/512/1024 typical
  int   smem_f    = 4096;       // shared floats per block (LDS)
  int   wss_kb    = 256;        // LDG working set (KB) for micro/warmup
};

static inline bool eq(const char* a, const char* b){ return std::strcmp(a,b)==0; }

// -------- device helpers --------
__device__ __forceinline__ unsigned long long rdclk() {
  unsigned long long c; asm volatile("mov.u64 %0, %%clock64;" : "=l"(c)); return c;
}

// LDG.128 with runtime cache flag (0=ca, 1=cg)
__device__ __forceinline__
void ldg128_once(const float* __restrict__ p,
                 float &x0, float &x1, float &x2, float &x3,
                 int use_cg)
{
  unsigned long long gaddr = __cvta_generic_to_global((void*)p);
  if (use_cg) {
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
      : "=&f"(x0), "=&f"(x1), "=&f"(x2), "=&f"(x3)
      : "l"(gaddr));
  } else {
    asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];"
      : "=&f"(x0), "=&f"(x1), "=&f"(x2), "=&f"(x3)
      : "l"(gaddr));
  }
}

// LDS.128
__device__ __forceinline__
void lds128_once(const float* __restrict__ p,
                 float &x0, float &x1, float &x2, float &x3)
{
  unsigned long long saddr = __cvta_generic_to_shared((void*)p);
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
    : "=&f"(x0), "=&f"(x1), "=&f"(x2), "=&f"(x3)
    : "l"(saddr));
}

// Unrolled LDG (UNROLL = 1/2/4/8)
template<int UNROLL>
__device__ __forceinline__
void do_unrolled_ldg(const float* __restrict__ gptr,
                     int &g_idx, const int g_wrap,
                     float &a0, float &a1, float &a2, float &a3,
                     int use_cg)
{
#pragma unroll
  for (int u=0; u<UNROLL; ++u) {
    float x0,x1,x2,x3;
    ldg128_once(gptr + g_idx, x0,x1,x2,x3, use_cg);
    a0 += x0; a1 += x1; a2 += x2; a3 += x3;
    g_idx += 4; if (g_idx >= g_wrap) g_idx = 0; // 16B stride
  }
}

// Unrolled LDS (UNROLL = 1/2/4/8)
template<int UNROLL>
__device__ __forceinline__
void do_unrolled_lds(const float* __restrict__ sptr,
                     int &s_idx, const int s_wrap,
                     float &a0, float &a1, float &a2, float &a3)
{
#pragma unroll
  for (int u=0; u<UNROLL; ++u) {
    float x0,x1,x2,x3;
    lds128_once(sptr + s_idx, x0,x1,x2,x3);
    a0 += x0; a1 += x1; a2 += x2; a3 += x3;
    s_idx += 4; if (s_idx >= s_wrap) s_idx = 0; // 16B stride
  }
}

// -------- MICRO mode (single warp, %clock64), UNROLL-specialized --------
template<int UNROLL>
__global__ void micro_ldg_kernel_spec(uint64_t* cycles, float* sink,
                                      const float* __restrict__ gbuf,
                                      int iters, int wss_elems, int use_cg)
{
  // exactly one warp
  const int lane = threadIdx.x & 31;
  int g_idx  = (lane * 4) % (wss_elems - 4);
  const int g_wrap = (wss_elems - 4);
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f;

  __syncthreads();
  unsigned long long t0 = rdclk();
#pragma unroll 1
  for (int i=0;i<iters;++i){
    do_unrolled_ldg<UNROLL>(gbuf, g_idx, g_wrap, a0,a1,a2,a3, use_cg);
  }
  __syncthreads();
  unsigned long long t1 = rdclk();

  if (threadIdx.x==0) cycles[0] = (t1 - t0);
  if (threadIdx.x==0) *sink = a0+a1+a2+a3;
}

template<int UNROLL>
__global__ void micro_lds_kernel_spec(uint64_t* cycles, float* sink,
                                      int iters, int smem_floats)
{
  extern __shared__ __align__(16) float smem[];
  for (int i=threadIdx.x;i<smem_floats;i+=blockDim.x) smem[i]=float(i&1023)*0.25f;
  __syncthreads();

  const int lane = threadIdx.x & 31;
  int s_idx  = (lane * 4) % (smem_floats - 4);
  const int s_wrap = (smem_floats - 4);
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f;

  __syncthreads();
  unsigned long long t0 = rdclk();
#pragma unroll 1
  for (int i=0;i<iters;++i){
    do_unrolled_lds<UNROLL>(smem, s_idx, s_wrap, a0,a1,a2,a3);
  }
  __syncthreads();
  unsigned long long t1 = rdclk();

  if (threadIdx.x==0) cycles[0] = (t1 - t0);
  if (threadIdx.x==0) *sink = a0+a1+a2+a3;
}

// -------- FULL mode (cooperative grid timing), UNROLL-specialized --------
template<int UNROLL>
__global__ void full_coop_ldg_kernel_spec(float* sink, uint64_t* grid_cycles,
                                          const float* __restrict__ gbuf,
                                          int iters, int wss_elems, int use_cg)
{
  cg::grid_group grid = cg::this_grid();

  int g_idx = ((blockIdx.x*blockDim.x + threadIdx.x) * 4) % (wss_elems - 4);
  const int g_wrap = (wss_elems - 4);
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f;

  grid.sync();
  unsigned long long t0=0,t1=0;
  if (grid.thread_rank()==0) t0 = rdclk();
  grid.sync();

#pragma unroll 1
  for (int i=0;i<iters;++i){
    do_unrolled_ldg<UNROLL>(gbuf, g_idx, g_wrap, a0,a1,a2,a3, use_cg);
  }

  grid.sync();
  if (grid.thread_rank()==0) { t1 = rdclk(); grid_cycles[0] = (t1 - t0); }
  float s = a0+a1+a2+a3; atomicAdd(sink, s);
}

template<int UNROLL>
__global__ void full_coop_lds_kernel_spec(float* sink, uint64_t* grid_cycles,
                                          int iters, int smem_floats)
{
  cg::grid_group grid = cg::this_grid();
  extern __shared__ __align__(16) float smem[];

  for (int i=threadIdx.x;i<smem_floats;i+=blockDim.x) smem[i]=float((i^(i>>3))&1023)*0.125f;

  int s_idx = ((blockIdx.x*blockDim.x + threadIdx.x) * 4) % (smem_floats - 4);
  const int s_wrap= (smem_floats - 4);
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f;

  grid.sync();
  unsigned long long t0=0,t1=0;
  if (grid.thread_rank()==0) t0 = rdclk();
  grid.sync();

#pragma unroll 1
  for (int i=0;i<iters;++i){
    do_unrolled_lds<UNROLL>(smem, s_idx, s_wrap, a0,a1,a2,a3);
  }

  grid.sync();
  if (grid.thread_rank()==0) { t1 = rdclk(); grid_cycles[0] = (t1 - t0); }
  float s = a0+a1+a2+a3; atomicAdd(sink, s);
}

// -------- FULL mode (events fallback), UNROLL-specialized --------
template<int UNROLL>
__global__ void full_events_ldg_kernel_spec(float* sink,
                                            const float* __restrict__ gbuf,
                                            int iters, int wss_elems, int use_cg)
{
  int g_idx = ((blockIdx.x*blockDim.x + threadIdx.x) * 4) % (wss_elems - 4);
  const int g_wrap = (wss_elems - 4);
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f;

#pragma unroll 1
  for (int i=0;i<iters;++i){
    do_unrolled_ldg<UNROLL>(gbuf, g_idx, g_wrap, a0,a1,a2,a3, use_cg);
  }
  float s = a0+a1+a2+a3; atomicAdd(sink, s);
}

template<int UNROLL>
__global__ void full_events_lds_kernel_spec(float* sink,
                                            int iters, int smem_floats)
{
  extern __shared__ __align__(16) float smem[];
  for (int i=threadIdx.x;i<smem_floats;i+=blockDim.x) smem[i]=float((i^(i>>2))&1023)*0.25f;

  int s_idx = ((blockIdx.x*blockDim.x + threadIdx.x) * 4) % (smem_floats - 4);
  const int s_wrap= (smem_floats - 4);
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f;

#pragma unroll 1
  for (int i=0;i<iters;++i){
    do_unrolled_lds<UNROLL>(smem, s_idx, s_wrap, a0,a1,a2,a3);
  }
  float s = a0+a1+a2+a3; atomicAdd(sink, s);
}

// -------- host --------
int main(int argc, char** argv) {
  Args a;
  for (int i=1;i<argc;i++){
    if (!std::strncmp(argv[i],"--mode",6) && i+1<argc) a.mode=argv[++i];
    else if (!std::strncmp(argv[i],"--op",4) && i+1<argc) a.op=argv[++i];
    else if (!std::strncmp(argv[i],"--cache",7) && i+1<argc) a.cache=argv[++i];
    else if (!std::strncmp(argv[i],"--iters",7) && i+1<argc) a.iters=std::atoi(argv[++i]);
    else if (!std::strncmp(argv[i],"--unroll",8) && i+1<argc) a.unroll=std::atoi(argv[++i]);
    else if (!std::strncmp(argv[i],"--threads",9) && i+1<argc) a.threads=std::atoi(argv[++i]);
    else if (!std::strncmp(argv[i],"--smem-floats",13) && i+1<argc) a.smem_f=std::atoi(argv[++i]);
    else if (!std::strncmp(argv[i],"--wss-kb",8) && i+1<argc) a.wss_kb=std::atoi(argv[++i]);
  }
  // clamp UNROLL to {1,2,4,8}
  //if (!(a.unroll==1 || a.unroll==2 || a.unroll==4 || a.unroll==8)) a.unroll=8;

  int dev=0; CHECK_CUDA(cudaSetDevice(dev));
  cudaDeviceProp prop{}; CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  const int sms = prop.multiProcessorCount;
  printf("Device: %s sm_%d%d | SMs=%d\n", prop.name, prop.major, prop.minor, sms);
  printf("Args: mode=%s op=%s cache=%s iters=%d unroll=%d threads=%d smem_f=%d wss_kb=%d\n",
         a.mode, a.op, a.cache, a.iters, a.unroll, a.threads, a.smem_f, a.wss_kb);

  // common buffers
  uint64_t *d_cycles=nullptr;
  float *d_sink=nullptr;
  CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(uint64_t)));
  CHECK_CUDA(cudaMalloc(&d_sink, sizeof(float)));
  CHECK_CUDA(cudaMemset(d_cycles, 0, sizeof(uint64_t)));
  CHECK_CUDA(cudaMemset(d_sink, 0, sizeof(float)));

  // global buf (LDG)
  float* d_gbuf=nullptr;
  int wss_elems = (a.wss_kb * 1024) / (int)sizeof(float);
  if (eq(a.op,"ldg")) {
    CHECK_CUDA(cudaMalloc(&d_gbuf, (size_t)wss_elems * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_gbuf, 0, (size_t)wss_elems * sizeof(float)));
  }
  const int use_cg = eq(a.cache,"cg") ? 1 : 0;

  // ------------- MICRO MODE -------------
  if (eq(a.mode,"micro")) {
    dim3 grid(1), block(32); // exactly one warp
    if (eq(a.op,"ldg")) {
      // warmup to prime cache
      micro_ldg_kernel_spec<4><<<grid, block>>>(d_cycles, d_sink, d_gbuf, 1024, wss_elems, use_cg);
      CHECK_CUDA(cudaDeviceSynchronize());

      switch (a.unroll) {
        case 1: micro_ldg_kernel_spec<1><<<grid, block>>>(d_cycles, d_sink, d_gbuf, a.iters, wss_elems, use_cg); break;
        case 2: micro_ldg_kernel_spec<2><<<grid, block>>>(d_cycles, d_sink, d_gbuf, a.iters, wss_elems, use_cg); break;
        case 4: micro_ldg_kernel_spec<4><<<grid, block>>>(d_cycles, d_sink, d_gbuf, a.iters, wss_elems, use_cg); break;
        default: micro_ldg_kernel_spec<8><<<grid, block>>>(d_cycles, d_sink, d_gbuf, a.iters, wss_elems, use_cg); break;
      }
    } else { // lds
      size_t shmem = (size_t)a.smem_f * sizeof(float);
      switch (a.unroll) {
        case 1: micro_lds_kernel_spec<1><<<grid, block, shmem>>>(d_cycles, d_sink, a.iters, a.smem_f); break;
        case 2: micro_lds_kernel_spec<2><<<grid, block, shmem>>>(d_cycles, d_sink, a.iters, a.smem_f); break;
        case 4: micro_lds_kernel_spec<4><<<grid, block, shmem>>>(d_cycles, d_sink, a.iters, a.smem_f); break;
        case 8: micro_lds_kernel_spec<8><<<grid, block, shmem>>>(d_cycles, d_sink, a.iters, a.smem_f); break;
        case 16: micro_lds_kernel_spec<16><<<grid, block, shmem>>>(d_cycles, d_sink, a.iters, a.smem_f); break;
        default: micro_lds_kernel_spec<32><<<grid, block, shmem>>>(d_cycles, d_sink, a.iters, a.smem_f); break;
      }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    uint64_t cyc=0; CHECK_CUDA(cudaMemcpy(&cyc, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    const double total_insts = double(a.iters) * double(a.unroll) * 32.0; // per-thread 128b insts
    const double cpi = double(cyc) / total_insts;
    printf("[MICRO] %s.128 CPI: %.8f  (cycles=%llu, insts=%.0f)\n",
           a.op, cpi, (unsigned long long)cyc, total_insts);
  }

  // ------------- FULL MODE -------------
  if (eq(a.mode,"full")) {
    if (a.threads < 64) a.threads = 256;
    const bool coop = prop.cooperativeLaunch;
    int blocksPerSM=0;
    size_t shmem = eq(a.op,"lds") ? (size_t)a.smem_f * sizeof(float) : 0;

    // compute occupancy for the exact kernel we will launch
    if (coop) {
      if (eq(a.op,"ldg")) {
        switch (a.unroll) {
          case 1: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_coop_ldg_kernel_spec<1>, a.threads, shmem)); break;
          case 2: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_coop_ldg_kernel_spec<2>, a.threads, shmem)); break;
          case 4: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_coop_ldg_kernel_spec<4>, a.threads, shmem)); break;
          default: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_coop_ldg_kernel_spec<8>, a.threads, shmem)); break;
        }
      } else {
        switch (a.unroll) {
          case 1: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_coop_lds_kernel_spec<1>, a.threads, shmem)); break;
          case 2: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_coop_lds_kernel_spec<2>, a.threads, shmem)); break;
          case 4: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_coop_lds_kernel_spec<4>, a.threads, shmem)); break;
          default: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_coop_lds_kernel_spec<8>, a.threads, shmem)); break;
        }
      }
    } else {
      if (eq(a.op,"ldg")) {
        switch (a.unroll) {
          case 1: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_events_ldg_kernel_spec<1>, a.threads, shmem)); break;
          case 2: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_events_ldg_kernel_spec<2>, a.threads, shmem)); break;
          case 4: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_events_ldg_kernel_spec<4>, a.threads, shmem)); break;
          default: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_events_ldg_kernel_spec<8>, a.threads, shmem)); break;
        }
      } else {
        switch (a.unroll) {
          case 1: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_events_lds_kernel_spec<1>, a.threads, shmem)); break;
          case 2: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_events_lds_kernel_spec<2>, a.threads, shmem)); break;
          case 4: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_events_lds_kernel_spec<4>, a.threads, shmem)); break;
          default: CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, full_events_lds_kernel_spec<8>, a.threads, shmem)); break;
        }
      }
    }

    const int grid_blocks = blocksPerSM * sms;
    printf("Occupancy target: %d blocks/SM → launching %d blocks × %d threads\n",
           blocksPerSM, grid_blocks, a.threads);

    if (coop) {
      // warmup + timed using cooperative kernels and %clock64
      if (eq(a.op,"ldg")) {
        void* args[] = { &d_sink, &d_cycles, &d_gbuf, &a.iters, &wss_elems, (void*)&use_cg };
        switch (a.unroll) {
          case 1: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_ldg_kernel_spec<1>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
          case 2: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_ldg_kernel_spec<2>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
          case 4: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_ldg_kernel_spec<4>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
          default: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_ldg_kernel_spec<8>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        // timed (launch again)
        switch (a.unroll) {
          case 1: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_ldg_kernel_spec<1>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
          case 2: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_ldg_kernel_spec<2>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
          case 4: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_ldg_kernel_spec<4>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
          default: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_ldg_kernel_spec<8>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
        }
      } else { // lds
        void* args[] = { &d_sink, &d_cycles, &a.iters, &a.smem_f };
        switch (a.unroll) {
          case 1: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_lds_kernel_spec<1>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
          case 2: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_lds_kernel_spec<2>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
          case 4: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_lds_kernel_spec<4>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
          default: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_lds_kernel_spec<8>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        // timed
        switch (a.unroll) {
          case 1: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_lds_kernel_spec<1>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
          case 2: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_lds_kernel_spec<2>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
          case 4: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_lds_kernel_spec<4>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
          default: CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_coop_lds_kernel_spec<8>, dim3(grid_blocks), dim3(a.threads), args, shmem)); break;
        }
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      uint64_t cyc=0; CHECK_CUDA(cudaMemcpy(&cyc, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost));
      long double total_cycles = (long double)cyc * (long double)sms; // cycles on one SM × SMs
      long double total_insts  = (long double)grid_blocks * (long double)a.threads *
                                 (long double)a.iters    * (long double)a.unroll;
      long double cpi = total_cycles / total_insts;
      printf("[FULL/coop] %s.128 CPI: %.8Lf  (grid_cycles=%llu on 1 SM → total=%.0Lf, insts=%.0Lf)\n",
             a.op, cpi, (unsigned long long)cyc, total_cycles, total_insts);
    } else {
      // events fallback (no cg::this_grid inside kernels)
      cudaEvent_t st, ed; CHECK_CUDA(cudaEventCreate(&st)); CHECK_CUDA(cudaEventCreate(&ed));
      // warmup
      if (eq(a.op,"ldg")) {
        switch (a.unroll) {
          case 1: full_events_ldg_kernel_spec<1><<<grid_blocks, a.threads, shmem>>>(d_sink, d_gbuf, a.iters, wss_elems, use_cg); break;
          case 2: full_events_ldg_kernel_spec<2><<<grid_blocks, a.threads, shmem>>>(d_sink, d_gbuf, a.iters, wss_elems, use_cg); break;
          case 4: full_events_ldg_kernel_spec<4><<<grid_blocks, a.threads, shmem>>>(d_sink, d_gbuf, a.iters, wss_elems, use_cg); break;
          default: full_events_ldg_kernel_spec<8><<<grid_blocks, a.threads, shmem>>>(d_sink, d_gbuf, a.iters, wss_elems, use_cg); break;
        }
      } else {
        switch (a.unroll) {
          case 1: full_events_lds_kernel_spec<1><<<grid_blocks, a.threads, shmem>>>(d_sink, a.iters, a.smem_f); break;
          case 2: full_events_lds_kernel_spec<2><<<grid_blocks, a.threads, shmem>>>(d_sink, a.iters, a.smem_f); break;
          case 4: full_events_lds_kernel_spec<4><<<grid_blocks, a.threads, shmem>>>(d_sink, a.iters, a.smem_f); break;
          default: full_events_lds_kernel_spec<8><<<grid_blocks, a.threads, shmem>>>(d_sink, a.iters, a.smem_f); break;
        }
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      // timed
      CHECK_CUDA(cudaEventRecord(st));
      if (eq(a.op,"ldg")) {
        switch (a.unroll) {
          case 1: full_events_ldg_kernel_spec<1><<<grid_blocks, a.threads, shmem>>>(d_sink, d_gbuf, a.iters, wss_elems, use_cg); break;
          case 2: full_events_ldg_kernel_spec<2><<<grid_blocks, a.threads, shmem>>>(d_sink, d_gbuf, a.iters, wss_elems, use_cg); break;
          case 4: full_events_ldg_kernel_spec<4><<<grid_blocks, a.threads, shmem>>>(d_sink, d_gbuf, a.iters, wss_elems, use_cg); break;
          default: full_events_ldg_kernel_spec<8><<<grid_blocks, a.threads, shmem>>>(d_sink, d_gbuf, a.iters, wss_elems, use_cg); break;
        }
      } else {
        switch (a.unroll) {
          case 1: full_events_lds_kernel_spec<1><<<grid_blocks, a.threads, shmem>>>(d_sink, a.iters, a.smem_f); break;
          case 2: full_events_lds_kernel_spec<2><<<grid_blocks, a.threads, shmem>>>(d_sink, a.iters, a.smem_f); break;
          case 4: full_events_lds_kernel_spec<4><<<grid_blocks, a.threads, shmem>>>(d_sink, a.iters, a.smem_f); break;
          default: full_events_lds_kernel_spec<8><<<grid_blocks, a.threads, shmem>>>(d_sink, a.iters, a.smem_f); break;
        }
      }
      CHECK_CUDA(cudaEventRecord(ed));
      CHECK_CUDA(cudaEventSynchronize(ed));
      float ms=0.f; CHECK_CUDA(cudaEventElapsedTime(&ms, st, ed));

      const double total_cycles = (double)ms * 1e-3 * (double)prop.clockRate * 1e3 * (double)sms; // ms * kHz * SMs
      const double total_insts  = (double)grid_blocks * (double)a.threads *
                                  (double)a.iters    * (double)a.unroll;
      const double cpi = total_cycles / total_insts;
      printf("[FULL/events] %s.128 CPI: %.8f  (elapsed=%.3f ms, total_cycles=%.0f, insts=%.0f)\n",
             a.op, cpi, ms, total_cycles, total_insts);

      CHECK_CUDA(cudaEventDestroy(st));
      CHECK_CUDA(cudaEventDestroy(ed));
    }
  }

  if (d_gbuf) cudaFree(d_gbuf);
  cudaFree(d_sink); cudaFree(d_cycles);
  return 0;
}

