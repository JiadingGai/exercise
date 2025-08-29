// cpi_ldx_bothmodes.cu
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
  int   unroll    = 8;          // # of LDx.128 per iteration per thread
  int   threads   = 32;         // micro: 32; full: typical 256/512/1024
  int   smem_f    = 4096;       // shared floats per block (LDS)
  int   wss_kb    = 256;        // LDG micro working set (KB)
};

static inline bool eq(const char* a, const char* b){ return std::strcmp(a,b)==0; }

// Read SM clock
__device__ __forceinline__ unsigned long long rdclk() {
  unsigned long long c;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(c)); return c;
}

// One 128-bit LDG (choose .cg or .ca at runtime via flag)
__device__ __forceinline__
void ldg128_once(const float* __restrict__ p, float &x0, float &x1, float &x2, float &x3, int use_cg) {
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

// One 128-bit LDS
__device__ __forceinline__
void lds128_once(const float* __restrict__ p, float &x0, float &x1, float &x2, float &x3) {
  unsigned long long saddr = __cvta_generic_to_shared((void*)p);
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
    : "=&f"(x0), "=&f"(x1), "=&f"(x2), "=&f"(x3)
    : "l"(saddr));
}


template<bool LDG, bool LDS, int UNROLL>
__device__ __forceinline__ void do_unrolled_loads_ldg_lds(
    const float* __restrict__ gptr,
    float* __restrict__ sptr,
    int &g_idx, int g_wrap,
    int &s_idx, int s_wrap,
    float &a0, float &a1, float &a2, float &a3,
    const char* cache_flag)
{
#pragma unroll
  for (int u=0; u<UNROLL; ++u) {
    float x0=0.f, x1=0.f, x2=0.f, x3=0.f;

    if constexpr (LDG) {
      // LDG path
      float x0,x1,x2,x3;
      ldg128_once(gptr + g_idx, x0,x1,x2,x3, /*use_cg=*/true);

      g_idx += 4; if (g_idx >= g_wrap) g_idx = 0;
    }

    if constexpr (LDS) {
      // LDS path
      lds128_once(sptr + s_idx, x0,x1,x2,x3);
      s_idx += 4; if (s_idx >= s_wrap) s_idx = 0;
    }

    // Consume
    a0 += x0; a1 += x1; a2 += x2; a3 += x3;
  }
}

//
// ----------------------- MICRO MODE KERNELS -----------------------
//

template<bool DO_LDG>
__global__ void micro_ldg_kernel(uint64_t* cycles, float* sink, const float* __restrict__ gbuf,
                                 int iters, int unroll, int wss_elems, const char* cache_flag)
{
  // one warp
  const int lane = threadIdx.x & 31;
  // per-lane start offset (16B aligned), stride around a small WSS so hits are in cache
  int g_idx  = (lane * 4) % (wss_elems - 4);
  int g_wrap = (wss_elems - 4);

  float a0=0.f,a1=0.f,a2=0.f,a3=0.f;

  __syncthreads();
  unsigned long long t0 = rdclk();

#pragma unroll 1
  for (int i=0;i<iters;++i){
    if (unroll==1) do_unrolled_loads_ldg_lds<DO_LDG,false,1>(gbuf,nullptr,g_idx,g_wrap,g_idx,g_wrap,a0,a1,a2,a3,cache_flag);
    else if (unroll==2) do_unrolled_loads_ldg_lds<DO_LDG,false,2>(gbuf,nullptr,g_idx,g_wrap,g_idx,g_wrap,a0,a1,a2,a3,cache_flag);
    else if (unroll==4) do_unrolled_loads_ldg_lds<DO_LDG,false,4>(gbuf,nullptr,g_idx,g_wrap,g_idx,g_wrap,a0,a1,a2,a3,cache_flag);
    else                do_unrolled_loads_ldg_lds<DO_LDG,false,8>(gbuf,nullptr,g_idx,g_wrap,g_idx,g_wrap,a0,a1,a2,a3,cache_flag);
  }

  __syncthreads();
  unsigned long long t1 = rdclk();

  if (threadIdx.x==0) cycles[0] = (t1 - t0);
  if (threadIdx.x==0) *sink = a0+a1+a2+a3;
}

__global__ void micro_lds_kernel(uint64_t* cycles, float* sink,
                                 int iters, int unroll, int smem_floats)
{
  extern __shared__ __align__(16) float smem[];
  // init
  for (int i=threadIdx.x;i<smem_floats;i+=blockDim.x) smem[i]=float(i&1023)*0.25f;
  __syncthreads();

  const int lane = threadIdx.x & 31;
  int s_idx  = (lane * 4) % (smem_floats - 4);
  int s_wrap = (smem_floats - 4);

  float a0=0.f,a1=0.f,a2=0.f,a3=0.f;

  __syncthreads();
  unsigned long long t0 = rdclk();

#pragma unroll 1
  for (int i=0;i<iters;++i){
    if (unroll==1) do_unrolled_loads_ldg_lds<false,true,1>(nullptr,smem,s_idx,s_wrap,s_idx,s_wrap,a0,a1,a2,a3,"");
    else if (unroll==2) do_unrolled_loads_ldg_lds<false,true,2>(nullptr,smem,s_idx,s_wrap,s_idx,s_wrap,a0,a1,a2,a3,"");
    else if (unroll==4) do_unrolled_loads_ldg_lds<false,true,4>(nullptr,smem,s_idx,s_wrap,s_idx,s_wrap,a0,a1,a2,a3,"");
    else                do_unrolled_loads_ldg_lds<false,true,8>(nullptr,smem,s_idx,s_wrap,s_idx,s_wrap,a0,a1,a2,a3,"");
  }

  __syncthreads();
  unsigned long long t1 = rdclk();

  if (threadIdx.x==0) cycles[0] = (t1 - t0);
  if (threadIdx.x==0) *sink = a0+a1+a2+a3;
}

//
// ----------------------- FULL MODE KERNELS -----------------------
//

template<bool DO_LDG>
__global__ void full_ldg_kernel(float* sink, uint64_t* grid_cycles,
                                const float* __restrict__ gbuf,
                                int iters, int unroll, int wss_elems, const char* cache_flag)
{
  cg::grid_group grid = cg::this_grid();

  // per-thread state
  int g_idx = ((blockIdx.x*blockDim.x + threadIdx.x) * 4) % (wss_elems - 4);
  int g_wrap= (wss_elems - 4);
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f;

  grid.sync();
  unsigned long long t0=0,t1=0;
  if (grid.thread_rank()==0) t0 = rdclk();
  grid.sync();

#pragma unroll 1
  for (int i=0;i<iters;++i){
    if (unroll==1) do_unrolled_loads_ldg_lds<DO_LDG,false,1>(gbuf,nullptr,g_idx,g_wrap,g_idx,g_wrap,a0,a1,a2,a3,cache_flag);
    else if (unroll==2) do_unrolled_loads_ldg_lds<DO_LDG,false,2>(gbuf,nullptr,g_idx,g_wrap,g_idx,g_wrap,a0,a1,a2,a3,cache_flag);
    else if (unroll==4) do_unrolled_loads_ldg_lds<DO_LDG,false,4>(gbuf,nullptr,g_idx,g_wrap,g_idx,g_wrap,a0,a1,a2,a3,cache_flag);
    else                do_unrolled_loads_ldg_lds<DO_LDG,false,8>(gbuf,nullptr,g_idx,g_wrap,g_idx,g_wrap,a0,a1,a2,a3,cache_flag);
  }

  grid.sync();
  if (grid.thread_rank()==0) { t1 = rdclk(); grid_cycles[0] = (t1 - t0); }

  // prevent DCE
  float s = a0+a1+a2+a3;
  atomicAdd(sink, s);
}

__global__ void full_lds_kernel(float* sink, uint64_t* grid_cycles,
                                int iters, int unroll, int smem_floats)
{
  cg::grid_group grid = cg::this_grid();
  extern __shared__ __align__(16) float smem[];

  // init
  for (int i=threadIdx.x;i<smem_floats;i+=blockDim.x) smem[i]=float((i^(i>>3))&1023)*0.125f;

  int s_idx = ((blockIdx.x*blockDim.x + threadIdx.x) * 4) % (smem_floats - 4);
  int s_wrap= (smem_floats - 4);
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f;

  grid.sync();
  unsigned long long t0=0,t1=0;
  if (grid.thread_rank()==0) t0 = rdclk();
  grid.sync();

#pragma unroll 1
  for (int i=0;i<iters;++i){
    if (unroll==1) do_unrolled_loads_ldg_lds<false,true,1>(nullptr,smem,s_idx,s_wrap,s_idx,s_wrap,a0,a1,a2,a3,"");
    else if (unroll==2) do_unrolled_loads_ldg_lds<false,true,2>(nullptr,smem,s_idx,s_wrap,s_idx,s_wrap,a0,a1,a2,a3,"");
    else if (unroll==4) do_unrolled_loads_ldg_lds<false,true,4>(nullptr,smem,s_idx,s_wrap,s_idx,s_wrap,a0,a1,a2,a3,"");
    else                do_unrolled_loads_ldg_lds<false,true,8>(nullptr,smem,s_idx,s_wrap,s_idx,s_wrap,a0,a1,a2,a3,"");
  }

  grid.sync();
  if (grid.thread_rank()==0) { t1 = rdclk(); grid_cycles[0] = (t1 - t0); }

  float s = a0+a1+a2+a3;
  atomicAdd(sink, s);
}

//
// ----------------------- HOST HARNESS -----------------------
//

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

  int dev=0; CHECK_CUDA(cudaSetDevice(dev));
  cudaDeviceProp prop{}; CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  const int sms = prop.multiProcessorCount;

  printf("Device: %s sm_%d%d | SMs=%d\n", prop.name, prop.major, prop.minor, sms);
  printf("Args: mode=%s op=%s cache=%s iters=%d unroll=%d threads=%d smem_f=%d wss_kb=%d\n",
         a.mode, a.op, a.cache, a.iters, a.unroll, a.threads, a.smem_f, a.wss_kb);

  // Buffers common
  uint64_t *d_cycles=nullptr;
  float *d_sink=nullptr;
  CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(uint64_t)));
  CHECK_CUDA(cudaMalloc(&d_sink, sizeof(float)));
  CHECK_CUDA(cudaMemset(d_cycles, 0, sizeof(uint64_t)));
  CHECK_CUDA(cudaMemset(d_sink, 0, sizeof(float)));

  // Global buf for LDG
  float* d_gbuf=nullptr;
  int wss_elems = (a.wss_kb * 1024) / sizeof(float);
  if (eq(a.op,"ldg")) {
    CHECK_CUDA(cudaMalloc(&d_gbuf, (size_t)wss_elems * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_gbuf, 0, (size_t)wss_elems * sizeof(float)));
  }

  // ---------------- MICRO MODE ----------------
  if (eq(a.mode,"micro")) {
    // force one block, 32 threads
    dim3 grid(1), block(32);
    if (eq(a.op,"ldg")) {
      // prime cache
      micro_ldg_kernel<true><<<grid, block>>>(d_cycles, d_sink, d_gbuf, 1024, 4, wss_elems, a.cache);
      CHECK_CUDA(cudaDeviceSynchronize());
      // timed
      if (a.unroll!=1 && a.unroll!=2 && a.unroll!=4) a.unroll=8;
      micro_ldg_kernel<true><<<grid, block>>>(d_cycles, d_sink, d_gbuf, a.iters, a.unroll, wss_elems, a.cache);
    } else { // lds
      size_t shmem = a.smem_f * sizeof(float);
      if (a.unroll!=1 && a.unroll!=2 && a.unroll!=4) a.unroll=8;
      micro_lds_kernel<<<grid, block, shmem>>>(d_cycles, d_sink, a.iters, a.unroll, a.smem_f);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    uint64_t cyc=0; CHECK_CUDA(cudaMemcpy(&cyc, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    // #instructions = iters * unroll * 32 threads
    double total_insts = double(a.iters) * double(a.unroll) * 32.0;
    double cpi = double(cyc) / total_insts;
    printf("[MICRO] %s.128 CPI: %.8f  (cycles=%llu, insts=%.0f)\n",
           a.op, cpi, (unsigned long long)cyc, total_insts);
  }

  // ---------------- FULL MODE ----------------
  if (eq(a.mode,"full")) {
    // occupancy to saturate SMs
    int blocksPerSM=0; size_t shmem = eq(a.op,"ldg") ? 0 : (size_t)a.smem_f*sizeof(float);
    // choose a sensible block size if user left threads small
    if (a.threads < 64) a.threads = 256;
    if (eq(a.op,"ldg")) {
      CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, full_ldg_kernel<true>, a.threads, shmem));
    } else {
      CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, full_lds_kernel, a.threads, shmem));
    }
    int grid_blocks = blocksPerSM * sms;
    printf("Occupancy: %d blocks/SM → launching %d blocks x %d threads\n",
           blocksPerSM, grid_blocks, a.threads);

    // cooperative timing with %clock64 if supported
    bool coop = prop.cooperativeLaunch;
    float ms = 0.f;

    if (coop) {
      void* args_ldg[] = { &d_sink, &d_cycles, &d_gbuf, &a.iters, &a.unroll, &wss_elems, &a.cache };
      void* args_lds[] = { &d_sink, &d_cycles, &a.iters, &a.unroll, &a.smem_f };
      if (eq(a.op,"ldg")) {
        CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_ldg_kernel<true>,
          dim3(grid_blocks), dim3(a.threads), args_ldg, shmem));
      } else {
        CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_lds_kernel,
          dim3(grid_blocks), dim3(a.threads), args_lds, shmem));
      }
      CHECK_CUDA(cudaDeviceSynchronize());
      uint64_t cyc=0; CHECK_CUDA(cudaMemcpy(&cyc, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost));
      // grid_cycles = cycles on ONE SM for the timed region; total cycles = cyc * sms
      long double total_cycles = (long double)cyc * (long double)sms;
      long double total_insts  = (long double)grid_blocks * (long double)a.threads *
                                 (long double)a.iters * (long double)a.unroll;
      long double cpi = total_cycles / total_insts;
      printf("[FULL/coop] %s.128 CPI: %.8Lf  (grid_cycles=%llu on 1 SM → total=%.0Lf, insts=%.0Lf)\n",
             a.op, cpi, (unsigned long long)cyc, total_cycles, total_insts);
    } else {
      // fallback: CUDA events
      cudaEvent_t st, ed; CHECK_CUDA(cudaEventCreate(&st)); CHECK_CUDA(cudaEventCreate(&ed));
      if (eq(a.op,"ldg")) {
        void* args[] = { &d_sink, &d_cycles, &d_gbuf, &a.iters, &a.unroll, &wss_elems, &a.cache };
        CHECK_CUDA(cudaEventRecord(st));
        CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_ldg_kernel<true>,
                   dim3(grid_blocks), dim3(a.threads), args, shmem)); // if this fails, fall back to normal launch:
        CHECK_CUDA(cudaEventRecord(ed));
      } else {
        void* args[] = { &d_sink, &d_cycles, &a.iters, &a.unroll, &a.smem_f };
        CHECK_CUDA(cudaEventRecord(st));
        CHECK_CUDA(cudaLaunchCooperativeKernel((void*)full_lds_kernel,
                   dim3(grid_blocks), dim3(a.threads), args, shmem));
        CHECK_CUDA(cudaEventRecord(ed));
      }
      CHECK_CUDA(cudaEventSynchronize(ed));
      CHECK_CUDA(cudaEventElapsedTime(&ms, st, ed));
      double total_cycles = ms * 1e-3 * (double)prop.clockRate * 1e3 * (double)sms; // ms * kHz * SMs
      double total_insts  = (double)grid_blocks * (double)a.threads *
                            (double)a.iters * (double)a.unroll;
      double cpi = total_cycles / total_insts;
      printf("[FULL/events] %s.128 CPI: %.8f  (elapsed=%.3f ms, total_cycles=%.0f, insts=%.0f)\n",
             a.op, cpi, ms, total_cycles, total_insts);
    }
  }

  if (d_gbuf) cudaFree(d_gbuf);
  cudaFree(d_sink); cudaFree(d_cycles);
  return 0;
}

