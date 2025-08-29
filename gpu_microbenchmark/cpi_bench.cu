// cpi_bench_with_patched_loader.cu
// Micro-only CPI benchmark for LDG/LDS with:
//  - LDS mask-wrap addressing (branchless)
//  - Overhead-subtracted "pure LDS" CPI
//  - Optional loading of a patched SASS cubin + kernel symbol via Driver API
//
// Build:
//   nvcc -O3 -arch=sm_86 -o cpi_bench cpi_bench_with_patched_loader.cu -lcuda
//
// Examples:
//   # normal compiled kernels
//   ./cpi_bench --sweep
//   ./cpi_bench --op lds --unroll 64 --iters 90000 --smem-floats 4096
//
//   # launch a patched SASS kernel (symbol must match the cubin)
//   ./cpi_bench --op lds --iters 90000 --smem-floats 4096 \
//     --patched-cubin cpi.patched.cubin \
//     --patched-symbol _Z16micro_lds_kernelILi64EEvPmS0_Pfii \
//     --patched-unroll 64

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  std::exit(1);} } while(0)
#define CHECK_CU(x) do { CUresult _e=(x); if(_e!=CUDA_SUCCESS){ \
  const char* n=nullptr; const char* s=nullptr; cuGetErrorName(_e,&n); cuGetErrorString(_e,&s); \
  fprintf(stderr,"CUDA Driver error %s:%d: %s (%s)\n", __FILE__, __LINE__, n?n:"?", s?s:"?"); \
  std::exit(1);} } while(0)

struct Args {
  bool  sweep     = false;      // run all combos
  const char* op  = "lds";     // "lds" or "ldg"
  const char* cache = "ca";    // ldg cache: "ca" or "cg"
  int   iters    = 90000;
  int   unroll   = 8;           // {1,2,4,8,16,32,64}
  int   threads  = 32;          // single warp
  int   smem_f   = 4096;        // floats
  int   wss_kb   = 256;         // ldg working set
  // Patched SASS support
  const char* patched_cubin  = nullptr;   // path to .cubin
  const char* patched_symbol = nullptr;   // mangled kernel symbol
  int         patched_unroll = 0;         // for CPI denominator only
};

static inline bool eq(const char* a, const char* b){ return std::strcmp(a,b)==0; }

__device__ __forceinline__ unsigned long long rdclk() {
  unsigned long long c; asm volatile("mov.u64 %0, %%clock64;" : "=l"(c)); return c;
}

// ---- 128-bit loads ----
__device__ __forceinline__
void ldg128_once(const float* __restrict__ p,
                 float &x0, float &x1, float &x2, float &x3,
                 int use_cg) // 0=ca, 1=cg
{
  unsigned long long gaddr = __cvta_generic_to_global((void*)p);
  if (use_cg) {
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
      : "=&f"(x0), "=&f"(x1), "=&f"(x2), "=&f"(x3) : "l"(gaddr));
  } else {
    asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];"
      : "=&f"(x0), "=&f"(x1), "=&f"(x2), "=&f"(x3) : "l"(gaddr));
  }
}
__device__ __forceinline__
void lds128_once(const float* __restrict__ p,
                 float &x0, float &x1, float &x2, float &x3)
{
  unsigned long long saddr = __cvta_generic_to_shared((void*)p);
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
    : "=&f"(x0), "=&f"(x1), "=&f"(x2), "=&f"(x3) : "l"(saddr));
}

// ---- helpers ----
__device__ __forceinline__ int pow2_floor(int x) {
  x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; return (x + 1) >> 1;
}

// ---- unrolled bodies ----
// LDG with compare wrap
template<int UNROLL>
__device__ __forceinline__
void do_unrolled_ldg(const float* __restrict__ gptr,
                     int &g_idx, const int g_wrap,
                     float &a0, float &a1, float &a2, float &a3,
                     int use_cg)
{
  static_assert(UNROLL==1 || UNROLL==2 || UNROLL==4 || UNROLL==8 || UNROLL==16 || UNROLL==32 || UNROLL==64,
                "UNROLL must be {1,2,4,8,16,32,64}");
#pragma unroll
  for (int u=0; u<UNROLL; ++u) {
    float x0,x1,x2,x3;
    ldg128_once(gptr + g_idx, x0,x1,x2,x3, use_cg);
    a0+=x0; a1+=x1; a2+=x2; a3+=x3;
    g_idx += 4; if (g_idx >= g_wrap) g_idx = 0; // 16B step
  }
}

// LDS mask-wrap (index in bytes)
template<int UNROLL>
__device__ __forceinline__
void do_unrolled_lds_mask(float* __restrict__ smem_base,
                          int &ofs_bytes, const int mask_bytes,
                          float &a0, float &a1, float &a2, float &a3)
{
  static_assert(UNROLL==1 || UNROLL==2 || UNROLL==4 || UNROLL==8 || UNROLL==16 || UNROLL==32 || UNROLL==64,
                "UNROLL must be {1,2,4,8,16,32,64}");
#pragma unroll
  for (int u=0; u<UNROLL; ++u) {
    float x0,x1,x2,x3;
    char* cptr = reinterpret_cast<char*>(smem_base) + ofs_bytes; // 16B-aligned
    lds128_once(reinterpret_cast<float*>(cptr), x0,x1,x2,x3);
    a0+=x0; a1+=x1; a2+=x2; a3+=x3;          // 4 FADDs per vector
    ofs_bytes = (ofs_bytes + 16) & mask_bytes; // (idx+16) & mask
  }
}

// Overhead-only body (no LDS): index math + 4 FADDs per step
template<int UNROLL>
__device__ __forceinline__
void do_overhead_mask(float* __restrict__ /*smem_base_unused*/,
                      int &ofs_bytes, const int mask_bytes,
                      float &a0, float &a1, float &a2, float &a3)
{
#pragma unroll
  for (int u=0; u<UNROLL; ++u) {
    asm volatile("add.f32 %0, %0, %1;" : "+f"(a0) : "f"(a1));
    asm volatile("add.f32 %0, %0, %1;" : "+f"(a1) : "f"(a2));
    asm volatile("add.f32 %0, %0, %1;" : "+f"(a2) : "f"(a3));
    asm volatile("add.f32 %0, %0, %1;" : "+f"(a3) : "f"(a0));
    ofs_bytes = (ofs_bytes + 16) & mask_bytes;
  }
}

// ---- MICRO kernels (single warp) ----
// LDG micro
template<int UNROLL>
__global__ void micro_ldg_kernel(uint64_t* cycles_with, float* sink,
                                 const float* __restrict__ gbuf,
                                 int iters, int wss_elems, int use_cg)
{
  const int lane = threadIdx.x & 31;
  int g_idx  = (lane * 4) % (wss_elems - 4);
  const int g_wrap = (wss_elems - 4);
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f;

  __syncthreads();
  const unsigned long long t0 = rdclk();
#pragma unroll 1
  for (int i=0;i<iters;++i){
    do_unrolled_ldg<UNROLL>(gbuf, g_idx, g_wrap, a0,a1,a2,a3, use_cg);
  }
  __syncthreads();
  const unsigned long long t1 = rdclk();
  if (threadIdx.x==0) cycles_with[0] = (t1 - t0);
  if (threadIdx.x==0) *sink = a0+a1+a2+a3;
}

// LDS micro with overhead-subtracted timing
template<int UNROLL>
__global__ void micro_lds_kernel(uint64_t* cycles_with, uint64_t* cycles_ovh, float* sink,
                                 int iters, int smem_floats)
{
  extern __shared__ __align__(16) float smem[];
  for (int i=threadIdx.x;i<smem_floats;i+=blockDim.x) smem[i]=float(i&1023)*0.25f;
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int smem_bytes  = smem_floats * (int)sizeof(float);
  const int ring_bytes  = pow2_floor(smem_bytes);
  const int mask_bytes  = ring_bytes - 16;
  int ofs0 = ((lane * 16) & mask_bytes);

  float a0=0.f,a1=0.f,a2=0.f,a3=0.f; int ofs_ovh = ofs0;
  __syncthreads(); unsigned long long t0 = rdclk();
#pragma unroll 1
  for (int i=0;i<iters;++i){ do_overhead_mask<UNROLL>(smem, ofs_ovh, mask_bytes, a0,a1,a2,a3); }
  __syncthreads(); unsigned long long t1 = rdclk();

  float b0=0.f,b1=0.f,b2=0.f,b3=0.f; int ofs_ld = ofs0;
  __syncthreads(); unsigned long long t2 = rdclk();
#pragma unroll 1
  for (int i=0;i<iters;++i){ do_unrolled_lds_mask<UNROLL>(smem, ofs_ld, mask_bytes, b0,b1,b2,b3); }
  __syncthreads(); unsigned long long t3 = rdclk();

  if (threadIdx.x==0) { cycles_ovh[0] = (t1 - t0); cycles_with[0]= (t3 - t2); }
  if (threadIdx.x==0) *sink = (a0+a1+a2+a3) + (b0+b1+b2+b3);
}

// ---- runtime dispatch ----
template<int U>
static void launch_micro_ldg(dim3 grid, dim3 block,
                             uint64_t* d_cycles_with, float* d_sink,
                             const float* d_gbuf,
                             int iters, int wss_elems, int use_cg)
{ micro_ldg_kernel<U><<<grid, block>>>(d_cycles_with, d_sink, d_gbuf, iters, wss_elems, use_cg); }

template<int U>
static void launch_micro_lds(dim3 grid, dim3 block, size_t shmem,
                             uint64_t* d_cycles_with, uint64_t* d_cycles_ovh, float* d_sink,
                             int iters, int smem_floats)
{ micro_lds_kernel<U><<<grid, block, shmem>>>(d_cycles_with, d_cycles_ovh, d_sink, iters, smem_floats); }

static void dispatch_ldg_unroll(int unroll, dim3 grid, dim3 block,
                                uint64_t* d_cycles_with, float* d_sink,
                                const float* d_gbuf,
                                int iters, int wss_elems, int use_cg)
{
  switch (unroll) {
    case 1:  launch_micro_ldg<1 >(grid, block, d_cycles_with, d_sink, d_gbuf, iters, wss_elems, use_cg); break;
    case 2:  launch_micro_ldg<2 >(grid, block, d_cycles_with, d_sink, d_gbuf, iters, wss_elems, use_cg); break;
    case 4:  launch_micro_ldg<4 >(grid, block, d_cycles_with, d_sink, d_gbuf, iters, wss_elems, use_cg); break;
    case 8:  launch_micro_ldg<8 >(grid, block, d_cycles_with, d_sink, d_gbuf, iters, wss_elems, use_cg); break;
    case 16: launch_micro_ldg<16>(grid, block, d_cycles_with, d_sink, d_gbuf, iters, wss_elems, use_cg); break;
    case 32: launch_micro_ldg<32>(grid, block, d_cycles_with, d_sink, d_gbuf, iters, wss_elems, use_cg); break;
    default: launch_micro_ldg<64>(grid, block, d_cycles_with, d_sink, d_gbuf, iters, wss_elems, use_cg); break;
  }
}

static void dispatch_lds_unroll(int unroll, dim3 grid, dim3 block, size_t shmem,
                                uint64_t* d_cycles_with, uint64_t* d_cycles_ovh, float* d_sink,
                                int iters, int smem_floats)
{
  switch (unroll) {
    case 1:  launch_micro_lds<1 >(grid, block, shmem, d_cycles_with, d_cycles_ovh, d_sink, iters, smem_floats); break;
    case 2:  launch_micro_lds<2 >(grid, block, shmem, d_cycles_with, d_cycles_ovh, d_sink, iters, smem_floats); break;
    case 4:  launch_micro_lds<4 >(grid, block, shmem, d_cycles_with, d_cycles_ovh, d_sink, iters, smem_floats); break;
    case 8:  launch_micro_lds<8 >(grid, block, shmem, d_cycles_with, d_cycles_ovh, d_sink, iters, smem_floats); break;
    case 16: launch_micro_lds<16>(grid, block, shmem, d_cycles_with, d_cycles_ovh, d_sink, iters, smem_floats); break;
    case 32: launch_micro_lds<32>(grid, block, shmem, d_cycles_with, d_cycles_ovh, d_sink, iters, smem_floats); break;
    default: launch_micro_lds<64>(grid, block, shmem, d_cycles_with, d_cycles_ovh, d_sink, iters, smem_floats); break;
  }
}

// ---- Driver API launcher for patched SASS LDS kernel ----
static void launch_patched_lds(const char* cubin, const char* symbol,
                               uint64_t* d_cycles_with, uint64_t* d_cycles_ovh,
                               float* d_sink, int iters, int smem_floats,
                               size_t shmem_bytes)
{
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CU(cuInit(0));
  CUcontext ctx=nullptr; CHECK_CU(cuCtxGetCurrent(&ctx));
  if (!ctx) { CUdevice dev; CHECK_CU(cuDeviceGet(&dev,0)); CHECK_CU(cuDevicePrimaryCtxRetain(&ctx,dev)); CHECK_CU(cuCtxSetCurrent(ctx)); }

  CUmodule mod; CHECK_CU(cuModuleLoad(&mod, cubin));
  CUfunction fn; CHECK_CU(cuModuleGetFunction(&fn, mod, symbol));

  void* params[] = { &d_cycles_with, &d_cycles_ovh, &d_sink, &iters, &smem_floats };
  CHECK_CU(cuLaunchKernel(fn, 1,1,1, 32,1,1, (unsigned)shmem_bytes, 0, params, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CU(cuModuleUnload(mod));
}

int main(int argc, char** argv) {
  Args a;
  for (int i=1;i<argc;i++){
    if (!std::strncmp(argv[i],"--sweep",7)) a.sweep = true;
    else if (!std::strncmp(argv[i],"--op",4) && i+1<argc) a.op=argv[++i];
    else if (!std::strncmp(argv[i],"--cache",7) && i+1<argc) a.cache=argv[++i];
    else if (!std::strncmp(argv[i],"--iters",7) && i+1<argc) a.iters=std::atoi(argv[++i]);
    else if (!std::strncmp(argv[i],"--unroll",8) && i+1<argc) a.unroll=std::atoi(argv[++i]);
    else if (!std::strncmp(argv[i],"--threads",9) && i+1<argc) a.threads=std::atoi(argv[++i]);
    else if (!std::strncmp(argv[i],"--smem-floats",13) && i+1<argc) a.smem_f=std::atoi(argv[++i]);
    else if (!std::strncmp(argv[i],"--wss-kb",8) && i+1<argc) a.wss_kb=std::atoi(argv[++i]);
    else if (!std::strncmp(argv[i],"--patched-cubin",15) && i+1<argc) a.patched_cubin=argv[++i];
    else if (!std::strncmp(argv[i],"--patched-symbol",16) && i+1<argc) a.patched_symbol=argv[++i];
    else if (!std::strncmp(argv[i],"--patched-unroll",16) && i+1<argc) a.patched_unroll=std::atoi(argv[++i]);
  }
  if (!(a.unroll==1 || a.unroll==2 || a.unroll==4 || a.unroll==8 || a.unroll==16 || a.unroll==32 || a.unroll==64)) a.unroll=8;
  a.threads = 32; // single warp

  int dev=0; CHECK_CUDA(cudaSetDevice(dev));
  cudaDeviceProp prop{}; CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s sm_%d%d | SMs=%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

  // Buffers common
  uint64_t *d_cycles_with=nullptr, *d_cycles_ovh=nullptr; float *d_sink=nullptr;
  CHECK_CUDA(cudaMalloc(&d_cycles_with, sizeof(uint64_t)));
  CHECK_CUDA(cudaMalloc(&d_cycles_ovh,  sizeof(uint64_t)));
  CHECK_CUDA(cudaMalloc(&d_sink, sizeof(float)));
  CHECK_CUDA(cudaMemset(d_cycles_with, 0, sizeof(uint64_t)));
  CHECK_CUDA(cudaMemset(d_cycles_ovh,  0, sizeof(uint64_t)));
  CHECK_CUDA(cudaMemset(d_sink, 0, sizeof(float)));

  // LDG global buf
  const int wss_elems = (a.wss_kb * 1024) / (int)sizeof(float);
  float* d_gbuf=nullptr; CHECK_CUDA(cudaMalloc(&d_gbuf, (size_t)wss_elems * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_gbuf, 0, (size_t)wss_elems * sizeof(float)));

  const int UNROLLS[7] = {1,2,4,8,16,32,64};
  dim3 grid(1), block(32);

  auto run_one = [&](const char* op, const char* cache, int unroll)->void{
    // warmup for LDG
    if (eq(op,"ldg")) {
      micro_ldg_kernel<4><<<grid, block>>>(d_cycles_with, d_sink, d_gbuf, 1024, wss_elems, eq(cache,"cg")?1:0);
      CHECK_CUDA(cudaDeviceSynchronize());
    }

    if (eq(op,"ldg")) {
      dispatch_ldg_unroll(unroll, grid, block, d_cycles_with, d_sink, d_gbuf,
                          a.iters, wss_elems, eq(cache,"cg")?1:0);
    } else { // lds
      size_t shmem = (size_t)a.smem_f * sizeof(float);
      if (a.patched_cubin && a.patched_symbol) {
        launch_patched_lds(a.patched_cubin, a.patched_symbol,
                           d_cycles_with, d_cycles_ovh, d_sink,
                           a.iters, a.smem_f, shmem);
      } else {
        dispatch_lds_unroll(unroll, grid, block, shmem,
                            d_cycles_with, d_cycles_ovh, d_sink,
                            a.iters, a.smem_f);
      }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // readback + print
    uint64_t cyc_with=0, cyc_ovh=0; CHECK_CUDA(cudaMemcpy(&cyc_with, d_cycles_with, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    if (eq(op,"lds")) CHECK_CUDA(cudaMemcpy(&cyc_ovh,  d_cycles_ovh,  sizeof(uint64_t), cudaMemcpyDeviceToHost));

    int used_unroll = (eq(op,"lds") && a.patched_cubin && a.patched_symbol && a.patched_unroll>0) ? a.patched_unroll : unroll;
    const double insts = double(a.iters) * double(used_unroll) * 32.0;
    const double cpi_t = double(cyc_with) / insts;
    const double cpi_w = cpi_t * 32.0;

    if (eq(op,"lds")) {
      const double pure_cycles = (cyc_with > cyc_ovh) ? double(cyc_with - cyc_ovh) : 0.0;
      const double cpi_pure_t  = pure_cycles / insts;
      const double cpi_pure_w  = cpi_pure_t * 32.0;
      printf("%-4s %-2s %6d %9d %14llu %14.0f %14.8f %14.8f %14.8f %14.8f\n",
             op, "--", used_unroll, a.iters,
             (unsigned long long)cyc_with, insts,
             cpi_t, cpi_w, cpi_pure_t, cpi_pure_w);
    } else {
      printf("%-4s %-2s %6d %9d %14llu %14.0f %14.8f %14.8f %14s %14s\n",
             op, eq(cache,"cg")?"cg":"ca", used_unroll, a.iters,
             (unsigned long long)cyc_with, insts,
             cpi_t, cpi_w, "--", "--");
    }
  };

  if (a.sweep) {
    printf("Args: sweep=on iters=%d smem_f=%d wss_kb=%d\n", a.iters, a.smem_f, a.wss_kb);
    printf("%-4s %-2s %6s %9s %14s %14s %14s %14s %14s %14s\n",
           "op","c","unroll","iters","cycles","insts","CPI(thread)","CPIx32(warp)","CPI_pure(t)","CPI_purex32");
    printf("%-4s %-2s %6s %9s %14s %14s %14s %14s %14s %14s\n",
           "----","--","------","---------","--------------","--------------","--------------","--------------","--------------","--------------");

    for (int u : UNROLLS) run_one("lds","--",u);
    for (const char* cflag : (const char*[]){"ca","cg"}) {
      for (int u : UNROLLS) run_one("ldg", cflag, u);
    }
  } else {
    printf("Args: op=%s cache=%s iters=%d unroll=%d smem_f=%d wss_kb=%d\n",
           a.op, a.cache, a.iters, a.unroll, a.smem_f, a.wss_kb);
    printf("%-4s %-2s %6s %9s %14s %14s %14s %14s %14s %14s\n",
           "op","c","unroll","iters","cycles","insts","CPI(thread)","CPIx32(warp)","CPI_pure(t)","CPI_purex32");
    printf("%-4s %-2s %6s %9s %14s %14s %14s %14s %14s %14s\n",
           "----","--","------","---------","--------------","--------------","--------------","--------------","--------------","--------------");
    run_one(a.op, a.cache, a.unroll);
  }

  if (d_gbuf) cudaFree(d_gbuf);
  cudaFree(d_sink); cudaFree(d_cycles_with); cudaFree(d_cycles_ovh);
  return 0;
}

