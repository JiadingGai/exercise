// cpi_ldx_micro_sweep.cu  (now prints CPI and CPIx32 = warp-level CPI)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda.h>

#define CHECK_CUDA(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  std::exit(1);} } while(0)

struct Args {
  bool  sweep     = false;      // run all combinations if true
  const char* op  = "lds";      // "lds" or "ldg" (single-run mode)
  const char* cache = "ca";     // ldg cache: "ca" or "cg" (single-run mode)
  int   iters    = 90000;       // loop iterations
  int   unroll   = 8;           // 1,2,4,8,16,32,64 (single-run mode)
  int   threads  = 32;          // fixed: single warp
  int   smem_f   = 4096;        // shared floats per block (LDS)
  int   wss_kb   = 256;         // LDG working set (KB)
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

// ---- unrolled bodies ----
template<int UNROLL>
__device__ __forceinline__
void do_unrolled_ldg(const float* __restrict__ gptr,
                     int &g_idx, const int g_wrap,
                     float &a0, float &a1, float &a2, float &a3,
                     int use_cg)
{
  static_assert(UNROLL==1 || UNROLL==2 || UNROLL==4 || UNROLL==8 ||
                UNROLL==16 || UNROLL==32 || UNROLL==64,
                "UNROLL must be {1,2,4,8,16,32,64}");
#pragma unroll
  for (int u=0; u<UNROLL; ++u) {
    float x0,x1,x2,x3;
    ldg128_once(gptr + g_idx, x0,x1,x2,x3, use_cg);
    a0+=x0; a1+=x1; a2+=x2; a3+=x3;
    g_idx += 4; if (g_idx >= g_wrap) g_idx = 0; // 16B stride
  }
}
template<int UNROLL>
__device__ __forceinline__
void do_unrolled_lds(const float* __restrict__ sptr,
                     int &s_idx, const int s_wrap,
                     float &a0, float &a1, float &a2, float &a3)
{
  static_assert(UNROLL==1 || UNROLL==2 || UNROLL==4 || UNROLL==8 ||
                UNROLL==16 || UNROLL==32 || UNROLL==64,
                "UNROLL must be {1,2,4,8,16,32,64}");
#pragma unroll
  for (int u=0; u<UNROLL; ++u) {
    float x0,x1,x2,x3;
    lds128_once(sptr + s_idx, x0,x1,x2,x3);
    a0+=x0; a1+=x1; a2+=x2; a3+=x3;
    s_idx += 4; if (s_idx >= s_wrap) s_idx = 0; // 16B stride
  }
}

// ---- MICRO kernels (single warp) ----
template<int UNROLL>
__global__ void micro_ldg_kernel(uint64_t* cycles, float* sink,
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
  if (threadIdx.x==0) cycles[0] = (t1 - t0);
  if (threadIdx.x==0) *sink = a0+a1+a2+a3;
}
template<int UNROLL>
__global__ void micro_lds_kernel(uint64_t* cycles, float* sink,
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
  const unsigned long long t0 = rdclk();
#pragma unroll 1
  for (int i=0;i<iters;++i){
    do_unrolled_lds<UNROLL>(smem, s_idx, s_wrap, a0,a1,a2,a3);
  }
  __syncthreads();
  const unsigned long long t1 = rdclk();
  if (threadIdx.x==0) cycles[0] = (t1 - t0);
  if (threadIdx.x==0) *sink = a0+a1+a2+a3;
}

// ---- runtime dispatch helpers ----
template<int UNROLL>
static void launch_micro_ldg(dim3 grid, dim3 block,
                             uint64_t* d_cycles, float* d_sink,
                             const float* d_gbuf,
                             int iters, int wss_elems, int use_cg)
{
  micro_ldg_kernel<UNROLL><<<grid, block>>>(d_cycles, d_sink, d_gbuf, iters, wss_elems, use_cg);
}
template<int UNROLL>
static void launch_micro_lds(dim3 grid, dim3 block, size_t shmem,
                             uint64_t* d_cycles, float* d_sink,
                             int iters, int smem_floats)
{
  micro_lds_kernel<UNROLL><<<grid, block, shmem>>>(d_cycles, d_sink, iters, smem_floats);
}

static void dispatch_ldg_unroll(int unroll, dim3 grid, dim3 block,
                                uint64_t* d_cycles, float* d_sink,
                                const float* d_gbuf,
                                int iters, int wss_elems, int use_cg)
{
  switch (unroll) {
    case 1:  launch_micro_ldg<1 >(grid, block, d_cycles, d_sink, d_gbuf, iters, wss_elems, use_cg); break;
    case 2:  launch_micro_ldg<2 >(grid, block, d_cycles, d_sink, d_gbuf, iters, wss_elems, use_cg); break;
    case 4:  launch_micro_ldg<4 >(grid, block, d_cycles, d_sink, d_gbuf, iters, wss_elems, use_cg); break;
    case 8:  launch_micro_ldg<8 >(grid, block, d_cycles, d_sink, d_gbuf, iters, wss_elems, use_cg); break;
    case 16: launch_micro_ldg<16>(grid, block, d_cycles, d_sink, d_gbuf, iters, wss_elems, use_cg); break;
    case 32: launch_micro_ldg<32>(grid, block, d_cycles, d_sink, d_gbuf, iters, wss_elems, use_cg); break;
    default: launch_micro_ldg<64>(grid, block, d_cycles, d_sink, d_gbuf, iters, wss_elems, use_cg); break;
  }
}
static void dispatch_lds_unroll(int unroll, dim3 grid, dim3 block, size_t shmem,
                                uint64_t* d_cycles, float* d_sink,
                                int iters, int smem_floats)
{
  switch (unroll) {
    case 1:  launch_micro_lds<1 >(grid, block, shmem, d_cycles, d_sink, iters, smem_floats); break;
    case 2:  launch_micro_lds<2 >(grid, block, shmem, d_cycles, d_sink, iters, smem_floats); break;
    case 4:  launch_micro_lds<4 >(grid, block, shmem, d_cycles, d_sink, iters, smem_floats); break;
    case 8:  launch_micro_lds<8 >(grid, block, shmem, d_cycles, d_sink, iters, smem_floats); break;
    case 16: launch_micro_lds<16>(grid, block, shmem, d_cycles, d_sink, iters, smem_floats); break;
    case 32: launch_micro_lds<32>(grid, block, shmem, d_cycles, d_sink, iters, smem_floats); break;
    default: launch_micro_lds<64>(grid, block, shmem, d_cycles, d_sink, iters, smem_floats); break;
  }
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
  }
  if (!(a.unroll==1 || a.unroll==2 || a.unroll==4 || a.unroll==8 ||
        a.unroll==16 || a.unroll==32 || a.unroll==64)) a.unroll=8;
  a.threads = 32;

  int dev=0; CHECK_CUDA(cudaSetDevice(dev));
  cudaDeviceProp prop{}; CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s sm_%d%d | SMs=%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

  uint64_t *d_cycles=nullptr; float *d_sink=nullptr;
  CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(uint64_t)));
  CHECK_CUDA(cudaMalloc(&d_sink, sizeof(float)));
  CHECK_CUDA(cudaMemset(d_cycles, 0, sizeof(uint64_t)));
  CHECK_CUDA(cudaMemset(d_sink, 0, sizeof(float)));

  const int wss_elems = (a.wss_kb * 1024) / (int)sizeof(float);
  float* d_gbuf=nullptr; CHECK_CUDA(cudaMalloc(&d_gbuf, (size_t)wss_elems * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_gbuf, 0, (size_t)wss_elems * sizeof(float)));

  const int UNROLLS[7] = {1,2,4,8,16,32,64};
  dim3 grid(1), block(32);

  auto run_one = [&](const char* op, const char* cache, int unroll)->void{
    // warmup for LDG
    if (eq(op,"ldg")) {
      micro_ldg_kernel<4><<<grid, block>>>(d_cycles, d_sink, d_gbuf, 1024, wss_elems, eq(cache,"cg")?1:0);
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    // launch
    if (eq(op,"ldg")) {
      dispatch_ldg_unroll(unroll, grid, block, d_cycles, d_sink, d_gbuf,
                          a.iters, wss_elems, eq(cache,"cg")?1:0);
    } else {
      size_t shmem = (size_t)a.smem_f * sizeof(float);
      dispatch_lds_unroll(unroll, grid, block, shmem, d_cycles, d_sink,
                          a.iters, a.smem_f);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // readback + print
    uint64_t cyc=0; CHECK_CUDA(cudaMemcpy(&cyc, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    const double insts = double(a.iters) * double(unroll) * 32.0;
    const double cpi_t = double(cyc) / insts;      // per-thread CPI
    const double cpi_w = cpi_t * 32.0;             // warp-level CPI
    printf("%-4s %-2s %6d %9d %14llu %14.0f %12.8f %12.8f\n",
           op, eq(op,"ldg")? (eq(cache,"cg")?"cg":"ca") : "--",
           unroll, a.iters, (unsigned long long)cyc, insts, cpi_t, cpi_w);
  };

  if (a.sweep) {
    printf("Args: sweep=on iters=%d smem_f=%d wss_kb=%d\n", a.iters, a.smem_f, a.wss_kb);
    printf("op   c  unroll     iters          cycles          insts       CPI(thread)   CPIx32(warp)\n");
    printf("---- -- ------ --------- -------------- -------------- -------------- --------------\n");
    // LDS
    for (int u: UNROLLS) run_one("lds","--",u);
    // LDG (ca & cg)
    for (const char* cflag : (const char*[]){"ca","cg"}) {
      for (int u: UNROLLS) run_one("ldg", cflag, u);
    }
  } else {
    printf("Args: op=%s cache=%s iters=%d unroll=%d smem_f=%d wss_kb=%d\n",
           a.op, a.cache, a.iters, a.unroll, a.smem_f, a.wss_kb);
    printf("op   c  unroll     iters          cycles          insts       CPI(thread)   CPIx32(warp)\n");
    printf("---- -- ------ --------- -------------- -------------- -------------- --------------\n");
    run_one(a.op, a.cache, a.unroll);
  }

  if (d_gbuf) cudaFree(d_gbuf);
  cudaFree(d_sink); cudaFree(d_cycles);
  return 0;
}

