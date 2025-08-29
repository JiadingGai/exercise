````markdown
# GPU Microbenchmark: CPI Overhead

This repository contains CUDA microbenchmarks for measuring CPI (cycles per instruction) overhead of different memory operations (`lds`, `ldg.ca`, `ldg.cg`) across GPU architectures.

---

## Build & Run

### Compile for Ampere (sm_86)

```bash
nvcc -O3 -o cpi_overhead cpi_overhead.cu --keep --arch=sm_86
./cpi_overhead --sweep
````

**Example output (RTX A5000, sm\_86):**

| op  | c  | unroll | iters | cycles   | insts     | CPI(thread) | CPIx32(warp) | CPI\_pure(t) | CPI\_purex32 |
| --- | -- | ------ | ----- | -------- | --------- | ----------- | ------------ | ------------ | ------------ |
| lds | -- | 1      | 90000 | 4950084  | 2880000   | 1.7188      | 55.0009      | 0.8750       | 28.0005      |
| lds | -- | 2      | 90000 | 5490082  | 5760000   | 0.9531      | 30.5005      | 0.5313       | 17.0003      |
| lds | -- | 4      | 90000 | 7470084  | 11520000  | 0.6484      | 20.7502      | 0.4063       | 13.0001      |
| lds | -- | 8      | 90000 | 12150043 | 23040000  | 0.5273      | 16.8751      | 0.3242       | 10.3750      |
| lds | -- | 16     | 90000 | 21330109 | 46080000  | 0.4629      | 14.8126      | 0.2773       | 8.8750       |
| lds | -- | 32     | 90000 | 39330237 | 92160000  | 0.4268      | 13.6563      | 0.2510       | 8.0313       |
| lds | -- | 64     | 90000 | 75420491 | 184320000 | 0.4092      | 13.0938      | 0.2373       | 7.5938       |
| ldg | ca | 1      | 90000 | 16830778 | 2880000   | 5.8440      | 187.0086     | --           | --           |
| ldg | ca | 2      | 90000 | 27123744 | 5760000   | 4.7090      | 150.6875     | --           | --           |
| ldg | ca | 4      | 90000 | 31302941 | 11520000  | 2.7173      | 86.9526      | --           | --           |
| ... |    |        |       |          |           |             |              |              |              |

---

### Compile for Blackwell (sm\_120)

```bash
nvcc -O3 -o cpi_overhead cpi_overhead.cu --keep --arch=sm_120
./cpi_overhead --sweep
```

**Example output (RTX 5090, sm\_120):**

| op  | c  | unroll | iters | cycles   | insts     | CPI(thread) | CPIx32(warp) | CPI\_pure(t) | CPI\_purex32 |
| --- | -- | ------ | ----- | -------- | --------- | ----------- | ------------ | ------------ | ------------ |
| lds | -- | 1      | 90000 | 6030064  | 2880000   | 2.0938      | 67.0007      | 1.1562       | 37.0000      |
| lds | -- | 2      | 90000 | 6210065  | 5760000   | 1.0781      | 34.5004      | 0.6094       | 19.4996      |
| lds | -- | 4      | 90000 | 8640065  | 11520000  | 0.7500      | 24.0002      | 0.4844       | 15.4998      |
| lds | -- | 8      | 90000 | 12060065 | 23040000  | 0.5234      | 16.7501      | 0.3125       | 9.9999       |
| lds | -- | 16     | 90000 | 20430065 | 46080000  | 0.4434      | 14.1875      | 0.2539       | 8.1249       |
| lds | -- | 32     | 90000 | 39150062 | 92160000  | 0.4248      | 13.5938      | 0.2471       | 7.9062       |
| lds | -- | 64     | 90000 | 76950050 | 184320000 | 0.4175      | 13.3594      | 0.2451       | 7.8437       |
| ldg | ca | 1      | 90000 | 21012079 | 2880000   | 7.2959      | 233.4675     | --           | --           |
| ldg | ca | 2      | 90000 | 37687294 | 5760000   | 6.5429      | 209.3739     | --           | --           |
| ldg | ca | 4      | 90000 | 43241280 | 11520000  | 3.7536      | 120.1147     | --           | --           |
| ... |    |        |       |          |           |             |              |              |              |

---

## Disassembly

Inspect kernels and instructions using `nvdisasm`:

```bash
nvdisasm -g -c ./cpi_2.sm_86.cubin | grep -n 'Function : micro_lds_kernel_spec'
nvdisasm -g -c ./cpi_2.sm_86.cubin | grep -n 'LDS\.128'
```

---

## SASS Editing with CuAssembler

### 1. Set up CuAssembler

```bash
export PATH=${PATH}:~/exercise/gpu_microbenchmark/CuAssembler/bin
export PYTHONPATH=${PYTHOPATH}:~/exercise/gpu_microbenchmark/CuAssembler/
```

### 2. Compile with `--keep`

```bash
nvcc -O3 -arch=sm_86 -o cpi_bench cpi_bench.cu -lcuda --keep
```

### 3. Disassemble

```bash
cuasm cpi_bench.sm_86.cubin -o cpi_bench.sm_86.cuasm
```

### 4. Edit SASS

Manually edit `cpi_bench.sm_86.cuasm` as needed.

### 5. Reassemble

```bash
cuasm cpi_bench.sm_86.cuasm -o cpi_bench.sm_86.patched.cubin
```

### 6. Run with patched cubin

```bash
./cpi_bench \
  --op lds --iters 90000 --smem-floats 4096 \
  --patched-cubin cpi_bench.sm_86.patched.cubin \
  --patched-symbol _Z16micro_lds_kernelILi64EEvPmS0_Pfii \
  --patched-unroll 64
```

---

## Notes on Registers & Alignment

* `SHI_REGISTERS` declares the register count usage for a kernel.
* If you change the destination register to a higher index, ensure it (and the next 3 for `.128`) are within the declared register count.
  *Example:* using `R32` requires at least **36 registers**.
* `.128` instructions require **16-byte aligned addresses**.
* To avoid WAW (write-after-write) serialization, rotate destination registers:

```sass
LDS.128 R48, [R35];
LDS.128 R52, [R35];
LDS.128 R56, [R35];
LDS.128 R48, [R35];
LDS.128 R52, [R35];
LDS.128 R56, [R35];
```

* Update `SHI_REGISTERS` accordingly (e.g., set to at least 64) in `_Z16micro_lds_kernelILi64EEvPmS0_Pfii`.

---
