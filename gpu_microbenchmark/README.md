#
nvcc -O3 -o cpi_overhead cpi_overhead.cu --keep --arch=sm_86

root@ceb39cf1eb54:~# ./cpi_overhead --sweep
Device: NVIDIA RTX A5000 sm_86 | SMs=64
Args: sweep=on iters=90000 smem_f=4096 wss_kb=256
op   c  unroll     iters         cycles          insts    CPI(thread)   CPIx32(warp)    CPI_pure(t)    CPI_purex32
---- -- ------ --------- -------------- -------------- -------------- -------------- -------------- --------------
lds  --      1     90000        4950084        2880000     1.71877917    55.00093333     0.87501667    28.00053333
lds  --      2     90000        5490082        5760000     0.95313924    30.50045556     0.53125833    17.00026667
lds  --      4     90000        7470084       11520000     0.64844479    20.75023333     0.40625391    13.00012500
lds  --      8     90000       12150043       23040000     0.52734562    16.87505972     0.32421836    10.37498750
lds  --     16     90000       21330109       46080000     0.46289299    14.81257569     0.27734377     8.87500069
lds  --     32     90000       39330237       92160000     0.42676038    13.65633229     0.25097688     8.03126007
lds  --     64     90000       75420491      184320000     0.40918235    13.09383524     0.23730507     7.59376215
ldg  ca      1     90000       16830778        2880000     5.84402014   187.00864444             --             --
ldg  ca      2     90000       27123744        5760000     4.70898333   150.68746667             --             --
ldg  ca      4     90000       31302941       11520000     2.71726918    86.95261389             --             --
ldg  ca      8     90000       61729604       23040000     2.67923628    85.73556111             --             --
ldg  ca     16     90000      111551868       46080000     2.42083047    77.46657500             --             --
ldg  ca     32     90000      213850321       92160000     2.32042449    74.25358368             --             --
ldg  ca     64     90000      417885561      184320000     2.26717427    72.54957656             --             --
ldg  cg      1     90000       26893387        2880000     9.33798160   298.81541111             --             --
ldg  cg      2     90000       28123003        5760000     4.88246580   156.23890556             --             --
ldg  cg      4     90000       31526773       11520000     2.73669905    87.57436944             --             --
ldg  cg      8     90000       80933184       23040000     3.51272500   112.40720000             --             --
ldg  cg     16     90000      116090514       46080000     2.51932539    80.61841250             --             --
ldg  cg     32     90000      225095601       92160000     2.44244359    78.15819479             --             --
ldg  cg     64     90000      442328081      184320000     2.39978343    76.79306962             --             --


120root@897174830b87:~# nvcc -O3 -o cpi_overhead cpi_overhead.cu --keep -arch sm_120
root@897174830b87:~# ./cpi_overhead --sweep
Device: NVIDIA GeForce RTX 5090 sm_120 | SMs=170
Args: sweep=on iters=90000 smem_f=4096 wss_kb=256
op   c  unroll     iters         cycles          insts    CPI(thread)   CPIx32(warp)    CPI_pure(t)    CPI_purex32
---- -- ------ --------- -------------- -------------- -------------- -------------- -------------- --------------
lds  --      1     90000        6030064        2880000     2.09377222    67.00071111     1.15622326    36.99914444
lds  --      2     90000        6210065        5760000     1.07813628    34.50036111     0.60936181    19.49957778
lds  --      4     90000        8640065       11520000     0.75000564    24.00018056     0.48436797    15.49977500
lds  --      8     90000       12060065       23040000     0.52344032    16.75009028     0.31249644     9.99988611
lds  --     16     90000       20430065       46080000     0.44336079    14.18754514     0.25390447     8.12494306
lds  --     32     90000       39150062       92160000     0.42480536    13.59377153     0.24706939     7.90622049
lds  --     64     90000       76950050      184320000     0.41748074    13.35938368     0.24511666     7.84373316
ldg  ca      1     90000       21012079        2880000     7.29586076   233.46754444             --             --
ldg  ca      2     90000       37687294        5760000     6.54293299   209.37385556             --             --
ldg  ca      4     90000       43241280       11520000     3.75358333   120.11466667             --             --
ldg  ca      8     90000       80402126       23040000     3.48967561   111.66961944             --             --
ldg  ca     16     90000      163921286       46080000     3.55731957   113.83422639             --             --
ldg  ca     32     90000      304164523       92160000     3.30039630   105.61268160             --             --
ldg  ca     64     90000      582697073      184320000     3.16133395   101.16268628             --             --
ldg  cg      1     90000       36067763        2880000    12.52352882   400.75292222             --             --
ldg  cg      2     90000       39314175        5760000     6.82537760   218.41208333             --             --
ldg  cg      4     90000       45232828       11520000     3.92646076   125.64674444             --             --
ldg  cg      8     90000       83215068       23040000     3.61176510   115.57648333             --             --
ldg  cg     16     90000      157736027       46080000     3.42309086   109.53890764             --             --
ldg  cg     32     90000      309969316       92160000     3.36338234   107.62823472             --             --
ldg  cg     64     90000      610985186      184320000     3.31480678   106.07381701             --             --

# 

nvdisasm -g -c ./cpi_2.sm_86.cubin | grep -n 'Function : micro_lds_kernel_spec'
nvdisasm -g -c ./cpi_2.sm_86.cubin | grep -n 'LDS\.128' -n

# sass editing

1. set up CuAssembler
   21  export PATH=${PATH}:~/exercise/gpu_microbenchmark/CuAssembler/bin
   22  export PYTHONPATH=${PYTHOPATH}:~/exercise/gpu_microbenchmark/CuAssembler/

2. nvcc compile cpi_bench.cu with --keep

nvcc -O3 -arch=sm_86 -o cpi_bench cpi_bench.cu -lcuda --keep 

3. run cuasm 

# disassemble
cuasm cpi_bench.sm_86.cubin -o cpi_bench.sm_86.cuasm
# edit cpi_bench.sm_86.cuasm
# reassemble
cuasm cpi_bench.sm_86.cuasm -o cpi_bench.sm_86.patched.cubin;
# run cpi_bench with cubin loading from patched cubin
./cpi_bench --op lds --iters 90000 --smem-floats 4096 --patched-cubin cpi_bench.sm_86.patched.cubin --patched-symbol _Z16micro_lds_kernelILi64EEvPmS0_Pfii --patched-unroll 64  

note that `SHI_REGISTERS` declared register count usage in a kernel.

If you change the dest to a higher register, make sure it (and the next 3) are within the kernel’s declared register count (SHI_REGISTERS). E.g., using R32 means you need at least 36 registers.

This register-index rule is separate from memory address alignment: for .128, keep the address 16-byte aligned. 

Don’t use the same dest quad back-to-back if you want throughput; rotate R20/R24/R28/... to avoid WAW serialization. As illustrate below, we are rotating R48/R52/R56/... over and over. But be
sure to set `SHI_REGISTERS` to at least 64

[B------:R-:W0:-:S01]         /*2da0*/                   LDS.128 R48, [R35] ;
[B------:R-:W0:-:S01]         /*2da0*/                   LDS.128 R52, [R35] ;
[B------:R-:W0:-:S01]         /*2da0*/                   LDS.128 R56, [R35] ;
[B------:R-:W0:-:S01]         /*2da0*/                   LDS.128 R48, [R35] ;
[B------:R-:W0:-:S01]         /*2da0*/                   LDS.128 R52, [R35] ;
[B------:R-:W0:-:S01]         /*2da0*/                   LDS.128 R56, [R35] ;
