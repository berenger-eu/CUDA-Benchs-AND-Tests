# What is this repo

This is a simple code where I played to perform a prefix-sum inside the warp.
More precisely, each thread in a warp has an unsigned int and wants to know
the sum of the unsigned ints of threads with lower ids.

```bash
# each thread has an scalar
unsigned int val = ...

# each thread want to have the sum of others

unsigned int sum = thread-0.val + ... + thread-X.val // for X < threadIdx.x
```

You will see a `NbLoops` in the code, this is only to do the same thing many times.

## With shared memory (`core_test`)

A simple implementation is that we perform the classic reduction pattern (in log2(32))
but we build all the prefix sum at the same time.
This implementation is called `core_test_sm` in the code.

```cpp
        #pragma unroll
        for(int idxIter = 1 ; idxIter <= 32 ; idxIter <<= 1){
            if(threadIdx.x & idxIter){
                ptr2[threadIdx.x] = ptr1[threadIdx.x] + ptr1[(threadIdx.x ^ idxIter) | (idxIter-1)];
            }
            else{
                ptr2[threadIdx.x] = ptr1[threadIdx.x];
            }
            swap(ptr1, ptr2);
        }  
```

## Using Warp Shuffle Functions V2 (`core_test_v2`)

The second approach is to use `__ballot_sync` followed by `__popc`.
The idea is to sum all bits at a given position to build the complete sum.
Using mask `subReductionMask` allow to filter the bits.

```cpp
        // sizeof(unsigned int)*8 can be changed if we know that it can be stoped earlier
        // or if threads exchange their values to know the max
        int prefixSum = 0;

        #pragma unroll
        for (int sumBitPosition = 0; sumBitPosition < sizeof(unsigned int)*8 ; ++sumBitPosition) {
            const unsigned int allBitsSum = __ballot_sync(allTrue, vec[threadIdx.x] & (1 << sumBitPosition));
            prefixSum += __popc(allBitsSum & subReductionMask) << sumBitPosition;
        }
        vec[threadIdx.x] = prefixSum;
```

Nice slides are available here:
https://pages.mini.pw.edu.pl/~kaczmarskik/gpca/resources/Part4-new-cuda-features.pdf

** It is important to understand that the first approach can not be optimized if we know that all the values
are lower than a given MAX. Whereas, the second approach (which sums the bits) could stop earlier
if it is known that there are not more 1s after a given position. **

# Results

Currently this is just too slow to use my algorithm with `__shfl_xor_sync` compared with the used of shared memory.
Even if the V2 is a little bit faster.

## My laptop (SM=75)

```bash
$ nvidia-smi 
Thu Sep 21 17:06:36 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA T600 Lap...  On   | 00000000:01:00.0 Off |                  N/A |
| N/A   53C    P0    12W /  35W |      5MiB /  4096MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2468      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Mon_Oct_24_19:12:58_PDT_2022
Cuda compilation tools, release 12.0, V12.0.76
Build cuda_12.0.r12.0/compiler.31968024_0
```

```bash
$ mkdir build && cd build && cmake .. -DCUDA_SM=75 -DCMAKE_BUILD_TYPE=RELEASE && make && ./main
# ...
Bench GPU:
core_test_cu = 1.34468
Done
Bench GPUv2:
core_test_cu = 2.19579
Done
```
# On a computing node (A100, SM=80)

```bash
$ nvidia-smi 
Thu Sep 21 17:17:45 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.43.04    Driver Version: 515.43.04    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-PCI...  On   | 00000000:21:00.0 Off |                    0 |
| N/A   50C    P0    71W / 250W |   5780MiB / 40960MiB |     99%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-PCI...  On   | 00000000:E2:00.0 Off |                    0 |
| N/A   32C    P0    33W / 250W |      2MiB / 40960MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     22633      C   ...n-3.11.0-cupy/bin/python3     2913MiB |
|    0   N/A  N/A     22703      C   ...n-3.11.0-cupy/bin/python3     2865MiB |
+-----------------------------------------------------------------------------+
```

```bash
$ module load compiler/gcc/11.2.0 compiler/cuda/11.6 build/cmake
$ mkdir build && cd build && cmake .. -DCUDA_SM=80 -DCMAKE_BUILD_TYPE=RELEASE && make && ./main
# ...
Bench GPU:
core_test_cu = 2.1448
Done
Bench GPUv2:
core_test_cu = 4.42243
Done
```
