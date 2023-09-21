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

* It is important to understand that the first approach can not be optimized if we know that all the values
are lower than a given MAX. Whereas, the second approach (which sums the bits) could stop earlier
if it is known that there are not more 1s after a given position. *

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
Thu Feb 23 10:25:59 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-PCI...  On   | 00000000:21:00.0 Off |                    0 |
| N/A   32C    P0    32W / 250W |      0MiB / 40960MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-PCI...  On   | 00000000:E2:00.0 Off |                    0 |
| N/A   31C    P0    32W / 250W |      0MiB / 40960MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

```bash
$ mkdir build && cd build && cmake .. -DCUDA_SM=80 -DCMAKE_BUILD_TYPE=RELEASE && make && ./main
# ...
TODO
```
