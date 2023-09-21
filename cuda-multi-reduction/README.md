# What is this repo

This is a simple code where I played to perform a multi-summation inside the warp.
More precisely, each thread in a warp has an array of 32 values,
and each value in this array should be summed by the thread of the same warp index.
Somehow, we have 32 values per thread, and each thread want to do a reduction.

```bash
# each thread has an array
double buffer[32] = ...

# each thread should do the sum of all the array at a correponding position

double sum = thread-0.buffer[threadIdxInWarp] + ... + thread-32.buffer[threadIdxInWarp]
```

You will see a `NbLoops` in the code, this is only to do the same thing many times.

## With shared memory

A simple implementation is that each thread puts its values in a shared memory array
such that all threads can then directly access the values they needs.
This implementation is called `core_test_sm` in the code.

## Using Warp Shuffle Functions (`core_test`)

This is where the fun is.
At the time of writing, the available instructions to play inside a warp are not really flexible.
So I used `__shfl_xor_sync` only and the code can be summarized as:

```cpp
        for(int idx = 1 ; idx < 32 ; idx *= 2){
            const int neighIdx = targetNeigh(threadIdxInWarp, idx);
            buffer[threadIdxInWarp] += __shfl_xor_sync(0xffffffff, buffer[neighIdx], idx, 32);

            const int step = idx*2;
            for(int idxCoverage = step ; idxCoverage < 32 ; idxCoverage += step){
                const int recvFor = (threadIdxInWarp + idxCoverage)%32;
                const int sendFor = (neighIdx + idxCoverage)%32;
                buffer[recvFor] += __shfl_xor_sync(0xffffffff, buffer[sendFor], idx, 32);
            }
        }
```

To better understand `__shfl_xor_sync`, Prof Wes Armour's slides are great:
https://people.maths.ox.ac.uk/gilesm/cuda/2019/lecture_04.pdf

Here I have threads that communicate directly (so a thread X sends a value to a thread Y, and at the same time Y sends to X).
Also the mask I use (`idx`) is a power of two, so there is no way to have all threads communicating with other threads (improved in V2), this is why the threads also maintain partial sum for other indices.

Clearly I am doing more `add` operations then expected in a reduction algorithm, each threads does 5 outter iterations,
and for each outer iteration it will do 15, 7, 3, 1, 0 inner iterations resp., leading to
31 adds and 31 `__shfl_xor_sync`.
The same as if no reduction is used, so the difference between both implementation is mostly coming from the cost of `__shfl_xor_sync` and the cost of memory acceses.

## Using Warp Shuffle Functions V2 (`core_test_v2`)

In this case, a thread of index X will send a value to thread of index X-i MOD 32 and receive from X+i MOD 32.

```
        for(unsigned int idx = 1 ; idx < 32 ; idx += 1){
            const unsigned int neighPosDest = (threadIdxInWarp-threadIdxInWarp+32)%32;
            const unsigned int neighPosSrc = (threadIdxInWarp+threadIdxInWarp)%32;
            const unsigned int neighIdxSrc = (threadIdxInWarp ^ neighPosSrc);

            buffer[neighPosSrc] += __shfl_xor_sync(0xffffffff, buffer[neighPosDest], neighIdxSrc, 32);
        }
```

# Results

Currently this is just too slow to use my algorithm with `__shfl_xor_sync` compared with the used of shared memory.
Even if the V2 is a little bit faster.

## My laptop (SM=75)

```bash
$ nvidia-smi 
Thu Feb 23 10:21:49 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA T600 Lap...  On   | 00000000:01:00.0 Off |                  N/A |
| N/A   45C    P0    12W /  35W |      5MiB /  4096MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2423      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Mon_Oct_24_19:12:58_PDT_2022
Cuda compilation tools, release 12.0, V12.0.76
Build cuda_12.0.r12.0/compiler.31968024_0
```

```bash
$ mkdir build && cd build && cmake .. -DCUDA_SM=75 && make && ./main
# ...
 nbBlocksTest 16 nbThreadsTest 32
WARP = 0.167259
WARP V2 = 0.128066
SM = 0.0194225
 nbBlocksTest 16 nbThreadsTest 64
WARP = 0.153312
WARP V2 = 0.1065
SM = 0.0209584
 nbBlocksTest 16 nbThreadsTest 96
WARP = 0.250599
WARP V2 = 0.156695
SM = 0.0295795
 nbBlocksTest 16 nbThreadsTest 128
WARP = 0.331384
WARP V2 = 0.208941
SM = 0.038042
 nbBlocksTest 16 nbThreadsTest 160
WARP = 0.442718
WARP V2 = 0.262484
SM = 0.047527
 nbBlocksTest 16 nbThreadsTest 192
WARP = 0.643131
WARP V2 = 0.316857
SM = 0.0570808
 nbBlocksTest 32 nbThreadsTest 32
WARP = 0.120001
WARP V2 = 0.0959928
SM = 0.0159745
 nbBlocksTest 32 nbThreadsTest 64
WARP = 0.286181
WARP V2 = 0.18124
SM = 0.0326997
 nbBlocksTest 32 nbThreadsTest 96
WARP = 0.538695
WARP V2 = 0.272083
SM = 0.0569924
 nbBlocksTest 32 nbThreadsTest 128
WARP = 1.39577
WARP V2 = 0.374911
SM = 0.0744463
 nbBlocksTest 32 nbThreadsTest 160
WARP = 2.42909
WARP V2 = 0.48324
SM = 0.0931494
 nbBlocksTest 32 nbThreadsTest 192
WARP = 3.25863
WARP V2 = 0.736545
SM = 0.094319
 nbBlocksTest 64 nbThreadsTest 32
WARP = 0.287408
WARP V2 = 0.18028
SM = 0.0324493
 # I stopped here, too slow
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
$ module load compiler/gcc/11.2.0 compiler/cuda/11.6 build/cmake
$ mkdir build && cd build && cmake .. -DCUDA_SM=80 && make && ./main
# ...
nbBlocksTest 16 nbThreadsTest 32
WARP = 0.0729452
WARP V2 = 0.0389188
SM = 0.0165246
 nbBlocksTest 16 nbThreadsTest 64
WARP = 0.0720733
WARP V2 = 0.0422257
SM = 0.00896862
 nbBlocksTest 16 nbThreadsTest 96
WARP = 0.0848011
WARP V2 = 0.0633451
SM = 0.0126176
 nbBlocksTest 16 nbThreadsTest 128
WARP = 0.113254
WARP V2 = 0.0848415
SM = 0.0164596
 nbBlocksTest 16 nbThreadsTest 160
WARP = 0.141396
WARP V2 = 0.105961
SM = 0.0205414
 nbBlocksTest 16 nbThreadsTest 192
WARP = 0.169629
WARP V2 = 0.12714
SM = 0.0245955
 nbBlocksTest 32 nbThreadsTest 32
WARP = 0.0397979
WARP V2 = 0.0343995
SM = 0.00899463
 nbBlocksTest 32 nbThreadsTest 64
WARP = 0.0671064
WARP V2 = 0.0487229
SM = 0.00899312
 nbBlocksTest 32 nbThreadsTest 96
WARP = 0.0963828
WARP V2 = 0.0679383
SM = 0.0126132
 nbBlocksTest 32 nbThreadsTest 128
WARP = 0.124495
WARP V2 = 0.0928421
SM = 0.0164449
 nbBlocksTest 32 nbThreadsTest 160
WARP = 0.14876
WARP V2 = 0.111326
SM = 0.0205541
 nbBlocksTest 32 nbThreadsTest 192
WARP = 0.180274
WARP V2 = 0.13143
SM = 0.0246052
 nbBlocksTest 64 nbThreadsTest 32
WARP = 0.0477083
WARP V2 = 0.0487272
SM = 0.00900954
 nbBlocksTest 64 nbThreadsTest 64
WARP = 0.0935239
WARP V2 = 0.0824946
SM = 0.00900658
 nbBlocksTest 64 nbThreadsTest 96
WARP = 0.134018
WARP V2 = 0.106273
SM = 0.0126239
 nbBlocksTest 64 nbThreadsTest 128
WARP = 0.173774
WARP V2 = 0.132816
SM = 0.016458
 nbBlocksTest 64 nbThreadsTest 160
WARP = 0.199154
WARP V2 = 0.162009
SM = 0.0205691
 nbBlocksTest 64 nbThreadsTest 192
WARP = 0.231156
WARP V2 = 0.187189
SM = 0.0246405
 nbBlocksTest 128 nbThreadsTest 32
WARP = 0.103275
WARP V2 = 0.082292
SM = 0.00899972
 nbBlocksTest 128 nbThreadsTest 64
WARP = 0.198394
WARP V2 = 0.159926
SM = 0.0173639
 nbBlocksTest 128 nbThreadsTest 96
WARP = 0.301652
WARP V2 = 0.237579
SM = 0.0248702
 nbBlocksTest 128 nbThreadsTest 128
WARP = 0.36252
WARP V2 = 0.286938
SM = 0.0328621
 nbBlocksTest 128 nbThreadsTest 160
WARP = 0.447427
WARP V2 = 0.373833
SM = 0.0419506
 nbBlocksTest 128 nbThreadsTest 192
WARP = 0.545733
WARP V2 = 0.444307
SM = 0.0492161
 nbBlocksTest 256 nbThreadsTest 32
WARP = 0.186084
WARP V2 = 0.145655
SM = 0.0126669
 nbBlocksTest 256 nbThreadsTest 64
WARP = 0.330942
WARP V2 = 0.259498
SM = 0.0247743
 nbBlocksTest 256 nbThreadsTest 96
WARP = 0.464152
WARP V2 = 0.40436
SM = 0.0381125
 nbBlocksTest 256 nbThreadsTest 128
WARP = 0.609647
WARP V2 = 0.505612
SM = 0.0492398
 nbBlocksTest 256 nbThreadsTest 160
WARP = 0.792544
WARP V2 = 0.594863
SM = 0.0616559
 nbBlocksTest 256 nbThreadsTest 192
WARP = 0.91727
WARP V2 = 0.684436
SM = 0.0735986
 nbBlocksTest 512 nbThreadsTest 32
WARP = 0.322835
WARP V2 = 0.255214
SM = 0.0218017
 nbBlocksTest 512 nbThreadsTest 64
WARP = 0.574491
WARP V2 = 0.470452
SM = 0.041111
 nbBlocksTest 512 nbThreadsTest 96
WARP = 0.879575
WARP V2 = 0.644996
SM = 0.0614433
 nbBlocksTest 512 nbThreadsTest 128
WARP = 1.36668
WARP V2 = 0.824134
SM = 0.0817656
 nbBlocksTest 512 nbThreadsTest 160
WARP = 2.15138
WARP V2 = 1.00895
SM = 0.102351
 nbBlocksTest 512 nbThreadsTest 192
WARP = 3.21934
WARP V2 = 1.21111
SM = 0.122636


```
