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

## With shared memory

A simple implementation is that each thread puts its values in a shared memory array
such that all threads can then directly access the values they needs.
This implementation is called `core_test_sm` in the code.

## Using Warp Shuffle Functions

This is where the fun is.
At the time of writing, the available instructions to play inside a warp are not really flexible.
So I used `__shfl_xor_sync` only and the code can be summarized as:

```cpp
    for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
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
    }
```

To better understand `__shfl_xor_sync`, Prof Wes Armour's slides are great:
https://people.maths.ox.ac.uk/gilesm/cuda/2019/lecture_04.pdf

Clearly I am doing more `add` operations then expected in a reduction algorithm, each threads does 5 outter iterations,
and for each outer iteration it will do 15, 7, 3, 1, 0 inner iterations resp., leading to
31 adds and 31 `__shfl_xor_sync`.
The same as if no reduction is used, so the difference between both implementation is mostly coming from the cost of `__shfl_xor_sync` and the cost of memory acceses.

# Results

Currently this is just really slow to use my algorithm with `__shfl_xor_sync` compared with the used of shared memory.

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
```

```bash
$ mkdir build && cd build && cmake .. -DCUDA_SM=75 && make && ./main
# ...
 nbBlocksTest 16 nbThreadsTest 32
WARP = 0.167268
SM = 0.0233758
 nbBlocksTest 16 nbThreadsTest 64
WARP = 0.235581
SM = 0.0188339
 nbBlocksTest 16 nbThreadsTest 96
WARP = 0.248089
SM = 0.0292978
 nbBlocksTest 16 nbThreadsTest 128
WARP = 0.34031
SM = 0.0377677
 nbBlocksTest 16 nbThreadsTest 160
WARP = 0.427255
SM = 0.0615797
 nbBlocksTest 16 nbThreadsTest 192
WARP = 0.635577
SM = 0.0566756
 nbBlocksTest 32 nbThreadsTest 32
WARP = 0.11897
SM = 0.0158536
 nbBlocksTest 32 nbThreadsTest 64
WARP = 0.279671
SM = 0.032455
 nbBlocksTest 32 nbThreadsTest 96
WARP = 0.533057
SM = 0.0482949
 nbBlocksTest 32 nbThreadsTest 128
WARP = 1.39346
SM = 0.0738548
 nbBlocksTest 32 nbThreadsTest 160
WARP = 2.35861
SM = 0.0930473
 nbBlocksTest 32 nbThreadsTest 192
WARP = 3.26089
SM = 0.0943431
 nbBlocksTest 64 nbThreadsTest 32
WARP = 0.284564
SM = 0.0324401
 nbBlocksTest 64 nbThreadsTest 64
WARP = 1.5471
SM = 0.0744361
 nbBlocksTest 64 nbThreadsTest 96
WARP = 3.22695
SM = 0.0938923
 nbBlocksTest 64 nbThreadsTest 128
WARP = 4.41329
SM = 0.124977
 nbBlocksTest 64 nbThreadsTest 160
WARP = 5.70358
SM = 0.156362
 nbBlocksTest 64 nbThreadsTest 192
WARP = 7.16628
SM = 0.187457
 nbBlocksTest 128 nbThreadsTest 32
WARP = 1.63069
SM = 0.0600818
 nbBlocksTest 128 nbThreadsTest 64
WARP = 4.46993
SM = 0.119148
 nbBlocksTest 128 nbThreadsTest 96
WARP = 7.25435
SM = 0.18809
 nbBlocksTest 128 nbThreadsTest 128
WARP = 10.5254
SM = 0.238243
 nbBlocksTest 128 nbThreadsTest 160
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
$ mkdir build && cd build && cmake .. -DCUDA_SM=80 && make && ./main
# ...
nbBlocksTest 16 nbThreadsTest 32
WARP = 0.0656336
SM = 0.0140495
 nbBlocksTest 16 nbThreadsTest 64
WARP = 0.0881981
SM = 0.0140611
 nbBlocksTest 16 nbThreadsTest 96
WARP = 0.100459
SM = 0.012553
 nbBlocksTest 16 nbThreadsTest 128
WARP = 0.113396
SM = 0.0165526
 nbBlocksTest 16 nbThreadsTest 160
WARP = 0.142203
SM = 0.0206947
 nbBlocksTest 16 nbThreadsTest 192
WARP = 0.170639
SM = 0.0247533
 nbBlocksTest 32 nbThreadsTest 32
WARP = 0.0400047
SM = 0.00905349
 nbBlocksTest 32 nbThreadsTest 64
WARP = 0.0623376
SM = 0.00906001
 nbBlocksTest 32 nbThreadsTest 96
WARP = 0.0911787
SM = 0.0126702
 nbBlocksTest 32 nbThreadsTest 128
WARP = 0.124881
SM = 0.0165394
 nbBlocksTest 32 nbThreadsTest 160
WARP = 0.14943
SM = 0.0206889
 nbBlocksTest 32 nbThreadsTest 192
WARP = 0.181649
SM = 0.024769
 nbBlocksTest 64 nbThreadsTest 32
WARP = 0.0500637
SM = 0.00905666
 nbBlocksTest 64 nbThreadsTest 64
WARP = 0.0905392
SM = 0.00905823
 nbBlocksTest 64 nbThreadsTest 96
WARP = 0.124255
SM = 0.0126675
 nbBlocksTest 64 nbThreadsTest 128
WARP = 0.174162
SM = 0.0165384
 nbBlocksTest 64 nbThreadsTest 160
WARP = 0.204312
SM = 0.020701
 nbBlocksTest 64 nbThreadsTest 192
WARP = 0.234229
SM = 0.0247723
 nbBlocksTest 128 nbThreadsTest 32
WARP = 0.117446
SM = 0.00906557
 nbBlocksTest 128 nbThreadsTest 64
WARP = 0.204035
SM = 0.0165431
 nbBlocksTest 128 nbThreadsTest 96
WARP = 0.309418
SM = 0.0250169
 nbBlocksTest 128 nbThreadsTest 128
WARP = 0.362847
SM = 0.0330439
 nbBlocksTest 128 nbThreadsTest 160
WARP = 0.452274
SM = 0.0423144
 nbBlocksTest 128 nbThreadsTest 192
WARP = 0.552338
SM = 0.0497066
 nbBlocksTest 256 nbThreadsTest 32
WARP = 0.187304
SM = 0.0127041
 nbBlocksTest 256 nbThreadsTest 64
WARP = 0.328517
SM = 0.0250122
 nbBlocksTest 256 nbThreadsTest 96
WARP = 0.438027
SM = 0.038508
 nbBlocksTest 256 nbThreadsTest 128
WARP = 0.613189
SM = 0.049555
 nbBlocksTest 256 nbThreadsTest 160
WARP = 0.798083
SM = 0.0622785
 nbBlocksTest 256 nbThreadsTest 192
WARP = 0.920881
SM = 0.0743121
 nbBlocksTest 512 nbThreadsTest 32
WARP = 0.340599
SM = 0.0220554
 nbBlocksTest 512 nbThreadsTest 64
WARP = 0.575721
SM = 0.0415136
 nbBlocksTest 512 nbThreadsTest 96
WARP = 0.856189
SM = 0.0620385
 nbBlocksTest 512 nbThreadsTest 128
WARP = 1.36432
SM = 0.082564
 nbBlocksTest 512 nbThreadsTest 160
WARP = 2.13154
SM = 0.103344
 nbBlocksTest 512 nbThreadsTest 192
WARP = 3.19269
SM = 0.123827

```
