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
```

```bash
$ mkdir build && cd build && cmake .. -DCUDA_SM=75 && make && ./main
# ...
 nbBlocksTest 16 nbThreadsTest 32
WARP = 0.167277
WARP V2 = 0.0996064
SM = 0.00978927
SM2 = 0.0138952
 nbBlocksTest 16 nbThreadsTest 64
WARP = 0.164665
WARP V2 = 0.103602
SM = 0.0186992
SM2 = 0.0142137
 nbBlocksTest 16 nbThreadsTest 96
WARP = 0.244573
WARP V2 = 0.156799
SM = 0.0291452
SM2 = 0.0198955
 nbBlocksTest 16 nbThreadsTest 128
WARP = 0.32475
WARP V2 = 0.204564
SM = 0.0375112
SM2 = 0.0265066
 nbBlocksTest 16 nbThreadsTest 160
WARP = 0.448334
WARP V2 = 0.262924
SM = 0.0616345
SM2 = 0.0332965
 nbBlocksTest 16 nbThreadsTest 192
WARP = 0.629645
WARP V2 = 0.313768
SM = 0.0566766
SM2 = 0.0399944
 nbBlocksTest 32 nbThreadsTest 32
WARP = 0.118362
WARP V2 = 0.0986685
SM = 0.0158618
SM2 = 0.0142534
 nbBlocksTest 32 nbThreadsTest 64
WARP = 0.283156
WARP V2 = 0.180433
SM = 0.0324563
SM2 = 0.0200381
 nbBlocksTest 32 nbThreadsTest 96
WARP = 0.487522
WARP V2 = 0.267265
SM = 0.0565747
SM2 = 0.0318266
 nbBlocksTest 32 nbThreadsTest 128
WARP = 1.39202
WARP V2 = 0.366394
SM = 0.0738455
SM2 = 0.0399972
 nbBlocksTest 32 nbThreadsTest 160
WARP = 2.38349
WARP V2 = 0.478834
SM = 0.09241
SM2 = 0.0499497
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
WARP = 0.0729483
WARP V2 = 0.0355367
SM = 0.0133832
SM2 = 0.014621
 nbBlocksTest 16 nbThreadsTest 64
WARP = 0.084084
WARP V2 = 0.0630039
SM = 0.0125061
SM2 = 0.00980389
 nbBlocksTest 16 nbThreadsTest 96
WARP = 0.0846235
WARP V2 = 0.0635117
SM = 0.0126443
SM2 = 0.00989732
 nbBlocksTest 16 nbThreadsTest 128
WARP = 0.113556
WARP V2 = 0.08514
SM = 0.0165012
SM2 = 0.0100041
 nbBlocksTest 16 nbThreadsTest 160
WARP = 0.141956
WARP V2 = 0.106443
SM = 0.0206506
SM2 = 0.0100399
 nbBlocksTest 16 nbThreadsTest 192
WARP = 0.170383
WARP V2 = 0.127679
SM = 0.0247379
SM2 = 0.0100796
 nbBlocksTest 32 nbThreadsTest 32
WARP = 0.0399818
WARP V2 = 0.0288567
SM = 0.00904189
SM2 = 0.00988617
 nbBlocksTest 32 nbThreadsTest 64
WARP = 0.0627149
WARP V2 = 0.0480735
SM = 0.00903754
SM2 = 0.0098843
 nbBlocksTest 32 nbThreadsTest 96
WARP = 0.0911497
WARP V2 = 0.0720534
SM = 0.0126877
SM2 = 0.00993257
 nbBlocksTest 32 nbThreadsTest 128
WARP = 0.124705
WARP V2 = 0.0931859
SM = 0.0165323
SM2 = 0.0100363
 nbBlocksTest 32 nbThreadsTest 160
WARP = 0.153543
WARP V2 = 0.110194
SM = 0.0206613
SM2 = 0.010105
 nbBlocksTest 32 nbThreadsTest 192
WARP = 0.1815
WARP V2 = 0.129197
SM = 0.0247127
SM2 = 0.0100746
 nbBlocksTest 64 nbThreadsTest 32
WARP = 0.0520662
WARP V2 = 0.0454065
SM = 0.00904157
SM2 = 0.00987938
 nbBlocksTest 64 nbThreadsTest 64
WARP = 0.0921966
WARP V2 = 0.0743044
SM = 0.0090407
SM2 = 0.00987695
 nbBlocksTest 64 nbThreadsTest 96
WARP = 0.133175
WARP V2 = 0.105679
SM = 0.0126733
SM2 = 0.00991303
 nbBlocksTest 64 nbThreadsTest 128
WARP = 0.174054
WARP V2 = 0.132985
SM = 0.0165216
SM2 = 0.01003
 nbBlocksTest 64 nbThreadsTest 160
WARP = 0.21724
WARP V2 = 0.152256
SM = 0.0206509
SM2 = 0.0101027
 nbBlocksTest 64 nbThreadsTest 192
WARP = 0.243197
WARP V2 = 0.196842
SM = 0.0247196
SM2 = 0.0101088
 nbBlocksTest 128 nbThreadsTest 32
WARP = 0.118947
WARP V2 = 0.0747889
SM = 0.00903398
SM2 = 0.00987687
 nbBlocksTest 128 nbThreadsTest 64
WARP = 0.206964
WARP V2 = 0.162856
SM = 0.0165261
SM2 = 0.0100577
 nbBlocksTest 128 nbThreadsTest 96
WARP = 0.290767
WARP V2 = 0.219352
SM = 0.0249552
SM2 = 0.0100962
 nbBlocksTest 128 nbThreadsTest 128
WARP = 0.362044
WARP V2 = 0.287138
SM = 0.0330275
SM2 = 0.0101414
 nbBlocksTest 128 nbThreadsTest 160
WARP = 0.466362
WARP V2 = 0.338635
SM = 0.0423134
SM2 = 0.010405
 nbBlocksTest 128 nbThreadsTest 192
WARP = 0.55288
WARP V2 = 0.432945
SM = 0.0496944
SM2 = 0.0110366
 nbBlocksTest 256 nbThreadsTest 32
WARP = 0.18725
WARP V2 = 0.14762
SM = 0.0127038
SM2 = 0.00995412
 nbBlocksTest 256 nbThreadsTest 64
WARP = 0.358519
WARP V2 = 0.257084
SM = 0.0249946
SM2 = 0.0101164
 nbBlocksTest 256 nbThreadsTest 96
WARP = 0.45411
WARP V2 = 0.405571
SM = 0.0384548
SM2 = 0.0102874
 nbBlocksTest 256 nbThreadsTest 128
WARP = 0.620208
WARP V2 = 0.504167
SM = 0.049537
SM2 = 0.01104
 nbBlocksTest 256 nbThreadsTest 160
WARP = 0.798689
WARP V2 = 0.58622
SM = 0.0622086
SM2 = 0.0137928
 nbBlocksTest 256 nbThreadsTest 192
WARP = 0.919715
WARP V2 = 0.675788
SM = 0.0742606
SM2 = 0.0218454
 nbBlocksTest 512 nbThreadsTest 32
WARP = 0.32478
WARP V2 = 0.243131
SM = 0.0219933
SM2 = 0.0101351
 nbBlocksTest 512 nbThreadsTest 64
WARP = 0.570501
WARP V2 = 0.461095
SM = 0.0415105
SM2 = 0.0103562
 nbBlocksTest 512 nbThreadsTest 96
WARP = 0.865157
WARP V2 = 0.64408
SM = 0.0620175
SM2 = 0.0137972
 nbBlocksTest 512 nbThreadsTest 128
WARP = 1.36356
WARP V2 = 0.826044
SM = 0.0825379
SM2 = 0.0291036
 nbBlocksTest 512 nbThreadsTest 160
WARP = 2.10066
WARP V2 = 1.00881
SM = 0.103347
SM2 = 0.0274404
 nbBlocksTest 512 nbThreadsTest 192
WARP = 3.19508
WARP V2 = 1.22514
SM = 0.123718
SM2 = 0.0328016


```
