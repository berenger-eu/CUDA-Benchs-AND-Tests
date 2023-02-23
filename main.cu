#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>
#include "SpTimer.hpp"

#include <cuda_runtime.h>

#define CUDA_ASSERT(X)\
{\
    cudaError_t ___resCuda = (X);\
    if ( cudaSuccess != ___resCuda ){\
    printf("Error: fails, %s (%s line %d)\nbCols", cudaGetErrorString(___resCuda), __FILE__, __LINE__ );\
    exit(1);\
    }\
    }

template <int Idx>
struct SwitchIdx{
    static constexpr int value = Idx;

    constexpr int getValue() const{
        return value;
    }

    operator int() const{
        return Idx;
    }
};

template <int CaseVal, int... Ints, class Func>
void Switch(const int choice, Func&& f){
    if(choice == CaseVal){
        f(SwitchIdx<CaseVal>());
    }
    else {
        if constexpr (sizeof...(Ints) >= 1){
            Switch<Ints...>(choice, std::forward<Func>(f));
        }
        else{
            std::cout << "Choice does not exist in switch " << choice << std::endl;
            exit(0);
        }
    }
}

/////////////////////////////////////////////////////////////

using ValType = double;

__device__ __host__ unsigned int targetNeigh(const unsigned int thread, const unsigned int idx){
    const unsigned int pos = (thread ^ idx);
    assert(0 <= pos);
    assert(pos < 32);
    return pos;
}

__global__ void core_test(const ValType* values, ValType* results, const int nbLoops){
    const int threadIdxInWarp = threadIdx.x%32;

    ValType buffer[32];
    for(int idxVal = 0 ; idxVal < 32 ; ++idxVal){
        buffer[idxVal] = values[threadIdxInWarp*32 + idxVal];
    }


    for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
        for(unsigned int idx = 1 ; idx < 32 ; idx *= 2){
            const unsigned int neighIdx = targetNeigh(threadIdxInWarp, idx);
            buffer[threadIdxInWarp] += __shfl_xor_sync(0xffffffff, buffer[neighIdx], idx, 32);

            const int step = idx*2;
            for(unsigned int idxCoverage = step ; idxCoverage < 32 ; idxCoverage += step){
                const unsigned int recvFor = (threadIdxInWarp + idxCoverage)%32u;
                const unsigned int sendFor = (neighIdx + idxCoverage)%32u;
                buffer[recvFor] += __shfl_xor_sync(0xffffffff, buffer[sendFor], idx, 32);
            }
        }
    }

    results[blockIdx.x*blockDim.x + threadIdx.x] = buffer[threadIdxInWarp];
}

auto test_cu_partition(const std::vector<ValType>& values,
                       const int nbGroupsTest,
                       const int nbThreadsTest,
                       const int NbLoops){
    assert(values.size() == 32*32);
    ValType* cuValues;
    CUDA_ASSERT( cudaMalloc(&cuValues, 32*32 * sizeof(ValType)) );
    CUDA_ASSERT( cudaMemcpy(cuValues, values.data(),
                            32*32 * sizeof(ValType),
                            cudaMemcpyHostToDevice) );

    ValType* cuResults;
    CUDA_ASSERT( cudaMalloc(&cuResults, nbThreadsTest*nbGroupsTest * sizeof(ValType)) );

    SpTimer timer;

    core_test<<<nbGroupsTest,nbThreadsTest>>>(cuValues, cuResults, NbLoops);
    CUDA_ASSERT(cudaDeviceSynchronize());

    timer.stop();
    std::cout << "WARP = " << timer.getElapsed() << std::endl;

    std::vector<ValType> results(nbThreadsTest*nbGroupsTest);
    CUDA_ASSERT( cudaMemcpy(results.data(), cuResults, nbThreadsTest*nbGroupsTest * sizeof(ValType),
                            cudaMemcpyDeviceToHost) );

    CUDA_ASSERT( cudaFree(cuValues) );

    return results;
}


template <int nbThreadsPerBlock>
__global__ void core_test_sm(const ValType* values, ValType* results, const int nbLoops){
    const int warpSize = 32;
    __shared__ ValType intermediateResultsAll[nbThreadsPerBlock/warpSize][warpSize][warpSize];

    const int idxWarpInBlock = (threadIdx.x/warpSize);
    const int idxThreadInWarp = (threadIdx.x%warpSize);

    const int threadIdxInWarp = threadIdx.x/32;

    ValType buffer[32];
    for(int idxVal = 0 ; idxVal < 32 ; ++idxVal){
        buffer[idxVal] = values[threadIdxInWarp*32 + idxVal];
    }


    ValType sum = 0;
    for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
        ValType (*intermediateResults)[warpSize] = intermediateResultsAll[idxWarpInBlock];

        for(int idxVal = 0 ; idxVal < 32 ; ++idxVal){
            intermediateResults[idxVal][idxThreadInWarp] = buffer[idxVal];
        }

        for(int idxVal = 0 ; idxVal < 32 ; ++idxVal){
            sum += intermediateResults[idxThreadInWarp][idxVal];
        }
    }

    results[blockIdx.x*blockDim.x + threadIdx.x] = sum;
}


auto test_cu_partition_sm(const std::vector<ValType>& values,
                          const int nbGroupsTest,
                          const int nbThreadsTest,
                          const int NbLoops){
    assert(values.size() == 32*32);
    ValType* cuValues;
    CUDA_ASSERT( cudaMalloc(&cuValues, 32*32 * sizeof(ValType)) );
    CUDA_ASSERT( cudaMemcpy(cuValues, values.data(),
                            32*32 * sizeof(ValType),
                            cudaMemcpyHostToDevice) );

    ValType* cuResults;
    CUDA_ASSERT( cudaMalloc(&cuResults, nbThreadsTest*nbGroupsTest * sizeof(ValType)) );

    SpTimer timer;

    Switch<32, 64, 96, 128, 160, 192>(nbThreadsTest, [&](auto idx){
        core_test_sm<idx.getValue()><<<nbGroupsTest,idx.getValue()>>>(cuValues, cuResults, NbLoops);
        CUDA_ASSERT(cudaDeviceSynchronize());
    });

    timer.stop();
    std::cout << "SM = " << timer.getElapsed() << std::endl;

    std::vector<ValType> results(nbThreadsTest*nbGroupsTest);
    CUDA_ASSERT( cudaMemcpy(results.data(), cuResults, nbThreadsTest*nbGroupsTest * sizeof(ValType),
                            cudaMemcpyDeviceToHost) );

    CUDA_ASSERT( cudaFree(cuValues) );

    return results;
}

/////////////////////////////////////////////////////////////



int main(){
    std::vector<ValType> values(32*32);
    for(int idx0 = 0 ; idx0 < 32 ; ++idx0){
        for(int idx1 = 0 ; idx1 < 32 ; ++idx1){
            values[idx0*32 + idx1] = idx1+1;//(idx0 == 0 ? 1 : 0);
        }
    }

    {
        const int nbBlocksTest = 1;
        const int nbThreadsTest = 32;
        auto results = test_cu_partition(values, nbBlocksTest, nbThreadsTest, 1);
        for(int idx0 = 0 ; idx0 < 32 ; ++idx0){
            std::cout << idx0 << ") " << results[idx0] << std::endl;
        }

        auto results_sm = test_cu_partition_sm(values, nbBlocksTest, nbThreadsTest, 1);
        for(int idx0 = 0 ; idx0 < 32 ; ++idx0){
            std::cout << idx0 << ") " << results_sm[idx0] << std::endl;
        }
    }

    const int NbLoops = 10000;

    std::vector<long int> nbThreadsPossibleValues{32, 64, 96, 128, 160, 192};
    for(long int nbBlocksTest = 16 ; nbBlocksTest < 1024/*2147483647*/ ; nbBlocksTest *= 2){
        for(long int nbThreadsTest : nbThreadsPossibleValues){
            std::cout << " nbBlocksTest " << nbBlocksTest
                      << " nbThreadsTest " << nbThreadsTest << std::endl;

            test_cu_partition(values, nbBlocksTest, nbThreadsTest, NbLoops);

            test_cu_partition_sm(values, nbBlocksTest, nbThreadsTest, NbLoops);
        }
    }

	return 0;
}
