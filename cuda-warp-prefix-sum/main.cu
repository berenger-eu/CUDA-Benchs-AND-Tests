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
    

auto numberToBinaryString(const unsigned int val){
    std::string result;
    for(int idx = 31 ; idx >= 0 ; --idx){
        result.append(val & (1U << idx) ? "1" : "0");
    }
    return result;
}

template <class T>
__host__ __device__ void swap(T& p1, T& p2){
    T tmp = p1;
    p1 = p2;
    p2 = tmp;
}

/////////////////////////////////////////////////////////////

using ValType = unsigned int;

__global__ void core_test(ValType* values, const int nbLoops){
    __shared__ ValType vec[32];
    __shared__ ValType buffer[32];

    vec[threadIdx.x] = values[threadIdx.x];

    for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){ 
        ValType* ptr1 = vec;
        ValType* ptr2 = buffer; 
        
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
        
        ptr2[threadIdx.x] = ptr1[threadIdx.x];
    }
    
    values[threadIdx.x] = vec[threadIdx.x];
}

auto core_test_cu(const std::vector<ValType>& values, const int NbLoops){
    assert(values.size() == 32);
    ValType* cuValues;
    CUDA_ASSERT( cudaMalloc(&cuValues, 32 * sizeof(ValType)) );
    CUDA_ASSERT( cudaMemcpy(cuValues, values.data(),
                            32 * sizeof(ValType),
                            cudaMemcpyHostToDevice) );

    SpTimer timer;

    core_test<<<1,32>>>(cuValues, NbLoops);
    CUDA_ASSERT(cudaDeviceSynchronize());

    timer.stop();
    std::cout << "core_test_cu = " << timer.getElapsed() << std::endl;

    std::vector<ValType> results(32);
    CUDA_ASSERT( cudaMemcpy(results.data(), cuValues, 
                            32 * sizeof(ValType),
                            cudaMemcpyDeviceToHost) );

    CUDA_ASSERT( cudaFree(cuValues) );

    return results;
}


void test_gpu_reduce(){
    std::vector<unsigned int> valuesToReduce(32);
    for(int idx = 0 ; idx < 32 ; ++idx){
        valuesToReduce[idx] = (1U << idx);
    }
    
    valuesToReduce = core_test_cu(valuesToReduce, 1);
    
    for(int idx = 0 ; idx < 32 ; ++idx){
        const unsigned int expectedVal = (idx+1 >= 32 ? 0xFFFFFFFFU : ~(0xFFFFFFFFU << (idx+1)));
        if(valuesToReduce[idx] != expectedVal){
            std::cerr << "Error at index " << idx << " should be " << expectedVal << " is " << valuesToReduce[idx] << std::endl;
            std::cerr << " - expected: " << numberToBinaryString(expectedVal) << std::endl;
            std::cerr << " - current:  " << numberToBinaryString(valuesToReduce[idx]) << std::endl;
        }
    }    
}

void bench_gpu_reduce(const int nbLoops){
    std::vector<unsigned int> valuesToReduce(32);
    for(int idx = 0 ; idx < 32 ; ++idx){
        valuesToReduce[idx] = 1;
    }
    
    valuesToReduce = core_test_cu(valuesToReduce, nbLoops);  
}



/////////////////////////////////////////////////////////////

__global__ void core_test_v2(ValType* values, const int nbLoops){
    const unsigned int allTrue = 0xFFFFFFFFu;
    const unsigned int subReductionMask = ~(0xFFFFFFFFu << (threadIdx.x+1));

    __shared__ ValType vec[32];

    vec[threadIdx.x] = values[threadIdx.x];

    for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){ 
        int prefixSum = 0;

        #pragma unroll
        for (int sumBitPosition = 0; sumBitPosition < sizeof(unsigned int)*8 ; ++sumBitPosition) {
            const unsigned int allBitsSum = __ballot_sync(allTrue, vec[threadIdx.x] & (1 << sumBitPosition));
            prefixSum += __popc(allBitsSum & subReductionMask) << sumBitPosition;
        }
        vec[threadIdx.x] = prefixSum;
    }
    
    values[threadIdx.x] = vec[threadIdx.x];
}

auto core_test_cu_v2(const std::vector<ValType>& values, const int NbLoops){
    assert(values.size() == 32);
    ValType* cuValues;
    CUDA_ASSERT( cudaMalloc(&cuValues, 32 * sizeof(ValType)) );
    CUDA_ASSERT( cudaMemcpy(cuValues, values.data(),
                            32 * sizeof(ValType),
                            cudaMemcpyHostToDevice) );

    SpTimer timer;

    core_test_v2<<<1,32>>>(cuValues, NbLoops);
    CUDA_ASSERT(cudaDeviceSynchronize());

    timer.stop();
    std::cout << "core_test_cu = " << timer.getElapsed() << std::endl;

    std::vector<ValType> results(32);
    CUDA_ASSERT( cudaMemcpy(results.data(), cuValues, 
                            32 * sizeof(ValType),
                            cudaMemcpyDeviceToHost) );

    CUDA_ASSERT( cudaFree(cuValues) );

    return results;
}


void test_gpu_reduce_v2(){
    std::vector<unsigned int> valuesToReduce(32);
    for(int idx = 0 ; idx < 32 ; ++idx){
        valuesToReduce[idx] = (1U << idx);
    }
    
    valuesToReduce = core_test_cu_v2(valuesToReduce, 1);
    
    for(int idx = 0 ; idx < 32 ; ++idx){
        const unsigned int expectedVal = (idx+1 >= 32 ? 0xFFFFFFFFU : ~(0xFFFFFFFFU << (idx+1)));
        if(valuesToReduce[idx] != expectedVal){
            std::cerr << "Error at index " << idx << " should be " << expectedVal << " is " << valuesToReduce[idx] << std::endl;
            std::cerr << " - expected: " << numberToBinaryString(expectedVal) << std::endl;
            std::cerr << " - current:  " << numberToBinaryString(valuesToReduce[idx]) << std::endl;
        }
    }    
}

void bench_gpu_reduce_v2(const int nbLoops){
    std::vector<unsigned int> valuesToReduce(32);
    for(int idx = 0 ; idx < 32 ; ++idx){
        valuesToReduce[idx] = 1;
    }
    
    valuesToReduce = core_test_cu_v2(valuesToReduce, nbLoops);  
}


/////////////////////////////////////////////////////////////


void test_cpu_reduce(){
    std::vector<unsigned int> valuesToReduce(32);
    for(int idx = 0 ; idx < 32 ; ++idx){
        valuesToReduce[idx] = (1U << idx);
    }
    
    {
        std::vector<unsigned int> buffer(32);
        
        unsigned int* ptr1 = valuesToReduce.data();
        unsigned int* ptr2 = buffer.data();
        
        for(int idxIter = 1 ; idxIter <= 32 ; idxIter <<= 1){
            for(int idxVal = 0 ; idxVal < 32 ; ++idxVal){
                if(idxVal & idxIter){
                    ptr2[idxVal] = ptr1[idxVal] + ptr1[(idxVal ^ idxIter) | (idxIter-1)];
                }
                else{
                    ptr2[idxVal] = ptr1[idxVal];
                }
            }
            
            std::swap(ptr1, ptr2);
        }
        
       
        for(int idxVal = 0 ; idxVal < 32 ; ++idxVal){
            ptr2[idxVal] = ptr1[idxVal];        
        }
    }
    
    for(int idx = 0 ; idx < 32 ; ++idx){
        const unsigned int expectedVal = (idx+1 >= 32 ? 0xFFFFFFFFU : ~(0xFFFFFFFFU << (idx+1)));
        if(valuesToReduce[idx] != expectedVal){
            std::cerr << "Error at index " << idx << " should be " << expectedVal << " is " << valuesToReduce[idx] << std::endl;
            std::cerr << " - expected: " << numberToBinaryString(expectedVal) << std::endl;
            std::cerr << " - current:  " << numberToBinaryString(valuesToReduce[idx]) << std::endl;
        }
    }    
}

int main(){
    std::cout << "Test CPU:" << std::endl;
    test_cpu_reduce();
    std::cout << "Done" << std::endl;
    
    std::cout << "Test GPU:" << std::endl;
    test_gpu_reduce();
    std::cout << "Done" << std::endl;
    
    std::cout << "Test GPUv2:" << std::endl;
    test_gpu_reduce_v2();
    std::cout << "Done" << std::endl;
    
    std::cout << "Bench GPU:" << std::endl;
    bench_gpu_reduce(10000000);
    std::cout << "Done" << std::endl;
    
    std::cout << "Bench GPUv2:" << std::endl;
    bench_gpu_reduce_v2(10000000);
    std::cout << "Done" << std::endl;

	return 0;
}
