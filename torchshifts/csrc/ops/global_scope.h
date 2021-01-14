#ifndef TORCHSHIFTS_GLOBAL_SCOPE
#define TORCHSHIFTS_GLOBAL_SCOPE

#define CUDA_THREADS 1024

#include <cmath>
#ifdef _SHIFTS_CPU
    #define ROUND(a) (std::round(a))
    #define FLOOR(a) (std::floor(a))
    #define CEIL(a) (std::ceil(a))
    #define MIN(a,b) (std::min(a,b))
    #define MAX(a,b) (std::max(a,b))
    #define ABS(a) (std::abs(a))
    #define ADD(a,b) (*a += b)
    #if (defined __cpp_inline_variables) || __cplusplus >= 201703L
        #define API_INLINE inline
    #else 
    #ifdef _MSC_VER
        #define API_INLINE __declspec(selectany)
    #else
        #define API_INLINE __attribute__((weak))
    #endif
    #endif
#endif
#ifdef _SHIFTS_CUDA
    #include <THC/THCAtomics.cuh>
    #define ROUND(a) (::round(a))
    #define FLOOR(a) (::floor(a))
    #define CEIL(a) (::ceil(a))
    #define MIN(a,b) (::min(a,b))
    #define MAX(a,b) (::max(a,b))
    #define ABS(a) (::abs(a))
    #define ADD(a,b) (gpuAtomicAdd(a,b))
    #define API_INLINE __device__ __forceinline__

    const int LOCAL_CUDA_NUM_THREADS = CUDA_THREADS;
    // taken from PyTorch
    inline int CUDA_BLOCKS(const int64_t N, const int64_t NUMTHREADS)
    {
        return static_cast<int>(N / NUMTHREADS + ((N % NUMTHREADS) == 0 ? 0 : 1));
    }
#endif


#endif 

