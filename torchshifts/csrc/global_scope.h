#ifndef TORCHSHIFTS_GLOBAL_SCOPE
#define TORCHSHIFTS_GLOBAL_SCOPE

#define CUDA_THREADS 1024




#ifdef _WIN32
#if defined(TORCHSHIFTS_EXPORTS)
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __declspec(dllimport)
#endif
#else
#define API_EXPORT
#endif


#include <cmath>
#ifdef _SHIFTS_CPU
    #define ROUND(a) (std::round(a))
    #define FLOOR(a) (std::floor(a))
    #define ABS(a) (std::abs(a))
    #define ADD(a,b) (*a += b)
    #define API_INLINE inline
#endif
#ifdef _SHIFTS_CUDA
    #include <THC/THCAtomics.cuh>
    #define ROUND(a) (::round(a))
    #define FLOOR(a) (::floor(a))
    #define ABS(a) (::abs(a))
    #define ADD(a,b) (gpuAtomicAdd(a,b))
    #define API_INLINE __device__ __forceinline__

    const int LOCAL_CUDA_NUM_THREADS = CUDA_THREADS;
    // taken from PyTorch
    inline int GET_CUDA_BLOCKS(const int64_t N)
    {
        return static_cast<int>(N / LOCAL_CUDA_NUM_THREADS + (N % LOCAL_CUDA_NUM_THREADS == 0 ? 0 : 1));
    }
#endif


#endif 

