#ifndef TORCHSHIFTS_GLOBAL_SCOPE
#define TORCHSHIFTS_GLOBAL_SCOPE

#ifdef _WIN32
#if defined(TORCHSHIFTS_EXPORTS)
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __declspec(dllimport)
#endif
#else
#define API_EXPORT
#endif



#if _SHIFTS_CPU
    #include <cmath>
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
    #define ADD(a,b) (atomicAdd(a,b))
    #define API_INLINE __device__ __forceinline__
#endif


#endif 

