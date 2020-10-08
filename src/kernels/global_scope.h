#ifdef _SHIFTS_CPU
    #include <cmath>
    #define ROUND(a) (std::round(a))
    #define FLOOR(a) (std::floor(a))
    #define ADD(a,b) (*a += b)
    #define FTYPE inline
#endif
#ifdef _SHIFTS_CUDA_KERNELS
    #include <THC/THCAtomics.cuh>
    #define ROUND(a) (::round(a))
    #define FLOOR(a) (::floor(a))
    #define ADD(a,b) (atomicAdd(a,b))
    #define FTYPE __device__ __forceinline__
#endif