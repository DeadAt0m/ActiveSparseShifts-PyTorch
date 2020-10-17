#pragma once
#include <torch/extension.h>
#ifdef _SHIFTS_CPU
    #include <cmath>
    #define ROUND(a) (std::round(a))
    #define FLOOR(a) (std::floor(a))
    #define ABS(a) (std::abs(a))
    #define ADD(a,b) (*a += b)
    #define FTYPE inline
#endif
#ifdef _SHIFTS_CUDA
    #include <THC/THCAtomics.cuh>
    #include <ATen/cuda/detail/IndexUtils.cuh>
    #include <ATen/cuda/detail/KernelUtils.h>
    #define ROUND(a) (::round(a))
    #define FLOOR(a) (::floor(a))
    #define ABS(a) (::abs(a))
    #define ADD(a,b) (atomicAdd(a,b))
    #define FTYPE __device__ __forceinline__
#endif