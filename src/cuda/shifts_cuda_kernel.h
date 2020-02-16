#ifndef _SHIFTS_CUDA_KERNEL
#define _SHIFTS_CUDA_KERNEL

#include <torch/extension.h>


enum class BIPadding {Zeros, Border};



at::Tensor _shift2d_gpu(const at::Tensor& input,
                        const at::Tensor& weights);

std::vector<at::Tensor> _shift2d_backward_gpu(const at::Tensor& grad,
                                              const at::Tensor& weights,
                                              const at::Tensor& input,
                                              int padding_mode);


#endif