#ifndef _SHIFTS_CUDA
#define _SHIFTS_CUDA


#include "shifts_cuda_kernel.h"


torch::Tensor shift2d_gpu(const torch::Tensor& input,
                          const torch::Tensor& weights);

std::vector<torch::Tensor> shift2d_backward_gpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                int padding_mode);

#endif
