#ifndef _SHIFTS_CUDA
#define _SHIFTS_CUDA


#include "shifts_cuda_kernel.h"


torch::Tensor shift1d_gpu(const torch::Tensor& input,
                          const torch::Tensor& weights,
                          int padding_mode,
                          bool active_flag);

std::vector<torch::Tensor> shift1d_backward_gpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                int padding_mode,
                                                bool active_flag);

torch::Tensor shift2d_gpu(const torch::Tensor& input,
                          const torch::Tensor& weights,
                          int padding_mode,
                          bool active_flag);

std::vector<torch::Tensor> shift2d_backward_gpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                int padding_mode,
                                                bool active_flag);

#endif
