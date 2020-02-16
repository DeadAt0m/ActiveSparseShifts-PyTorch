#ifndef _SHIFTS_CPU
#define _SHIFTS_CPU

#include <torch/extension.h>


// padding used during bilinear interpolation(backward computation)
enum class BIPadding {Zeros, Border};

torch::Tensor shift2d_cpu(const torch::Tensor& input,
                          const torch::Tensor& weights);

std::vector<torch::Tensor> shift2d_backward_cpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                int padding_mode);

#endif