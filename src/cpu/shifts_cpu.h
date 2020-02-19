#ifndef _SHIFTS_CPU
#define _SHIFTS_CPU

#include <torch/extension.h>


// padding used during bilinear interpolation(backward computation)
enum class BIPadding {Zeros, Border, Reflect, Symmetric};


torch::Tensor shift1d_cpu(const torch::Tensor& input,
                          const torch::Tensor& weights,
                          int padding_mode,
                          bool active_flag);

std::vector<torch::Tensor> shift1d_backward_cpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                int padding_mode,
                                                bool active_flag);


torch::Tensor shift2d_cpu(const torch::Tensor& input,
                          const torch::Tensor& weights,
                          int padding_mode,
                          bool active_flag);

std::vector<torch::Tensor> shift2d_backward_cpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                int padding_mode,
                                                bool active_flag);

#endif