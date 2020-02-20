#ifndef _SHIFTS_CUDA_KERNEL
#define _SHIFTS_CUDA_KERNEL

#include <torch/extension.h>

// padding used during bilinear interpolation
enum class BIPadding {Zeros, Border, Reflect, Symmetric};



at::Tensor _shift1d_gpu(const at::Tensor& input,
                        const at::Tensor& weights,
                        int padding_mode,
                        bool active_flag);

std::vector<at::Tensor> _shift1d_backward_gpu(const at::Tensor& grad,
                                              const at::Tensor& weights,
                                              const at::Tensor& input,
                                              int padding_mode,
                                              bool active_flag);

at::Tensor _shift2d_gpu(const at::Tensor& input,
                        const at::Tensor& weights,
                        int padding_mode,
                        bool active_flag);

std::vector<at::Tensor> _shift2d_backward_gpu(const at::Tensor& grad,
                                              const at::Tensor& weights,
                                              const at::Tensor& input,
                                              int padding_mode,
                                              bool active_flag);


#endif