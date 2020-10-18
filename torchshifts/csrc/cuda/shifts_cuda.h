
#pragma once
#include <torch/extension.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include "../global_scope.h"


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


API_EXPORT torch::Tensor shift1d_forward_cuda(const torch::Tensor& input,
                                              const torch::Tensor& weights,
                                              int padding_mode,
                                              bool active_flag);


API_EXPORT torch::Tensor shift2d_forward_cuda(const torch::Tensor& input,
                                              const torch::Tensor& weights,
                                              int padding_mode,
                                              bool active_flag);


API_EXPORT torch::Tensor shift3d_forward_cuda(const torch::Tensor& input,
                                              const torch::Tensor& weights,
                                              int padding_mode,
                                              bool active_flag);

API_EXPORT std::vector<torch::Tensor> shift1d_backward_cuda(const torch::Tensor& grad,
                                                            const torch::Tensor& weights,
                                                            const torch::Tensor& input,
                                                            int padding_mode,
                                                            bool active_flag);

API_EXPORT std::vector<torch::Tensor> shift2d_backward_cuda(const torch::Tensor& grad,
                                                            const torch::Tensor& weights,
                                                            const torch::Tensor& input,
                                                            int padding_mode,
                                                            bool active_flag);


API_EXPORT std::vector<torch::Tensor> shift3d_backward_cuda(const torch::Tensor& grad,
                                                            const torch::Tensor& weights,
                                                            const torch::Tensor& input,
                                                            int padding_mode,
                                                            bool active_flag);

