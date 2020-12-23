#pragma once
#include <torch/extension.h>
#include "../global_scope.h"




API_EXPORT torch::Tensor shift1d_forward_cuda(const torch::Tensor& input,
                                              const torch::Tensor& weights,
                                              int64_t padding_mode,
                                              bool active_flag);


API_EXPORT torch::Tensor shift2d_forward_cuda(const torch::Tensor& input,
                                              const torch::Tensor& weights,
                                              int64_t padding_mode,
                                              bool active_flag);


API_EXPORT torch::Tensor shift3d_forward_cuda(const torch::Tensor& input,
                                              const torch::Tensor& weights,
                                              int64_t padding_mode,
                                              bool active_flag);

API_EXPORT std::vector<torch::Tensor> shift1d_backward_cuda(const torch::Tensor& grad,
                                                            const torch::Tensor& weights,
                                                            const torch::Tensor& input,
                                                            int64_t padding_mode,
                                                            bool active_flag);

API_EXPORT std::vector<torch::Tensor> shift2d_backward_cuda(const torch::Tensor& grad,
                                                            const torch::Tensor& weights,
                                                            const torch::Tensor& input,
                                                            int64_t padding_mode,
                                                            bool active_flag);


API_EXPORT std::vector<torch::Tensor> shift3d_backward_cuda(const torch::Tensor& grad,
                                                            const torch::Tensor& weights,
                                                            const torch::Tensor& input,
                                                            int64_t padding_mode,
                                                            bool active_flag);

