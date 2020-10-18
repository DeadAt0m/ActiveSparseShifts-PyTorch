#pragma once
#include <torch/extension.h>
#include "../global_scope.h"



API_EXPORT torch::Tensor shift1d_forward_cpu(const torch::Tensor& input,
                                             const torch::Tensor& weights,
                                             int padding_mode,
                                             bool active_flag);


API_EXPORT torch::Tensor shift2d_forward_cpu(const torch::Tensor& input,
                                             const torch::Tensor& weights,
                                             int padding_mode,
                                             bool active_flag);


API_EXPORT torch::Tensor shift3d_forward_cpu(const torch::Tensor& input,
                                             const torch::Tensor& weights,
                                             int padding_mode,
                                             bool active_flag);

API_EXPORT std::vector<torch::Tensor> shift1d_backward_cpu(const torch::Tensor& grad,
                                                           const torch::Tensor& weights,
                                                           const torch::Tensor& input,
                                                           int padding_mode,
                                                           bool active_flag);

API_EXPORT std::vector<torch::Tensor> shift2d_backward_cpu(const torch::Tensor& grad,
                                                           const torch::Tensor& weights,
                                                           const torch::Tensor& input,
                                                           int padding_mode,
                                                           bool active_flag);


API_EXPORT std::vector<torch::Tensor> shift3d_backward_cpu(const torch::Tensor& grad,
                                                           const torch::Tensor& weights,
                                                           const torch::Tensor& input,
                                                           int padding_mode,
                                                           bool active_flag);

API_EXPORT torch::Tensor q_shift1d_cpu(const torch::Tensor& input,
                                       const torch::Tensor& weights,
                                       int padding_mode);

API_EXPORT torch::Tensor q_shift1d_cpu(const torch::Tensor& input,
                                       const torch::Tensor& weights,
                                       int padding_mode);

API_EXPORT torch::Tensor q_shift1d_cpu(const torch::Tensor& input,
                                       const torch::Tensor& weights,
                                       int padding_mode);  