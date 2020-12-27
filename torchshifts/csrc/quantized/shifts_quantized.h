#pragma once
#include <torch/extension.h>
#include "../global_scope.h"


API_EXPORT torch::Tensor q_shift1d_cpu(const torch::Tensor& input,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& borders,
                                       const std::vector<int64_t>& new_size,
                                       int64_t padding_mode);

API_EXPORT torch::Tensor q_shift2d_cpu(const torch::Tensor& input,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& borders,
                                       const std::vector<int64_t>& new_size,
                                       int64_t padding_mode);

API_EXPORT torch::Tensor q_shift3d_cpu(const torch::Tensor& input,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& borders,
                                       const std::vector<int64_t>& new_size,
                                       int64_t padding_mode);  