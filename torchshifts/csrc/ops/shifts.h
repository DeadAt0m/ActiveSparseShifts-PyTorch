#pragma once

#include <torch/extension.h>
#include <torch/library.h>
#include "../macros.h"

namespace shifts {
namespace ops {

    
    
API_EXPORT torch::Tensor shift1d(const torch::Tensor& input,
                                 const torch::Tensor& weights,
                                 const torch::Tensor& borders,
                                 int64_t padding_mode,
                                 bool active_flag);  
    
API_EXPORT torch::Tensor shift2d(const torch::Tensor& input,
                                 const torch::Tensor& weights,
                                 const torch::Tensor& borders,
                                 int64_t padding_mode,
                                 bool active_flag);   

API_EXPORT torch::Tensor shift3d(const torch::Tensor& input,
                                 const torch::Tensor& weights,
                                 const torch::Tensor& borders,
                                 int64_t padding_mode,
                                 bool active_flag);   




namespace detail {

torch::Tensor _shift1d_forward(const torch::Tensor& input,
                               const torch::Tensor& weights,
                               const torch::Tensor& borders,
                               const std::vector<int64_t>& new_size,
                               int64_t padding_mode,
                               bool active_flag);  

torch::Tensor _shift2d_forward(const torch::Tensor& input,
                               const torch::Tensor& weights,
                               const torch::Tensor& borders,
                               const std::vector<int64_t>& new_size,
                               int64_t padding_mode,
                               bool active_flag);
    
torch::Tensor _shift3d_forward(const torch::Tensor& input,
                               const torch::Tensor& weights,
                               const torch::Tensor& borders,
                               const std::vector<int64_t>& new_size,
                               int64_t padding_mode,
                               bool active_flag);

std::tuple<torch::Tensor, torch::Tensor> _shift1d_backward(const torch::Tensor& grad,
                                                           const torch::Tensor& weights,
                                                           const torch::Tensor& input,
                                                           const torch::Tensor& borders,
                                                           int64_t padding_mode,
                                                           bool active_flag);

std::tuple<torch::Tensor, torch::Tensor> _shift2d_backward(const torch::Tensor& grad,
                                                           const torch::Tensor& weights,
                                                           const torch::Tensor& input,
                                                           const torch::Tensor& borders,
                                                           int64_t padding_mode,
                                                           bool active_flag);

std::tuple<torch::Tensor, torch::Tensor> _shift3d_backward(const torch::Tensor& grad,
                                                           const torch::Tensor& weights,
                                                           const torch::Tensor& input,
                                                           const torch::Tensor& borders,
                                                           int64_t padding_mode,
                                                           bool active_flag);

} // namespace detail

} // namespace ops
} // namespace shifts