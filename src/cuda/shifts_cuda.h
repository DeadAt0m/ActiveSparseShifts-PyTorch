#include <torch/extension.h>

torch::Tensor _shift1d_cuda(const torch::Tensor& input,
                            const torch::Tensor& weights,
                            int padding_mode,
                            bool active_flag);

torch::Tensor _shift2d_cuda(const torch::Tensor& input,
                            const torch::Tensor& weights,
                            int padding_mode,
                            bool active_flag);

torch::Tensor _shift3d_cuda(const torch::Tensor& input,
                            const torch::Tensor& weights,
                            int padding_mode,
                            bool active_flag);

std::vector<torch::Tensor> _shift1d_backward_cuda(const torch::Tensor& grad,
                                                  const torch::Tensor& weights,
                                                  const torch::Tensor& input,
                                                  int padding_mode,
                                                  bool active_flag);

std::vector<torch::Tensor> _shift2d_backward_cuda(const torch::Tensor& grad,
                                                  const torch::Tensor& weights,
                                                  const torch::Tensor& input,
                                                  int padding_mode,
                                                  bool active_flag);


std::vector<torch::Tensor> _shift3d_backward_cuda(const torch::Tensor& grad,
                                                  const torch::Tensor& weights,
                                                  const torch::Tensor& input,
                                                  int padding_mode,
                                                  bool active_flag);

