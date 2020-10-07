#ifndef _SHIFTS_CUDA
#define _SHIFTS_CUDA

#include <torch/extension.h>
#include <torch/script.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor shift1d_cuda(const torch::Tensor& input,
                           const torch::Tensor& weights,
                           int padding_mode,
                           bool active_flag){
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    return _shift1d_cuda(input, weights, padding_mode, active_flag);                    
}

torch::Tensor shift2d_cuda(const torch::Tensor& input,
                           const torch::Tensor& weights,
                           int padding_mode,
                           bool active_flag){
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    return _shift2d_cuda(input, weights, padding_mode, active_flag);                    
}

torch::Tensor shift3d_cuda(const torch::Tensor& input,
                           const torch::Tensor& weights,
                           int padding_mode,
                           bool active_flag){
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    return _shift3d_cuda(input, weights, padding_mode, active_flag);                    
}


std::vector<torch::Tensor> shift1d_backward_cuda(const torch::Tensor& grad,
                                                 const torch::Tensor& weights,
                                                 const torch::Tensor& input,
                                                 int padding_mode,
                                                 bool active_flag){
    CHECK_INPUT(grad); 
    CHECK_INPUT(weights);
    CHECK_INPUT(input);
    return  _shift1d_backward_cuda(grad, weights, input, padding_mode, active_flag);                                       
}

std::vector<torch::Tensor> shift2d_backward_cuda(const torch::Tensor& grad,
                                                 const torch::Tensor& weights,
                                                 const torch::Tensor& input,
                                                 int padding_mode,
                                                 bool active_flag){
    CHECK_INPUT(grad);
    CHECK_INPUT(weights);
    CHECK_INPUT(input);
    return  _shift2d_backward_cuda(grad, weights, input, padding_mode, active_flag);                                       
}

std::vector<torch::Tensor> shift3d_backward_cuda(const torch::Tensor& grad,
                                                 const torch::Tensor& weights,
                                                 const torch::Tensor& input,
                                                 int padding_mode,
                                                 bool active_flag){
    CHECK_INPUT(grad);
    CHECK_INPUT(weights);
    CHECK_INPUT(input);
    return  _shift3d_backward_cuda(grad, weights, input, padding_mode, active_flag);                                       
}

// TORCH_LIBRARY(shifts_cuda, m) {
//     m.def("shift1d_cuda", &shift1d_cuda);
//     m.def("shift2d_cuda", &shift2d_cuda);
//     m.def("shift3d_cuda", &shift3d_cuda);
//     m.def("shift1d_backward_cuda", &shift1d_backward_cuda);
//     m.def("shift2d_backward_cuda", &shift2d_backward_cuda);
//     m.def("shift3d_backward_cuda", &shift3d_backward_cuda); 
// }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("shift1d_cuda", &shift1d_cuda, "1D Shift operation forward (cuda)");
    m.def("shift2d_cuda", &shift2d_cuda, "2D Shift operation forward (cuda)");
    m.def("shift3d_cuda", &shift3d_cuda, "3D Shift operation forward (cuda)");
    m.def("shift1d_backward_cuda", &shift1d_backward_cuda, "1D Shift operator backward (cuda)");
    m.def("shift2d_backward_cuda", &shift2d_backward_cuda, "2D Shift operator backward (cuda)");
    m.def("shift3d_backward_cuda", &shift3d_backward_cuda, "3D Shift operator backward (cuda)");
};



#endif
