#include "shifts_cuda.h"


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor shift2d_gpu(const torch::Tensor& input,
                          const torch::Tensor& weights){
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    return _shift2d_gpu(input, weights);
}

std::vector<torch::Tensor> shift2d_backward_gpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                int padding_mode){
    CHECK_INPUT(grad);
    CHECK_INPUT(weights);
    CHECK_INPUT(input);
    return _shift2d_backward_gpu(input, weights,input,padding_mode);   
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("shift2d_gpu", &shift2d_gpu, "2D Shift operation forward (gpu)");
    m.def("shift2d_backward_gpu", &shift2d_backward_gpu, "2D Shift operator backward (gpu) ");
}