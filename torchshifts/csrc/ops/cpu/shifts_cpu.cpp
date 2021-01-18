#ifndef _SHIFTS_CPU
#define _SHIFTS_CPU

#include <torch/extension.h>
#include "../global_scope.h"
#include "../kernels/shifts_kernels.h"
#include <torch/library.h>

namespace shifts {
namespace ops {

namespace {


template <typename scalar_t, int kSpatialDim = 1, 
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
API_INLINE void shiftnd_forward_kernel(const torch::Tensor& input, const torch::Tensor& iweights,
                                       const torch::Tensor& dweights,
                                       const torch::Tensor& borders,
                                       torch::Tensor& output){
    const int64_t sizeN = input.size(0);
    const int64_t sizeC = input.size(1);
    const int64_t sizeH = input.size(2);
    const int64_t sizeW = kSpatialDim < 2 ? 1 : input.size(3);
    const int64_t sizeD = kSpatialDim < 3 ? 1 : input.size(4);
    const int64_t input_sN = input.stride(0);
    const int64_t input_sC = input.stride(1);
    const int64_t input_sH = input.stride(2);
    const int64_t input_sW = kSpatialDim < 2 ? 0 : input.stride(3);
    const int64_t input_sD = kSpatialDim < 3 ? 0 : input.stride(4);
    const int64_t output_sN = output.stride(0);
    const int64_t output_sC = output.stride(1);
    const int64_t output_sH = output.stride(2);
    const int64_t output_sW = kSpatialDim < 2 ? 0 : output.stride(3);
    const int64_t output_sD = kSpatialDim < 3 ? 0 : output.stride(4);
    scalar_t* input_ptr = input.data_ptr<scalar_t>();
    scalar_t* output_ptr = output.data_ptr<scalar_t>();
    int64_t* weights_ptr = iweights.data_ptr<int64_t>();
    const int64_t weights_sC = iweights.stride(0);
    const int64_t weights_sS = iweights.stride(1);
    scalar_t* dweights_ptr = dweights.data_ptr<scalar_t>();
    const int64_t dweights_sC = dweights.stride(0);
    const int64_t dweights_sS = dweights.stride(1);
    
    int64_t* borders_data = borders.data_ptr<int64_t>();
    const int64_t i_left_border = borders_data[0];
    const int64_t i_right_border = borders_data[1];
    const int64_t j_left_border = kSpatialDim < 2 ? 0 : borders_data[2];
    const int64_t j_right_border = kSpatialDim < 2 ? 1 : borders_data[3];
    const int64_t k_left_border =  kSpatialDim < 3 ? 0 : borders_data[4];
    const int64_t k_right_border =  kSpatialDim < 3 ? 1 : borders_data[5];

    
    if (input.is_contiguous(c10::MemoryFormat::ChannelsLast) || input.is_contiguous(c10::MemoryFormat::ChannelsLast3d))
    {// Path for NDHWC
        at::parallel_for(0, sizeN, 0, [&](int64_t start, int64_t end){
            for (int64_t n = start; n < end; ++n) {
                for (int64_t i = 0; i < sizeH; ++i){
                    for (int64_t j = 0; j < sizeW; ++j){
                        for (int64_t k = 0; k < sizeD; ++k){
                            shift_forward_kernel_nhwdc<scalar_t, int64_t, kSpatialDim,
                                                       padding_mode, active>(
                                        input_ptr, output_ptr, weights_ptr, dweights_ptr,
                                        n, i, j, k, sizeC, sizeH, sizeW, sizeD,
                                        input_sN, input_sC, input_sH, input_sW, input_sD,
                                        output_sN, output_sC, output_sH, output_sW, output_sD,
                                        weights_sC, weights_sS, dweights_sC, dweights_sS,
                                        i_left_border, j_left_border, k_left_border,
                                        i_right_border, j_right_border, k_right_border);
                        }
                    }
                }
            }
        });
    } else
    {
        at::parallel_for(0, sizeN*sizeC, 0, [&](int64_t start, int64_t end){
            for (int64_t index = start; index < end; ++index) {
                const int64_t c = index % sizeC;
                const int64_t n = index / sizeC;
                for (int64_t i = 0; i < sizeH; ++i){
                    for (int64_t j = 0; j < sizeW; ++j){
                        for (int64_t k = 0; k < sizeD; ++k){
                            shift_forward_kernel_nchwd<scalar_t, int64_t, kSpatialDim,
                                                       padding_mode, active>(
                                        input_ptr, output_ptr, weights_ptr, dweights_ptr,
                                        n, c, i, j, k, sizeH, sizeW, sizeD,
                                        input_sN, input_sC, input_sH, input_sW, input_sD,
                                        output_sN, output_sC, output_sH, output_sW, output_sD,
                                        weights_sC, weights_sS, dweights_sC, dweights_sS,
                                        i_left_border, j_left_border, k_left_border,
                                        i_right_border, j_right_border, k_right_border);
                        }
                    }
                }
            }
        });
    }
}


template <typename scalar_t, int kSpatialDim = 1, 
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
API_INLINE void shiftnd_backward_kernel(const torch::Tensor& grad_input, 
                                        const torch::Tensor& iweights,
                                        const torch::Tensor& dweights,
                                        const torch::Tensor& input, 
                                        const torch::Tensor& borders,
                                        torch::Tensor& grad_output,
                                        torch::Tensor& grad_weights)
{
    const int64_t sizeN = input.size(0);
    const int64_t sizeC = input.size(1);
    const int64_t sizeH = input.size(2);
    const int64_t sizeW = kSpatialDim < 2 ? 1 : input.size(3);
    const int64_t sizeD = kSpatialDim < 3 ? 1 : input.size(4);
    const int64_t grad_input_sN = grad_input.stride(0);
    const int64_t grad_input_sC = grad_input.stride(1);
    const int64_t grad_input_sH = grad_input.stride(2);
    const int64_t grad_input_sW = kSpatialDim < 2 ? 0 : grad_input.stride(3);
    const int64_t grad_input_sD = kSpatialDim < 3 ? 0 : grad_input.stride(4);
    const int64_t input_sN = input.stride(0);
    const int64_t input_sC = input.stride(1);
    const int64_t input_sH = input.stride(2);
    const int64_t input_sW = kSpatialDim < 2 ? 0 : input.stride(3);
    const int64_t input_sD = kSpatialDim < 3 ? 0 : input.stride(4);
    const int64_t grad_output_sN = grad_output.stride(0);
    const int64_t grad_output_sC = grad_output.stride(1);
    const int64_t grad_output_sH = grad_output.stride(2);
    const int64_t grad_output_sW = kSpatialDim < 2 ? 0 : grad_output.stride(3);
    const int64_t grad_output_sD = kSpatialDim < 3 ? 0 : grad_output.stride(4);
    int64_t* weights_ptr = iweights.data_ptr<int64_t>();
    const int64_t weights_sC = iweights.stride(0);
    const int64_t weights_sS = iweights.stride(1);
    scalar_t* dweights_ptr = dweights.data_ptr<scalar_t>();
    const int64_t dweights_sC = dweights.stride(0);
    const int64_t dweights_sS = dweights.stride(1);
    const int64_t grad_weights_sC = grad_weights.stride(0);
    const int64_t grad_weights_sS = grad_weights.stride(1);
    scalar_t* grad_weights_ptr = grad_weights.data_ptr<scalar_t>();
    scalar_t* grad_input_ptr = grad_input.data_ptr<scalar_t>();
    scalar_t* input_ptr = input.data_ptr<scalar_t>();
    scalar_t* grad_output_ptr = grad_output.data_ptr<scalar_t>();
    
    int64_t* borders_data = borders.data_ptr<int64_t>();
    const int64_t i_left_border = borders_data[0];
    const int64_t i_right_border = borders_data[1];
    const int64_t j_left_border = kSpatialDim < 2 ? 0 : borders_data[2];
    const int64_t j_right_border = kSpatialDim < 2 ? 1 : borders_data[3];
    const int64_t k_left_border = kSpatialDim < 3 ? 0 : borders_data[4];
    const int64_t k_right_border = kSpatialDim < 3 ? 1 : borders_data[5];

    
    if (input.is_contiguous(c10::MemoryFormat::ChannelsLast) || input.is_contiguous(c10::MemoryFormat::ChannelsLast3d))
    {// Path for NDHWC
        at::parallel_for(0, sizeN, 0, [&](int64_t start, int64_t end){
            for (int64_t n = start; n < end; ++n) {
                for (int64_t i = 0; i < sizeH; ++i){
                    for (int64_t j = 0; j < sizeW; ++j){
                        for (int64_t k = 0; k < sizeD; ++k){
                            shift_backward_kernel_nhwdc<scalar_t, int64_t, kSpatialDim, 
                                                        padding_mode, active>(
                                    grad_input_ptr, input_ptr, grad_output_ptr,
                                    weights_ptr, dweights_ptr, grad_weights_ptr,
                                    n, i, j, k, sizeC, sizeH, sizeW, sizeD,
                                    grad_input_sN, grad_input_sC, grad_input_sH,
                                    grad_input_sW, grad_input_sD,
                                    input_sN, input_sC, input_sH, input_sW, input_sD,
                                    grad_output_sN, grad_output_sC, grad_output_sH,
                                    grad_output_sW, grad_output_sD,
                                    weights_sC, weights_sS, dweights_sC, dweights_sS, 
                                    grad_weights_sC, grad_weights_sS,
                                    i_left_border, j_left_border, k_left_border,
                                    i_right_border, j_right_border, k_right_border);
                        }
                    }
                }
            }
        });
    } else
    {
        at::parallel_for(0, sizeN*sizeC, 0, [&](int64_t start, int64_t end){
            for (int64_t index = start; index < end; ++index) {
                const int64_t c = index % sizeC;
                const int64_t n = index / sizeC;
                for (int64_t i = 0; i < sizeH; ++i){
                    for (int64_t j = 0; j < sizeW; ++j){
                        for (int64_t k = 0; k < sizeD; ++k){
                            shift_backward_kernel_nchwd<scalar_t, int64_t, kSpatialDim, 
                                                        padding_mode, active>(
                                    grad_input_ptr, input_ptr, grad_output_ptr,
                                    weights_ptr, dweights_ptr, grad_weights_ptr,
                                    n, c, i, j, k, sizeC, sizeH, sizeW, sizeD,
                                    grad_input_sN, grad_input_sC, grad_input_sH,
                                    grad_input_sW, grad_input_sD,
                                    input_sN, input_sC, input_sH, input_sW, input_sD,
                                    grad_output_sN, grad_output_sC, grad_output_sH, 
                                    grad_output_sW, grad_output_sD,
                                    weights_sC, weights_sS, dweights_sC, dweights_sS,
                                    grad_weights_sC, grad_weights_sS,
                                    i_left_border, j_left_border, k_left_border,
                                    i_right_border, j_right_border, k_right_border);
                        }
                    }
                }
            }
        });
    }
}


template <int nD, BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
torch::Tensor shiftnd_forward(const torch::Tensor& input,
                              const torch::Tensor& weights,
                              const torch::Tensor& borders,
                              const std::vector<int64_t>& new_size){
    std::string name = "shift"+std::to_string(nD)+"d_forward_cpu";
    
    torch::Tensor output =  torch::empty(new_size, input.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    torch::Tensor iweights = (active?torch::floor(weights):torch::round(weights)).to(torch::kLong);
    torch::Tensor dweights = active?(weights - iweights):torch::zeros_like(weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
              
    torch::Tensor _borders = borders.to(torch::kLong);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), name, [&] {
        shiftnd_forward_kernel<scalar_t, nD, padding_mode, active>(input, iweights, dweights, _borders, output);
    });
    return output;
}


template <int nD, BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
std::tuple<torch::Tensor, torch::Tensor> shiftnd_backward(const torch::Tensor& grad,
                                                          const torch::Tensor& weights,
                                                          const torch::Tensor& input,
                                                          const torch::Tensor& borders) {
    std::string name = "shift"+std::to_string(nD)+"d_backward_cpu";
    
    torch::Tensor dweights = active?(weights - torch::floor(weights)):torch::where(weights>0,weights - torch::floor(weights), 
                                                                                             weights - torch::ceil(weights));          
    torch::Tensor iweights = (active?(weights - dweights):torch::round(weights)).to(torch::kLong);

    torch::Tensor out_grad = torch::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor weights_grad = torch::zeros_like(weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    
    torch::Tensor _borders = borders.to(torch::kLong);

    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), name, [&] {
        shiftnd_backward_kernel<scalar_t, nD, padding_mode, active>(grad, iweights, dweights, input, _borders, out_grad, weights_grad);
    });
    return std::make_tuple(out_grad, weights_grad);
}


// TEMPLATE DISPATCHERS
              
torch::Tensor shift1d_forward(const torch::Tensor& input,
                              const torch::Tensor& weights,
                              const torch::Tensor& borders,
                              const std::vector<int64_t>& new_size,
                              int64_t padding_mode,
                              bool active_flag){
    torch::Tensor ret;
    switch (padding_mode){        
        case 0:
            ret = active_flag?shiftnd_forward<1,BIPadding::Zeros,true>(input, weights, borders, new_size):
                              shiftnd_forward<1,BIPadding::Zeros,false>(input, weights, borders, new_size);
            break;
        case 1:
            ret = active_flag?shiftnd_forward<1,BIPadding::Border,true>(input, weights, borders, new_size):
                              shiftnd_forward<1,BIPadding::Border,false>(input, weights, borders, new_size);
            break;
        case 2:
            ret = active_flag?shiftnd_forward<1,BIPadding::Periodic,true>(input, weights, borders, new_size):
                              shiftnd_forward<1,BIPadding::Periodic,false>(input, weights, borders, new_size);
            break;
        case 3:
            ret = active_flag?shiftnd_forward<1,BIPadding::Reflect,true>(input, weights, borders, new_size):
                              shiftnd_forward<1,BIPadding::Reflect,false>(input, weights, borders, new_size);
            break;
        case 4:
            ret = active_flag?shiftnd_forward<1,BIPadding::Symmetric,true>(input, weights, borders, new_size):
                              shiftnd_forward<1,BIPadding::Symmetric,false>(input, weights, borders, new_size);
            break;
    }
    return ret;                  
}

torch::Tensor shift2d_forward(const torch::Tensor& input,
                              const torch::Tensor& weights,
                              const torch::Tensor& borders,
                              const std::vector<int64_t>& new_size,
                              int64_t padding_mode,
                              bool active_flag){
    torch::Tensor ret;
    switch (padding_mode){        
        case 0:
            ret = active_flag?shiftnd_forward<2,BIPadding::Zeros,true>(input, weights, borders, new_size):
                              shiftnd_forward<2,BIPadding::Zeros,false>(input, weights, borders, new_size);
            break;
        case 1:
            ret = active_flag?shiftnd_forward<2,BIPadding::Border,true>(input, weights, borders, new_size):
                              shiftnd_forward<2,BIPadding::Border,false>(input, weights, borders, new_size);
            break;
        case 2:
            ret = active_flag?shiftnd_forward<2,BIPadding::Periodic,true>(input, weights, borders, new_size):
                              shiftnd_forward<2,BIPadding::Periodic,false>(input, weights, borders, new_size);
            break;
        case 3:
            ret = active_flag?shiftnd_forward<2,BIPadding::Reflect,true>(input, weights, borders, new_size):
                              shiftnd_forward<2,BIPadding::Reflect,false>(input, weights, borders, new_size);
            break;
        case 4:
            ret = active_flag?shiftnd_forward<2,BIPadding::Symmetric,true>(input, weights, borders, new_size):
                              shiftnd_forward<2,BIPadding::Symmetric,false>(input, weights, borders, new_size);
            break;
    }
    return ret;                         
}

torch::Tensor shift3d_forward(const torch::Tensor& input,
                              const torch::Tensor& weights,
                              const torch::Tensor& borders,
                              const std::vector<int64_t>& new_size,
                              int64_t padding_mode,
                              bool active_flag){
    torch::Tensor ret;
    switch (padding_mode){        
        case 0:
            ret = active_flag?shiftnd_forward<3,BIPadding::Zeros,true>(input, weights, borders, new_size):
                              shiftnd_forward<3,BIPadding::Zeros,false>(input, weights, borders, new_size);
            break;
        case 1:
            ret = active_flag?shiftnd_forward<3,BIPadding::Border,true>(input, weights, borders, new_size):
                              shiftnd_forward<3,BIPadding::Border,false>(input, weights, borders, new_size);
            break;
        case 2:
            ret = active_flag?shiftnd_forward<3,BIPadding::Periodic,true>(input, weights, borders, new_size):
                              shiftnd_forward<3,BIPadding::Periodic,false>(input, weights, borders, new_size);
            break;
        case 3:
            ret = active_flag?shiftnd_forward<3,BIPadding::Reflect,true>(input, weights, borders, new_size):
                              shiftnd_forward<3,BIPadding::Reflect,false>(input, weights, borders, new_size);
            break;
        case 4:
            ret = active_flag?shiftnd_forward<3,BIPadding::Symmetric,true>(input, weights, borders, new_size):
                              shiftnd_forward<3,BIPadding::Symmetric,false>(input, weights, borders, new_size);
            break;
    }
    return ret;                        
}


std::tuple<torch::Tensor, torch::Tensor> shift1d_backward(const torch::Tensor& grad,
                                                          const torch::Tensor& weights,
                                                          const torch::Tensor& input,
                                                          const torch::Tensor& borders,
                                                          int64_t padding_mode,
                                                          bool active_flag){
    std::tuple<torch::Tensor, torch::Tensor> ret;
    switch (padding_mode){        
        case 0:
            ret = active_flag?shiftnd_backward<1,BIPadding::Zeros,true>(grad, weights, input, borders):
                              shiftnd_backward<1,BIPadding::Zeros,false>(grad, weights, input, borders);
            break;
        case 1:
            ret = active_flag?shiftnd_backward<1,BIPadding::Border,true>(grad, weights, input, borders):
                              shiftnd_backward<1,BIPadding::Border,false>(grad, weights, input, borders);
            break;
        case 2:
            ret = active_flag?shiftnd_backward<1,BIPadding::Periodic,true>(grad, weights, input, borders):
                              shiftnd_backward<1,BIPadding::Periodic,false>(grad, weights, input, borders);
            break;
        case 3:
            ret = active_flag?shiftnd_backward<1,BIPadding::Reflect,true>(grad, weights, input, borders):
                              shiftnd_backward<1,BIPadding::Reflect,false>(grad, weights, input, borders);
            break;
        case 4:
            ret = active_flag?shiftnd_backward<1,BIPadding::Symmetric,true>(grad, weights, input, borders):
                              shiftnd_backward<1,BIPadding::Symmetric,false>(grad, weights, input, borders);
            break;
    }
    return ret;                                            
}

std::tuple<torch::Tensor, torch::Tensor> shift2d_backward(const torch::Tensor& grad,
                                                          const torch::Tensor& weights,
                                                          const torch::Tensor& input,
                                                          const torch::Tensor& borders,
                                                          int64_t padding_mode,
                                                          bool active_flag){
    std::tuple<torch::Tensor, torch::Tensor> ret;
    switch (padding_mode){        
        case 0:
            ret = active_flag?shiftnd_backward<2,BIPadding::Zeros,true>(grad, weights, input, borders):
                              shiftnd_backward<2,BIPadding::Zeros,false>(grad, weights, input, borders);
            break;
        case 1:
            ret = active_flag?shiftnd_backward<2,BIPadding::Border,true>(grad, weights, input, borders):
                              shiftnd_backward<2,BIPadding::Border,false>(grad, weights, input, borders);
            break;
        case 2:
            ret = active_flag?shiftnd_backward<2,BIPadding::Periodic,true>(grad, weights, input, borders):
                              shiftnd_backward<2,BIPadding::Periodic,false>(grad, weights, input, borders);
            break;
        case 3:
            ret = active_flag?shiftnd_backward<2,BIPadding::Reflect,true>(grad, weights, input, borders):
                              shiftnd_backward<2,BIPadding::Reflect,false>(grad, weights, input, borders);
            break;
        case 4:
            ret = active_flag?shiftnd_backward<2,BIPadding::Symmetric,true>(grad, weights, input, borders):
                              shiftnd_backward<2,BIPadding::Symmetric,false>(grad, weights, input, borders);
            break;
    }
    return ret;    
}

std::tuple<torch::Tensor, torch::Tensor> shift3d_backward(const torch::Tensor& grad,
                                                          const torch::Tensor& weights,
                                                          const torch::Tensor& input,
                                                          const torch::Tensor& borders,
                                                          int64_t padding_mode,
                                                          bool active_flag){
    std::tuple<torch::Tensor, torch::Tensor> ret;
    switch (padding_mode){        
        case 0:
            ret = active_flag?shiftnd_backward<3,BIPadding::Zeros,true>(grad, weights, input, borders):
                              shiftnd_backward<3,BIPadding::Zeros,false>(grad, weights, input, borders);
            break;
        case 1:
            ret = active_flag?shiftnd_backward<3,BIPadding::Border,true>(grad, weights, input, borders):
                              shiftnd_backward<3,BIPadding::Border,false>(grad, weights, input, borders);
            break;
        case 2:
            ret = active_flag?shiftnd_backward<3,BIPadding::Periodic,true>(grad, weights, input, borders):
                              shiftnd_backward<3,BIPadding::Periodic,false>(grad, weights, input, borders);
            break;
        case 3:
            ret = active_flag?shiftnd_backward<3,BIPadding::Reflect,true>(grad, weights, input, borders):
                              shiftnd_backward<3,BIPadding::Reflect,false>(grad, weights, input, borders);
            break;
        case 4:
            ret = active_flag?shiftnd_backward<3,BIPadding::Symmetric,true>(grad, weights, input, borders):
                              shiftnd_backward<3,BIPadding::Symmetric,false>(grad, weights, input, borders);
            break;
    }
    return ret;                                        
}


        
} // end of anonymous namespace


TORCH_LIBRARY_IMPL(torchshifts, CPU, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift1d_forward"),
        TORCH_FN(shift1d_forward));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift1d_backward"),
        TORCH_FN(shift1d_backward));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift2d_forward"),
        TORCH_FN(shift2d_forward));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift2d_backward"),
        TORCH_FN(shift2d_backward));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift3d_forward"),
        TORCH_FN(shift3d_forward));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift3d_backward"),
        TORCH_FN(shift3d_backward));
}

} // namespace ops
} // namespace shifts


#endif
