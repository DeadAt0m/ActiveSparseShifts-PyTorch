#ifndef SHIFTS_CPU
#define SHIFTS_CPU


#include <torch/extension.h>
#include "../global_scope.h"
#include "../kernels/shifts_kernels.h"


namespace shifts {
namespace ops {

namespace {


template <typename scalar_t, int kSpatialDim=1,
          BIPadding padding_mode = BIPadding::Zeros>
API_INLINE void qshiftnd_kernel(const torch::Tensor& input,
                                const torch::Tensor& weights,
                                const torch::Tensor& borders,
                                torch::Tensor& output,
                                int64_t weights_zero_point){
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
    const scalar_t zero_point  = static_cast<scalar_t>(input.q_zero_point());
    scalar_t* output_ptr = output.data_ptr<scalar_t>();
    int64_t* weights_ptr = weights.data_ptr<int64_t>();
    const int64_t weights_sC = weights.stride(0);
    const int64_t weights_sS = weights.stride(1);
    
    
    
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
                            shift_forward_kernel_nhwdc_q<scalar_t, int64_t, kSpatialDim, padding_mode>(
                                        input_ptr, output_ptr, weights_ptr,
                                        n, i, j, k, sizeC, sizeH, sizeW, sizeD,
                                        input_sN, input_sC, input_sH, input_sW, input_sD,
                                        output_sN, output_sC, output_sH, output_sW, output_sD,
                                        weights_sC, weights_sS,
                                        i_left_border, j_left_border, k_left_border,
                                        i_right_border, j_right_border, k_right_border,
                                        zero_point, weights_zero_point);
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
                            shift_forward_kernel_nchwd_q<scalar_t, int64_t, kSpatialDim, padding_mode>(
                                        input_ptr, output_ptr, weights_ptr,
                                        n, c, i, j, k, sizeH, sizeW, sizeD,
                                        input_sN, input_sC, input_sH, input_sW, input_sD,
                                        output_sN, output_sC, output_sH, output_sW, output_sD,
                                        weights_sC, weights_sS,
                                        i_left_border, j_left_border, k_left_border,
                                        i_right_border, j_right_border, k_right_border,
                                        zero_point,  weights_zero_point);
                        }
                    }
                }
            }
        });
    }
}






template <int nD, BIPadding padding_mode = BIPadding::Zeros>
torch::Tensor qshiftnd(const torch::Tensor& input,
                       const torch::Tensor& weights,
                       const torch::Tensor& borders,
                       const std::vector<int64_t>& new_size){
    std::string name = "q_shift"+std::to_string(nD)+"d_cpu";
    torch::Tensor output;
    int64_t weights_zero_point = static_cast<int64_t>(weights.q_zero_point());
    torch::Tensor iweights = weights.int_repr().to(torch::kLong);
    
    torch::Tensor _borders = borders.to(torch::kLong);
    
    if (input.is_contiguous(c10::MemoryFormat::ChannelsLast) || input.is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
        output = at::_empty_affine_quantized(new_size, input.options().memory_format(input.suggest_memory_format()),
                                             input.q_scale(), input.q_zero_point(), c10::nullopt);
    }
    else {
        output = at::_empty_affine_quantized(new_size, input.options(), input.q_scale(), input.q_zero_point());
    }
    AT_DISPATCH_QINT_TYPES(input.scalar_type(), name, [&] {
                qshiftnd_kernel<scalar_t, nD, padding_mode>(input, iweights, _borders, output, weights_zero_point);
    }); 
    return output;
}

 
// TEMPLATE DISPATCHERS   


torch::Tensor qshift1d(const torch::Tensor& input,
                       const torch::Tensor& weights,
                       const torch::Tensor& borders,
                       const std::vector<int64_t>& new_size,
                       int64_t padding_mode,
                       bool active_flag){ //active_flag not used here, but needs for API compatibility
    torch::Tensor ret;
    switch (padding_mode){        
        case 0:
            ret = qshiftnd<1, BIPadding::Zeros>(input, weights, borders, new_size);
            break;
        case 1:
            ret = qshiftnd<1, BIPadding::Border>(input, weights, borders, new_size);
            break;
        case 2:
            ret = qshiftnd<1, BIPadding::Periodic>(input, weights, borders, new_size);
            break;
        case 3:
            ret = qshiftnd<1, BIPadding::Reflect>(input, weights, borders, new_size);
            break;
        case 4:
            ret = qshiftnd<1, BIPadding::Symmetric>(input, weights, borders, new_size);
            break;
    }
    return ret;                  
}


torch::Tensor qshift2d(const torch::Tensor& input,
                       const torch::Tensor& weights,
                       const torch::Tensor& borders,
                       const std::vector<int64_t>& new_size,
                       int64_t padding_mode,
                       bool active_flag){ //active_flag not used here, but needs for API compatibility
    torch::Tensor ret;
    switch (padding_mode){        
        case 0:
            ret = qshiftnd<2, BIPadding::Zeros>(input, weights, borders, new_size);
            break;
        case 1:
            ret = qshiftnd<2, BIPadding::Border>(input, weights, borders, new_size);
            break;
        case 2:
            ret = qshiftnd<2, BIPadding::Periodic>(input, weights, borders, new_size);
            break;
        case 3:
            ret = qshiftnd<2, BIPadding::Reflect>(input, weights, borders, new_size);
            break;
        case 4:
            ret = qshiftnd<2, BIPadding::Symmetric>(input, weights, borders, new_size);
            break;
    }
    return ret;                  
}

torch::Tensor qshift3d(const torch::Tensor& input,
                       const torch::Tensor& weights,
                       const torch::Tensor& borders,
                       const std::vector<int64_t>& new_size,
                       int64_t padding_mode,
                       bool active_flag){ //active_flag not used here, but needs for API compatibility
    torch::Tensor ret;
    switch (padding_mode){        
        case 0:
            ret = qshiftnd<3, BIPadding::Zeros>(input, weights, borders, new_size);
            break;
        case 1:
            ret = qshiftnd<3, BIPadding::Border>(input, weights, borders, new_size);
            break;
        case 2:
            ret = qshiftnd<3, BIPadding::Periodic>(input, weights, borders, new_size);
            break;
        case 3:
            ret = qshiftnd<3, BIPadding::Reflect>(input, weights, borders, new_size);
            break;
        case 4:
            ret = qshiftnd<3, BIPadding::Symmetric>(input, weights, borders, new_size);
            break;
    }
    return ret;                  
}
    
std::tuple<torch::Tensor, torch::Tensor> qshiftnd_backward(const torch::Tensor& grad,
                                                           const torch::Tensor& weights,
                                                           const torch::Tensor& input,
                                                           const torch::Tensor& borders,
                                                           int64_t padding_mode,
                                                           bool active_flag){
    TORCH_CHECK(0, "backwards on quantized tensor are not supported");
}

} // end of anonymous namespace


TORCH_LIBRARY_IMPL(torchshifts, QuantizedCPU, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift1d_forward"),
        TORCH_FN(qshift1d));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift1d_backward"),
        TORCH_FN(qshiftnd_backward));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift2d_forward"),
        TORCH_FN(qshift2d));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift2d_backward"),
        TORCH_FN(qshiftnd_backward));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift3d_forward"),
        TORCH_FN(qshift3d));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift3d_backward"),
        TORCH_FN(qshiftnd_backward));
}

} // namespace ops
} // namespace shifts

    
#endif
