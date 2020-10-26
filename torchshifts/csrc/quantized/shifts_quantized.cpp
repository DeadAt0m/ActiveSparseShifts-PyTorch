#ifndef _SHIFTS_CPU
#define _SHIFTS_CPU

#include "shifts_quantized.h"
#include "../kernels/shifts_kernels.h"




template <typename scalar_t, int32_t kSpatialDim>
API_INLINE void _q_shifts_cpu(const torch::Tensor& input, const torch::Tensor& weights,
                              torch::Tensor& output, BIPadding padding_mode){
    int64_t sizeN = input.size(0);
    int64_t sizeC = input.size(1);
    int64_t sizeH = input.size(2);
    int64_t sizeW = kSpatialDim < 2 ? 1 : input.size(3);
    int64_t sizeD = kSpatialDim < 3 ? 1 : input.size(4);
    int64_t input_sN = input.stride(0);
    int64_t input_sC = input.stride(1);
    int64_t input_sH = input.stride(2);
    int64_t input_sW = kSpatialDim < 2 ? 0 : input.stride(3);
    int64_t input_sD = kSpatialDim < 3 ? 0 : input.stride(4);
    int64_t output_sN = output.stride(0);
    int64_t output_sC = output.stride(1);
    int64_t output_sH = output.stride(2);
    int64_t output_sW = kSpatialDim < 2 ? 0 : output.stride(3);
    int64_t output_sD = kSpatialDim < 3 ? 0 : output.stride(4);
    scalar_t *input_ptr = input.data_ptr<scalar_t>();
    scalar_t zero_point  = static_cast<scalar_t>(input.q_zero_point());
    int64_t weights_zero_point = static_cast<int64_t>(weights.q_zero_point());
    scalar_t *output_ptr = output.data_ptr<scalar_t>();
    int64_t *weights_ptr = nullptr;
    init_weights<scalar_t, int64_t, true, false>(weights.data_ptr<scalar_t>(), weights_ptr, (int)weights.numel());
    int64_t weights_sC = weights.stride(0);
    int64_t weights_sS = weights.stride(1);
    scalar_t *dweights_ptr = nullptr;
    int64_t dweights_sC = 0;
    int64_t dweights_sS = 0;
    if (input.is_contiguous(c10::MemoryFormat::ChannelsLast) || input.is_contiguous(c10::MemoryFormat::ChannelsLast3d))
    {// Path for NDHWC
        at::parallel_for(0, sizeN*sizeH*sizeW*sizeD, 0, [&](int64_t start, int64_t end){
            for (int64_t index = start; index < end; ++index) {
                int64_t k = index % sizeD;
                int64_t j = (index / sizeD) % sizeW;
                int64_t i = (index / (sizeD*sizeW)) % sizeH;
                int64_t n = (index / (sizeD*sizeW*sizeH));
                shift_forward_kernel_nhwdc<scalar_t, int64_t, true, false>(input_ptr, output_ptr, weights_ptr, dweights_ptr,
                                                                                 n, i, j, k, sizeC, sizeH, sizeW, sizeD,
                                                                                 input_sN, input_sC, input_sH, input_sW, input_sD,
                                                                                 output_sN, output_sC, output_sH, output_sW, output_sD,
                                                                                 weights_sC, weights_sS, dweights_sC, dweights_sS,
                                                                                 zero_point,  weights_zero_point, padding_mode);
            }
        });
    } else
    {
        at::parallel_for(0, sizeN*sizeC*sizeH*sizeW*sizeD, 0, [&](int64_t start, int64_t end){
            for (int64_t index = start; index < end; ++index) {
                int64_t k = index % sizeD;
                int64_t j = (index / sizeD) % sizeW;
                int64_t i = (index / (sizeD*sizeW)) % sizeH;
                int64_t c = (index / (sizeD*sizeW*sizeH)) % sizeC;
                int64_t n = (index / (sizeD*sizeW*sizeH*sizeC));
                shift_forward_kernel_nchwd<scalar_t, int64_t, true, false>(input_ptr, output_ptr, weights_ptr, dweights_ptr,
                                                                           n, c, i, j, k, sizeH, sizeW, sizeD,
                                                                           input_sN, input_sC, input_sH, input_sW, input_sD,
                                                                           output_sN, output_sC, output_sH, output_sW, output_sD,
                                                                           weights_sC, weights_sS, dweights_sC, dweights_sS,
                                                                           zero_point,  weights_zero_point, padding_mode);
            }
        });
    }
    delete weights_ptr;
}


template <int nD>
torch::Tensor q_shiftnd_cpu(const torch::Tensor& input,
                            const torch::Tensor& weights,
                            int64_t padding_mode){
    std::string name = "q_shift"+std::to_string(nD)+"d_cpu";
    torch::Tensor output;
    if (input.is_contiguous(c10::MemoryFormat::ChannelsLast) || input.is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
        output = at::_empty_affine_quantized(input.sizes(), input.options().memory_format(input.suggest_memory_format()),
                                             input.q_scale(), input.q_zero_point(), c10::nullopt);
    }
    else {
        output = at::_empty_affine_quantized(input.sizes(), input.options(), input.q_scale(), input.q_zero_point());
    }

    AT_DISPATCH_QINT_TYPES(input.scalar_type(), name, [&] {
            _q_shifts_cpu<scalar_t, nD>(input, weights, output,
                                        static_cast<BIPadding>(padding_mode));
        }); 
    return output;
}




torch::Tensor q_shift1d_cpu(const torch::Tensor& input,
                            const torch::Tensor& weights,
                            int64_t padding_mode){
    return q_shiftnd_cpu<1>(input, weights, padding_mode);                    
}

torch::Tensor q_shift2d_cpu(const torch::Tensor& input,
                            const torch::Tensor& weights,
                            int64_t padding_mode){
    return q_shiftnd_cpu<2>(input, weights, padding_mode);                    
}

torch::Tensor q_shift3d_cpu(const torch::Tensor& input,
                            const torch::Tensor& weights,
                            int64_t padding_mode){
    return q_shiftnd_cpu<3>(input, weights, padding_mode);                    
}

#endif
