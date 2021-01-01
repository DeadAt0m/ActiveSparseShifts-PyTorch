#ifndef _SHIFTS_CPU
#define _SHIFTS_CPU

#include "shifts_quantized.h"
#include "../kernels/shifts_kernels.h"




template <typename scalar_t, int kSpatialDim=1,
          BIPadding padding_mode = BIPadding::Zeros>
API_INLINE void _q_shifts_cpu(const torch::Tensor& input, const torch::Tensor& weights,
                              const torch::Tensor& borders,
                              torch::Tensor& output,
                              int64_t weights_zero_point){
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
    scalar_t *output_ptr = output.data_ptr<scalar_t>();
    int64_t *weights_ptr = weights.data_ptr<int64_t>();
    int64_t weights_sC = weights.stride(0);
    int64_t weights_sS = weights.stride(1);
    
    
    
    int64_t *borders_data = borders.data_ptr<int64_t>();
    int64_t i_left_border = MAX(static_cast<int64_t>(0), borders_data[0]);
    int64_t i_right_border = MIN(sizeH, borders_data[1]);
    int64_t j_left_border = kSpatialDim < 2 ? 0 : MAX(static_cast<int64_t>(0), borders_data[2]);
    int64_t j_right_border = kSpatialDim < 2 ? 1 : MIN(sizeW, borders_data[3]);
    int64_t k_left_border =  kSpatialDim < 3 ? 0 : MAX(static_cast<int64_t>(0), borders_data[4]);
    int64_t k_right_border =  kSpatialDim < 3 ? 1 : MIN(sizeD, borders_data[5]);
    
    int64_t sizeWH = sizeW*sizeH;
    int64_t sizeWD = sizeD*sizeW;
    int64_t sizeDWH = sizeWD*sizeH;
    int64_t sizeDWHC = sizeDWH * sizeC;
    int64_t sizeHC = sizeH*sizeC;
    int64_t sizeWHC = sizeW*sizeH*sizeC;

    if (input.is_contiguous(c10::MemoryFormat::ChannelsLast) || input.is_contiguous(c10::MemoryFormat::ChannelsLast3d))
    {// Path for NDHWC
        at::parallel_for(0, sizeN*sizeH*sizeW*sizeD, 0, [&](int64_t start, int64_t end){
            for (int64_t index = start; index < end; ++index) {
                int64_t k, j, i, n;
                switch (kSpatialDim){
                    case 1:
                        k = 0;
                        j = 0;
                        i = index % sizeH;
                        n = index / sizeH;
                        break;
                    case 2:
                        k = 0;
                        j = index % sizeW;
                        i = (index / sizeW) % sizeH;
                        n = index / sizeWH;
                        break;
                    case 3:
                        k = index % sizeD;
                        j = (index / sizeD) % sizeW;
                        i = (index / sizeWD) % sizeH;
                        n = (index / sizeDWH);
                        break;
                }  
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
        });
    } else
    {
        at::parallel_for(0, sizeN*sizeC*sizeH*sizeW*sizeD, 0, [&](int64_t start, int64_t end){
            for (int64_t index = start; index < end; ++index) {
                int64_t k, j, i, c, n;
                switch (kSpatialDim){
                    case 1:
                        k = 0;
                        j = 0;
                        i = index % sizeH;
                        c = (index / sizeH) % sizeC;
                        n =  index / sizeHC;
                        break;
                    case 2:
                        k = 0;
                        j = index % sizeW;
                        i = (index / sizeW) % sizeH;
                        c = (index / sizeWH) % sizeC; 
                        n = index / sizeWHC;
                        break;
                    case 3:
                        k = index % sizeD;
                        j = (index / sizeD) % sizeW;
                        i = (index / sizeWD) % sizeH;
                        c = (index / sizeDWH) % sizeC;
                        n = index / sizeDWHC;
                        break;
                }            
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
        });
    }
}






template <int nD>
torch::Tensor q_shiftnd_cpu(const torch::Tensor& input,
                            const torch::Tensor& weights,
                            const torch::Tensor& borders,
                            const std::vector<int64_t>& new_size,
                            int64_t padding_mode){
    std::string name = "q_shift"+std::to_string(nD)+"d_cpu";
    torch::Tensor output;
    int64_t weights_zero_point = static_cast<int64_t>(weights.q_zero_point());
    torch::Tensor iweights = weights.int_repr().to(torch::kLong);
    

    if (input.is_contiguous(c10::MemoryFormat::ChannelsLast) || input.is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
        output = at::_empty_affine_quantized(new_size, input.options().memory_format(input.suggest_memory_format()),
                                             input.q_scale(), input.q_zero_point(), c10::nullopt);
    }
    else {
        output = at::_empty_affine_quantized(new_size, input.options(), input.q_scale(), input.q_zero_point());
    }
    switch (padding_mode){        
        case 0:
            AT_DISPATCH_QINT_TYPES(input.scalar_type(), name, [&] {
                _q_shifts_cpu<scalar_t, nD, BIPadding::Zeros>(input, iweights, borders, output, weights_zero_point);
            }); 
            break;
        case 1:
            AT_DISPATCH_QINT_TYPES(input.scalar_type(), name, [&] {
                _q_shifts_cpu<scalar_t, nD, BIPadding::Border>(input, iweights, borders, output, weights_zero_point);
            }); 
            break;
        case 2:
            AT_DISPATCH_QINT_TYPES(input.scalar_type(), name, [&] {
                _q_shifts_cpu<scalar_t, nD, BIPadding::Periodic>(input, iweights, borders, output, weights_zero_point);
            }); 
            break;
        case 3:
            AT_DISPATCH_QINT_TYPES(input.scalar_type(), name, [&] {
                _q_shifts_cpu<scalar_t, nD, BIPadding::Reflect>(input, iweights, borders, output, weights_zero_point);
            }); 
            break;
        case 4:
            AT_DISPATCH_QINT_TYPES(input.scalar_type(), name, [&] {
                _q_shifts_cpu<scalar_t, nD, BIPadding::Symmetric>(input, iweights, borders, output, weights_zero_point);
            }); 
            break;
    }
    return output;
}




torch::Tensor q_shift1d_cpu(const torch::Tensor& input,
                            const torch::Tensor& weights,
                            const torch::Tensor& borders,
                            const std::vector<int64_t>& new_size,
                            int64_t padding_mode){
    return q_shiftnd_cpu<1>(input, weights, borders, new_size, padding_mode);                    
}

torch::Tensor q_shift2d_cpu(const torch::Tensor& input,
                            const torch::Tensor& weights,
                            const torch::Tensor& borders,
                            const std::vector<int64_t>& new_size,
                            int64_t padding_mode){
    return q_shiftnd_cpu<2>(input, weights, borders, new_size, padding_mode);                    
}

torch::Tensor q_shift3d_cpu(const torch::Tensor& input,
                            const torch::Tensor& weights,
                            const torch::Tensor& borders,
                            const std::vector<int64_t>& new_size,
                            int64_t padding_mode){
    return q_shiftnd_cpu<3>(input, weights, borders, new_size, padding_mode);                    
}

#endif
