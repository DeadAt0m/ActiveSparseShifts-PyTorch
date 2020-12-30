#ifndef _SHIFTS_CPU
#define _SHIFTS_CPU

#include "shifts_cpu.h"
#include "../kernels/shifts_kernels.h"




template <typename scalar_t, int kSpatialDim = 1, 
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
API_INLINE void _shifts_forward_cpu(const torch::Tensor& input, const torch::Tensor& iweights,
                                    const torch::Tensor& dweights,
                                    const torch::Tensor& borders,
                                    torch::Tensor& output){
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
    scalar_t *output_ptr = output.data_ptr<scalar_t>();
    int64_t *weights_ptr = iweights.data_ptr<int64_t>();
    int64_t weights_sC = iweights.stride(0);
    int64_t weights_sS = iweights.stride(1);
    scalar_t *dweights_ptr = dweights.data_ptr<scalar_t>();
    int64_t dweights_sC = dweights.stride(0);
    int64_t dweights_sS = dweights.stride(1);
    
    int64_t *borders_data = borders.data_ptr<int64_t>();
    int64_t i_left_border = MAX(static_cast<int64_t>(0), borders_data[0]);
    int64_t i_right_border = MIN(sizeH, borders_data[1]);
    int64_t j_left_border = kSpatialDim < 2 ? 0 : MAX(static_cast<int64_t>(0), borders_data[2]);
    int64_t j_right_border = kSpatialDim < 2 ? 1 : MIN(sizeW, borders_data[3]);
    int64_t k_left_border =  kSpatialDim < 3 ? 0 : MAX(static_cast<int64_t>(0), borders_data[4]);
    int64_t k_right_border =  kSpatialDim < 3 ? 1 : MIN(sizeD, borders_data[5]);
    
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
                        n = index / (sizeW*sizeH);
                        break;
                    case 3:
                        k = index % sizeD;
                        j = (index / sizeD) % sizeW;
                        i = (index / (sizeD*sizeW)) % sizeH;
                        n = (index / (sizeD*sizeW*sizeH));
                        break;
                }                
                shift_forward_kernel_nhwdc<scalar_t, int64_t, kSpatialDim, padding_mode, active>(
                                        input_ptr, output_ptr, weights_ptr, dweights_ptr,
                                        n, i, j, k, sizeC, sizeH, sizeW, sizeD,
                                        input_sN, input_sC, input_sH, input_sW, input_sD,
                                        output_sN, output_sC, output_sH, output_sW, output_sD,
                                        weights_sC, weights_sS, dweights_sC, dweights_sS,
                                        i_left_border, j_left_border, k_left_border,
                                        i_right_border, j_right_border, k_right_border);
            }
        });
    } else
    {
        at::parallel_for(0, sizeN*sizeC*sizeH*sizeW*sizeD, 0, [&](int64_t start, int64_t end){
            for (int64_t index = start; index < end; ++index) {
                int64_t k, j, i, c n;
                switch (kSpatialDim){
                    case 1:
                        k = 0;
                        j = 0;
                        i = index % sizeH;
                        c = (index / sizeH) % sizeC;
                        n =  index / (sizeH*sizeC);
                        break;
                    case 2:
                        k = 0;
                        j = index % sizeW;
                        i = (index / sizeW) % sizeH;
                        c = (index / (sizeW*sizeH)) % sizeC; 
                        n = index / (sizeW*sizeH*sizeC);
                        break;
                    case 3:
                        k = index % sizeD;
                        j = (index / sizeD) % sizeW;
                        i = (index / (sizeD*sizeW)) % sizeH;
                        c = (index / (sizeD*sizeW*sizeH)) % sizeC;
                        n = (index / (sizeD*sizeW*sizeH*sizeC));
                        break;
                }                
                shift_forward_kernel_nchwd<scalar_t, int64_t, kSpatialDim, padding_mode, active>(
                                        input_ptr, output_ptr, weights_ptr, dweights_ptr,
                                        n, c, i, j, k, sizeH, sizeW, sizeD,
                                        input_sN, input_sC, input_sH, input_sW, input_sD,
                                        output_sN, output_sC, output_sH, output_sW, output_sD,
                                        weights_sC, weights_sS, dweights_sC, dweights_sS,
                                        i_left_border, j_left_border, k_left_border,
                                        i_right_border, j_right_border, k_right_border);
            }
        });
    }
}


template <typename scalar_t, int kSpatialDim = 1, 
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
API_INLINE void _shifts_backward_cpu(const torch::Tensor& grad_input, 
                                     const torch::Tensor& iweights,
                                     const torch::Tensor& dweights,
                                     const torch::Tensor& input, 
                                     const torch::Tensor& borders,
                                     torch::Tensor& grad_output,
                                     torch::Tensor& grad_weights)
{
    int64_t sizeN = grad_input.size(0);
    int64_t sizeC = grad_input.size(1);
    int64_t sizeH = grad_input.size(2);
    int64_t sizeW = kSpatialDim < 2 ? 1 : grad_input.size(3);
    int64_t sizeD = kSpatialDim < 3 ? 1 : grad_input.size(4);
    int64_t grad_input_sN = grad_input.stride(0);
    int64_t grad_input_sC = grad_input.stride(1);
    int64_t grad_input_sH = grad_input.stride(2);
    int64_t grad_input_sW = kSpatialDim < 2 ? 0 : grad_input.stride(3);
    int64_t grad_input_sD = kSpatialDim < 3 ? 0 : grad_input.stride(4);
    int64_t input_sN = input.stride(0);
    int64_t input_sC = input.stride(1);
    int64_t input_sH = input.stride(2);
    int64_t input_sW = kSpatialDim < 2 ? 0 : input.stride(3);
    int64_t input_sD = kSpatialDim < 3 ? 0 : input.stride(4);
    int64_t grad_output_sN = grad_output.stride(0);
    int64_t grad_output_sC = grad_output.stride(1);
    int64_t grad_output_sH = grad_output.stride(2);
    int64_t grad_output_sW = kSpatialDim < 2 ? 0 : grad_output.stride(3);
    int64_t grad_output_sD = kSpatialDim < 3 ? 0 : grad_output.stride(4);
    int64_t *weights_ptr = iweights.data_ptr<int64_t>();
    int64_t weights_sC = iweights.stride(0);
    int64_t weights_sS = iweights.stride(1);
    scalar_t *dweights_ptr = dweights.data_ptr<scalar_t>();
    int64_t dweights_sC = dweights.stride(0);
    int64_t dweights_sS = dweights.stride(1);
    int64_t grad_weights_sC = grad_weights.stride(0);
    int64_t grad_weights_sS = grad_weights.stride(1);
    scalar_t *grad_weights_ptr = grad_weights.data_ptr<scalar_t>();
    scalar_t *grad_input_ptr = grad_input.data_ptr<scalar_t>();
    scalar_t *input_ptr = input.data_ptr<scalar_t>();
    scalar_t *grad_output_ptr = grad_output.data_ptr<scalar_t>();
    
    int64_t *borders_data = borders.data_ptr<int64_t>();
    int64_t i_left_border = MAX(static_cast<int64_t>(0), borders_data[0]);
    int64_t i_right_border = MIN(input.size(2), borders_data[1]);
    int64_t j_left_border = kSpatialDim < 2 ? 0 : MAX(static_cast<int64_t>(0), borders_data[2]);
    int64_t j_right_border = kSpatialDim < 2 ? 1 : MIN(input.size(3), borders_data[3]);
    int64_t k_left_border = kSpatialDim < 3 ? 0 : MAX(static_cast<int64_t>(0), borders_data[4]);
    int64_t k_right_border = kSpatialDim < 3 ? 1 : MIN(input.size(4), borders_data[5]);
    
    
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
                        n = index / (sizeW*sizeH);
                        break;
                    case 3:
                        k = index % sizeD;
                        j = (index / sizeD) % sizeW;
                        i = (index / (sizeD*sizeW)) % sizeH;
                        n = (index / (sizeD*sizeW*sizeH));
                        break;
                }      
                shift_backward_kernel_nhwdc<scalar_t, int64_t, kSpatialDim, padding_mode, active>(
                                    grad_input_ptr, input_ptr, grad_output_ptr,
                                    weights_ptr, dweights_ptr, grad_weights_ptr,
                                    n, i, j, k, sizeC, sizeH, sizeW, sizeD,
                                    grad_input_sN, grad_input_sC, grad_input_sH, grad_input_sW, grad_input_sD,
                                    input_sN, input_sC, input_sH, input_sW, input_sD,
                                    grad_output_sN, grad_output_sC, grad_output_sH, grad_output_sW, grad_output_sD,
                                    weights_sC, weights_sS, dweights_sC, dweights_sS, grad_weights_sC, grad_weights_sS,
                                    i_left_border, j_left_border, k_left_border,
                                    i_right_border, j_right_border, k_right_border);
            }
        });
    } else
    {
        at::parallel_for(0, sizeN*sizeC*sizeH*sizeW*sizeD, 0, [&](int64_t start, int64_t end){
            for (int64_t index = start; index < end; ++index) {
                int64_t k, j, i, c n;
                switch (kSpatialDim){
                    case 1:
                        k = 0;
                        j = 0;
                        i = index % sizeH;
                        c = (index / sizeH) % sizeC;
                        n =  index / (sizeH*sizeC);
                        break;
                    case 2:
                        k = 0;
                        j = index % sizeW;
                        i = (index / sizeW) % sizeH;
                        c = (index / (sizeW*sizeH)) % sizeC; 
                        n = index / (sizeW*sizeH*sizeC);
                        break;
                    case 3:
                        k = index % sizeD;
                        j = (index / sizeD) % sizeW;
                        i = (index / (sizeD*sizeW)) % sizeH;
                        c = (index / (sizeD*sizeW*sizeH)) % sizeC;
                        n = (index / (sizeD*sizeW*sizeH*sizeC));
                        break;
                }      
                shift_backward_kernel_nchwd<scalar_t, int64_t, kSpatialDim, padding_mode, active>(
                                    grad_input_ptr, input_ptr, grad_output_ptr,
                                    weights_ptr, dweights_ptr, grad_weights_ptr,
                                    n, c, i, j, k, sizeH, sizeW, sizeD,
                                    grad_input_sN, grad_input_sC, grad_input_sH, grad_input_sW, grad_input_sD,
                                    input_sN, input_sC, input_sH, input_sW, input_sD,
                                    grad_output_sN, grad_output_sC, grad_output_sH, grad_output_sW, grad_output_sD,
                                    weights_sC, weights_sS, dweights_sC, dweights_sS, grad_weights_sC, grad_weights_sS,
                                    i_left_border, j_left_border, k_left_border,
                                    i_right_border, j_right_border, k_right_border);
            }
        });
    }
}


template <int nD>
torch::Tensor shiftnd_forward_cpu(const torch::Tensor& input,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& borders,
                                  const std::vector<int64_t>& new_size,
                                  int64_t padding_mode,
                                  bool active_flag){
    std::string name = "shift"+std::to_string(nD)+"d_forward_cpu";
    
    torch::Tensor output = torch::zeros(new_size, input.options());
    
    torch::Tensor iweights = (active_flag?torch::floor(weights):torch::round(weights)).to(torch::kLong);
    torch::Tensor dweights = torch::empty_like(weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    if (active_flag){
        dweights = weights - torch::floor(weights);
    }
    switch (padding_mode){        
        case 0:
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), name, [&] {
                (active_flag)?_shifts_forward_cpu<scalar_t, nD, BIPadding::Zeros, true>(input, iweights, dweights, borders, output):
                              _shifts_forward_cpu<scalar_t, nD, BIPadding::Zeros, false>(input, iweights, dweights, borders, output);
            });
            break;
        case 1:
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), name, [&] {
                (active_flag)?_shifts_forward_cpu<scalar_t, nD, BIPadding::Border, true>(input, iweights, dweights, borders, output):
                              _shifts_forward_cpu<scalar_t, nD,  BIPadding::Border, false>(input, iweights, dweights, borders output);
            });
            break;
        case 2:
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), name, [&] {
                (active_flag)?_shifts_forward_cpu<scalar_t, nD, BIPadding::Periodic, true>(input, iweights, dweights, borders, output):
                              _shifts_forward_cpu<scalar_t, nD, BIPadding::Periodic, false>(input, iweights, dweights, borders, output);
            });
            break;
        case 3:
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), name, [&] {
                (active_flag)?_shifts_forward_cpu<scalar_t, nD, BIPadding::Reflect, true>(input, iweights, dweights, borders, output):
                              _shifts_forward_cpu<scalar_t, nD, BIPadding::Reflect, false>(input, iweights, dweights, borders, output);
            });
            break;
        case 4:
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), name, [&] {
                (active_flag)?_shifts_forward_cpu<scalar_t, nD, BIPadding::Symmetric, true>(input, iweights, dweights, borders, output):
                              _shifts_forward_cpu<scalar_t, nD, BIPadding::Symmetric, false>(input, iweights, dweights, borders, output);
            }); 
            break;
    }
    return output;
}


template <int nD>
std::vector<torch::Tensor> shiftnd_backward_cpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                const torch::Tensor& borders,
                                                int64_t padding_mode,
                                                bool active_flag) {
    std::string name = "shift"+std::to_string(nD)+"d_backward_cpu";
    
    torch::Tensor iweights = (active_flag?torch::floor(weights):torch::round(weights)).to(torch::kLong);
    torch::Tensor dweights = weights - torch::floor(weights);
    
    torch::Tensor out_grad = torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor weights_grad = torch::zeros_like(weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    switch (padding_mode){        
        case 0:
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), name, [&] {
                (active_flag)?_shifts_backward_cpu<scalar_t, nD, BIPadding::Zeros, true>(grad, iweights, dweights, input, borders, out_grad, weights_grad):
                              _shifts_backward_cpu<scalar_t, nD, BIPadding::Zeros, false>(grad, iweights, dweights, input, borders, out_grad, weights_grad);
            });
            break;
        case 1:
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), name, [&] {
                (active_flag)?_shifts_backward_cpu<scalar_t, nD, BIPadding::Border, true>(grad, iweights, dweights, input, borders, out_grad, weights_grad):
                              _shifts_backward_cpu<scalar_t, nD,  BIPadding::Border, false>(grad, iweights, dweights, input, borders, out_grad, weights_grad);
            });
            break;
        case 2:
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), name, [&] {
                (active_flag)?_shifts_backward_cpu<scalar_t, nD, BIPadding::Periodic, true>(grad, iweights, dweights, input, borders, out_grad, weights_grad):
                              _shifts_backward_cpu<scalar_t, nD, BIPadding::Periodic, false>(grad, iweights, dweights, input, borders, out_grad, weights_grad);
            });
            break;
        case 3:
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), name, [&] {
                (active_flag)?_shifts_backward_cpu<scalar_t, nD, BIPadding::Reflect, true>(grad, iweights, dweights, input, borders, out_grad, weights_grad):
                              _shifts_backward_cpu<scalar_t, nD, BIPadding::Reflect, false>(grad, iweights, dweights, input, borders, out_grad, weights_grad);
            });
            break;
        case 4:
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), name, [&] {
                (active_flag)?_shifts_backward_cpu<scalar_t, nD, BIPadding::Symmetric, true>(grad, iweights, dweights, input, borders, out_grad, weights_grad):
                              _shifts_backward_cpu<scalar_t, nD, BIPadding::Symmetric, false>(grad, iweights, dweights, input, borders, out_grad, weights_grad);
            });
            break;
    } 
    return {out_grad, weights_grad};
}




torch::Tensor shift1d_forward_cpu(const torch::Tensor& input,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& borders,
                                  const std::vector<int64_t>& new_size,
                                  int64_t padding_mode,
                                  bool active_flag){
    return shiftnd_forward_cpu<1>(input, weights, borders, new_size, padding_mode, active_flag);                    
}

torch::Tensor shift2d_forward_cpu(const torch::Tensor& input,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& borders,
                                  const std::vector<int64_t>& new_size,
                                  int64_t padding_mode,
                                  bool active_flag){
    return shiftnd_forward_cpu<2>(input, weights, borders, new_size, padding_mode, active_flag);                    
}

torch::Tensor shift3d_forward_cpu(const torch::Tensor& input,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& borders,
                                  const std::vector<int64_t>& new_size,
                                  int64_t padding_mode,
                                  bool active_flag){
    return shiftnd_forward_cpu<3>(input, weights, borders, new_size, padding_mode, active_flag);                    
}


std::vector<torch::Tensor> shift1d_backward_cpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                const torch::Tensor& borders,
                                                int64_t padding_mode,
                                                bool active_flag){
    return  shiftnd_backward_cpu<1>(grad, weights, input, borders, padding_mode, active_flag);                                       
}

std::vector<torch::Tensor> shift2d_backward_cpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                const torch::Tensor& borders,
                                                int64_t padding_mode,
                                                bool active_flag){
    return  shiftnd_backward_cpu<2>(grad, weights, input, borders, padding_mode, active_flag);                                       
}

std::vector<torch::Tensor> shift3d_backward_cpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                const torch::Tensor& borders,
                                                int64_t padding_mode,
                                                bool active_flag){
    return  shiftnd_backward_cpu<3>(grad, weights, input, borders, padding_mode, active_flag);                                       
}



#endif