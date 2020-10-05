#ifndef _SHIFTS_CPU
#define _SHIFTS_CPU

#include <torch/extension.h>
#include "shifts_kernels.h"

template <typename scalar_t, int32_t kSpatialDim, bool quantized=false, bool active=false>
inline void _shifts_cpu(const torch::Tensor& input, const torch::Tensor& weights,
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
    scalar_t zero_point = quantized ? static_cast<scalar_t>(input.q_zero_point()) : 0;
    scalar_t *output_ptr = output.data_ptr<scalar_t>();
    int64_t *weights_ptr = init_weights<scalar_t, int64_t, quantized, active>(weights.data_ptr<scalar_t>(), (int)weights.numel());
    int64_t weights_zero_point = quantized ? static_cast<int64_t>(weights.q_zero_point()) : 0;
    int64_t weights_sC = weights.stride(0);
    int64_t weights_sS = weights.stride(1);
    scalar_t *dweights_ptr = active ? init_weight_offsets<scalar_t>(weights.data_ptr<scalar_t>(), (int)weights.numel()) : &zero_point;
    int64_t dweights_sC = active ? weights_sC : 0;
    int64_t dweights_sS = active ? weights_sS : 0;
    if (input.is_contiguous(c10::MemoryFormat::ChannelsLast) || input.is_contiguous(c10::MemoryFormat::ChannelsLast3d))
    {// Path for NDHWC
        at::parallel_for(0, N*H*W*D, 0, [&](int64_t start, int64_t end){
            for (int64_t index = start; index < end; ++idx) {
                int64_t k = index % D;
                int64_t j = (index / D) % W;
                int64_t i = (index / (D*W)) % H;
                int64_t n = (index / (D*W*H));
                shift_forward_kernel_nhwdc<scalar_t, int64_t, quantized, active>(input_ptr, output_ptr, weights_ptr, dweights_ptr,
                                                                                 n, i, j, k, sizeC, sizeH, sizeW, sizeD,
                                                                                 input_sN, input_sC, input_sH, input_sW, input_sD,
                                                                                 output_sN, output_sC, output_sH, output_sW, output_sD,
                                                                                 weights_sC, weights_sS, dweights_sC, dweights_sS,
                                                                                 zero_point,  weights_zero_point, padding_mode);
            }
        }
    } else
    {
        at::parallel_for(0, N*C*H*W*D, 0, [&](int64_t start, int64_t end){
            for (int64_t index = start; index < end; ++idx) {
                int64_t k = index % D;
                int64_t j = (index / D) % W;
                int64_t i = (index / (D*W)) % H;
                int64_t c = (index / (D*W*H)) % C;
                int64_t n = (index / (D*W*H*C));
                shift_forward_kernel_nchwd<scalar_t, int64_t, quantized, active>(input_ptr, output_ptr, weights_ptr, dweights_ptr,
                                                                                 n, c, i, j, k, sizeH, sizeW, sizeD,
                                                                                 input_sN, input_sC, input_sH, input_sW, input_sD,
                                                                                 output_sN, output_sC, output_sH, output_sW, output_sD,
                                                                                 weights_sC, weights_sS, dweights_sC, dweights_sS,
                                                                                 zero_point,  weights_zero_point, padding_mode);
            }
        }
    }
    delete weights_ptr;
    if (active) {delete dweights_ptr;}
}

template <typename scalar_t, int32_t kSpatialDim, bool active=false>
inline void _shifts_backward_cpu(const torch::Tensor& grad_input, const torch::Tensor& weights,
                                 const torch::Tensor& input, torch::Tensor& grad_output,
                                 torch::Tensor& grad_weights, BIPadding padding_mode)
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
    int64_t weights_sC = weights.stride(0);
    int64_t weights_sS = weights.stride(1);
    int64_t grad_weights_sC = grad_weights.stride(0);
    int64_t grad_weights_sS = grad_weights.stride(1);
    scalar_t *grad_input_ptr = grad_input.data_ptr<scalar_t>();
    scalar_t *input_ptr = input.data_ptr<scalar_t>();
    scalar_t *grad_output_ptr = grad_output.data_ptr<scalar_t>();
    int64_t *weights_ptr = init_weights<scalar_t, int64_t, quantized, active>(weights.data_ptr<scalar_t>(), (int)weights.numel());
    scalar_t *grad_weights_ptr = grad_weights.data_ptr<scalar_t>();
    int64_t dweights_sC = active ? weights_sC : 0;
    int64_t dweights_sS = active ? weights_sS : 0;
    scalar_t *dweights_ptr = active ? init_weight_offsets<scalar_t>(weights.data_ptr<scalar_t>(), (int)weights.numel()) : &dweights_sC;
    if (input.is_contiguous(c10::MemoryFormat::ChannelsLast) || input.is_contiguous(c10::MemoryFormat::ChannelsLast3d))
    {// Path for NDHWC
        at::parallel_for(0, N*H*W*D, 0, [&](int64_t start, int64_t end){
            for (int64_t index = start; index < end; ++idx) {
                int64_t k = index % D;
                int64_t j = (index / D) % W;
                int64_t i = (index / (D*W)) % H;
                int64_t n = (index / (D*W*H));
                shift_backward_kernel_nhwdc<scalar_t, int64_t, active>(grad_input_ptr, input_ptr, grad_output_ptr,
                                                                       weights_ptr, dweights_ptr, grad_weights_ptr,
                                                                       n, i, j, k, sizeC, sizeH, sizeW, sizeD,
                                                                       grad_input_sN, grad_input_sC, grad_input_sH, grad_input_sW, grad_input_sD,
                                                                       input_sN, input_sC, input_sH, input_sW, input_sD,
                                                                       grad_output_sN, grad_output_sC, grad_output_sH, grad_output_sW, grad_output_sD,
                                                                       weights_sC, weights_sS, dweights_sC, dweights_sS, grad_weights_sC, grad_weights_sS,
                                                                       padding_mode);
            }
        }
    } else
    {
        at::parallel_for(0, N*C*H*W*D, 0, [&](int64_t start, int64_t end){
            for (int64_t index = start; index < end; ++idx) {
                int64_t k = index % D;
                int64_t j = (index / D) % W;
                int64_t i = (index / (D*W)) % H;
                int64_t c = (index / (D*W*H)) % C;
                int64_t n = (index / (D*W*H*C));
                shift_backward_kernel_nchwd<scalar_t, int64_t, active>(grad_input_ptr, input_ptr, grad_output_ptr,
                                                                       weights_ptr, dweights_ptr, grad_weights_ptr,
                                                                       n, c, i, j, k, sizeH, sizeW, sizeD,
                                                                       grad_input_sN, grad_input_sC, grad_input_sH, grad_input_sW, grad_input_sD,
                                                                       input_sN, input_sC, input_sH, input_sW, input_sD,
                                                                       grad_output_sN, grad_output_sC, grad_output_sH, grad_output_sW, grad_output_sD,
                                                                       weights_sC, weights_sS, dweights_sC, dweights_sS, grad_weights_sC, grad_weights_sS,
                                                                       padding_mode);
            }
        }
    }
    delete weights_ptr;
    if (active) {delete dweights_ptr;}
}


template <int nD, std::str name>
torch::Tensor shiftnd_cpu(const torch::Tensor& input,
                          const torch::Tensor& weights,
                          int padding_mode,
                          bool active_flag){
    torch::Tensor output;
    bool is_quantized = input.is_quantized();
    if (is_quantized){
        if (input.is_contiguous(c10::MemoryFormat::ChannelsLast) || input.is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
            output = at::_empty_affine_quantized(input.sizes(), input.options().memory_format(input.suggest_memory_format()),
                                                 input.q_scale(), input.q_zero_point(), c10::nullopt);
        }
        else
        {
            output = at::_empty_affine_quantized(input.sizes(), input.options(), input.q_scale(), input.q_zero_point());
        }
    }
    else {
        output = torch::zeros_like(input, input.options());
    }
    if (is_quantized){ 
        AT_DISPATCH_QINT_TYPES(input.scalar_type(), name, [&] {
            _shifts_cpu<scalar_t, nD, true, false>(input, weights, output,
                                                   static_cast<BIPadding>(padding_mode));
        }); 
    }
    else {
        if (active_flag){
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), name, [&] {
                _shifts_cpu<scalar_t, nD, false, true>(input, weights, output,
                                                       static_cast<BIPadding>(padding_mode));
            });   
        }
        else {
             AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), name, [&] {
                _shifts_cpu<scalar_t, nD, false, false>(input, weights, output,
                                                        static_cast<BIPadding>(padding_mode));
            }); 
        }
    }
    return output;
}

template <int nD, std::str name>
std::vector<torch::Tensor> shiftnd_backward_cpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                int padding_mode,
                                                bool active_flag) {
  auto out_grad = torch::zeros_like(grad, grad.options());
  auto weights_grad = torch::zeros_like(weights, weights.options());
  
  if (active_flag){
      AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), name, [&] {
                                 _shifts_backward_cpu<scalar_t, nD, true>(grad, weights, input, out_grad, weights_grad,
                                                                         static_cast<BIPadding>(padding_mode));
      });
  }
  else {
      AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), name, [&] {
                                 _shifts_backward_cpu<scalar_t, nD, false>(grad, weights, input, out_grad, weights_grad,
                                                                          static_cast<BIPadding>(padding_mode));
      });
  }
  return {out_grad, weights_grad};
}

torch::Tensor shift1d_cpu(const torch::Tensor& input,
                          const torch::Tensor& weights,
                          int padding_mode,
                          bool active_flag){
    return shiftnd_cpu<1,"shift1d_cpu">(input, weights, padding_mode, active_flag);                    
}

torch::Tensor shift2d_cpu(const torch::Tensor& input,
                          const torch::Tensor& weights,
                          int padding_mode,
                          bool active_flag){
    return shiftnd_cpu<2,"shift2d_cpu">(input, weights, padding_mode, active_flag);                    
}

torch::Tensor shift3d_cpu(const torch::Tensor& input,
                          const torch::Tensor& weights,
                          int padding_mode,
                          bool active_flag){
    return shiftnd_cpu<3,"shift3d_cpu">(input, weights, padding_mode, active_flag);                    
}


std::vector<torch::Tensor> shift1d_backward_cpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                int padding_mode,
                                                bool active_flag){
    return  shiftnd_backward_cpu<1,"shift1d_backward_cpu">(grad, weights, input, padding_mode, active_flag);                                       
}

std::vector<torch::Tensor> shift2d_backward_cpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                int padding_mode,
                                                bool active_flag){
    return  shiftnd_backward_cpu<2,"shift2d_backward_cpu">(grad, weights, input, padding_mode, active_flag);                                       
}

std::vector<torch::Tensor> shift3d_backward_cpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                int padding_mode,
                                                bool active_flag){
    return  shiftnd_backward_cpu<3,"shift3d_backward_cpu">(grad, weights, input, padding_mode, active_flag);                                       
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("shift1d_cpu", &shift1d_cpu, "1D Shift operation forward (cpu)");
    m.def("shift2d_cpu", &shift2d_cpu, "2D Shift operation forward (cpu)");
    m.def("shift3d_cpu", &shift3d_cpu, "3D Shift operation forward (cpu)");
    m.def("shift1d_backward_cpu", &shift1d_backward_cpu, "1D Shift operator backward (cpu)");
    m.def("shift2d_backward_cpu", &shift2d_backward_cpu, "2D Shift operator backward (cpu)");
    m.def("shift3d_backward_cpu", &shift3d_backward_cpu, "3D Shift operator backward (cpu)");
};


// TORCH_LIBRARY(srrqops, m) {
//   m.def("quantized_adaptive_avg2dpool", &quantized_adaptive_avg_pool2d_cpu);
//   m.def("quantized_adaptive_avg3dpool", &quantized_adaptive_avg_pool3d_cpu);
// }

#endif