#ifndef _SHIFTS_CUDA
#define _SHIFTS_CUDA

#include "shifts_cuda.h"
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
using namespace at::cuda::detail;

namespace {
#include "../kernels/shifts_kernels.h"

template <typename scalar_t, int kSpatialDim, bool active>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void _shifts_cuda(const int n_threads,
                             TensorInfo<scalar_t, int> input,
                             TensorInfo<scalar_t, int> weights,
                             TensorInfo<scalar_t, int> output,
                             const int weights_size,
                             const BIPadding padding_mode){
    int sizeC = input.sizes[1];
    int sizeH = input.sizes[2];
    int sizeW = kSpatialDim < 2 ? 1 : input.sizes[3];
    int sizeD = kSpatialDim < 3 ? 1 : input.sizes[4];
    int input_sN = input.strides[0];
    int input_sC = input.strides[1];
    int input_sH = input.strides[2];
    int input_sW = kSpatialDim < 2 ? 0 : input.strides[3];
    int input_sD = kSpatialDim < 3 ? 0 : input.strides[4];
    int output_sN = output.strides[0];
    int output_sC = output.strides[1];
    int output_sH = output.strides[2];
    int output_sW = kSpatialDim < 2 ? 0 : output.strides[3];
    int output_sD = kSpatialDim < 3 ? 0 : output.strides[4];
    scalar_t *input_ptr = input.data;
    scalar_t zero_point = static_cast<scalar_t>(0);
    int weights_zero_point = 0;
    scalar_t *output_ptr = output.data;
    int *weights_ptr = NULL;
    init_weights<scalar_t, int, false, active>(weights.data, weights_ptr, weights_size);
    int weights_sC = weights.strides[0];
    int weights_sS = weights.strides[1];
    scalar_t *dweights_ptr = NULL;
    int dweights_sC = 0;
    int dweights_sS = 0;
    STATIC_IF(active){
        init_weight_offsets<scalar_t>(weights.data, dweights_ptr, weights_size);
        dweights_sC = weights_sC;
        dweights_sS = weights_sS;
    } STATIC_ENDIF
    CUDA_KERNEL_LOOP(index, n_threads){
        const int k = index % sizeD;
        const int j = (index / sizeD) % sizeW;
        const int i = (index / (sizeD*sizeW)) % sizeH;
        const int c = (index / (sizeD*sizeW*sizeH)) % sizeC;
        const int n = (index / (sizeD*sizeW*sizeH*sizeC));
        shift_forward_kernel_nchwd<scalar_t, int, false, active>(input_ptr, output_ptr, weights_ptr, dweights_ptr,
                                                                 n, c, i, j, k, sizeH, sizeW, sizeD,
                                                                 input_sN, input_sC, input_sH, input_sW, input_sD,
                                                                 output_sN, output_sC, output_sH, output_sW, output_sD,
                                                                 weights_sC, weights_sS, dweights_sC, dweights_sS,
                                                                 zero_point,  weights_zero_point, padding_mode);
         
    }
        delete weights_ptr;
        STATIC_IF(active){delete dweights_ptr;} STATIC_ENDIF
}

template <typename scalar_t, int kSpatialDim, bool active>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void _shifts_backward_cuda(const int n_threads, 
                                      TensorInfo<scalar_t, int> grad_input, 
                                      TensorInfo<scalar_t, int> weights,
                                      TensorInfo<scalar_t, int> input, 
                                      TensorInfo<scalar_t, int> grad_output,
                                      TensorInfo<scalar_t, int> grad_weights,
                                      const int weights_size,
                                      const BIPadding padding_mode)
{
    int sizeC = grad_input.sizes[1];
    int sizeH = grad_input.sizes[2];
    int sizeW = kSpatialDim < 2 ? 1 : grad_input.sizes[3];
    int sizeD = kSpatialDim < 3 ? 1 : grad_input.sizes[4];
    int grad_input_sN = grad_input.strides[0];
    int grad_input_sC = grad_input.strides[1];
    int grad_input_sH = grad_input.strides[2];
    int grad_input_sW = kSpatialDim < 2 ? 0 : grad_input.strides[3];
    int grad_input_sD = kSpatialDim < 3 ? 0 : grad_input.strides[4];
    int input_sN = input.strides[0];
    int input_sC = input.strides[1];
    int input_sH = input.strides[2];
    int input_sW = kSpatialDim < 2 ? 0 : input.strides[3];
    int input_sD = kSpatialDim < 3 ? 0 : input.strides[4];
    int grad_output_sN = grad_output.strides[0];
    int grad_output_sC = grad_output.strides[1];
    int grad_output_sH = grad_output.strides[2];
    int grad_output_sW = kSpatialDim < 2 ? 0 : grad_output.strides[3];
    int grad_output_sD = kSpatialDim < 3 ? 0 : grad_output.strides[4];
    int weights_sC = weights.strides[0];
    int weights_sS = weights.strides[1];
    int grad_weights_sC = grad_weights.strides[0];
    int grad_weights_sS = grad_weights.strides[1];
    scalar_t *grad_input_ptr = grad_input.data;
    scalar_t *input_ptr = input.data;
    scalar_t *grad_output_ptr = grad_output.data;
    int *weights_ptr = NULL;
    init_weights<scalar_t, int, false, active>(weights.data, weights_ptr, weights_size);
    scalar_t *grad_weights_ptr = grad_weights.data;
    scalar_t *dweights_ptr = NULL;
    int dweights_sC = 0;
    int dweights_sS = 0;
    STATIC_IF(active){
        init_weight_offsets<scalar_t>(weights.data, dweights_ptr, weights_size);
        dweights_sC = weights_sC;
        dweights_sS = weights_sS;
    } STATIC_ENDIF
    CUDA_KERNEL_LOOP(index, n_threads){
        const int k = index % sizeD;
        const int j = (index / sizeD) % sizeW;
        const int i = (index / (sizeD*sizeW)) % sizeH;
        const int c = (index / (sizeD*sizeW*sizeH)) % sizeC;
        const int n = (index / (sizeD*sizeW*sizeH*sizeC));
        shift_backward_kernel_nchwd<scalar_t, int, active>(grad_input_ptr, input_ptr, grad_output_ptr,
                                                           weights_ptr, dweights_ptr, grad_weights_ptr,
                                                           n, c, i, j, k, sizeH, sizeW, sizeD,
                                                           grad_input_sN, grad_input_sC, grad_input_sH, grad_input_sW, grad_input_sD,
                                                           input_sN, input_sC, input_sH, input_sW, input_sD,
                                                           grad_output_sN, grad_output_sC, grad_output_sH, grad_output_sW, grad_output_sD,
                                                           weights_sC, weights_sS, dweights_sC, dweights_sS, grad_weights_sC, grad_weights_sS,
                                                           padding_mode);
    }
    delete weights_ptr;
    STATIC_IF(active){delete dweights_ptr;} STATIC_ENDIF
}

//end of anonymous namespace        
}

template <int nD>
torch::Tensor shiftnd_forward_cuda(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   int64_t padding_mode,
                                   bool active_flag){
    std::string name = "shift"+std::to_string(nD)+"d_forward_cpu";
    torch::Tensor output = torch::zeros_like(input, input.options());
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = (nD<2)?1:input.size(3);
    int64_t D = (nD<3)?1:input.size(4);
    
    int weights_size = static_cast<int>(weights.numel());
    int count = static_cast<int>(N*C*H*W*D);
    
    if (count > 0) {
        if (active_flag){
             AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), name, [&] {
                _shifts_cuda<scalar_t, nD, true>
                <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                getTensorInfo<scalar_t, int>(input),
                getTensorInfo<scalar_t, int>(weights),
                getTensorInfo<scalar_t, int>(output),
                weights_size, static_cast<BIPadding>(padding_mode));
             });
        }
        else {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), name, [&] {
                _shifts_cuda<scalar_t, nD, false>
                <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                getTensorInfo<scalar_t, int>(input),
                getTensorInfo<scalar_t, int>(weights),
                getTensorInfo<scalar_t, int>(output),
                weights_size, static_cast<BIPadding>(padding_mode));
             });
        }
    }
    return output;
}

template <int nD>
std::vector<torch::Tensor> shiftnd_backward_cuda(const torch::Tensor& grad,
                                                 const torch::Tensor& weights,
                                                 const torch::Tensor& input,
                                                 int64_t padding_mode,
                                                 bool active_flag) {
    std::string name = "shift"+std::to_string(nD)+"d_backward_cpu";
    torch::Tensor out_grad = torch::zeros_like(grad, grad.options());
    torch::Tensor weights_grad = torch::zeros_like(weights, weights.options());
  
    int64_t N = grad.size(0);
    int64_t C = grad.size(1);
    int64_t H = grad.size(2);
    int64_t W = (nD<2)?1:input.size(3);
    int64_t D = (nD<3)?1:input.size(4);
    
    int weights_size = static_cast<int>(weights.numel());
    int count = static_cast<int>(N*C*H*W*D);

    if (count > 0) {
        if (active_flag) {
             AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), name, [&] {
                _shifts_backward_cuda<scalar_t, nD, true>
                <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                getTensorInfo<scalar_t, int>(grad),
                getTensorInfo<scalar_t, int>(weights),
                getTensorInfo<scalar_t, int>(input),
                getTensorInfo<scalar_t, int>(out_grad),
                getTensorInfo<scalar_t, int>(weights_grad),
                weights_size, static_cast<BIPadding>(padding_mode));
             });
        }
        else {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), name, [&] {
                _shifts_backward_cuda<scalar_t, nD, false>
                <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                getTensorInfo<scalar_t, int>(grad),
                getTensorInfo<scalar_t, int>(weights),
                getTensorInfo<scalar_t, int>(input),
                getTensorInfo<scalar_t, int>(out_grad),
                getTensorInfo<scalar_t, int>(weights_grad),
                weights_size, static_cast<BIPadding>(padding_mode));
             });
        }
    }
    return {out_grad, weights_grad};
}


torch::Tensor shift1d_forward_cuda(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   int64_t padding_mode,
                                   bool active_flag){
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    return shiftnd_forward_cuda<1>(input, weights, padding_mode, active_flag);                    
}

torch::Tensor shift2d_forward_cuda(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   int64_t padding_mode,
                                   bool active_flag){
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    return shiftnd_forward_cuda<2>(input, weights, padding_mode, active_flag);                     
}

torch::Tensor shift3d_forward_cuda(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   int64_t padding_mode,
                                   bool active_flag){
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    return shiftnd_forward_cuda<3>(input, weights, padding_mode, active_flag);                     
}


std::vector<torch::Tensor> shift1d_backward_cuda(const torch::Tensor& grad,
                                                 const torch::Tensor& weights,
                                                 const torch::Tensor& input,
                                                 int64_t padding_mode,
                                                 bool active_flag){
    CHECK_INPUT(grad); 
    CHECK_INPUT(weights);
    CHECK_INPUT(input);
    return  shiftnd_backward_cuda<1>(grad, weights, input, padding_mode, active_flag);                                        
}

std::vector<torch::Tensor> shift2d_backward_cuda(const torch::Tensor& grad,
                                                 const torch::Tensor& weights,
                                                 const torch::Tensor& input,
                                                 int64_t padding_mode,
                                                 bool active_flag){
    CHECK_INPUT(grad);
    CHECK_INPUT(weights);
    CHECK_INPUT(input);
    return  shiftnd_backward_cuda<2>(grad, weights, input, padding_mode, active_flag);     
}

std::vector<torch::Tensor> shift3d_backward_cuda(const torch::Tensor& grad,
                                                 const torch::Tensor& weights,
                                                 const torch::Tensor& input,
                                                 int64_t padding_mode,
                                                 bool active_flag){
    CHECK_INPUT(grad);
    CHECK_INPUT(weights);
    CHECK_INPUT(input);
    return  shiftnd_backward_cuda<3>(grad, weights, input, padding_mode, active_flag);                                        
}

#endif