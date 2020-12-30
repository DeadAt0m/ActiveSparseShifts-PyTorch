#ifndef _SHIFTS_CUDA
#define _SHIFTS_CUDA


#include "shifts_cuda.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/macros/Macros.h>



using namespace at::cuda::detail;


namespace {
#include "../kernels/shifts_kernels.h"

template <typename scalar_t, int kSpatialDim = 1, typename idx_t,
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
C10_LAUNCH_BOUNDS_1(CUDA_THREADS)
__global__ void _shifts_cuda(const idx_t n_threads,
                             TensorInfo<scalar_t, idx_t> input,
                             TensorInfo<idx_t, idx_t> iweights,
                             TensorInfo<scalar_t, idx_t> dweights,
                             TensorInfo<idx_t, idx_t> borders,
                             TensorInfo<scalar_t, idx_t> output){
    idx_t sizeC = input.sizes[1];
    idx_t sizeH = input.sizes[2];
    idx_t sizeW = kSpatialDim < 2 ? 1 : input.sizes[3];
    idx_t sizeD = kSpatialDim < 3 ? 1 : input.sizes[4];
    idx_t input_sN = input.strides[0];
    idx_t input_sC = input.strides[1];
    idx_t input_sH = input.strides[2];
    idx_t input_sW = kSpatialDim < 2 ? 0 : input.strides[3];
    idx_t input_sD = kSpatialDim < 3 ? 0 : input.strides[4];
    idx_t output_sN = output.strides[0];
    idx_t output_sC = output.strides[1];
    idx_t output_sH = output.strides[2];
    idx_t output_sW = kSpatialDim < 2 ? 0 : output.strides[3];
    idx_t output_sD = kSpatialDim < 3 ? 0 : output.strides[4];
    scalar_t *input_ptr = input.data;
    scalar_t *output_ptr = output.data;
    idx_t *weights_ptr = iweights.data;
    idx_t weights_sC = iweights.strides[0];
    idx_t weights_sS = iweights.strides[1];
    scalar_t *dweights_ptr = dweights.data;
    idx_t dweights_sC = dweights.strides[0];
    idx_t dweights_sS = dweights.strides[1];
    
    idx_t *borders_data = borders.data;
    idx_t i_left_border = MAX(static_cast<idx_t>(0), borders_data[0]);
    idx_t i_right_border = MIN(sizeH, borders_data[1]);
    idx_t j_left_border = kSpatialDim < 2 ? 0 : MAX(static_cast<idx_t>(0), borders_data[2]);
    idx_t j_right_border = kSpatialDim < 2 ? 1 : MIN(sizeW, borders_data[3]);
    idx_t k_left_border =  kSpatialDim < 3 ? 0 : MAX(static_cast<idx_t>(0), borders_data[4]);
    idx_t k_right_border =  kSpatialDim < 3 ? 1 : MIN(sizeD, borders_data[5]);
    
    CUDA_KERNEL_LOOP_TYPE(index, n_threads, idx_t){
        const idx_t k, j, i, c n;
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
        shift_forward_kernel_nchwd<scalar_t, idx_t, kSpatialDim, padding_mode, active>(
                    input_ptr, output_ptr, weights_ptr, dweights_ptr,
                    n, c, i, j, k, sizeH, sizeW, sizeD,
                    input_sN, input_sC, input_sH, input_sW, input_sD,
                    output_sN, output_sC, output_sH, output_sW, output_sD,
                    weights_sC, weights_sS, dweights_sC, dweights_sS,
                    i_left_border, j_left_border, k_left_border,
                    i_right_border, j_right_border, k_right_border);
         
    }
}

template <typename scalar_t, int kSpatialDim=1, typename idx_t,
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
C10_LAUNCH_BOUNDS_1(CUDA_THREADS)
__global__ void _shifts_backward_cuda(const idx_t n_threads, 
                                      TensorInfo<scalar_t, idx_t> grad_input,
                                      TensorInfo<idx_t, idx_t> iweights,
                                      TensorInfo<scalar_t, idx_t> dweights,
                                      TensorInfo<scalar_t, idx_t> input,
                                      TensorInfo<idx_t, idx_t> borders,
                                      TensorInfo<scalar_t, idx_t> grad_output,
                                      TensorInfo<scalar_t, idx_t> grad_weights)
{
    idx_t sizeC = grad_input.sizes[1];
    idx_t sizeH = grad_input.sizes[2];
    idx_t sizeW = kSpatialDim < 2 ? 1 : grad_input.sizes[3];
    idx_t sizeD = kSpatialDim < 3 ? 1 : grad_input.sizes[4];
    idx_t grad_input_sN = grad_input.strides[0];
    idx_t grad_input_sC = grad_input.strides[1];
    idx_t grad_input_sH = grad_input.strides[2];
    idx_t grad_input_sW = kSpatialDim < 2 ? 0 : grad_input.strides[3];
    idx_t grad_input_sD = kSpatialDim < 3 ? 0 : grad_input.strides[4];
    idx_t input_sN = input.strides[0];
    idx_t input_sC = input.strides[1];
    idx_t input_sH = input.strides[2];
    idx_t input_sW = kSpatialDim < 2 ? 0 : input.strides[3];
    idx_t input_sD = kSpatialDim < 3 ? 0 : input.strides[4];
    idx_t grad_output_sN = grad_output.strides[0];
    idx_t grad_output_sC = grad_output.strides[1];
    idx_t grad_output_sH = grad_output.strides[2];
    idx_t grad_output_sW = kSpatialDim < 2 ? 0 : grad_output.strides[3];
    idx_t grad_output_sD = kSpatialDim < 3 ? 0 : grad_output.strides[4];
    idx_t grad_weights_sC = grad_weights.strides[0];
    idx_t grad_weights_sS = grad_weights.strides[1];
    scalar_t *grad_input_ptr = grad_input.data;
    scalar_t *input_ptr = input.data;
    scalar_t *grad_output_ptr = grad_output.data;
    scalar_t *grad_weights_ptr = grad_weights.data;
    idx_t *weights_ptr = iweights.data;
    idx_t weights_sC = iweights.strides[0];
    idx_t weights_sS = iweights.strides[1];
    scalar_t *dweights_ptr = dweights.data;
    idx_t dweights_sC = dweights.strides[0];
    idx_t dweights_sS = dweights.strides[1];
    

    idx_t *borders_data = borders.data;
    idx_t i_left_border = MAX(static_cast<idx_t>(0), borders_data[0]);
    idx_t i_right_border = MIN(input.sizes[2], borders_data[1]);
    idx_t j_left_border = kSpatialDim < 2 ? 0 : MAX(static_cast<idx_t>(0), borders_data[2]);
    idx_t j_right_border = kSpatialDim < 2 ? 1 : MIN(input.sizes[3], borders_data[3]);
    idx_t k_left_border = kSpatialDim < 3 ? 0 : MAX(static_cast<idx_t>(0), borders_data[4]);
    idx_t k_right_border = kSpatialDim < 3 ? 1 : MIN(input.sizes[4], borders_data[5]);
    

    CUDA_KERNEL_LOOP_TYPE(index, n_threads, idx_t){
        const idx_t k, j, i, c n;
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
        shift_backward_kernel_nchwd<scalar_t, idx_t, kSpatialDim, padding_mode, active>(
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
}

//end of anonymous namespace        
}

template <int nD>
torch::Tensor shiftnd_forward_cuda(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& borders,
                                   const std::vector<int64_t>& new_size,
                                   int64_t padding_mode,
                                   bool active_flag){
    std::string name = "shift"+std::to_string(nD)+"d_forward_cpu";
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");                              
    torch::TensorArg input_t{input, "input", 1}, weights_t{weights, "weights", 2};                                 
    torch::CheckedFrom c = name.c_str();
    
    torch::checkAllSameGPU(c, {input_t, weights_t});
    torch::checkAllSameType(c, {input_t, weights_t});
    at::cuda::CUDAGuard device_guard(input.device());
    
    
    torch::Tensor output = torch::zeros(new_size, input.options());
    
    torch::Tensor iweights = (active_flag?torch::floor(weights):torch::round(weights));
    torch::Tensor dweights = torch::empty_like(weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    if (active_flag){
        dweights = (weights - torch::floor(weights));
    }
    
    bool int32bit_cond = canUse32BitIndexMath(input) && canUse32BitIndexMath(iweights) &&
                         canUse32BitIndexMath(dweights) && canUse32BitIndexMath(output);
                         
    iweights = int32bit_cond?iweights.to(torch::kInt):iweights.to(torch::kLong);
    torch::Tensor _borders = int32bit_cond?borders.to(torch::kInt):borders.to(torch::kLong);
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = (nD<2)?1:input.size(3);
    int64_t D = (nD<3)?1:input.size(4);
    
  
    int64_t count = N*C*H*W*D;
    
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();


    switch (padding_mode){        
        case 0:
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), name, [&] {
                if (int32bit_cond){
                    active_flag?_shifts_cuda<scalar_t, nD, int, BIPadding::Zeros, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(output)):
                                _shifts_cuda<scalar_t, nD, int,  BIPadding::Zeros, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(output));
                }
                else{
                    active_flag?_shifts_cuda<scalar_t, nD, int64_t, BIPadding::Zeros, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(output)):
                                _shifts_cuda<scalar_t, nD, int64_t, BIPadding::Zeros, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(output));
                }
            });
            break;
        case 1:
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), name, [&] {
                if (int32bit_cond){
                    active_flag?_shifts_cuda<scalar_t, nD, int, BIPadding::Border, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(output)):
                                _shifts_cuda<scalar_t, nD, int,  BIPadding::Border, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(output));
                }
                else{
                    active_flag?_shifts_cuda<scalar_t, nD, int64_t, BIPadding::Border, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(output)):
                                _shifts_cuda<scalar_t, nD, int64_t, BIPadding::Border, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(output));
                }
            });
            break;
        case 2:
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), name, [&] {
                if (int32bit_cond){
                    active_flag?_shifts_cuda<scalar_t, nD, int, BIPadding::Periodic, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(output)):
                                _shifts_cuda<scalar_t, nD, int,  BIPadding::Periodic, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(output));
                }
                else{
                    active_flag?_shifts_cuda<scalar_t, nD, int64_t, BIPadding::Periodic, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(output)):
                                _shifts_cuda<scalar_t, nD, int64_t, BIPadding::Periodic, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(output));
                }
            });
            break;
        case 3:
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), name, [&] {
                if (int32bit_cond){
                    active_flag?_shifts_cuda<scalar_t, nD, int, BIPadding::Reflect, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(output)):
                                _shifts_cuda<scalar_t, nD, int,  BIPadding::Reflect, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(output));
                }
                else{
                    active_flag?_shifts_cuda<scalar_t, nD, int64_t, BIPadding::Reflect, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(output)):
                                _shifts_cuda<scalar_t, nD, int64_t, BIPadding::Reflect, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(output));
                }
            });
            break;
        case 4:
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), name, [&] {
                if (int32bit_cond){
                    active_flag?_shifts_cuda<scalar_t, nD, int, BIPadding::Symmetric, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(output)):
                                _shifts_cuda<scalar_t, nD, int,  BIPadding::Symmetric, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(output));
                }
                else{
                    active_flag?_shifts_cuda<scalar_t, nD, int64_t, BIPadding::Symmetric, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(output)):
                                _shifts_cuda<scalar_t, nD, int64_t, BIPadding::Symmetric, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(output));
                }
            });
            break;
    }

    AT_CUDA_CHECK(cudaGetLastError());
 
    return output;
}

template <int nD>
std::vector<torch::Tensor> shiftnd_backward_cuda(const torch::Tensor& grad,
                                                 const torch::Tensor& weights,
                                                 const torch::Tensor& input,
                                                 const torch::Tensor& borders,
                                                 int64_t padding_mode,
                                                 bool active_flag) {
    std::string name = "shift"+std::to_string(nD)+"d_backward_cpu";
    at::globalContext().alertNotDeterministic(name.c_str());
    
    TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");                               
    torch::TensorArg grad_t{grad, "grad", 1}, weights_t{weights, "weights", 2}, input_t{input, "input", 3};
    torch::CheckedFrom c = name.c_str();
    
    torch::checkAllSameGPU(c, {grad_t, input_t, weights_t});
    torch::checkAllSameType(c, {grad_t, input_t, weights_t});
    at::cuda::CUDAGuard device_guard(grad.device());
    

    torch::Tensor out_grad = torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor weights_grad = torch::zeros_like(weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    
    torch::Tensor iweights = (active_flag?torch::floor(weights):torch::round(weights));
    torch::Tensor dweights = weights - torch::floor(weights);

    
    bool int32bit_cond = canUse32BitIndexMath(grad) && canUse32BitIndexMath(iweights) &&
                         canUse32BitIndexMath(dweights) && canUse32BitIndexMath(input) && 
                         canUse32BitIndexMath(out_grad) && canUse32BitIndexMath(weights_grad);
    
    iweights = int32bit_cond?iweights.to(torch::kInt):iweights.to(torch::kLong);
    torch::Tensor _borders = int32bit_cond?borders.to(torch::kInt):borders.to(torch::kLong);
  
    int64_t N = grad.size(0);
    int64_t C = grad.size(1);
    int64_t H = grad.size(2);
    int64_t W = (nD<2)?1:grad.size(3);
    int64_t D = (nD<3)?1:grad.size(4);
    

    int64_t count = N*C*H*W*D;
    

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (padding_mode){        
        case 0:
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), name, [&] {
                if (int32bit_cond){
                    active_flag?_shifts_backward_cuda<scalar_t, nD, int, BIPadding::Zeros, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(grad),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(out_grad),
                                    getTensorInfo<scalar_t, int>(weights_grad)):
                                _shifts_backward_cuda<scalar_t, nD, int,  BIPadding::Zeros, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(grad),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(out_grad),
                                    getTensorInfo<scalar_t, int>(weights_grad));
                }
                else{
                    active_flag?_shifts_backward_cuda<scalar_t, nD, int64_t, BIPadding::Zeros, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(grad),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(out_grad),
                                    getTensorInfo<scalar_t, int64_t>(weights_grad)):
                                _shifts_backward_cuda<scalar_t, nD, int64_t, BIPadding::Zeros, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(grad),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(out_grad),
                                    getTensorInfo<scalar_t, int64_t>(weights_grad));
                }
            });
            break;
        case 1:
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), name, [&] {
                if (int32bit_cond){
                    active_flag?_shifts_backward_cuda<scalar_t, nD, int, BIPadding::Border, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(grad),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(out_grad),
                                    getTensorInfo<scalar_t, int>(weights_grad)):
                                _shifts_backward_cuda<scalar_t, nD, int,  BIPadding::Border, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(grad),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(out_grad),
                                    getTensorInfo<scalar_t, int>(weights_grad));
                }
                else{
                    active_flag?_shifts_backward_cuda<scalar_t, nD, int64_t, BIPadding::Border, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(grad),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(out_grad),
                                    getTensorInfo<scalar_t, int64_t>(weights_grad)):
                                _shifts_backward_cuda<scalar_t, nD, int64_t, BIPadding::Border, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(grad),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(out_grad),
                                    getTensorInfo<scalar_t, int64_t>(weights_grad));
                }
            });
            break;
        case 2:
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), name, [&] {
                if (int32bit_cond){
                    active_flag?_shifts_backward_cuda<scalar_t, nD, int, BIPadding::Periodic, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(grad),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(out_grad),
                                    getTensorInfo<scalar_t, int>(weights_grad)):
                                _shifts_backward_cuda<scalar_t, nD, int,  BIPadding::Periodic, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(output));
                }
                else{
                    active_flag?_shifts_backward_cuda<scalar_t, nD, int64_t, BIPadding::Periodic, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(grad),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(out_grad),
                                    getTensorInfo<scalar_t, int64_t>(weights_grad)):
                                _shifts_backward_cuda<scalar_t, nD, int64_t, BIPadding::Periodic, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(grad),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(out_grad),
                                    getTensorInfo<scalar_t, int64_t>(weights_grad));
                }
            });
            break;
        case 3:
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), name, [&] {
                if (int32bit_cond){
                    active_flag?_shifts_backward_cuda<scalar_t, nD, int, BIPadding::Reflect, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(grad),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(out_grad),
                                    getTensorInfo<scalar_t, int>(weights_grad)):
                                _shifts_backward_cuda<scalar_t, nD, int,  BIPadding::Reflect, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(grad),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(out_grad),
                                    getTensorInfo<scalar_t, int>(weights_grad));
                }
                else{
                    active_flag?_shifts_backward_cuda<scalar_t, nD, int64_t, BIPadding::Reflect, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(grad),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(out_grad),
                                    getTensorInfo<scalar_t, int64_t>(weights_grad)):
                                _shifts_backward_cuda<scalar_t, nD, int64_t, BIPadding::Reflect, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(grad),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(out_grad),
                                    getTensorInfo<scalar_t, int64_t>(weights_grad));
                }
            });
            break;
        case 4:
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), name, [&] {
                if (int32bit_cond){
                    active_flag?_shifts_backward_cuda<scalar_t, nD, int, BIPadding::Symmetric, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    sstatic_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(grad),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(out_grad),
                                    getTensorInfo<scalar_t, int>(weights_grad)):
                                _shifts_backward_cuda<scalar_t, nD, int,  BIPadding::Symmetric, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    static_cast<int>(count),
                                    getTensorInfo<scalar_t, int>(grad),
                                    getTensorInfo<int, int>(iweights),
                                    getTensorInfo<scalar_t, int>(dweights),
                                    getTensorInfo<scalar_t, int>(input),
                                    getTensorInfo<int, int>(_borders),
                                    getTensorInfo<scalar_t, int>(out_grad),
                                    getTensorInfo<scalar_t, int>(weights_grad));
                }
                else{
                    active_flag?_shifts_backward_cuda<scalar_t, nD, int64_t, BIPadding::Symmetric, true>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(grad),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(out_grad),
                                    getTensorInfo<scalar_t, int64_t>(weights_grad)):
                                _shifts_backward_cuda<scalar_t, nD, int64_t, BIPadding::Symmetric, false>
                                <<<GET_CUDA_BLOCKS(count), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                                    count,
                                    getTensorInfo<scalar_t, int64_t>(grad),
                                    getTensorInfo<int64_t, int64_t>(iweights),
                                    getTensorInfo<scalar_t, int64_t>(dweights),
                                    getTensorInfo<scalar_t, int64_t>(input),
                                    getTensorInfo<int64_t, int64_t>(_borders),
                                    getTensorInfo<scalar_t, int64_t>(out_grad),
                                    getTensorInfo<scalar_t, int64_t>(weights_grad));
                }
            });
            break;
    }

    AT_CUDA_CHECK(cudaGetLastError());
    
    return {out_grad, weights_grad};
}


torch::Tensor shift1d_forward_cuda(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& borders,
                                   const std::vector<int64_t>& new_size,
                                   int64_t padding_mode,
                                   bool active_flag){
    return shiftnd_forward_cuda<1>(input, weights, borders, new_size, padding_mode, active_flag);                    
}

torch::Tensor shift2d_forward_cuda(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& borders,
                                   const std::vector<int64_t>& new_size,
                                   int64_t padding_mode,
                                   bool active_flag){
    return shiftnd_forward_cuda<2>(input, weights, borders, new_size, padding_mode, active_flag);                     
}

torch::Tensor shift3d_forward_cuda(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& borders,
                                   const std::vector<int64_t>& new_size,
                                   int64_t padding_mode,
                                   bool active_flag){
    return shiftnd_forward_cuda<3>(input, weights, borders, new_size, padding_mode, active_flag);                     
}


std::vector<torch::Tensor> shift1d_backward_cuda(const torch::Tensor& grad,
                                                 const torch::Tensor& weights,
                                                 const torch::Tensor& input,
                                                 const torch::Tensor& borders,
                                                 int64_t padding_mode,
                                                 bool active_flag){
    return  shiftnd_backward_cuda<1>(grad, weights, input, borders, padding_mode, active_flag);                                        
}

std::vector<torch::Tensor> shift2d_backward_cuda(const torch::Tensor& grad,
                                                 const torch::Tensor& weights,
                                                 const torch::Tensor& input,
                                                 const torch::Tensor& borders,
                                                 int64_t padding_mode,
                                                 bool active_flag){
    return  shiftnd_backward_cuda<2>(grad, weights, input, borders, padding_mode, active_flag);     
}

std::vector<torch::Tensor> shift3d_backward_cuda(const torch::Tensor& grad,
                                                 const torch::Tensor& weights,
                                                 const torch::Tensor& input,
                                                 const torch::Tensor& borders,
                                                 int64_t padding_mode,
                                                 bool active_flag){
    return  shiftnd_backward_cuda<3>(grad, weights, input, borders, padding_mode, active_flag);                                        
}

#endif