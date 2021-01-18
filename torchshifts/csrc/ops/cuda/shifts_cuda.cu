#ifndef _SHIFTS_CUDA
#define _SHIFTS_CUDA


#include <torch/extension.h>
#include "../global_scope.h"
#include <thrust/tuple.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/macros/Macros.h>
#include <torch/library.h>

using namespace at::cuda::detail;

namespace shifts {
namespace ops {

namespace {
#include "../kernels/shifts_kernels.h"
    
    
template <typename scalar_t, int kSpatialDim = 1, typename idx_t,
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
C10_LAUNCH_BOUNDS_1(CUDA_THREADS)
__global__ void shiftnd_forward_kernel(const idx_t n_threads,
                                       TensorInfo<scalar_t, idx_t> input,
                                       TensorInfo<idx_t, idx_t> iweights,
                                       TensorInfo<scalar_t, idx_t> dweights,
                                       TensorInfo<idx_t, idx_t> borders,
                                       TensorInfo<scalar_t, idx_t> output){
    const idx_t sizeC = input.sizes[1];
    const idx_t sizeH = input.sizes[2];
    const idx_t sizeW = kSpatialDim < 2 ? 1 : input.sizes[3];
    const idx_t sizeD = kSpatialDim < 3 ? 1 : input.sizes[4];
    const idx_t input_sN = input.strides[0];
    const idx_t input_sC = input.strides[1];
    const idx_t input_sH = input.strides[2];
    const idx_t input_sW = kSpatialDim < 2 ? 0 : input.strides[3];
    const idx_t input_sD = kSpatialDim < 3 ? 0 : input.strides[4];
    const idx_t output_sN = output.strides[0];
    const idx_t output_sC = output.strides[1];
    const idx_t output_sH = output.strides[2];
    const idx_t output_sW = kSpatialDim < 2 ? 0 : output.strides[3];
    const idx_t output_sD = kSpatialDim < 3 ? 0 : output.strides[4];
    scalar_t* input_ptr = input.data;
    scalar_t* output_ptr = output.data;
    idx_t* weights_ptr = iweights.data;
    const idx_t weights_sC = iweights.strides[0];
    const idx_t weights_sS = iweights.strides[1];
    scalar_t* dweights_ptr = dweights.data;
    const idx_t dweights_sC = dweights.strides[0];
    const idx_t dweights_sS = dweights.strides[1];
    
    idx_t* borders_data = borders.data;
    const idx_t i_left_border = borders_data[0];
    const idx_t i_right_border = borders_data[1];
    const idx_t j_left_border = kSpatialDim < 2 ? 0 : borders_data[2];
    const idx_t j_right_border = kSpatialDim < 2 ? 1 : borders_data[3];
    const idx_t k_left_border =  kSpatialDim < 3 ? 0 : borders_data[4];
    const idx_t k_right_border =  kSpatialDim < 3 ? 1 : borders_data[5];

    const idx_t sizeDW = (kSpatialDim > 1)?(sizeD*sizeW):1;
    const idx_t sizeDWH = sizeDW*sizeH;
    const idx_t sizeDWHC = sizeDWH*sizeC; 
    
    CUDA_KERNEL_LOOP(index, n_threads){
        const int k = (kSpatialDim > 2)? (index % sizeD):0;
        const int j = (kSpatialDim > 1)? ((index / sizeD) % sizeW): 0;
        const int i = (index / sizeDW) % sizeH;
        const int c = (index / sizeDWH) % sizeC;
        const int n = (index / sizeDWHC);
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
__global__ void shiftnd_backward_kernel(const idx_t n_threads, 
                                        TensorInfo<scalar_t, idx_t> grad_input,
                                        TensorInfo<idx_t, idx_t> iweights,
                                        TensorInfo<scalar_t, idx_t> dweights,
                                        TensorInfo<scalar_t, idx_t> input,
                                        TensorInfo<idx_t, idx_t> borders,
                                        TensorInfo<scalar_t, idx_t> grad_output,
                                        TensorInfo<scalar_t, idx_t> grad_weights)
{
    const idx_t sizeC = input.sizes[1];
    const idx_t sizeH = input.sizes[2];
    const idx_t sizeW = kSpatialDim < 2 ? 1 : input.sizes[3];
    const idx_t sizeD = kSpatialDim < 3 ? 1 : input.sizes[4];
    const idx_t grad_input_sN = grad_input.strides[0];
    const idx_t grad_input_sC = grad_input.strides[1];
    const idx_t grad_input_sH = grad_input.strides[2];
    const idx_t grad_input_sW = kSpatialDim < 2 ? 0 : grad_input.strides[3];
    const idx_t grad_input_sD = kSpatialDim < 3 ? 0 : grad_input.strides[4];
    const idx_t input_sN = input.strides[0];
    const idx_t input_sC = input.strides[1];
    const idx_t input_sH = input.strides[2];
    const idx_t input_sW = kSpatialDim < 2 ? 0 : input.strides[3];
    const idx_t input_sD = kSpatialDim < 3 ? 0 : input.strides[4];
    const idx_t grad_output_sN = grad_output.strides[0];
    const idx_t grad_output_sC = grad_output.strides[1];
    const idx_t grad_output_sH = grad_output.strides[2];
    const idx_t grad_output_sW = kSpatialDim < 2 ? 0 : grad_output.strides[3];
    const idx_t grad_output_sD = kSpatialDim < 3 ? 0 : grad_output.strides[4];
    const idx_t grad_weights_sC = grad_weights.strides[0];
    const idx_t grad_weights_sS = grad_weights.strides[1];
    scalar_t* grad_input_ptr = grad_input.data;
    scalar_t* input_ptr = input.data;
    scalar_t* grad_output_ptr = grad_output.data;
    scalar_t* grad_weights_ptr = grad_weights.data;
    idx_t* weights_ptr = iweights.data;
    const idx_t weights_sC = iweights.strides[0];
    const idx_t weights_sS = iweights.strides[1];
    scalar_t* dweights_ptr = dweights.data;
    const idx_t dweights_sC = dweights.strides[0];
    const idx_t dweights_sS = dweights.strides[1];
    

    idx_t* borders_data = borders.data;
    const idx_t i_left_border = borders_data[0];
    const idx_t i_right_border = borders_data[1];
    const idx_t j_left_border = kSpatialDim < 2 ? 0 : borders_data[2];
    const idx_t j_right_border = kSpatialDim < 2 ? 1 : borders_data[3];
    const idx_t k_left_border =  kSpatialDim < 3 ? 0 : borders_data[4];
    const idx_t k_right_border =  kSpatialDim < 3 ? 1 : borders_data[5];    
    
    const idx_t sizeDW = (kSpatialDim > 1)?(sizeD*sizeW):1;
    const idx_t sizeDWH = sizeDW*sizeH;
    const idx_t sizeDWHC = sizeDWH*sizeC; 
    
    CUDA_KERNEL_LOOP(index, n_threads){
        const int k = (kSpatialDim > 2)? (index % sizeD):0;
        const int j = (kSpatialDim > 1)? ((index / sizeD) % sizeW): 0;
        const int i = (index / sizeDW) % sizeH;
        const int c = (index / sizeDWH) % sizeC;
        const int n = (index / sizeDWHC);
        shift_backward_kernel_nchwd<scalar_t, idx_t, kSpatialDim, padding_mode, active>(
                    grad_input_ptr, input_ptr, grad_output_ptr,
                    weights_ptr, dweights_ptr, grad_weights_ptr,
                    n, c, i, j, k, sizeC, sizeH, sizeW, sizeD,
                    grad_input_sN, grad_input_sC, grad_input_sH, grad_input_sW, grad_input_sD,
                    input_sN, input_sC, input_sH, input_sW, input_sD,
                    grad_output_sN, grad_output_sC, grad_output_sH, grad_output_sW, grad_output_sD,
                    weights_sC, weights_sS, dweights_sC, dweights_sS, grad_weights_sC, grad_weights_sS,
                    i_left_border, j_left_border, k_left_border,
                    i_right_border, j_right_border, k_right_border);
    }
}

        
template <typename scalar_t, bool active = false>
inline void weights_init_forward(const torch::Tensor& weights,
                                 torch::Tensor iweights,
                                 torch::Tensor dweights){
              
    torch::TensorIterator iter = torch::TensorIteratorConfig().add_output(iweights)
                                                              .add_output(dweights)
                                                              .add_input(weights).build();          
    
    at::native::gpu_kernel_multiple_outputs(iter, [=] GPU_LAMBDA (scalar_t src) -> thrust::tuple<scalar_t, scalar_t> {
                scalar_t iw = active?((src>0)?static_cast<scalar_t>(FLOOR(src)):
                                              static_cast<scalar_t>(CEIL(src))):
                                       static_cast<scalar_t>(ROUND(src));
                scalar_t dw = active?(src - iw):static_cast<scalar_t>(0);
        
                return {iw, dw};
    });          
} 
    
template <typename scalar_t, bool active = false>
inline void weights_init_backward(const torch::Tensor& weights,
                                  torch::Tensor iweights,
                                  torch::Tensor dweights){
    torch::TensorIterator iter = torch::TensorIteratorConfig().add_output(iweights)
                                                              .add_output(dweights)
                                                              .add_input(weights).build();          
    
    at::native::gpu_kernel_multiple_outputs(iter, [=] GPU_LAMBDA (scalar_t src) -> thrust::tuple<scalar_t, scalar_t> {
                scalar_t dw = (src>0)?(src - static_cast<scalar_t>FLOOR(src)):(src - static_cast<scalar_t>CEIL(src));
                scalar_t iw = active?(src-dw):static_cast<scalar_t>(ROUND(src));
                return {iw, dw};
    });          
} 


template <int nD, BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
torch::Tensor shiftnd_forward(const torch::Tensor& input,
                              const torch::Tensor& weights,
                              const torch::Tensor& borders,
                              const std::vector<int64_t>& new_size){
    std::string name = "shift"+std::to_string(nD)+"d_forward_cpu";
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");                              
    torch::TensorArg input_t{input, "input", 1}, weights_t{weights, "weights", 2};                                 
    torch::CheckedFrom c = name.c_str();
    
    torch::checkAllSameGPU(c, {input_t, weights_t});
    torch::checkAllSameType(c, {input_t, weights_t});
    at::cuda::CUDAGuard device_guard(input.device());
    
    bool int32bit_cond = canUse32BitIndexMath(input) && canUse32BitIndexMath(weights);    

    torch::Tensor _weights = weights.contiguous(LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor _borders = int32bit_cond?borders.to(torch::kInt):borders.to(torch::kLong);
    
    torch::Tensor output = torch::empty(new_size, input.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor _iweights = torch::empty_like(_weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor dweights = torch::empty_like(_weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
              
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(_weights.scalar_type(), 'weights_init_cuda_forward', [&] {
        weights_init_forward<scalar_t, active>(_weights, _iweights, dweights);
    });
    torch::Tensor iweights = int32bit_cond?_iweights.to(torch::kInt):_iweights.to(torch::kLong);
                  
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = (nD<2)?1:input.size(3);
    const int64_t D = (nD<3)?1:input.size(4);
              
    const int64_t count = N*C*H*W*D;
    const dim3 blocks(CUDA_BLOCKS(count, LOCAL_CUDA_NUM_THREADS), 1, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), name, [&] {
        if (int32bit_cond){
            shiftnd_forward_kernel<scalar_t, nD, int, padding_mode, active>
                   <<<blocks, LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                    static_cast<int>(count),
                    getTensorInfo<scalar_t, int>(input),
                    getTensorInfo<int, int>(iweights),
                    getTensorInfo<scalar_t, int>(dweights),
                    getTensorInfo<int, int>(_borders),
                    getTensorInfo<scalar_t, int>(output));
        } else {
            shiftnd_forward_kernel<scalar_t, nD, int64_t, padding_mode, active>
                <<<blocks, LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                    count,
                    getTensorInfo<scalar_t, int64_t>(input),
                    getTensorInfo<int64_t, int64_t>(iweights),
                    getTensorInfo<scalar_t, int64_t>(dweights),
                    getTensorInfo<int64_t, int64_t>(_borders),
                    getTensorInfo<scalar_t, int64_t>(output));
        }
    });
              
    AT_CUDA_CHECK(cudaGetLastError());
 
    return output;
}

template <int nD, BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
std::tuple<torch::Tensor, torch::Tensor> shiftnd_backward(const torch::Tensor& grad,
                                                          const torch::Tensor& weights,
                                                          const torch::Tensor& input,
                                                          const torch::Tensor& borders) {
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
    
    
    bool int32bit_cond = canUse32BitIndexMath(grad) && canUse32BitIndexMath(weights) &&
                         canUse32BitIndexMath(input);

    torch::Tensor _weights = weights.contiguous(LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor _borders = int32bit_cond?borders.to(torch::kInt):borders.to(torch::kLong);
    
    torch::Tensor out_grad = torch::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor weights_grad = torch::zeros_like(_weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor _iweights = torch::empty_like(_weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor dweights = torch::empty_like(_weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
              
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(_weights.scalar_type(), 'weights_init_cuda_backward', [&] {
        weights_init_backward<scalar_t, active>(_weights, _iweights, dweights);
    });
    torch::Tensor iweights = int32bit_cond?_iweights.to(torch::kInt):_iweights.to(torch::kLong);
              
        
    //Yes it's not a mistake, iteration happens under input size
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = (nD<2)?1:input.size(3);
    const int64_t D = (nD<3)?1:input.size(4);
     
    const int64_t count = N*C*H*W*D;
    const dim3 blocks(CUDA_BLOCKS(count, LOCAL_CUDA_NUM_THREADS), 1, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
              
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), name, [&] {
        if (int32bit_cond){
            shiftnd_backward_kernel<scalar_t, nD, int, padding_mode, active>
                        <<<blocks, LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                            static_cast<int>(count),
                            getTensorInfo<scalar_t, int>(grad),
                            getTensorInfo<int, int>(iweights),
                            getTensorInfo<scalar_t, int>(dweights),
                            getTensorInfo<scalar_t, int>(input),
                            getTensorInfo<int, int>(_borders),
                            getTensorInfo<scalar_t, int>(out_grad),
                            getTensorInfo<scalar_t, int>(weights_grad));
        } else {
            shiftnd_backward_kernel<scalar_t, nD, int64_t, padding_mode, active>
                         <<<blocks, LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
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

    AT_CUDA_CHECK(cudaGetLastError());
    
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


TORCH_LIBRARY_IMPL(torchshifts, CUDA, m) {
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