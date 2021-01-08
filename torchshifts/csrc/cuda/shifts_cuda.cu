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

    
template <typename scalar_t, typename idx_t,
          bool active = false>
C10_LAUNCH_BOUNDS_1(CUDA_THREADS)
__global__ void weight_init_cuda(const idx_t n_threads,
                                 TensorInfo<scalar_t, idx_t> src,
                                 TensorInfo<idx_t, idx_t> iw,
                                 TensorInfo<scalar_t, idx_t> dw)
{
    scalar_t* src_ptr = src.data;
    idx_t* iw_ptr = iw.data;
    scalar_t* dw_ptr = dw.data;
              
    CUDA_KERNEL_LOOP_TYPE(i, n_threads, idx_t){         
         weight_init_kernel<scalar_t, idx_t, active>(src_ptr, iw_ptr, dw_ptr, i);
    }                
}
    
    
    
template <typename scalar_t, int kSpatialDim = 1, typename idx_t,
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
C10_LAUNCH_BOUNDS_1(CUDA_THREADS)
__global__ void _shifts_cuda(const idx_t sizeNC,
                             TensorInfo<scalar_t, idx_t> input,
                             TensorInfo<idx_t, idx_t> iweights,
                             TensorInfo<scalar_t, idx_t> dweights,
                             TensorInfo<idx_t, idx_t> borders,
                             TensorInfo<scalar_t, idx_t> output){
    const idx_t sizeN = input.sizes[0];
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

//     const idx_t sizeDW = sizeD*sizeW;
    
    idx_t nc = blockIdx.x * blockDim.x + threadIdx.x;
    const idx_t c = nc / sizeN;
    const idx_t n = nc / sizeC;
    idx_t i = blockIdx.y * blockDim.y + threadIdx.y;
    idx_t jk = (kSpatialDim > 1)?(blockIdx.z * blockDim.z + threadIdx.z):0;
    const idx_t j = (kSpatialDim > 1)?(jk / sizeD):0;
    const idx_t k = (kSpatialDim > 2)?(jk / sizeW):0;
    shift_forward_kernel_nchwd<scalar_t, idx_t, kSpatialDim, padding_mode, active>(
        input_ptr, output_ptr, weights_ptr, dweights_ptr,
        n, c, i, j, k, sizeH, sizeW, sizeD,
        input_sN, input_sC, input_sH, input_sW, input_sD,
        output_sN, output_sC, output_sH, output_sW, output_sD,
        weights_sC, weights_sS, dweights_sC, dweights_sS,
        i_left_border, j_left_border, k_left_border,
        i_right_border, j_right_border, k_right_border);
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
    const idx_t sizeN = input.sizes[0];
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
    
    //     const idx_t sizeDW = sizeD*sizeW;
    
    idx_t nc = blockIdx.x * blockDim.x + threadIdx.x;
    const idx_t c = nc / sizeN;
    const idx_t n = nc / sizeC;
    idx_t i = blockIdx.y * blockDim.y + threadIdx.y;
    idx_t jk = (kSpatialDim > 1)?(blockIdx.z * blockDim.z + threadIdx.z):0;
    const idx_t j = (kSpatialDim > 1)?(jk / sizeD):0;
    const idx_t k = (kSpatialDim > 2)?(jk / sizeW):0;         
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

//end of anonymous namespace        
}



template <int nD, BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
torch::Tensor shiftnd_forward_cuda(const torch::Tensor& input,
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
    torch::Tensor iweights = torch::empty(_weights.sizes(), _weights.options().dtype(int32bit_cond?torch::kInt:torch::kLong),
                                          LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor dweights = torch::empty_like(_weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    
    int64_t count = _weights.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(_weights.scalar_type(), 'weights_init_cuda_f', [&] {
        if (int32bit_cond){
            weight_init_cuda<scalar_t,int,active>
                <<<CUDA_BLOCKS(count,LOCAL_CUDA_NUM_THREADS), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                    static_cast<int>(count),
                    getTensorInfo<scalar_t, int>(_weights),
                    getTensorInfo<int, int>(iweights),
                    getTensorInfo<scalar_t, int>(dweights));
        }
        else{
             weight_init_cuda<scalar_t,int64_t,active>
                <<<CUDA_BLOCKS(count,LOCAL_CUDA_NUM_THREADS), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                    count,
                    getTensorInfo<scalar_t, int64_t>(_weights),
                    getTensorInfo<int64_t, int64_t>(iweights),
                    getTensorInfo<scalar_t, int64_t>(dweights));
        }
    });
    
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = (nD<2)?1:input.size(3);
    const int64_t D = (nD<3)?1:input.size(4);
    
    const int threads_x = 64;
    const int threads_y = 4*((nD>1)?1:4);
    const int threads_z = (nD>1)?4:1;
              
   
    const dim3 blocks(CUDA_BLOCKS(N*C, threads_x),
                      CUDA_BLOCKS(H, threads_y),
                      (nD>1)?CUDA_BLOCKS(W*D,threads_z):1);
    const dim3 threads(threads_x,threads_y,threads_z);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), name, [&] {
        if (int32bit_cond){
            _shifts_cuda<scalar_t, nD, int, padding_mode, active>
                   <<<blocks, threads, 0, stream>>>(
                    static_cast<int>(N*C),
                    getTensorInfo<scalar_t, int>(input),
                    getTensorInfo<int, int>(iweights),
                    getTensorInfo<scalar_t, int>(dweights),
                    getTensorInfo<int, int>(_borders),
                    getTensorInfo<scalar_t, int>(output));
        } else {
            _shifts_cuda<scalar_t, nD, int64_t, padding_mode, active>
                <<<blocks, threads, 0, stream>>>(
                    N*C,
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
std::vector<torch::Tensor> shiftnd_backward_cuda(const torch::Tensor& grad,
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

    torch::Tensor _weights = int32bit_cond?weights.to(torch::kInt):weights.to(torch::kLong);
    torch::Tensor _borders = int32bit_cond?borders.to(torch::kInt):borders.to(torch::kLong);
    
    torch::Tensor out_grad = torch::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor weights_grad = torch::empty_like(_weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor iweights = torch::empty_like(_weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor dweights = torch::empty_like(_weights, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    
    int64_t count = _weights.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(_weights.scalar_type(), 'weights_init_cuda_b', [&] {
        if (int32bit_cond){
            weight_init_cuda<scalar_t,int,active>
                <<<CUDA_BLOCKS(count,LOCAL_CUDA_NUM_THREADS), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                    static_cast<int>(count),
                    getTensorInfo<scalar_t, int>(_weights),
                    getTensorInfo<int, int>(iweights),
                    getTensorInfo<scalar_t, int>(dweights));
        }
        else{
             weight_init_cuda<scalar_t,int64_t,active>
                <<<CUDA_BLOCKS(count,LOCAL_CUDA_NUM_THREADS), LOCAL_CUDA_NUM_THREADS, 0, stream>>>(
                    count,
                    getTensorInfo<scalar_t, int64_t>(_weights),
                    getTensorInfo<int64_t, int64_t>(iweights),
                    getTensorInfo<scalar_t, int64_t>(dweights));
        }
    });
        
    //Yes it's not a mistake, iteration happens under input size
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = (nD<2)?1:input.size(3);
    const int64_t D = (nD<3)?1:input.size(4);
     

    const int threads_x = 64;
    const int threads_y = 4*((nD>1)?1:4);
    const int threads_z = (nD>1)?4:1;
              
   
    const dim3 blocks(CUDA_BLOCKS(N*C, threads_x),
                      CUDA_BLOCKS(H, threads_y),
                      (nD>1)?CUDA_BLOCKS(W*D,threads_z):1);
    const dim3 threads(threads_x,threads_y,threads_z);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), name, [&] {
        if (int32bit_cond){
            _shifts_backward_cuda<scalar_t, nD, int, padding_mode, active>
                        <<<blocks, threads, 0, stream>>>(
                            static_cast<int>(N*C),
                            getTensorInfo<scalar_t, int>(grad),
                            getTensorInfo<int, int>(iweights),
                            getTensorInfo<scalar_t, int>(dweights),
                            getTensorInfo<scalar_t, int>(input),
                            getTensorInfo<int, int>(_borders),
                            getTensorInfo<scalar_t, int>(out_grad),
                            getTensorInfo<scalar_t, int>(weights_grad));
        } else {
            _shifts_backward_cuda<scalar_t, nD, int64_t, padding_mode, active>
                         <<<blocks, threads, 0, stream>>>(
                            N*C,
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
    
    return {out_grad, weights_grad};
}


// DISPATCHES!



torch::Tensor shift1d_forward_cuda(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& borders,
                                   const std::vector<int64_t>& new_size,
                                   int64_t padding_mode,
                                   bool active_flag){
    torch::Tensor ret;
    switch (padding_mode){        
        case 0:
            ret = active_flag?shiftnd_forward_cuda<1,BIPadding::Zeros,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<1,BIPadding::Zeros,false>(input, weights, borders, new_size);
            break;
        case 1:
            ret = active_flag?shiftnd_forward_cuda<1,BIPadding::Border,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<1,BIPadding::Border,false>(input, weights, borders, new_size);
            break;
        case 2:
            ret = active_flag?shiftnd_forward_cuda<1,BIPadding::Periodic,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<1,BIPadding::Periodic,false>(input, weights, borders, new_size);
            break;
        case 3:
            ret = active_flag?shiftnd_forward_cuda<1,BIPadding::Reflect,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<1,BIPadding::Reflect,false>(input, weights, borders, new_size);
            break;
        case 4:
            ret = active_flag?shiftnd_forward_cuda<1,BIPadding::Symmetric,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<1,BIPadding::Symmetric,false>(input, weights, borders, new_size);
            break;
    }
    return ret;                  
}

torch::Tensor shift2d_forward_cuda(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& borders,
                                   const std::vector<int64_t>& new_size,
                                   int64_t padding_mode,
                                   bool active_flag){
    torch::Tensor ret;
    switch (padding_mode){        
        case 0:
            ret = active_flag?shiftnd_forward_cuda<2,BIPadding::Zeros,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<2,BIPadding::Zeros,false>(input, weights, borders, new_size);
            break;
        case 1:
            ret = active_flag?shiftnd_forward_cuda<2,BIPadding::Border,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<2,BIPadding::Border,false>(input, weights, borders, new_size);
            break;
        case 2:
            ret = active_flag?shiftnd_forward_cuda<2,BIPadding::Periodic,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<2,BIPadding::Periodic,false>(input, weights, borders, new_size);
            break;
        case 3:
            ret = active_flag?shiftnd_forward_cuda<2,BIPadding::Reflect,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<2,BIPadding::Reflect,false>(input, weights, borders, new_size);
            break;
        case 4:
            ret = active_flag?shiftnd_forward_cuda<2,BIPadding::Symmetric,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<2,BIPadding::Symmetric,false>(input, weights, borders, new_size);
            break;
    }
    return ret;                         
}

torch::Tensor shift3d_forward_cuda(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& borders,
                                   const std::vector<int64_t>& new_size,
                                   int64_t padding_mode,
                                   bool active_flag){
    torch::Tensor ret;
    switch (padding_mode){        
        case 0:
            ret = active_flag?shiftnd_forward_cuda<3,BIPadding::Zeros,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<3,BIPadding::Zeros,false>(input, weights, borders, new_size);
            break;
        case 1:
            ret = active_flag?shiftnd_forward_cuda<3,BIPadding::Border,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<3,BIPadding::Border,false>(input, weights, borders, new_size);
            break;
        case 2:
            ret = active_flag?shiftnd_forward_cuda<3,BIPadding::Periodic,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<3,BIPadding::Periodic,false>(input, weights, borders, new_size);
            break;
        case 3:
            ret = active_flag?shiftnd_forward_cuda<3,BIPadding::Reflect,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<3,BIPadding::Reflect,false>(input, weights, borders, new_size);
            break;
        case 4:
            ret = active_flag?shiftnd_forward_cuda<3,BIPadding::Symmetric,true>(input, weights, borders, new_size):
                              shiftnd_forward_cuda<3,BIPadding::Symmetric,false>(input, weights, borders, new_size);
            break;
    }
    return ret;                        
}


std::vector<torch::Tensor> shift1d_backward_cuda(const torch::Tensor& grad,
                                                 const torch::Tensor& weights,
                                                 const torch::Tensor& input,
                                                 const torch::Tensor& borders,
                                                 int64_t padding_mode,
                                                 bool active_flag){
    std::vector<torch::Tensor> ret;
    switch (padding_mode){        
        case 0:
            ret = active_flag?shiftnd_backward_cuda<1,BIPadding::Zeros,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<1,BIPadding::Zeros,false>(grad, weights, input, borders);
            break;
        case 1:
            ret = active_flag?shiftnd_backward_cuda<1,BIPadding::Border,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<1,BIPadding::Border,false>(grad, weights, input, borders);
            break;
        case 2:
            ret = active_flag?shiftnd_backward_cuda<1,BIPadding::Periodic,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<1,BIPadding::Periodic,false>(grad, weights, input, borders);
            break;
        case 3:
            ret = active_flag?shiftnd_backward_cuda<1,BIPadding::Reflect,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<1,BIPadding::Reflect,false>(grad, weights, input, borders);
            break;
        case 4:
            ret = active_flag?shiftnd_backward_cuda<1,BIPadding::Symmetric,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<1,BIPadding::Symmetric,false>(grad, weights, input, borders);
            break;
    }
    return ret;                                            
}

std::vector<torch::Tensor> shift2d_backward_cuda(const torch::Tensor& grad,
                                                 const torch::Tensor& weights,
                                                 const torch::Tensor& input,
                                                 const torch::Tensor& borders,
                                                 int64_t padding_mode,
                                                 bool active_flag){
    std::vector<torch::Tensor> ret;
    switch (padding_mode){        
        case 0:
            ret = active_flag?shiftnd_backward_cuda<2,BIPadding::Zeros,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<2,BIPadding::Zeros,false>(grad, weights, input, borders);
            break;
        case 1:
            ret = active_flag?shiftnd_backward_cuda<2,BIPadding::Border,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<2,BIPadding::Border,false>(grad, weights, input, borders);
            break;
        case 2:
            ret = active_flag?shiftnd_backward_cuda<2,BIPadding::Periodic,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<2,BIPadding::Periodic,false>(grad, weights, input, borders);
            break;
        case 3:
            ret = active_flag?shiftnd_backward_cuda<2,BIPadding::Reflect,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<2,BIPadding::Reflect,false>(grad, weights, input, borders);
            break;
        case 4:
            ret = active_flag?shiftnd_backward_cuda<2,BIPadding::Symmetric,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<2,BIPadding::Symmetric,false>(grad, weights, input, borders);
            break;
    }
    return ret;    
}

std::vector<torch::Tensor> shift3d_backward_cuda(const torch::Tensor& grad,
                                                 const torch::Tensor& weights,
                                                 const torch::Tensor& input,
                                                 const torch::Tensor& borders,
                                                 int64_t padding_mode,
                                                 bool active_flag){
    std::vector<torch::Tensor> ret;
    switch (padding_mode){        
        case 0:
            ret = active_flag?shiftnd_backward_cuda<3,BIPadding::Zeros,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<3,BIPadding::Zeros,false>(grad, weights, input, borders);
            break;
        case 1:
            ret = active_flag?shiftnd_backward_cuda<3,BIPadding::Border,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<3,BIPadding::Border,false>(grad, weights, input, borders);
            break;
        case 2:
            ret = active_flag?shiftnd_backward_cuda<3,BIPadding::Periodic,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<3,BIPadding::Periodic,false>(grad, weights, input, borders);
            break;
        case 3:
            ret = active_flag?shiftnd_backward_cuda<3,BIPadding::Reflect,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<3,BIPadding::Reflect,false>(grad, weights, input, borders);
            break;
        case 4:
            ret = active_flag?shiftnd_backward_cuda<3,BIPadding::Symmetric,true>(grad, weights, input, borders):
                              shiftnd_backward_cuda<3,BIPadding::Symmetric,false>(grad, weights, input, borders);
            break;
    }
    return ret;                                        
}

#endif