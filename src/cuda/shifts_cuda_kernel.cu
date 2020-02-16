#include "shifts_cuda_kernel.h"

#include <THC/THCAtomics.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>

using namespace at::cuda::detail;

namespace {
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void shift2d_gpu_kernel(const int n_threads,
                                   TensorInfo<scalar_t, int> input,
                                   TensorInfo<scalar_t, int> weights,
                                   TensorInfo<scalar_t, int> output){
    int C = input.sizes[1];
    int H = input.sizes[2];
    int W = input.sizes[3];
    
    int input_sN = input.strides[0];
    int input_sC = input.strides[1];
    int input_sH = input.strides[2];
    int input_sW = input.strides[3];
    
    int output_sN = output.strides[0];
    int output_sC = output.strides[1];
    int output_sH = output.strides[2];
    int output_sW = output.strides[3];
    
    int w_sC = weights.strides[0];
    int w_sS = weights.strides[1];
    
    CUDA_KERNEL_LOOP(index, n_threads){
        const int w = index % W;
        const int h = (index / W) % H;
        const int c = (index / (H*W)) % C;
        const int n = index / (H*W*C);
        
        const int weights_offset = c * w_sC;
        
        scalar_t *inp_ptr_NC = input.data + n * input_sN + c * input_sC;
        scalar_t shift_H = weights.data[weights_offset];
        scalar_t shift_W = weights.data[weights_offset+w_sS];
        
        int rounded_shift_H = static_cast<int>(::round(shift_H));
        int rounded_shift_W = static_cast<int>(::round(shift_W));
        
        int H_shifted = ::max(H - ::abs(rounded_shift_H), static_cast<int>(0));
        int W_shifted = ::max(W - ::abs(rounded_shift_W), static_cast<int>(0));
        
        if  ((h < H_shifted)&&(h >= 0)){
            int inp_h = h;
            int out_h = h;               
            if (rounded_shift_H > 0){
                out_h += ::abs(rounded_shift_H);
            }
            if (rounded_shift_H < 0){
                inp_h += ::abs(rounded_shift_H);
            }
            if ((w < W_shifted)&&(w >= 0)){
                int inp_w = w;
                int out_w = w;
                if (rounded_shift_W > 0){
                    out_w += ::abs(rounded_shift_W);
                }
                if (rounded_shift_W < 0){
                    inp_w += ::abs(rounded_shift_W);
                }
                scalar_t *output_ptr_NCHW = output.data + n * output_sN + c * output_sC + out_h * output_sH + out_w * output_sW;
                *output_ptr_NCHW = inp_ptr_NC[inp_h * input_sH + inp_w * input_sW]; 
            } 
        }
        
    }
}   

__device__ __forceinline__ int detect_borders(int index, int threshold){                
    return (index > (threshold-1)) ? -index : index;
}

__device__ __forceinline__ int infer_ind_border(int index, int threshold){
    int tmp =  ::max((1-threshold), index);
    return (tmp < 0) ? 0 : ::abs(tmp);
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void shift2d_backward_gpu_kernel(const int n_threads,
                                            TensorInfo<scalar_t, int> grad,
                                            TensorInfo<scalar_t, int> weights,
                                            TensorInfo<scalar_t, int> input,
                                            TensorInfo<scalar_t, int> out_grad,
                                            TensorInfo<scalar_t, int> weights_grad,
                                            const BIPadding padding_mode)
{
    int C = grad.sizes[1];
    int H = grad.sizes[2];
    int W = grad.sizes[3];
    
    int grad_sN = grad.strides[0];
    int grad_sC = grad.strides[1];
    int grad_sH = grad.strides[2];
    int grad_sW = grad.strides[3];
    
    int out_grad_sN = out_grad.strides[0];
    int out_grad_sC = out_grad.strides[1];
    int out_grad_sH = out_grad.strides[2];
    int out_grad_sW = out_grad.strides[3];
    
    int input_sN = input.strides[0];
    int input_sC = input.strides[1];
    int input_sH = input.strides[2];
    int input_sW = input.strides[3];
    
    int w_sC = weights.strides[0];
    int w_sS = weights.strides[1];
    int grad_w_sC = weights_grad.strides[0];
    int grad_w_sS = weights_grad.strides[1];
    
    CUDA_KERNEL_LOOP(index, n_threads){
        const int w = index % W;
        const int h = (index / W) % H;
        const int c = (index / (H*W)) % C;
        const int n = index / (H*W*C);
        
        scalar_t *grad_ptr_NC = grad.data + n * grad_sN + c * grad_sC;
        scalar_t *input_ptr_NC = input.data + n * input_sN + c * input_sC;
        
        scalar_t* grad_w_ptr_C_H = weights_grad.data + c * grad_w_sC;
        scalar_t* grad_w_ptr_C_W = weights_grad.data + c * grad_w_sC + grad_w_sS;

        
        // init shifts; we use round for shift and floor for bilinear interpolation
        const int weights_offset = c * w_sC;
        scalar_t shift_H = weights.data[weights_offset];
        scalar_t shift_W = weights.data[weights_offset+w_sS];
        
        //reverse rounded shifts for grad backward 
        int rounded_shift_H = -1 * static_cast<int>(::round(shift_H));
        int rounded_shift_W = -1 * static_cast<int>(::round(shift_W));
        int H_shifted = ::max(H - ::abs(rounded_shift_H), static_cast<int>(0));
        int W_shifted = ::max(W - ::abs(rounded_shift_W), static_cast<int>(0));
        // floor shifts for weights backward
        int floor_shift_H = static_cast<int>(::floor(shift_H));
        int floor_shift_W = static_cast<int>(::floor(shift_W));
        scalar_t diff_shift_H = shift_H - static_cast<scalar_t>(floor_shift_H);
        scalar_t diff_shift_W = shift_W - static_cast<scalar_t>(floor_shift_W);
        
        // gradients for output
        if  ((h < H_shifted)&&(h >= 0)){
            int inp_h = h;
            int out_h = h;               
            if (rounded_shift_H > 0){
                out_h += ::abs(rounded_shift_H);
            }
            if (rounded_shift_H < 0){
                inp_h += ::abs(rounded_shift_H);
            }
            if ((w < W_shifted)&&(w >= 0)){
                int inp_w = w;
                int out_w = w;
                if (rounded_shift_W > 0){
                    out_w += ::abs(rounded_shift_W);
                }
                if (rounded_shift_W < 0){
                    inp_w += ::abs(rounded_shift_W);
                }
                scalar_t *out_grad_ptr_NCHW = out_grad.data + n * out_grad_sN + c * out_grad_sC + out_h * out_grad_sH + out_w * out_grad_sW;
                *out_grad_ptr_NCHW = grad_ptr_NC[inp_h * grad_sH + inp_w * grad_sW]; 
            } 
        }
        
        // gradients for weights
        int bl_h = detect_borders(h+floor_shift_H, H);
        int bl_w = detect_borders(w+floor_shift_W, W);
        int br_h = detect_borders(h+floor_shift_H, H);
        int br_w = detect_borders(w+floor_shift_W+1, W);
        int ul_h = detect_borders(h+floor_shift_H+1, H);
        int ul_w = detect_borders(w+floor_shift_W, W);
        int ur_h = detect_borders(h+floor_shift_H+1, H);
        int ur_w = detect_borders(w+floor_shift_W+1, W);
    
        scalar_t bl_v = 0;
        scalar_t br_v = 0;
        scalar_t ul_v = 0;
        scalar_t ur_v = 0;
        
        // threating indexes regarding of padding mode and get values
        if (padding_mode == BIPadding::Border){
            bl_h = infer_ind_border(bl_h,H);
            bl_w = infer_ind_border(bl_w,W);
            br_h = infer_ind_border(br_h,H);
            br_w = infer_ind_border(br_w,W);
            ul_h = infer_ind_border(ul_h,H);
            ul_w = infer_ind_border(ul_w,W);
            ur_h = infer_ind_border(ur_h,H);
            ur_w = infer_ind_border(ur_w,W);
        // values
            bl_v = input_ptr_NC[bl_h*input_sH + bl_w*input_sW];
            br_v = input_ptr_NC[br_h*input_sH + br_w*input_sW];
            ul_v = input_ptr_NC[ul_h*input_sH + ul_w*input_sW];
            ur_v = input_ptr_NC[ur_h*input_sH + ur_w*input_sW];
        }
        if (padding_mode == BIPadding::Zeros){
            if ((bl_h >= 0)&&(bl_w >= 0)){
                bl_v = input_ptr_NC[bl_h*input_sH + bl_w*input_sW];
            }
            if ((br_h >= 0)&&(br_w >= 0)){
                br_v = input_ptr_NC[br_h*input_sH + br_w*input_sW];
            }
            if ((ul_h >= 0)&&(ul_w >= 0)){
                ul_v = input_ptr_NC[ul_h*input_sH + ul_w*input_sW];
            }
            if ((ur_h >= 0)&&(ur_w >= 0)){
                ur_v = input_ptr_NC[ur_h*input_sH + ur_w*input_sW];
            }
        }
        //compute grads
        scalar_t grad_v = grad_ptr_NC[h*grad_sH + w*grad_sW];
        atomicAdd(grad_w_ptr_C_H, grad_v * ((1-diff_shift_W)*(ul_v-bl_v)+diff_shift_W*(ur_v-br_v)));
        atomicAdd(grad_w_ptr_C_W, grad_v * ((1-diff_shift_H)*(br_v-bl_v)+diff_shift_H*(ur_v-ul_v)));
    }
}
}
   
    

at::Tensor _shift2d_gpu(const at::Tensor& input,
                        const at::Tensor& weights){
    
    auto output = at::zeros_like(input, input.options());
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    
    int count = static_cast<int>(N*C*H*W);
    
    if (count > 0) {
         AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "_shift2d_gpu", [&] {
            shift2d_gpu_kernel<scalar_t>
            <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<scalar_t, int>(input),
            getTensorInfo<scalar_t, int>(weights),
            getTensorInfo<scalar_t, int>(output));
         });
    }
    return  output;
}


std::vector<at::Tensor> _shift2d_backward_gpu(const at::Tensor& grad,
                                              const at::Tensor& weights,
                                              const at::Tensor& input,
                                              int padding_mode){
    
    auto out_grad = at::zeros_like(grad, grad.options());
    auto weights_grad = at::zeros_like(weights, weights.options());
    
    int64_t N = grad.size(0);
    int64_t C = grad.size(1);
    int64_t H = grad.size(2);
    int64_t W = grad.size(3);
    
    int count = static_cast<int>(N*C*H*W);
    
    if (count > 0) {
         AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "_shift2d_backward_gpu", [&] {
            shift2d_backward_gpu_kernel<scalar_t>
            <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<scalar_t, int>(grad),
            getTensorInfo<scalar_t, int>(weights),
            getTensorInfo<scalar_t, int>(input),
            getTensorInfo<scalar_t, int>(out_grad),
            getTensorInfo<scalar_t, int>(weights_grad),
            static_cast<BIPadding>(padding_mode));
         });
    }
    return {out_grad,weights_grad};
}
