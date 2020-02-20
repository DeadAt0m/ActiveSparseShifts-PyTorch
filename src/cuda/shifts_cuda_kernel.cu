#include "shifts_cuda_kernel.h"

#include <THC/THCAtomics.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <thrust/tuple.h>


using namespace at::cuda::detail;

namespace {
    
// UTILS
__device__ __forceinline__ int infer_ind_zero(int index, int threshold){
    if ((index < threshold)&&(index >= 0)){
        return index;
    }
    return -1;
}
             
__device__ __forceinline__ int infer_ind_border(int index, int threshold){
    if ((index < threshold)&&(index >= 0)){
        return index;
    }
    if (index >= threshold){
        return threshold - 1;
    }
    return 0;   
}
            
__device__ __forceinline__ int infer_ind_reflect(int index, int threshold, int offset){
    if ((index < threshold)&&(index >= 0)){
        return index;
    }
    int temp = index;
    if (index < 0){
        temp = ::abs(index-offset) - 1;
    }
    return ((temp / (threshold - offset)) % 2 == 1) ? (threshold - 1 - (temp / (threshold - offset))) : (temp / (threshold - offset));
}

template <typename scalar_t>
__device__ __forceinline__ thrust::tuple<scalar_t,scalar_t> infer_linear_values(int i, int shift,
                                                                                int H,  int stride,
                                                                                scalar_t* vector_pointer,
                                                                                BIPadding padding_mode){
    int l = i+shift;
    int r = i+shift+1;
    
    if (padding_mode == BIPadding::Zeros){
        l = infer_ind_zero(l,H);
        r = infer_ind_zero(r,H);
     }
    if (padding_mode == BIPadding::Border){
        l = infer_ind_border(l,H);
        r = infer_ind_border(r,H);
     }
    if (padding_mode == BIPadding::Reflect){
        l = infer_ind_reflect(l,H,0);
        r = infer_ind_reflect(r,H,0);
     }
    if (padding_mode == BIPadding::Symmetric){
        l = infer_ind_reflect(l,H,1);
        r = infer_ind_reflect(r,H,1);
     }
     scalar_t l_v = 0;
     scalar_t r_v = 0;    
     if (l_v >= 0){
         l_v = vector_pointer[l*stride];
     }
     if (r_v >= 0){
         r_v = vector_pointer[r*stride];
     }
    return thrust::make_tuple(l_v, r_v);
}

template <typename scalar_t>
__device__ __forceinline__ thrust::tuple<scalar_t,scalar_t,scalar_t,scalar_t> infer_bilinear_values(int i, int j,
                                                                                                    int shift_H, int shift_W,
                                                                                                    int H, int W,
                                                                                                    int strideH, int strideW,
                                                                                                    scalar_t* vector_pointer,
                                                                                                    BIPadding padding_mode){
    // init indexes for interpolation: bl-bottom left, br-bottom right, ul-upper left, ur-upper right   
    int bl_h = i+shift_H;
    int bl_w = j+shift_W;
    int br_h = i+shift_H;
    int br_w = j+shift_W+1;
    int ul_h = i+shift_H+1;
    int ul_w = j+shift_W;
    int ur_h = i+shift_H+1;
    int ur_w = j+shift_W+1;
    
    if (padding_mode == BIPadding::Zeros){
        bl_h = infer_ind_zero(bl_h,H);
        bl_w = infer_ind_zero(bl_w,W);
        br_h = infer_ind_zero(br_h,H);
        br_w = infer_ind_zero(br_w,W);
        ul_h = infer_ind_zero(ul_h,H);
        ul_w = infer_ind_zero(ul_w,W);
        ur_h = infer_ind_zero(ur_h,H);
        ur_w = infer_ind_zero(ur_w,W);
     }
    if (padding_mode == BIPadding::Border){
        bl_h = infer_ind_border(bl_h,H);
        bl_w = infer_ind_border(bl_w,W);
        br_h = infer_ind_border(br_h,H);
        br_w = infer_ind_border(br_w,W);
        ul_h = infer_ind_border(ul_h,H);
        ul_w = infer_ind_border(ul_w,W);
        ur_h = infer_ind_border(ur_h,H);
        ur_w = infer_ind_border(ur_w,W);
     }
    if (padding_mode == BIPadding::Reflect){
        bl_h = infer_ind_reflect(bl_h,H,0);
        bl_w = infer_ind_reflect(bl_w,W,0);
        br_h = infer_ind_reflect(br_h,H,0);
        br_w = infer_ind_reflect(br_w,W,0);
        ul_h = infer_ind_reflect(ul_h,H,0);
        ul_w = infer_ind_reflect(ul_w,W,0);
        ur_h = infer_ind_reflect(ur_h,H,0);
        ur_w = infer_ind_reflect(ur_w,W,0);
     }
    if (padding_mode == BIPadding::Symmetric){
        bl_h = infer_ind_reflect(bl_h,H,1);
        bl_w = infer_ind_reflect(bl_w,W,1);
        br_h = infer_ind_reflect(br_h,H,1);
        br_w = infer_ind_reflect(br_w,W,1);
        ul_h = infer_ind_reflect(ul_h,H,1);
        ul_w = infer_ind_reflect(ul_w,W,1);
        ur_h = infer_ind_reflect(ur_h,H,1);
        ur_w = infer_ind_reflect(ur_w,W,1);
     }
     scalar_t bl_v = 0;
     scalar_t br_v = 0;
     scalar_t ul_v = 0;
     scalar_t ur_v = 0;
    
     if ((bl_h >= 0)&&(bl_w >= 0)){
         bl_v = vector_pointer[bl_h*strideH + bl_w*strideW];
     }
     if ((br_h >= 0)&&(br_w >= 0)){
         br_v = vector_pointer[br_h*strideH + br_w*strideW];
     }
     if ((ul_h >= 0)&&(ul_w >= 0)){
         ul_v = vector_pointer[ul_h*strideH + ul_w*strideW];
     }
     if ((ur_h >= 0)&&(ur_w >= 0)){
         ur_v = vector_pointer[ur_h*strideH + ur_w*strideW];
     }
    return thrust::make_tuple(bl_v, br_v, ul_v, ur_v);
}    
///SHIFT1D
///
///FORWARD PASS
////   
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void shift1d_gpu_kernel_active(const int n_threads,
                                          TensorInfo<scalar_t, int> input,
                                          TensorInfo<scalar_t, int> weights,
                                          TensorInfo<scalar_t, int> output,
                                          const BIPadding padding_mode){
    int C = input.sizes[1];
    int H = input.sizes[2];

    int input_sN = input.strides[0];
    int input_sC = input.strides[1];
    int input_sH = input.strides[2];
    
    int output_sN = output.strides[0];
    int output_sC = output.strides[1];
    int output_sH = output.strides[2];
    
    int w_sC = weights.strides[0];
    
    CUDA_KERNEL_LOOP(index, n_threads){
        const int h = index % H;
        const int c = (index / H) % C;
        const int n = index / (H*C);

        const int weights_offset = c * w_sC;
        
        scalar_t *inp_ptr_NC = input.data + n * input_sN + c * input_sC;
        scalar_t shift = weights.data[weights_offset];
        
        int floor_shift = static_cast<int>(::floor(shift));
        scalar_t diff_shift = shift - static_cast<scalar_t>(floor_shift);
        
        thrust::tuple<scalar_t,scalar_t> values = infer_linear_values<scalar_t>(h, floor_shift,
                                                                                H, input_sH,
                                                                                inp_ptr_NC, padding_mode);
        scalar_t *out_ptr_NCH = output.data + n * output_sN + c * output_sC  + h * output_sH;
        *out_ptr_NCH = (1-diff_shift) * thrust::get<0>(values) +
                       diff_shift * thrust::get<1>(values);
    }
}
    
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void shift1d_gpu_kernel_quantized(const int n_threads,
                                             TensorInfo<scalar_t, int> input,
                                             TensorInfo<scalar_t, int> weights,
                                             TensorInfo<scalar_t, int> output){
    int C = input.sizes[1];
    int H = input.sizes[2];

    int input_sN = input.strides[0];
    int input_sC = input.strides[1];
    int input_sH = input.strides[2];

    
    int output_sN = output.strides[0];
    int output_sC = output.strides[1];
    int output_sH = output.strides[2];
    
    int w_sC = weights.strides[0];
    
    CUDA_KERNEL_LOOP(index, n_threads){
        const int h = index % H;
        const int c = (index / H) % C;
        const int n = index / (H*C);
        
        const int weights_offset = c * w_sC;
        
        scalar_t *inp_ptr_NC = input.data + n * input_sN + c * input_sC;
        scalar_t shift = weights.data[weights_offset];
        
        int rounded_shift = static_cast<int>(::round(shift));
        int H_shifted = ::max(H - ::abs(rounded_shift), static_cast<int>(0));
        
        if  ((h < H_shifted)&&(h >= 0)){
            int inp_h = h;
            int out_h = h;               
            if (rounded_shift > 0){
                out_h += ::abs(rounded_shift);
            }
            if (rounded_shift < 0){
                inp_h += ::abs(rounded_shift);
            }
            scalar_t *output_ptr_NCH = output.data + n * output_sN + c * output_sC + out_h * output_sH ;
            *output_ptr_NCH = inp_ptr_NC[inp_h * input_sH]; 
        }  
    }
}       
///BACKWARD    
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void shift1d_backward_gpu_kernel_active(const int n_threads,
                                                   TensorInfo<scalar_t, int> grad,
                                                   TensorInfo<scalar_t, int> weights,
                                                   TensorInfo<scalar_t, int> input,
                                                   TensorInfo<scalar_t, int> out_grad,
                                                   TensorInfo<scalar_t, int> weights_grad,
                                                   const BIPadding padding_mode)
{
    int C = grad.sizes[1];
    int H = grad.sizes[2];
    
    int grad_sN = grad.strides[0];
    int grad_sC = grad.strides[1];
    int grad_sH = grad.strides[2];

    int out_grad_sN = out_grad.strides[0];
    int out_grad_sC = out_grad.strides[1];
    int out_grad_sH = out_grad.strides[2];
    
    int input_sN = input.strides[0];
    int input_sC = input.strides[1];
    int input_sH = input.strides[2];
    
    int w_sC = weights.strides[0];
    int grad_w_sC = weights_grad.strides[0];

    CUDA_KERNEL_LOOP(index, n_threads){
        const int h = index % H;
        const int c = (index / H) % C;
        const int n = index / (H*C);
        
        scalar_t *grad_ptr_NC = grad.data + n * grad_sN + c * grad_sC;
        scalar_t *input_ptr_NC = input.data + n * input_sN + c * input_sC;
        
        scalar_t* grad_w_ptr_C_H = weights_grad.data + c * grad_w_sC;

        
        // init shifts for weights backward
        const int weights_offset = c * w_sC;
        scalar_t shift = weights.data[weights_offset];

        int floor_shift = static_cast<int>(::floor(shift));
        // init reversed shifts for grad backward
        scalar_t rshift = -shift;
        int floor_rshift = static_cast<int>(::floor(rshift));
        scalar_t diff_rshift = rshift - static_cast<scalar_t>(floor_rshift);
        
        // grad backward
        thrust::tuple<scalar_t,scalar_t> rvalues = infer_linear_values<scalar_t>(h, floor_rshift,
                                                                                 H, grad_sH,
                                                                                 grad_ptr_NC, padding_mode);
        
        scalar_t *out_grad_ptr_NCH = out_grad.data + n * out_grad_sN + c * out_grad_sC + h * out_grad_sH;
        *out_grad_ptr_NCH = (1-diff_rshift)* thrust::get<0>(rvalues) +
                            diff_rshift * thrust::get<1>(rvalues);
        // weight backward
        thrust::tuple<scalar_t,scalar_t> values = infer_linear_values<scalar_t>(h, floor_shift,
                                                                                H, input_sH, 
                                                                                input_ptr_NC, padding_mode);
        
        scalar_t local_grad_w_H = thrust::get<1>(values)-thrust::get<0>(values);
       
        //compute grads
        scalar_t grad_v = grad_ptr_NC[h*grad_sH];
        atomicAdd(grad_w_ptr_C_H, grad_v * local_grad_w_H);
    }
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void shift1d_backward_gpu_kernel_quantized(const int n_threads,
                                                      TensorInfo<scalar_t, int> grad,
                                                      TensorInfo<scalar_t, int> weights,
                                                      TensorInfo<scalar_t, int> input,
                                                      TensorInfo<scalar_t, int> out_grad,
                                                      TensorInfo<scalar_t, int> weights_grad,
                                                      const BIPadding padding_mode)
{
    int C = grad.sizes[1];
    int H = grad.sizes[2];
 
    int grad_sN = grad.strides[0];
    int grad_sC = grad.strides[1];
    int grad_sH = grad.strides[2];

    int out_grad_sN = out_grad.strides[0];
    int out_grad_sC = out_grad.strides[1];
    int out_grad_sH = out_grad.strides[2];

    
    int input_sN = input.strides[0];
    int input_sC = input.strides[1];
    int input_sH = input.strides[2];
    
    int w_sC = weights.strides[0];
    int grad_w_sC = weights_grad.strides[0];
    
    CUDA_KERNEL_LOOP(index, n_threads){
        const int h = index % H;
        const int c = (index / H) % C;
        const int n = index / (H*C);
        
        scalar_t *grad_ptr_NC = grad.data + n * grad_sN + c * grad_sC;
        scalar_t *input_ptr_NC = input.data + n * input_sN + c * input_sC;
        
        scalar_t* grad_w_ptr_C_H = weights_grad.data + c * grad_w_sC;
        
        // init shifts; we use round for shift and floor for bilinear interpolation
        const int weights_offset = c * w_sC;
        scalar_t shift = weights.data[weights_offset];
        
        //reverse rounded shifts for grad backward 
        int rounded_shift = -1 * static_cast<int>(::round(shift));
        int H_shifted = ::max(H - ::abs(rounded_shift), static_cast<int>(0));
        // floor shifts for weights backward
        int floor_shift = static_cast<int>(::floor(shift));
        
        // gradients for output
        if  ((h < H_shifted)&&(h >= 0)){
            int inp_h = h;
            int out_h = h;               
            if (rounded_shift > 0){
                out_h += ::abs(rounded_shift);
            }
            if (rounded_shift < 0){
                inp_h += ::abs(rounded_shift);
            }
           
            scalar_t *out_grad_ptr_NCH = out_grad.data + n * out_grad_sN + c * out_grad_sC + out_h * out_grad_sH ;
            *out_grad_ptr_NCH = grad_ptr_NC[inp_h * grad_sH]; 
   
        }
        thrust::tuple<scalar_t,scalar_t> values = infer_linear_values<scalar_t>(h, floor_shift,
                                                                                H, input_sH,
                                                                                input_ptr_NC, padding_mode);
        
        scalar_t local_grad_w_H = thrust::get<1>(values)-thrust::get<0>(values);
       
        //compute grads
        scalar_t grad_v = grad_ptr_NC[h*grad_sH];
        atomicAdd(grad_w_ptr_C_H, grad_v * local_grad_w_H);
    }
}       
//SHIFT2D
///
///FORWARD PASS
////   
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void shift2d_gpu_kernel_active(const int n_threads,
                                          TensorInfo<scalar_t, int> input,
                                          TensorInfo<scalar_t, int> weights,
                                          TensorInfo<scalar_t, int> output,
                                          const BIPadding padding_mode){
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
        
        int floor_shift_H = static_cast<int>(::floor(shift_H));
        int floor_shift_W = static_cast<int>(::floor(shift_W));
        scalar_t diff_shift_H = shift_H - static_cast<scalar_t>(floor_shift_H);
        scalar_t diff_shift_W = shift_W - static_cast<scalar_t>(floor_shift_W);
        
        thrust::tuple<scalar_t,scalar_t,scalar_t,scalar_t> values = infer_bilinear_values<scalar_t>(h, w, floor_shift_H, floor_shift_W,
                                                                                                    H, W, input_sH, input_sW,
                                                                                                    inp_ptr_NC, padding_mode);
        scalar_t *out_ptr_NCHW =  output.data + n * output_sN + c * output_sC + h * output_sH + w * output_sW;
        *out_ptr_NCHW = (1-diff_shift_H)*(1-diff_shift_W) * thrust::get<0>(values) +
                        (1-diff_shift_H)*diff_shift_W * thrust::get<1>(values) +
                        diff_shift_H*(1-diff_shift_W) * thrust::get<2>(values) +
                        diff_shift_H*diff_shift_W  * thrust::get<3>(values);
        
    }
}
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void shift2d_gpu_kernel_quantized(const int n_threads,
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
// BACKWARD
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void shift2d_backward_gpu_kernel_active(const int n_threads,
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

        
        // init shifts for weights backward
        const int weights_offset = c * w_sC;
        scalar_t shift_H = weights.data[weights_offset];
        scalar_t shift_W = weights.data[weights_offset+w_sS];
        
        int floor_shift_H = static_cast<int>(::floor(shift_H));
        int floor_shift_W = static_cast<int>(::floor(shift_W));
        scalar_t diff_shift_H = shift_H - static_cast<scalar_t>(floor_shift_H);
        scalar_t diff_shift_W = shift_W - static_cast<scalar_t>(floor_shift_W);
        // init reversed shifts for grad backward
        scalar_t rshift_H = -shift_H;
        scalar_t rshift_W = -shift_W;
        int floor_rshift_H = static_cast<int>(::floor(rshift_H));
        int floor_rshift_W = static_cast<int>(::floor(rshift_W));
        scalar_t diff_rshift_H = rshift_H - static_cast<scalar_t>(floor_rshift_H);
        scalar_t diff_rshift_W = rshift_W - static_cast<scalar_t>(floor_rshift_W);
        
        // grad backward
        thrust::tuple<scalar_t,scalar_t,scalar_t,scalar_t> rvalues = infer_bilinear_values<scalar_t>(h, w, floor_rshift_H, floor_rshift_W,
                                                                                                     H, W, grad_sH, grad_sW,
                                                                                                     grad_ptr_NC, padding_mode);
        scalar_t *out_grad_ptr_NCHW = out_grad.data + n * out_grad_sN + c * out_grad_sC + h * out_grad_sH + w * out_grad_sW;
        *out_grad_ptr_NCHW = (1-diff_rshift_H)*(1-diff_rshift_W) * thrust::get<0>(rvalues) +
                             (1-diff_rshift_H)*diff_rshift_W * thrust::get<1>(rvalues) +
                             diff_rshift_H*(1-diff_rshift_W) * thrust::get<2>(rvalues) +
                             diff_rshift_H*diff_rshift_W  * thrust::get<3>(rvalues);
        // weight backward
        thrust::tuple<scalar_t,scalar_t,scalar_t,scalar_t> values = infer_bilinear_values<scalar_t>(h, w, floor_shift_H, floor_shift_W,
                                                                                                    H, W, input_sH, input_sW,
                                                                                                    input_ptr_NC, padding_mode);
        
        scalar_t local_grad_w_H = (1-diff_shift_W)*(thrust::get<2>(values)-thrust::get<0>(values)) + 
                                  diff_shift_W*(thrust::get<3>(values)-thrust::get<1>(values));
                    
        scalar_t local_grad_w_W = (1-diff_shift_H)*(thrust::get<1>(values)-thrust::get<0>(values)) + 
                                  diff_shift_H*(thrust::get<3>(values)-thrust::get<2>(values));
       
        //compute grads
        scalar_t grad_v = grad_ptr_NC[h*grad_sH + w*grad_sW];
        atomicAdd(grad_w_ptr_C_H, grad_v * local_grad_w_H);
        atomicAdd(grad_w_ptr_C_W, grad_v * local_grad_w_W);
    }
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void shift2d_backward_gpu_kernel_quantized(const int n_threads,
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
        
        thrust::tuple<scalar_t,scalar_t,scalar_t,scalar_t> values = infer_bilinear_values<scalar_t>(h, w, floor_shift_H, floor_shift_W,
                                                                                                    H, W, input_sH, input_sW,
                                                                                                    input_ptr_NC, padding_mode);
        scalar_t local_grad_w_H = (1-diff_shift_W)*(thrust::get<2>(values)-thrust::get<0>(values)) + 
                                  diff_shift_W*(thrust::get<3>(values)-thrust::get<1>(values));
                    
        scalar_t local_grad_w_W = (1-diff_shift_H)*(thrust::get<1>(values)-thrust::get<0>(values)) + 
                                  diff_shift_H*(thrust::get<3>(values)-thrust::get<2>(values));
       
        //compute grads
        scalar_t grad_v = grad_ptr_NC[h*grad_sH + w*grad_sW];
        atomicAdd(grad_w_ptr_C_H, grad_v * local_grad_w_H);
        atomicAdd(grad_w_ptr_C_W, grad_v * local_grad_w_W);
    }
}
        
}
/// INTERFACES
at::Tensor _shift1d_gpu(const at::Tensor& input,
                        const at::Tensor& weights,
                        int padding_mode,
                        bool active_flag){
    
    auto output = at::zeros_like(input, input.options());
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    
    int count = static_cast<int>(N*C*H);
    
    if (count > 0) {
        if (active_flag){
             AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "_shift1d_gpu", [&] {
                shift1d_gpu_kernel_active<scalar_t>
                <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                getTensorInfo<scalar_t, int>(input),
                getTensorInfo<scalar_t, int>(weights),
                getTensorInfo<scalar_t, int>(output),
                static_cast<BIPadding>(padding_mode));
             });
        }
        else {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "_shift1d_gpu", [&] {
                shift1d_gpu_kernel_quantized<scalar_t>
                <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                getTensorInfo<scalar_t, int>(input),
                getTensorInfo<scalar_t, int>(weights),
                getTensorInfo<scalar_t, int>(output));
             });
        }
    }
    return  output;
}


std::vector<at::Tensor> _shift1d_backward_gpu(const at::Tensor& grad,
                                              const at::Tensor& weights,
                                              const at::Tensor& input,
                                              int padding_mode,
                                              bool active_flag){
    
    auto out_grad = at::zeros_like(grad, grad.options());
    auto weights_grad = at::zeros_like(weights, weights.options());
    
    int64_t N = grad.size(0);
    int64_t C = grad.size(1);
    int64_t H = grad.size(2);
    
    int count = static_cast<int>(N*C*H);
    
    if (count > 0) {
        if (active_flag) {
             AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "_shift1d_backward_gpu", [&] {
                shift1d_backward_gpu_kernel_active<scalar_t>
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
        else {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "_shift1d_backward_gpu", [&] {
                shift1d_backward_gpu_kernel_quantized<scalar_t>
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
    }
    return {out_grad,weights_grad};
}

at::Tensor _shift2d_gpu(const at::Tensor& input,
                        const at::Tensor& weights,
                        int padding_mode,
                        bool active_flag){
    
    auto output = at::zeros_like(input, input.options());
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    
    int count = static_cast<int>(N*C*H*W);
    
    if (count > 0) {
        if (active_flag){
             AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "_shift2d_gpu", [&] {
                shift2d_gpu_kernel_active<scalar_t>
                <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                getTensorInfo<scalar_t, int>(input),
                getTensorInfo<scalar_t, int>(weights),
                getTensorInfo<scalar_t, int>(output),
                static_cast<BIPadding>(padding_mode));
             });
        }
        else {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "_shift2d_gpu", [&] {
                shift2d_gpu_kernel_quantized<scalar_t>
                <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                getTensorInfo<scalar_t, int>(input),
                getTensorInfo<scalar_t, int>(weights),
                getTensorInfo<scalar_t, int>(output));
             });
        }
    }
    return  output;
}


std::vector<at::Tensor> _shift2d_backward_gpu(const at::Tensor& grad,
                                              const at::Tensor& weights,
                                              const at::Tensor& input,
                                              int padding_mode,
                                              bool active_flag){
    
    auto out_grad = at::zeros_like(grad, grad.options());
    auto weights_grad = at::zeros_like(weights, weights.options());
    
    int64_t N = grad.size(0);
    int64_t C = grad.size(1);
    int64_t H = grad.size(2);
    int64_t W = grad.size(3);
    
    int count = static_cast<int>(N*C*H*W);
    
    if (count > 0) {
        if (active_flag) {
             AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "_shift2d_backward_gpu", [&] {
                shift2d_backward_gpu_kernel_active<scalar_t>
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
        else {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "_shift2d_backward_gpu", [&] {
                shift2d_backward_gpu_kernel_quantized<scalar_t>
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
    }
    return {out_grad,weights_grad};
}
