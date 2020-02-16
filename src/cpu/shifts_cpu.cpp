#include "shifts_cpu.h"

template <typename scalar_t>
torch::Tensor shift2d_cpu_kernel(const torch::Tensor& input,
                                 const torch::Tensor& weights){
    
    auto output = torch::zeros_like(input, input.options());

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    
    int64_t input_sN = input.stride(0);
    int64_t input_sC = input.stride(1);
    int64_t input_sH = input.stride(2);
    int64_t input_sW = input.stride(3);
    
    int64_t output_sN = output.stride(0);
    int64_t output_sC = output.stride(1);
    int64_t output_sH = output.stride(2);
    int64_t output_sW = output.stride(3);
    
    int64_t w_sC = weights.stride(0);
    int64_t w_sS = weights.stride(1);

    scalar_t *inp_ptr = input.data_ptr<scalar_t>();
    scalar_t *out_ptr = output.data_ptr<scalar_t>();
    scalar_t *w_ptr = weights.data_ptr<scalar_t>();
    
    at::parallel_for(0, N*C, 0, [&](int64_t start, int64_t end){
        for (int64_t idx = start; idx < end; ++idx) {
            int64_t n = idx / C;
            int64_t c = idx % C;
            scalar_t *inp_ptr_NC = inp_ptr + n * input_sN + c * input_sC;
            scalar_t *out_ptr_NC = out_ptr + n * output_sN + c * output_sC;
            scalar_t *w_ptr_shifts = w_ptr + c * w_sC;
            
            int64_t shift_H = static_cast<int64_t>(std::round(*w_ptr_shifts));
            int64_t shift_W = static_cast<int64_t>(std::round(w_ptr_shifts[w_sS]));
            int64_t H_shifted = std::max(H - std::abs(shift_H),static_cast<int64_t>(0));
            int64_t W_shifted = std::max(W - std::abs(shift_W),static_cast<int64_t>(0));
            
            
            for (int64_t h = 0; h < H_shifted; ++h) {
                int64_t inp_h = h;
                int64_t out_h = h;               
                if (shift_H > 0){
                    out_h += std::abs(shift_H);
                }
                if (shift_H < 0){
                    inp_h += std::abs(shift_H);
                }
                for (int64_t w = 0; w < W_shifted; ++w) {
                    int64_t inp_w = w;
                    int64_t out_w = w;     
                    if (shift_W > 0){
                        out_w += std::abs(shift_W);
                    }
                    if (shift_W < 0){
                        inp_w += std::abs(shift_W);
                    }
                    scalar_t *out_ptr_NCHW = out_ptr_NC + out_h * output_sH + out_w * output_sW;
                    *out_ptr_NCHW = inp_ptr_NC[inp_h * input_sH + inp_w * input_sW];                   
                }
            }
        }    
    });
    return output;
}


inline int64_t detect_borders(int64_t index, int64_t threshold){                
    return (index > (threshold-1)) ? -index : index;
}

inline int64_t infer_ind_border(int64_t index, int64_t threshold){
    int64_t tmp =  std::max((1-threshold), index);
    return (tmp < 0) ? 0 : std::abs(tmp);
}

template <typename scalar_t>
inline std::tuple<scalar_t,scalar_t> infer_weight_grads(int64_t i, int64_t j,
                                                        int64_t shift_H, int64_t shift_W,
                                                        int64_t H, int64_t W,
                                                        int64_t strideH, int64_t strideW,
                                                        scalar_t diff_shift_H, scalar_t diff_shift_W,
                                                        scalar_t* vector_pointer,
                                                        BIPadding padding_mode){
    // init indexes for interpolation: bl-bottom left, br-bottom right, ul-upper left, ur-upper right
    int64_t bl_h = detect_borders(i+shift_H, H);
    int64_t bl_w = detect_borders(j+shift_W, W);
    int64_t br_h = detect_borders(i+shift_H, H);
    int64_t br_w = detect_borders(j+shift_W+1, W);
    int64_t ul_h = detect_borders(i+shift_H+1, H);
    int64_t ul_w = detect_borders(j+shift_W, W);
    int64_t ur_h = detect_borders(i+shift_H+1, H);
    int64_t ur_w = detect_borders(j+shift_W+1, W);
    
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
        bl_v = vector_pointer[bl_h*strideH + bl_w*strideW];
        br_v = vector_pointer[br_h*strideH + br_w*strideW];
        ul_v = vector_pointer[ul_h*strideH + ul_w*strideW];
        ur_v = vector_pointer[ur_h*strideH + ur_w*strideW];
    }
    if (padding_mode == BIPadding::Zeros){
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
    }
                                                                            
    //compute grads
    scalar_t local_grad_w_H = (1-diff_shift_W)*(ul_v-bl_v)+diff_shift_W*(ur_v-br_v);
    scalar_t local_grad_w_W = (1-diff_shift_H)*(br_v-bl_v)+diff_shift_H*(ur_v-ul_v);
    
    return std::make_tuple(local_grad_w_H,local_grad_w_W);    
}


template <typename scalar_t>
std::vector<torch::Tensor> shift2d_cpu_backward_kernel(const torch::Tensor& grad,
                                                       const torch::Tensor& weights,
                                                       const torch::Tensor& input,
                                                       BIPadding padding_mode){
    auto out_grad = torch::zeros_like(grad, grad.options());
    auto weights_grad = torch::zeros_like(weights, weights.options());
    
    
    int64_t N = grad.size(0);
    int64_t C = grad.size(1);
    int64_t H = grad.size(2);
    int64_t W = grad.size(3);
    
    int64_t grad_sN = grad.stride(0);
    int64_t grad_sC = grad.stride(1);
    int64_t grad_sH = grad.stride(2);
    int64_t grad_sW = grad.stride(3);
    
    int64_t out_grad_sN = out_grad.stride(0);
    int64_t out_grad_sC = out_grad.stride(1);
    int64_t out_grad_sH = out_grad.stride(2);
    int64_t out_grad_sW = out_grad.stride(3);
    
    int64_t input_sN = input.stride(0);
    int64_t input_sC = input.stride(1);
    int64_t input_sH = input.stride(2);
    int64_t input_sW = input.stride(3);
    
    int64_t w_sC = weights.stride(0);
    int64_t w_sS = weights.stride(1);
    int64_t grad_w_sC = weights_grad.stride(0);
    int64_t grad_w_sS = weights_grad.stride(1);

    scalar_t *grad_ptr = grad.data_ptr<scalar_t>();
    scalar_t *out_grad_ptr = out_grad.data_ptr<scalar_t>();
    scalar_t *input_ptr = input.data_ptr<scalar_t>();

    scalar_t *w_ptr = weights.data_ptr<scalar_t>();
    scalar_t *grad_w_ptr = weights_grad.data_ptr<scalar_t>();
    
    at::parallel_for(0, N*C, 0, [&](int64_t start, int64_t end){
        for (int64_t idx = start; idx < end; ++idx) {
            int64_t n = idx / C;
            int64_t c = idx % C;

            // init pointers for main tensors
            scalar_t *grad_ptr_NC = grad_ptr + n * grad_sN + c * grad_sC;
            scalar_t *out_grad_ptr_NC = out_grad_ptr + n * out_grad_sN + c * out_grad_sC;
            scalar_t *inp_ptr_NC = input_ptr + n * input_sN + c * input_sC;
                   
            
            // init pointers for weights
            scalar_t *w_ptr_C = w_ptr + c * w_sC;
            scalar_t *grad_w_ptr_C= grad_w_ptr + c * grad_w_sC;
            scalar_t *grad_w_ptr_C_H = grad_w_ptr_C;
            scalar_t *grad_w_ptr_C_W = grad_w_ptr_C + grad_w_sS;

            
            // init shifts; we use round for shift and floor for bilinear interpolation
            scalar_t shift_H = *w_ptr_C;
            scalar_t shift_W = w_ptr_C[w_sS];
            //reverse rounded shifts for grad backward 
            int64_t rounded_shift_H = -1 * static_cast<int64_t>(std::round(shift_H));
            int64_t rounded_shift_W = -1 * static_cast<int64_t>(std::round(shift_W));
            int64_t H_shifted = std::max(H - std::abs(rounded_shift_H),static_cast<int64_t>(0));
            int64_t W_shifted = std::max(W - std::abs(rounded_shift_W),static_cast<int64_t>(0));
            // floor shifts for weights backward
            int64_t floor_shift_H = static_cast<int64_t>(std::floor(shift_H));
            int64_t floor_shift_W = static_cast<int64_t>(std::floor(shift_W));
            scalar_t diff_shift_H = shift_H - static_cast<scalar_t>(floor_shift_H);
            scalar_t diff_shift_W = shift_W - static_cast<scalar_t>(floor_shift_W);
            
                
            for (int64_t h = 0; h < H_shifted; ++h) {
                int64_t grad_h = h;
                int64_t out_h = h;               
                if (rounded_shift_H > 0){
                    out_h += std::abs(rounded_shift_H);
                }
                if (rounded_shift_H < 0){
                    grad_h += std::abs(rounded_shift_H);
                }
                for (int64_t w = 0; w < W_shifted; ++w) {
                    int64_t grad_w = w;
                    int64_t out_w = w;     
                    if (rounded_shift_W > 0){
                        out_w += std::abs(rounded_shift_W);
                    }
                    if (rounded_shift_W < 0){
                        grad_w += std::abs(rounded_shift_W);
                    }
                    // compute main grad
                    scalar_t *out_grad_ptr_NCHW = out_grad_ptr_NC + out_h * out_grad_sH + out_w * out_grad_sW;
                    *out_grad_ptr_NCHW = grad_ptr_NC[grad_h * grad_sH + grad_w * grad_sW];    
                    // compute weights grad
                    std::tuple<scalar_t,scalar_t> local_grads = infer_weight_grads<scalar_t>(h, w, floor_shift_H, floor_shift_W,
                                                                                             H, W, input_sH, input_sW,
                                                                                             diff_shift_H, diff_shift_W,
                                                                                             inp_ptr_NC, padding_mode);
                    scalar_t grad_v = grad_ptr_NC[h * grad_sH + w * grad_sW];
                    *grad_w_ptr_C_H += (grad_v * std::get<0>(local_grads));
                    *grad_w_ptr_C_W += (grad_v * std::get<1>(local_grads));
                }
                // such as main cycle mainly using to compute both gradients to save cycles
                // but we need extra small cycles to finish weights grads computations
                for (int64_t w = W_shifted; w < W; ++w){
                    std::tuple<scalar_t,scalar_t> local_grads = infer_weight_grads<scalar_t>(h, w, floor_shift_H, floor_shift_W,
                                                                                             H, W, input_sH, input_sW,
                                                                                             diff_shift_H, diff_shift_W,
                                                                                             inp_ptr_NC, padding_mode);
                    scalar_t grad_v = grad_ptr_NC[h*grad_sH + w*grad_sW];
                    *grad_w_ptr_C_H += (grad_v * std::get<0>(local_grads));
                    *grad_w_ptr_C_W += (grad_v * std::get<1>(local_grads));
                }    
            }
            for (int64_t h = H_shifted; h < H; ++h){
                for (int64_t w = 0; w < W; ++w){
                    std::tuple<scalar_t,scalar_t> local_grads = infer_weight_grads<scalar_t>(h, w, floor_shift_H, floor_shift_W,
                                                                                             H, W, input_sH, input_sW,
                                                                                             diff_shift_H, diff_shift_W,
                                                                                             inp_ptr_NC, padding_mode);
                    scalar_t grad_v = grad_ptr_NC[h*grad_sH + w*grad_sW];
                    *grad_w_ptr_C_H += (grad_v * std::get<0>(local_grads));
                    *grad_w_ptr_C_W += (grad_v * std::get<1>(local_grads));
                }
            }
        }    
    });
    return {out_grad,weights_grad};
}


torch::Tensor shift2d_cpu(const torch::Tensor& input,
                          const torch::Tensor& weights) {
  return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "shift2d_cpu", [&] {
    return shift2d_cpu_kernel<scalar_t>(input, weights);
  });
}


std::vector<torch::Tensor> shift2d_backward_cpu(const torch::Tensor& grad,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& input,
                                                int padding_mode) {
  return AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "shift2d_backward_cpu", [&] {
    return shift2d_cpu_backward_kernel<scalar_t>(grad, weights, input, static_cast<BIPadding>(padding_mode));
  });
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("shift2d_cpu", &shift2d_cpu, "2D Shift operation forward (cpu)");
    m.def("shift2d_backward_cpu", &shift2d_backward_cpu, "2D Shift operator backward (cpu) ");
}