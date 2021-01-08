#pragma once
#include "cpu/shifts_cpu.h"
#include "quantized/shifts_quantized.h"
#ifdef WITH_CUDA
    #include "cuda/shifts_cuda.h"
#endif


using namespace torch::indexing;

template<int nD = 1>
torch::Tensor shiftnd_forward(const torch::Tensor& input,
                              const torch::Tensor& weights,
                              const torch::Tensor& borders,
                              const std::vector<int64_t>& new_size,
                              int64_t padding_mode,
                              bool active_flag){
    if (input.is_cuda()) {
        #ifdef WITH_CUDA
            if constexpr(nD == 3){
                return shift3d_forward_cuda(input, weights, borders, new_size, padding_mode, active_flag);
            } else if constexpr(nD == 2){
                return shift2d_forward_cuda(input, weights, borders, new_size, padding_mode, active_flag);
            } else {
                return shift1d_forward_cuda(input, weights, borders, new_size, padding_mode, active_flag);
            }
        #else
            TORCH_CHECK(false, "Not compiled with GPU support");
        #endif
    }      
    if constexpr(nD == 3){
        return shift3d_forward_cpu(input, weights, borders.to(torch::kLong), new_size, padding_mode, active_flag);
    } else if constexpr(nD == 2){
        return shift2d_forward_cpu(input, weights, borders.to(torch::kLong), new_size, padding_mode, active_flag);
    } else {
        return shift1d_forward_cpu(input, weights, borders.to(torch::kLong), new_size, padding_mode, active_flag);
    }
}

template<int nD = 1>
std::vector<torch::Tensor> shiftnd_backward(const torch::Tensor& grad,
                                            const torch::Tensor& weights,
                                            const torch::Tensor& input,
                                            const torch::Tensor& borders,
                                            int64_t padding_mode,
                                            bool active_flag){
    if (input.is_cuda()) {
        #ifdef WITH_CUDA
            if constexpr(nD == 3){
                return shift3d_backward_cuda(grad, weights, input, borders, padding_mode, active_flag);

            } else if constexpr(nD == 2){
                return shift2d_backward_cuda(grad, weights, input, borders, padding_mode, active_flag);
            } else {
                return shift1d_backward_cuda(grad, weights, input, borders, padding_mode, active_flag);
            }
        #else
            TORCH_CHECK(false, "Not compiled with GPU support");
        #endif
    }      
    if constexpr(nD == 3){
        return shift3d_backward_cpu(grad, weights, input, borders.to(torch::kLong), padding_mode, active_flag);
    } else if constexpr(nD == 2){
        return shift2d_backward_cpu(grad, weights, input, borders.to(torch::kLong), padding_mode, active_flag);
    } else {
        return shift1d_backward_cpu(grad, weights, input, borders.to(torch::kLong), padding_mode, active_flag);
    }
}



class Shift1dFunction : public torch::autograd::Function<Shift1dFunction> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                     torch::Tensor input,
                                     torch::Tensor weight,
                                     torch::Tensor borders,
                                     std::vector<int64_t> new_size,
                                     int64_t padding_mode, bool active_flag){
            ctx->saved_data["padding_mode"] = padding_mode;
            ctx->saved_data["active_flag"] = active_flag;
            ctx->save_for_backward({input, weight, borders});
            return shiftnd_forward<1>(input, weight, borders, new_size, padding_mode, active_flag);
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                       torch::autograd::variable_list grad_output) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto weight = saved[1];
            auto borders = saved[2];
            auto result = shiftnd_backward<1>(grad_output[0], weight, input, borders,
                                              ctx->saved_data["padding_mode"].toInt(),
                                              ctx->saved_data["active_flag"].toBool());
            auto grad_in = result[0];
            auto grad_weight = result[1];
            return {grad_in, grad_weight, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
                
        }
};




class Shift2dFunction : public torch::autograd::Function<Shift2dFunction> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                     torch::Tensor input,
                                     torch::Tensor weight,
                                     torch::Tensor borders,
                                     std::vector<int64_t> new_size,
                                     int64_t padding_mode, bool active_flag){
            ctx->saved_data["padding_mode"] = padding_mode;
            ctx->saved_data["active_flag"] = active_flag;
            ctx->save_for_backward({input, weight, borders});
            return shiftnd_forward<2>(input, weight, borders, new_size, padding_mode, active_flag);
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                       torch::autograd::variable_list grad_output) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto weight = saved[1];
            auto borders = saved[2];
            auto result = shiftnd_backward<2>(grad_output[0], weight, input, borders,
                                              ctx->saved_data["padding_mode"].toInt(),
                                              ctx->saved_data["active_flag"].toBool());
            auto grad_in = result[0];
            auto grad_weight = result[1];
            return {grad_in, grad_weight, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
                
        }
};


class Shift3dFunction : public torch::autograd::Function<Shift3dFunction> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                     torch::Tensor input,
                                     torch::Tensor weight,
                                     torch::Tensor borders,
                                     std::vector<int64_t> new_size,
                                     int64_t padding_mode, bool active_flag){
            ctx->saved_data["padding_mode"] = padding_mode;
            ctx->saved_data["active_flag"] = active_flag;
            ctx->save_for_backward({input, weight, borders});
            return shiftnd_forward<3>(input, weight, borders, new_size, padding_mode, active_flag);
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                       torch::autograd::variable_list grad_output) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto weight = saved[1];
            auto borders = saved[2];
            auto result = shiftnd_backward<3>(grad_output[0], weight, input, borders,
                                              ctx->saved_data["padding_mode"].toInt(),
                                              ctx->saved_data["active_flag"].toBool());
            auto grad_in = result[0];
            auto grad_weight = result[1];
            return {grad_in, grad_weight, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
                
        }
};


std::tuple<torch::Tensor, std::vector<int64_t>> check_borders(const torch::Tensor& input,
                                                              const torch::Tensor& borders,
                                                              const int64_t idim){
    auto sizes = input.sizes();
    const int dim = static_cast<int>(idim);
    const int shift = (((dim + 1) == (int)sizes.size())?1:2);
    const int hdim = 3; // hardcoded for pass no more than, 5D tensor
    const int _dim = std::min(hdim,dim);
    auto dev = input.device();
    torch::Tensor std_borders = torch::empty({hdim*2}, borders.options().dtype(torch::kInt).device(torch::kCPU));
    int* std_borders_data = std_borders.data_ptr<int>();
    for (int i=0 ; i < hdim; ++i){
        std_borders_data[i*2] = 0;
        std_borders_data[i*2+1] = ((i+1)>dim)?1:sizes[i+shift];
    }
    if (borders.numel() != 0){
        auto _borders = borders.to(torch::kInt).to(torch::kCPU);
        int* borders_data = _borders.data_ptr<int>();
        for (int i=0 ; i < _dim; ++i){
            std_borders_data[i*2+1] -= borders_data[i*2+1];
            std_borders_data[i*2] = borders_data[i*2]; 
            if ((std_borders_data[i*2+1] - std_borders_data[i*2]) < 1){
                std_borders_data[i*2+1] = std_borders_data[i*2] + 1;
            }
            if (std_borders_data[i*2] == static_cast<int>(sizes[i+shift])){
                std_borders_data[i*2] = static_cast<int>(sizes[i+shift]) - 1;
                std_borders_data[i*2+1] = std_borders_data[i*2] + 1;
            }
            if (std_borders_data[i*2+1] == 0){
                std_borders_data[i*2] = 0;
                std_borders_data[i*2+1] = 1;
            } 
            std_borders_data[i*2] = std::max(static_cast<int>(0), std_borders_data[i*2]);
            std_borders_data[i*2+1] = std::min(static_cast<int>(sizes[i+shift]), std_borders_data[i*2+1]);
        } 
    }
    std::vector<int64_t> new_sizes(shift+_dim);
    std::copy(sizes.begin(), sizes.begin()+shift, new_sizes.begin()); 
    for (int i=0 ; i < _dim; ++i){ 
        new_sizes[i+shift] = static_cast<int64_t>(std_borders_data[i*2+1] - std_borders_data[i*2]); 
    }
    return  std::make_tuple(std_borders.to(dev), new_sizes);
}


template <int nD = 1>
torch::Tensor shiftnd(const torch::Tensor& input,
                      const torch::Tensor& weights,
                      const torch::Tensor& borders,
                      int64_t padding_mode, bool active_flag){
    
    if (input.is_quantized()) {
        if constexpr(nD == 3){
            auto bands = check_borders(input, borders, 3);
            return q_shift3d_cpu(input, weights, 
                                 std::get<0>(bands).to(torch::kLong), std::get<1>(bands),
                                 padding_mode);
        } else if constexpr(nD == 2){
            auto bands = check_borders(input, borders, 2);
            return q_shift2d_cpu(input, weights,
                                 std::get<0>(bands).to(torch::kLong), std::get<1>(bands),
                                 padding_mode);
        } else {
            auto bands = check_borders(input, borders, 1);
            return q_shift1d_cpu(input, weights, 
                                 std::get<0>(bands).to(torch::kLong), std::get<1>(bands),
                                 padding_mode);
        }
    }
    else {
        if constexpr(nD == 3){
            auto bands = check_borders(input, borders, 3);
            return Shift3dFunction::apply(input, weights, 
                                          std::get<0>(bands), std::get<1>(bands),
                                          padding_mode, active_flag);
        } else if constexpr(nD == 2){
            auto bands = check_borders(input, borders, 2);
            return Shift2dFunction::apply(input, weights,
                                          std::get<0>(bands), std::get<1>(bands), 
                                          padding_mode, active_flag);
        } else {
            auto bands = check_borders(input, borders, 1);
            return Shift1dFunction::apply(input, weights,
                                          std::get<0>(bands), std::get<1>(bands),
                                          padding_mode, active_flag);
        }
    }
}

torch::Tensor shift1d(const torch::Tensor& input,
                      const torch::Tensor& weights,
                      const torch::Tensor& borders,
                      int64_t padding_mode, bool active_flag){
    return shiftnd<1>(input, weights, borders, padding_mode, active_flag);
}

torch::Tensor shift2d(const torch::Tensor& input,
                      const torch::Tensor& weights,
                      const torch::Tensor& borders,
                      int64_t padding_mode, bool active_flag){
    return shiftnd<2>(input, weights, borders, padding_mode, active_flag);
}

torch::Tensor shift3d(const torch::Tensor& input,
                      const torch::Tensor& weights,
                      const torch::Tensor& borders,
                      int64_t padding_mode, bool active_flag){
    return shiftnd<3>(input, weights, borders, padding_mode, active_flag);
}