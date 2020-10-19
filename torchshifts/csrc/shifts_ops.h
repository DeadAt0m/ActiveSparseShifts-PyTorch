#pragma once
#include "cpu/shifts_cpu.h"
#ifdef WITH_CUDA
    #include "cuda/shifts_cuda.h"
#endif


template<int nD = 1>
torch::Tensor shiftnd_forward(const torch::Tensor& input,
                              const torch::Tensor& weights,
                              int64_t padding_mode,
                              bool active_flag){
    if (input.is_cuda()) {
        #ifdef WITH_CUDA
            if constexpr(nD == 3){
                return shift3d_forward_cuda(input, weights, padding_mode, active_flag);

            } else if constexpr(nD == 2){
                return shift2d_forward_cuda(input, weights, padding_mode, active_flag);
            } else {
                return shift1d_forward_cuda(input, weights, padding_mode, active_flag);
            }
        #else
            TORCH_CHECK(false, "Not compiled with GPU support");
        #endif
    }      
    if constexpr(nD == 3){
        return shift3d_forward_cpu(input, weights, padding_mode, active_flag);
    } else if constexpr(nD == 2){
        return shift2d_forward_cpu(input, weights, padding_mode, active_flag);
    } else {
        return shift1d_forward_cpu(input, weights, padding_mode, active_flag);
    }
}

template<int nD = 1>
std::vector<torch::Tensor> shiftnd_backward(const torch::Tensor& grad,
                                            const torch::Tensor& weights,
                                            const torch::Tensor& input,
                                            int64_t padding_mode,
                                            bool active_flag){
    if (input.is_cuda()) {
        #ifdef WITH_CUDA
            if constexpr(nD == 3){
                return shift3d_backward_cuda(grad, weights, input, padding_mode, active_flag);

            } else if constexpr(nD == 2){
                return shift2d_backward_cuda(grad, weights, input, padding_mode, active_flag);
            } else {
                return shift1d_backward_cuda(grad, weights, input, padding_mode, active_flag);
            }
        #else
            TORCH_CHECK(false, "Not compiled with GPU support");
        #endif
    }      
    if constexpr(nD == 3){
        return shift3d_backward_cpu(grad, weights, input, padding_mode, active_flag);
    } else if constexpr(nD == 2){
        return shift2d_backward_cpu(grad, weights, input, padding_mode, active_flag);
    } else {
        return shift1d_backward_cpu(grad, weights, input, padding_mode, active_flag);
    }
}



class Shift1dFunction : public torch::autograd::Function<Shift1dFunction> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                     torch::Tensor input,
                                     torch::Tensor weight,
                                     int64_t padding_mode, bool active_flag){
            ctx->saved_data["padding_mode"] = padding_mode;
            ctx->saved_data["active_flag"] = active_flag;
            ctx->save_for_backward({input, weight});
            return shiftnd_forward<1>(input, weight, padding_mode, active_flag);
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                       torch::autograd::variable_list grad_output) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto weight = saved[1];
            auto result = shiftnd_backward<1>(grad_output[0], weight, input,
                                              ctx->saved_data["padding_mode"].toInt(),
                                              ctx->saved_data["active_flag"].toBool());
            auto grad_in = result[0];
            auto grad_weight = result[1];
            return {grad_in, grad_weight, torch::Tensor(), torch::Tensor()};
                
        }
};




class Shift2dFunction : public torch::autograd::Function<Shift2dFunction> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                     torch::Tensor input,
                                     torch::Tensor weight,
                                     int64_t padding_mode, bool active_flag){
            ctx->saved_data["padding_mode"] = padding_mode;
            ctx->saved_data["active_flag"] = active_flag;
            ctx->save_for_backward({input, weight});
            return shiftnd_forward<2>(input, weight, padding_mode, active_flag);
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                       torch::autograd::variable_list grad_output) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto weight = saved[1];
            auto result = shiftnd_backward<2>(grad_output[0], weight, input,
                                              ctx->saved_data["padding_mode"].toInt(),
                                              ctx->saved_data["active_flag"].toBool());
            auto grad_in = result[0];
            auto grad_weight = result[1];
            return {grad_in, grad_weight, torch::Tensor(), torch::Tensor()};
                
        }
};


class Shift3dFunction : public torch::autograd::Function<Shift3dFunction> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                     torch::Tensor input,
                                     torch::Tensor weight,
                                     int64_t padding_mode, bool active_flag){
            ctx->saved_data["padding_mode"] = padding_mode;
            ctx->saved_data["active_flag"] = active_flag;
            ctx->save_for_backward({input, weight});
            return shiftnd_forward<3>(input, weight, padding_mode, active_flag);
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                       torch::autograd::variable_list grad_output) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto weight = saved[1];
            auto result = shiftnd_backward<3>(grad_output[0], weight, input,
                                              ctx->saved_data["padding_mode"].toInt(),
                                              ctx->saved_data["active_flag"].toBool());
            auto grad_in = result[0];
            auto grad_weight = result[1];
            return {grad_in, grad_weight, torch::Tensor(), torch::Tensor()};
                
        }
};


template <int nD = 1>
torch::Tensor shiftnd(const torch::Tensor& input,
                      const torch::Tensor& weights,
                      int64_t padding_mode, bool active_flag){
    if (input.is_quantized()) {
        if constexpr(nD == 3){
            return q_shift3d_cpu(input, weights, padding_mode);
        } else if constexpr(nD == 2){
            return q_shift2d_cpu(input, weights, padding_mode);
        } else {
            return q_shift1d_cpu(input, weights, padding_mode);
        }
    }
    else {
        if constexpr(nD == 3){
            return Shift3dFunction::apply(input, weights, padding_mode, active_flag);
        } else if constexpr(nD == 2){
            return Shift2dFunction::apply(input, weights, padding_mode, active_flag);
        } else {
            return Shift1dFunction::apply(input, weights, padding_mode, active_flag);
        }
    }
}

torch::Tensor shift1d(const torch::Tensor& input,
                      const torch::Tensor& weights,
                      int64_t padding_mode, bool active_flag){
    return shiftnd<1>(input, weights, padding_mode, active_flag);
}

torch::Tensor shift2d(const torch::Tensor& input,
                      const torch::Tensor& weights,
                      int64_t padding_mode, bool active_flag){
    return shiftnd<2>(input, weights, padding_mode, active_flag);
}

torch::Tensor shift3d(const torch::Tensor& input,
                      const torch::Tensor& weights,
                      int64_t padding_mode, bool active_flag){
    return shiftnd<3>(input, weights, padding_mode, active_flag);
}