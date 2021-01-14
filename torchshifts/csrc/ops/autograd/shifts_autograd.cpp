#ifndef _SHIFTS_CPU
#define _SHIFTS_CPU

#include "../shifts.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace shifts {
namespace ops {

namespace {


class Shift1dFunction : public torch::autograd::Function<Shift1dFunction> {
    public:
        static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                                      const torch::Tensor& input,
                                                      const torch::Tensor& weight,
                                                      const torch::Tensor& borders,
                                                      const std::vector<int64_t>& new_size,
                                                      int64_t padding_mode, bool active_flag){
            at::AutoNonVariableTypeMode g;
            auto output = detail::_shift1d_forward(input, weight, borders, new_size, padding_mode, active_flag);
            ctx->saved_data["padding_mode"] = padding_mode;
            ctx->saved_data["active_flag"] = active_flag;
            ctx->save_for_backward({input, weight, borders});
            return {output};
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                       const torch::autograd::variable_list& grad_output) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto weight = saved[1];
            auto borders = saved[2];
            auto padding_mode = ctx->saved_data["padding_mode"].toInt();
            auto active_flag = ctx->saved_data["active_flag"].toBool();
            
            auto result = detail::_shift1d_backward(grad_output[0], weight, input, borders,
                                                    padding_mode, active_flag);
            auto grad_input =  std::get<0>(result);
            auto grad_weight =  std::get<1>(result);
            return {grad_input, grad_weight, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
                
        }
};

// Hack for backward working during dispatch
class Shift1dBackwardFunction: public torch::autograd::Function<Shift1dBackwardFunction> {
    public:
        static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                                      const torch::Tensor& grad,
                                                      const torch::Tensor& weights,
                                                      const torch::Tensor& input,
                                                      const torch::Tensor& borders,
                                                      int64_t padding_mode,
                                                      bool active_flag) {
            at::AutoNonVariableTypeMode g;
            auto result = detail::_shift1d_backward(grad, weights, input, borders,
                                                    padding_mode, active_flag);
            auto grad_input =  std::get<0>(result);
            auto grad_weight =  std::get<1>(result);

            return { grad_input,  grad_weight };
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                      const torch::autograd::variable_list& grad_output) {
            TORCH_CHECK(0, "double backwards on shift1d not supported");
        }
};    
    
    
    
    
class Shift2dFunction : public torch::autograd::Function<Shift2dFunction> {
    public:
        static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                                      const torch::Tensor& input,
                                                      const torch::Tensor& weight,
                                                      const torch::Tensor& borders,
                                                      const std::vector<int64_t>& new_size,
                                                      int64_t padding_mode, bool active_flag){
            at::AutoNonVariableTypeMode g;
            auto output = detail::_shift2d_forward(input, weight, borders, new_size, padding_mode, active_flag);
            ctx->saved_data["padding_mode"] = padding_mode;
            ctx->saved_data["active_flag"] = active_flag;
            ctx->save_for_backward({input, weight, borders});
            return {output};
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                       const torch::autograd::variable_list& grad_output) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto weight = saved[1];
            auto borders = saved[2];
            auto padding_mode = ctx->saved_data["padding_mode"].toInt();
            auto active_flag = ctx->saved_data["active_flag"].toBool();
            
            auto result = detail::_shift2d_backward(grad_output[0], weight, input, borders,
                                                    padding_mode, active_flag);
            auto grad_input =  std::get<0>(result);
            auto grad_weight =  std::get<1>(result);
            return {grad_input, grad_weight, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
                
        }
};


// Hack for backward working during dispatch
class Shift2dBackwardFunction: public torch::autograd::Function<Shift2dBackwardFunction> {
    public:
        static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                                      const torch::Tensor& grad,
                                                      const torch::Tensor& weights,
                                                      const torch::Tensor& input,
                                                      const torch::Tensor& borders,
                                                      int64_t padding_mode,
                                                      bool active_flag) {
            at::AutoNonVariableTypeMode g;
            auto result = detail::_shift2d_backward(grad, weights, input, borders,
                                                    padding_mode, active_flag);
            auto grad_input =  std::get<0>(result);
            auto grad_weight =  std::get<1>(result);

            return { grad_input,  grad_weight };
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                      const torch::autograd::variable_list& grad_output) {
            TORCH_CHECK(0, "double backwards on shift2d not supported");
        }
};    
    
    
    
    
class Shift3dFunction : public torch::autograd::Function<Shift3dFunction> {
    public:
        static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                                      const torch::Tensor& input,
                                                      const torch::Tensor& weight,
                                                      const torch::Tensor& borders,
                                                      const std::vector<int64_t>& new_size,
                                                      int64_t padding_mode, bool active_flag){
            at::AutoNonVariableTypeMode g;
            auto output = detail::_shift3d_forward(input, weight, borders, new_size, padding_mode, active_flag);
            ctx->saved_data["padding_mode"] = padding_mode;
            ctx->saved_data["active_flag"] = active_flag;
            ctx->save_for_backward({input, weight, borders});
            return { output };
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                       const torch::autograd::variable_list& grad_output) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto weight = saved[1];
            auto borders = saved[2];
            auto padding_mode = ctx->saved_data["padding_mode"].toInt();
            auto active_flag = ctx->saved_data["active_flag"].toBool();
            
            auto result = detail::_shift3d_backward(grad_output[0], weight, input, borders,
                                                    padding_mode, active_flag);
            auto grad_in =  std::get<0>(result);
            auto grad_weight =  std::get<1>(result);
            return {grad_in, grad_weight, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
                
        }
};

    
// Hack for backward working during dispatch
class Shift3dBackwardFunction: public torch::autograd::Function<Shift3dBackwardFunction> {
    public:
        static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                                      const torch::Tensor& grad,
                                                      const torch::Tensor& weights,
                                                      const torch::Tensor& input,
                                                      const torch::Tensor& borders,
                                                      int64_t padding_mode,
                                                      bool active_flag) {
            at::AutoNonVariableTypeMode g;
            auto result = detail::_shift3d_backward(grad, weights, input, borders,
                                                    padding_mode, active_flag);
            auto grad_input =  std::get<0>(result);
            auto grad_weight =  std::get<1>(result);

            return { grad_input,  grad_weight };
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                      const torch::autograd::variable_list& grad_output) {
            TORCH_CHECK(0, "double backwards on shift3d not supported");
        }
};        
    

torch::Tensor shift1d_autograd(const torch::Tensor& input,
                               const torch::Tensor& weights,
                               const torch::Tensor& borders,
                               const std::vector<int64_t>& new_size,
                               int64_t padding_mode,
                               bool active_flag){
    return Shift1dFunction::apply(input, weights, borders, new_size, padding_mode, active_flag)[0];
}
    
torch::Tensor shift2d_autograd(const torch::Tensor& input,
                               const torch::Tensor& weights,
                               const torch::Tensor& borders,
                               const std::vector<int64_t>& new_size,
                               int64_t padding_mode,
                               bool active_flag){
    return Shift2dFunction::apply(input, weights, borders, new_size, padding_mode, active_flag)[0];
}    
    
torch::Tensor shift3d_autograd(const torch::Tensor& input,
                               const torch::Tensor& weights,
                               const torch::Tensor& borders,
                               const std::vector<int64_t>& new_size,
                               int64_t padding_mode,
                               bool active_flag){
    return Shift3dFunction::apply(input, weights, borders, new_size, padding_mode, active_flag)[0];
}    
    
std::tuple<torch::Tensor, torch::Tensor> shift1d_autograd_backward(const torch::Tensor& grad,
                                                                   const torch::Tensor& weights,
                                                                   const torch::Tensor& input,
                                                                   const torch::Tensor& borders,
                                                                   int64_t padding_mode,
                                                                   bool active_flag){
    auto result = Shift1dBackwardFunction::apply(grad, weights, input, borders, padding_mode, active_flag);
    return std::make_tuple(result[0], result[1]);
} 
    
std::tuple<torch::Tensor, torch::Tensor> shift2d_autograd_backward(const torch::Tensor& grad,
                                                                   const torch::Tensor& weights,
                                                                   const torch::Tensor& input,
                                                                   const torch::Tensor& borders,
                                                                   int64_t padding_mode,
                                                                   bool active_flag){
    auto result = Shift2dBackwardFunction::apply(grad, weights, input, borders, padding_mode, active_flag);
    return std::make_tuple(result[0], result[1]);
}  
    
std::tuple<torch::Tensor, torch::Tensor> shift3d_autograd_backward(const torch::Tensor& grad,
                                                                   const torch::Tensor& weights,
                                                                   const torch::Tensor& input,
                                                                   const torch::Tensor& borders,
                                                                   int64_t padding_mode,
                                                                   bool active_flag){
    auto result = Shift3dBackwardFunction::apply(grad, weights, input, borders, padding_mode, active_flag);
    return std::make_tuple(result[0], result[1]);
}  
    

       
} // end of anonymous namespace

TORCH_LIBRARY_IMPL(torchshifts, Autograd, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift1d_forward"),
        TORCH_FN(shift1d_autograd));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift1d_backward"),
        TORCH_FN(shift1d_autograd_backward));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift2d_forward"),
        TORCH_FN(shift2d_autograd));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift2d_backward"),
        TORCH_FN(shift2d_autograd_backward));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift3d_forward"),
        TORCH_FN(shift3d_autograd));
    m.impl(
        TORCH_SELECTIVE_NAME("torchshifts::_shift3d_backward"),
        TORCH_FN(shift3d_autograd_backward));
}

} // namespace ops
} // namespace shifts

#endif