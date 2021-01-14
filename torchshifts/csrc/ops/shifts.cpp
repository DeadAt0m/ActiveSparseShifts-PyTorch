#include "shifts.h"

#include <torch/types.h>

namespace shifts {
namespace ops {
    
namespace detail {
  
torch::Tensor _shift1d_forward(const torch::Tensor& input,
                               const torch::Tensor& weights,
                               const torch::Tensor& borders,
                               const std::vector<int64_t>& new_size,
                               int64_t padding_mode,
                               bool active_flag){
    static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchshifts::_shift1d_forward", "")
          .typed<decltype(_shift1d_forward)>();
    return op.call(input, weights, borders, new_size, padding_mode, active_flag);
}   
    
std::tuple<torch::Tensor, torch::Tensor> _shift1d_backward(const torch::Tensor& grad,
                                                           const torch::Tensor& weights,
                                                           const torch::Tensor& input,
                                                           const torch::Tensor& borders,
                                                           int64_t padding_mode,
                                                           bool active_flag){
    static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchshifts::_shift1d_backward", "")
          .typed<decltype(_shift1d_backward)>();
    return op.call(grad, weights, input, borders, padding_mode, active_flag);
}     
    
    
torch::Tensor _shift2d_forward(const torch::Tensor& input,
                               const torch::Tensor& weights,
                               const torch::Tensor& borders,
                               const std::vector<int64_t>& new_size,
                               int64_t padding_mode,
                               bool active_flag){
    static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchshifts::_shift2d_forward", "")
          .typed<decltype(_shift2d_forward)>();
    return op.call(input, weights,  borders, new_size, padding_mode, active_flag);
}   
    
std::tuple<torch::Tensor, torch::Tensor> _shift2d_backward(const torch::Tensor& grad,
                                                           const torch::Tensor& weights,
                                                           const torch::Tensor& input,
                                                           const torch::Tensor& borders,
                                                           int64_t padding_mode,
                                                           bool active_flag){
    static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchshifts::_shift2d_backward", "")
          .typed<decltype(_shift2d_backward)>();
    return op.call(grad, weights, input, borders, padding_mode, active_flag);
}        
    
    
torch::Tensor _shift3d_forward(const torch::Tensor& input,
                               const torch::Tensor& weights,
                               const torch::Tensor& borders,
                               const std::vector<int64_t>& new_size,
                               int64_t padding_mode,
                               bool active_flag){
    static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchshifts::_shift3d_forward", "")
          .typed<decltype(_shift3d_forward)>();
    return op.call(input, weights,  borders, new_size, padding_mode, active_flag);
}   
    
std::tuple<torch::Tensor, torch::Tensor> _shift3d_backward(const torch::Tensor& grad,
                                                           const torch::Tensor& weights,
                                                           const torch::Tensor& input,
                                                           const torch::Tensor& borders,
                                                           int64_t padding_mode,
                                                           bool active_flag){
    static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchshifts::_shift3d_backward", "")
          .typed<decltype(_shift3d_backward)>();
    return op.call(grad, weights, input, borders, padding_mode, active_flag);
}            

} // namespace detail
 
using namespace torch::indexing;
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

    
torch::Tensor shift1d(const torch::Tensor& input,
                      const torch::Tensor& weights,
                      const torch::Tensor& borders,
                      int64_t padding_mode, bool active_flag){
    auto bands = check_borders(input, borders, 1);
    auto _borders  = std::get<0>(bands);
    auto new_size = std::get<1>(bands);
    return detail::_shift1d_forward(input, weights, _borders, new_size, padding_mode, active_flag);
}
    
torch::Tensor shift2d(const torch::Tensor& input,
                      const torch::Tensor& weights,
                      const torch::Tensor& borders,
                      int64_t padding_mode, bool active_flag){
    auto bands = check_borders(input, borders, 2);
    auto _borders  = std::get<0>(bands);
    auto new_size = std::get<1>(bands);
    return detail::_shift2d_forward(input, weights, _borders, new_size, padding_mode, active_flag);
}

torch::Tensor shift3d(const torch::Tensor& input,
                      const torch::Tensor& weights,
                      const torch::Tensor& borders,
                      int64_t padding_mode, bool active_flag){
    auto bands = check_borders(input, borders, 3);
    auto _borders  = std::get<0>(bands);
    auto new_size = std::get<1>(bands);
    return detail::_shift3d_forward(input, weights, _borders, new_size, padding_mode, active_flag);
}

TS_TORCH_LIBRARY_FRAGMENT(torchshifts, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchshifts::_shift1d_forward(Tensor input, Tensor weights, Tensor borders, int[] new_size, int padding_mode, bool active_flag) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchshifts::_shift1d_backward(Tensor grad, Tensor weights, Tensor input, Tensor borders, int padding_mode, bool active_flag) -> (Tensor, Tensor)"));
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchshifts::_shift2d_forward(Tensor input, Tensor weights, Tensor borders, int[] new_size, int padding_mode, bool active_flag) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchshifts::_shift2d_backward(Tensor grad, Tensor weights, Tensor input, Tensor borders, int padding_mode, bool active_flag) -> (Tensor, Tensor)"));
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchshifts::_shift3d_forward(Tensor input, Tensor weights, Tensor borders, int[] new_size, int padding_mode, bool active_flag) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchshifts::_shift3d_backward(Tensor grad, Tensor weights, Tensor input, Tensor borders, int padding_mode, bool active_flag) -> (Tensor, Tensor)"));
}    
    
    
} // namespace ops
} // namespace shifts
