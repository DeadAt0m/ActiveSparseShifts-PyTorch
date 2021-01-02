import torch
import math
import copy
import torchshifts.modules.shifts as shifts
from torchshifts.quantized.functional import shift1d_quantized, shift2d_quantized, shift3d_quantized


rp_dict = {v: k for k, v in shifts.paddings_dict.items()}

def quantize_shift_weights(weight):
    scale = math.ceil((weight.max().item() - weight.min().item()) / 255.)
    return torch.quantize_per_tensor(weight, scale, 128, torch.quint8)

class Shift1d(shifts.Shift1d):
    def __init__(self, in_channels, padding='zeros'):
        super(Shift1d, self).__init__(in_channels, padding, 1, 0, False)
        self.qweight = quantize_shift_weights(self.weight.float())

    def forward(self, input):
        return self._reduction_fn(shift1d_quantized(input, self.qweight, self.padding, self.cut_borders))

    def _get_name(self):
        return 'QuantizedShift1D'

    @staticmethod
    def from_float(mod):
        qshift = Shift1d(mod.in_channels, rp_dict[mod.padding])
        qshift.cut_borders = mod.cut_borders
        qshift._reduction_fn = mod._reduction_fn
        qshift.weight = mod.weight
        qshift.qweight = quantize_shift_weights(mod.weight.float())
        return qshift


class Shift2d(shifts.Shift2d):
    def __init__(self, in_channels, padding='zeros'):
        super(Shift2d, self).__init__(in_channels, padding, 1, 0, False)
        self.qweight = quantize_shift_weights(self.weight.float())

    def forward(self, input):
        return self._reduction_fn(shift2d_quantized(input, self.qweight, self.padding, self.cut_borders))

    def _get_name(self):
        return 'QuantizedShift2D'

    @staticmethod
    def from_float(mod):
        qshift = Shift2d(mod.in_channels, rp_dict[mod.padding])
        qshift.cut_borders = mod.cut_borders
        qshift._reduction_fn = mod._reduction_fn
        qshift.weight = mod.weight
        qshift.qweight = quantize_shift_weights(mod.weight.float())
        return qshift
    
class Shift3d(shifts.Shift3d):
    def __init__(self, in_channels, padding='zeros'):
        super(Shift3d, self).__init__(in_channels, padding, 1, 0, False)
        self.qweight = quantize_shift_weights(self.weight.float())

    def forward(self, input):
        return self._reduction_fn(shift3d_quantized(input, self.qweight, self.padding, self.cut_borders))

    def _get_name(self):
        return 'QuantizedShift3D'

    @staticmethod
    def from_float(mod):
        qshift = Shift3d(mod.in_channels, rp_dict[mod.padding])
        qshift.cut_borders = mod.cut_borders
        qshift._reduction_fn = mod._reduction_fn
        qshift.weight = mod.weight
        qshift.qweight = quantize_shift_weights(mod.weight.float())
        return qshift
    

quant_mapping = copy.deepcopy(torch.quantization.quantization_mappings.get_static_quant_module_mappings())
quant_mapping.update({shifts.Shift1d: Shift1d, shifts.Shift2d: Shift2d, shifts.Shift3d: Shift3d})
