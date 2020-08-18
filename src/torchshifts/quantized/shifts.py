import torch
import torchshifts.shifts as shifts
import math

#  import the CPU shifts
SHIFTS_CPU_AVAILABLE = True
try:
    from torchshifts.cpu import shift1d_cpu, shift2d_cpu
except ImportError:
    SHIFTS_CPU_AVAILABLE = False
    
def shift1d_quantized(input, weight, padding_mode):
    if not input.is_quantized:
        raise ValueError("Input to 'shift1d_quantized' must be quantized!")
    return shift1d_cpu(input, weight, padding_mode, False)
    
def shift2d_quantized(input, weight, padding_mode):
    if not input.is_quantized:
        raise ValueError("Input to 'shift2d_quantized' must be quantized!")
    return shift2d_cpu(input, weight, padding_mode, False)
    

def quantize_shift_weights(weight):
    scale = math.ceil((weight.max().item() - weight.min().item()) / 255.)
    return torch.quantize_per_tensor(weight, scale, 128, torch.quint8)

class Shift2D(shifts.Shift2D):
    def __init__(self, in_channels, padding='zeros'):
        super(Shift2D, self).__init__(in_channels, padding, 1, 0, False)
        self.qweight = quantize_shift_weights(self.weight.float())

    def forward(self, input):
        return shift2d_quantized(input, self.qweight, self.padding)

    def _get_name(self):
        return 'QuantizedShift2D'

    @staticmethod
    def from_float(mod):
        padding_dict = {0:'zeros', 1:'border', 2:'reflect', 3:'symmetric'}
        qshift = Shift2D(mod.in_channels, padding_dict[mod.padding])
        qshift.weight = mod.weight
        qshift.qweight = quantize_shifts_weights(mod.weight.float())
        return qshift
    
class Shift1D(shifts.Shift1D):
    def __init__(self, in_channels, padding='zeros'):
        super(Shift1D, self).__init__(in_channels, padding, 1, 0, False)
        self.qweight = quantize_shift_weights(self.weight.float())

    def forward(self, input):
        return shift1d_quantized(input, self.qweight, self.padding)

    def _get_name(self):
        return 'QuantizedShift1D'

    @staticmethod
    def from_float(mod):
        padding_dict = {0:'zeros', 1:'border', 2:'reflect', 3:'symmetric'}
        qshift = Shift1D(mod.in_channels, padding_dict[mod.padding])
        qshift.weight = mod.weight
        qshift.qweight = quantize_shifts_weights(mod.weight.float())
        return qshift

quant_mapping = torch.quantization.default_mappings.DEFAULT_MODULE_MAPPING
quant_mapping.update({shifts.Shift1D: Shift1D, shifts.Shift2D: Shift2D})