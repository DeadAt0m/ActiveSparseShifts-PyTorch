import torch
import shifts 
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
    

class Shift2D(shifts.Shift2D):
    def __init__(self, in_channels, padding='zeros'):
        super(Shift2D, self).__init__(in_channels, padding, 1, 0, False)

    def forward(self, input):
        return shift2d_quantized(input, self.weight, self.padding_mode)

    def _get_name(self):
        return 'QuantizedShift2D'

    @staticmethod
    def from_float(mod):
        padding_dict = {0:'zeros':0, 1:'border', 2:'reflect', 3:'symmetric'}
        qshift = Shift2D(mod.in_channels, padding_dict[mod.padding])
        weight = mod.weight.float()
        scale = math.ceil((weight.max().item() - weight.min().item()) / 255.)
        qshift.weight = torch.quantize_per_tensor(weight, scale, 0, torch.qint8) 
        return qshift
 

class Shift1D(shifts.Shift1D):
    def __init__(self, in_channels, padding='zeros'):
        super(Shift1D, self).__init__(in_channels, padding, 1, 0, False)

    def forward(self, input):
        return shift1d_quantized(input, self.weight, self.padding_mode)

    def _get_name(self):
        return 'QuantizedShift1D'

    @staticmethod
    def from_float(mod):
        padding_dict = {0:'zeros':0, 1:'border', 2:'reflect', 3:'symmetric'}
        qshift = Shift1D(mod.in_channels, padding_dict[mod.padding])
        weight = mod.weight.float()
        scale = math.ceil((weight.max().item() - weight.min().item()) / 255.)
        qshift.weight = torch.quantize_per_tensor(weight, scale, 0, torch.qint8) 
        return qshift
