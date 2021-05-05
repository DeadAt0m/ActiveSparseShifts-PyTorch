new_quant_mapping = {}

from .shifts import Shift1d, Shift2d, Shift3d
import torchshifts.modules.shifts as shifts

new_quant_mapping.update({shifts.Shift1d: Shift1d, shifts.Shift2d: Shift2d, shifts.Shift3d: Shift3d})
