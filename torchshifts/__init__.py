import warnings
from pathlib import Path
from .extension import _HAS_OPS

try:
    from .version import __version__
except ImportError:
    pass

# Check if torchshifts is being imported within the root folder
if (not _HAS_OPS and Path(__file__).parent.resolve() == (Path.cwd() / 'torchshifts')):
    message = (f'You are importing torchshifts within its own root folder ({Path.cwd() / "torchshifts"}). '
               'This is not expected to work and may give errors. Please exit the '
               'torchshifts project source and relaunch your python interpreter.')
    warnings.warn(message)

from torchshifts.modules import Shift1d, Shift2d, Shift3d
from torchshifts.quantized import quant_mapping