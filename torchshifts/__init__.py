import warnings
from pathlib import path
from .extension import _HAS_OPS


# from .modules import *


try:
    from .version import __version__
except ImportError:
    pass

# Check if torchshifts is being imported within the root folder
if (not _HAS_OPS and Path(__file__).parent.resolve() == (Path.cwd() / 'torchshifts')):
    message = (f'You are importing torchshifts within its own root folder ({Path.cwd()}). '
               'This is not expected to work and may give errors. Please exit the '
               'torchshifts project source and relaunch your python interpreter.')
    warnings.warn(message)
