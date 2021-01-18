MODULE_NAME = 'torchshifts' 
MODULE_VERSION = '3.0'
#DO  NOT CHANGE ON EARLIER STANDARDS PLEASE
#(We use c++17 for using "constexpr" in our code)
STD_VERSION = "c++17"
PYTORCH_VERSION = "1.7"


import sys, os, copy
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME
from torch.utils.cpp_extension import CppExtension, CUDAExtension
from torch.cuda import is_available as cuda_available
from torch import version as torch_version
from pathlib import Path
import torch
from setup_utils import check_for_openmp, clean
import subprocess
cwd = Path.cwd()
torch_ver = torch.__version__
torch_ver = torch_ver.split('+')[0] if '+' in torch_ver else torch_ver


requirements = [f'torch >= {PYTORCH_VERSION}']

if torch_ver < '1.8':
    from torch_patch import  patch_torch_infer_schema_h
    __SUCC =  patch_torch_infer_schema_h()
    if not __SUCC:
        print('Something went wrong during patching! The CUDA build have chance to fail!')


#cuda
cuda_avail =  (cuda_available() and (CUDA_HOME is not None)) or os.getenv('FORCE_CUDA', '0') == '1'
if cuda_avail:
    cu_ver = ''
    if CUDA_HOME is not None:
        cu_ver = Path(CUDA_HOME).resolve().name.strip('cuda-')
    elif cuda_available():
        cu_ver = torch_version.cuda
    if cu_ver:
        cu_ver = '+cu' + cu_ver
    cu_ver = cu_ver.replace('.','')
    MODULE_VERSION += cu_ver


version = copy.copy(MODULE_VERSION)
sha = 'Unknown'
try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(cwd)).decode('ascii').strip()
except Exception:
    pass
if sha != 'Unknown':
    version += '+' + sha[:7]
print(f'Building wheel {MODULE_NAME}-{version}')

version_path = cwd / MODULE_NAME / 'version.py'
if version_path.exists():
    version_path.unlink()
version_path.touch()
version_path = version_path.open("a")
version_path.write(f"__version__ = '{version}'\n")
version_path.write(f"git_version = {repr(sha)}\n")
version_path.write(f"from {MODULE_NAME}.extension import _check_cuda_version\n")
version_path.write("if _check_cuda_version() > 0:\n")
version_path.write("    cuda = _check_cuda_version()\n")
version_path.close()

def get_extensions():
    extensions_dir = cwd / MODULE_NAME / 'csrc'

    sources = list(extensions_dir.glob('*.cpp'))
    sources += list((extensions_dir / 'ops').glob('*.cpp'))
    sources += list((extensions_dir / 'ops' / 'autograd').glob('*.cpp'))
    sources += list((extensions_dir / 'ops' / 'cpu').glob('*.cpp'))
    sources += list((extensions_dir / 'ops' / 'quantized').glob('*.cpp'))
    
    extension = CppExtension

    define_macros = []
    extra_compile_args = {'cxx':[f'-std={STD_VERSION}', '-O3']}

    parallel_method = ['-DAT_PARALLEL_NATIVE=1']
    if sys.platform == 'win32':
        parallel_method = ['-DAT_PARALLEL_NATIVE_TBB=1']
        extra_compile_args['cxx'].append('/MP')
        define_macros += [('TORCHSHIFTS_EXPORTS', None)]
    if sys.platform == 'linux':
        extra_compile_args['cxx'].append('-Wno-unused-but-set-variable')
        extra_compile_args['cxx'].append('-Wno-unused-variable')
        if check_for_openmp():
            parallel_method = ['-fopenmp','-DAT_PARALLEL_OPENMP=1']
    extra_compile_args['cxx'].extend(parallel_method)

    if cuda_avail:
        print('Building with CUDA')
        extension = CUDAExtension
        sources += list((extensions_dir / 'ops' / 'cuda').glob('*.cu'))
        define_macros += [('WITH_CUDA', None)]
        extra_compile_args['nvcc'] = ['-O3', '-DNDEBUG', '--expt-extended-lambda']
        if os.getenv('NVCC_FLAGS', '') != '':
            extra_compile_args['nvcc'].extend(os.getenv('NVCC_FLAGS', '').split(' '))
  
    if torch_ver >= '1.8':
        define_macros += [('TORCH1.8', None)]

    sources = list(set(map(lambda x: str(x.resolve()), sources)))
    include_dirs = [str(extensions_dir)]
    ext_modules = [
        extension(
            f'{MODULE_NAME}._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    # Metadata
    name=MODULE_NAME,
    version=MODULE_VERSION,
    description='Implementation of Sparse Active Shift https://arxiv.org/pdf/1903.05285.pdf for PyTorch',
    keywords=['shifts','activeshifts', 'shiftspytorch'],
    author='Ignatii Dubyshkin aka DeadAt0m',
    author_email='kheldi@yandex.ru',
    url='https://github.com/DeadAt0m/ActiveSparseShifts-PyTorch',
    license='BSD',

    # Package info
    packages=find_packages(),
    package_dir={MODULE_NAME : MODULE_NAME},
    package_data={ MODULE_NAME:['*.dll', '*.dylib', '*.so'] },
    zip_safe=False,
    install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True),
        'clean': clean,
    }
)
