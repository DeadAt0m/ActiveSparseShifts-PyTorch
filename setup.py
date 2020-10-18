MODULE_NAME = 'torchshifts' 
MODULE_VERSION = '2.1'
#DO  NOT CHANGE ON EARLIER STANDARDS PLEASE
#(We use c++17 for using "constexpr" in our code)
STD_VERSION = "c++17"
PYTORCH_VERSION = "1.6"


import sys, os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME
from torch.utils.cpp_extension import CppExtension, CUDAExtension
from torch.cuda import is_available as cuda_available
from pathlib import Path
from setup_utils import check_for_openmp, clean


requirements = [f'torch >= {PYTORCH_VERSION}']


def get_extensions():
    extensions_dir = Path(__file__).resolve().parent / MODULE_NAME / 'csrc'

    sources = extensions_dir.glob('*.cpp')
    sources += (extensions_dir / 'cpu').glob('*.cpp')

    extension = CppExtension

    define_macros = []
    extra_compile_args = {'cxx':[f'-std={STD_VERSION}']}

    parallel_method = ['-DAT_PARALLEL_NATIVE=1']
    if sys.platform == 'win32':
        parallel_method = ['-DAT_PARALLEL_NATIVE_TBB=1']
        extra_compile_args['cxx'].append('/MP')
        define_macros += [('TORCHSHIFTS_EXPORTS', None)]
    if sys.platform == 'linux':
        if check_for_openmp():
            parallel_method = ['-fopenmp','-DAT_PARALLEL_OPENMP=1']
    extra_compile_args['cxx'].extend(parallel_method)

    if (cuda_available() and (CUDA_HOME is not None)) or os.getenv('FORCE_CUDA', '0') == '1':
        print('Building with CUDA')
        extension = CUDAExtension
        sources += (extensions_dir / 'cuda').glob('*.cu')
        define_macros += [('WITH_CUDA', None)]
        extra_compile_args['nvcc'] = [] if os.getenv('NVCC_FLAGS', '') == '' else os.getenv('NVCC_FLAGS', '').split(' ')


    sources = list(map(lambda x: str(x.resolve()), sources))
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
    packages=find_packages(where=MODULE_NAME),
    package_dir={"": MODULE_NAME},
    package_data={ MODULE_NAME:['*.dll', '*.dylib', '*.so'] },
    zip_safe=False,
    install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True),
        'clean': clean,
    }
)
