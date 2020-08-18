from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME
from torch.utils.cpp_extension import CppExtension, CUDAExtension

p_opt_dict = {'native':'-DAT_PARALLEL_NATIVE=1',
              'openmp':'-DAT_PARALLEL_OPENMP=1',
              'windows':'-DAT_PARALLEL_NATIVE_TBB=1',
               None:''}


p_method='openmp'
print("We use openmp for parallelization on CPU, look inside setup.py to change it if needed")

modules = [
    CppExtension('torchshifts.cpu',
                 ['src/cpu/shifts_cpu.cpp'],
                 extra_compile_args=['-fopenmp',p_opt_dict[p_method]]),
]

# If nvcc is available, add the CUDA extension
if CUDA_HOME:
    modules.append(
        CUDAExtension('torchshifts.cuda',
                      ['src/cuda/shifts_cuda.cpp',
                       'src/cuda/shifts_cuda_kernel.cu'])
    )
print(f'Building with{"" if CUDA_HOME else "out"} CUDA')

setup(
    name='torchshifts',
    version='1.2',
    description='Implementation of Sparse Active Shift https://arxiv.org/pdf/1903.05285.pdf for PyTorch',
    keywords=['shifts','activeshifts', 'shiftspytorch'],
    author='Ignatii Dubyshkin',
    author_email='kheldi@yandex.ru',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    ext_modules=modules,
    cmdclass={
        'build_ext': BuildExtension
    })
