from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME
from torch.utils.cpp_extension import CppExtension, CUDAExtension
from pathlib import Path
lib_path = "-I"+str(Path(__file__).resolve() / 'src')


p_opt_dict = {'native':'-DAT_PARALLEL_NATIVE=1',
              'openmp':'-DAT_PARALLEL_OPENMP=1',
              'windows':'-DAT_PARALLEL_NATIVE_TBB=1',
               None:''}

#DO CHANGE ON EARLIER STANDARDS PLEASE
#(We use c++17 for using "constexpr" in our code)
STDversion = "c++17"
# assert int(STDversion.strip('c++')) >= 17, "DO CHANGE ON EARLIER STANDARDS PLEASE"

p_method='openmp'
print("We use openmp for parallelization on CPU, look inside setup.py to change it if needed")

modules = [
    CppExtension('torchshifts.shifts_cpu',
                 ['src/cpu/shifts_cpu.cpp'],
                 extra_compile_args=[f'-std={STDversion}','-fopenmp',p_opt_dict[p_method], lib_path]),
]

# If nvcc is available, add the CUDA extension
print(f'Building with{"" if CUDA_HOME else "out"} CUDA')

if CUDA_HOME:
    modules.append(
        CUDAExtension('torchshifts.shifts_cuda',
                      ['src/cuda/shifts_kernels.cu',
                       'src/cuda/shifts_cuda.cpp'])
    )

setup(
    name='torchshifts',
    version='2.0',
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
