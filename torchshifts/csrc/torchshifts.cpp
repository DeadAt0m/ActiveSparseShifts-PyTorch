#include "torchshifts.h"

#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/library.h>

#include "ops/ops.h"

#ifdef WITH_CUDA
    #include <cuda.h>
#endif



#ifdef _WIN32
    #if PY_MAJOR_VERSION < 3
        PyMODINIT_FUNC init_C(void) {return NULL;}
    #else
        PyMODINIT_FUNC PyInit__C(void) {return NULL;}
    #endif
#endif

namespace shifts {
    
    int64_t cuda_version() {
        #ifdef WITH_CUDA
            return CUDA_VERSION;
        #else
            return -1;
        #endif
    }

}

TS_TORCH_LIBRARY_FRAGMENT(torchshifts, m) {
        m.def("_cuda_version", &shifts::cuda_version);
        m.def("shift1d", &shifts::ops::shift1d);
        m.def("shift2d", &shifts::ops::shift2d);
        m.def("shift3d", &shifts::ops::shift3d);    
}
