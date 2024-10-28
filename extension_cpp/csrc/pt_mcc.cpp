#include <torch/script.h>

namespace extension_cpp
{

    void register_aabb(torch::Library &m);
    void register_muladd(torch::Library &m);
    void impl_muladd(torch::Library &m);

    // Registers _C as a Python extension module.
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
    {
    }

    // Defines the operators
    TORCH_LIBRARY(extension_cpp, m)
    {
        pt_ops::register_aabb(m);
        pt_ops::register_muladd(m);
    }

    // Registers CUDA implementations for mymuladd, mymul, myadd_out
    TORCH_LIBRARY_IMPL(extension_cpp, CPU, m)
    {
        pt_ops::impl_muladd(m);
    }
}
