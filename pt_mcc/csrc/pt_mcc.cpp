#include <torch/extension.h>

namespace pt_mcc
{

    // void register_aabb(torch::Library &m);
    void register_muladd(torch::Library &m);
}

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
}

// Defines the operators
TORCH_LIBRARY(pt_mcc, m)
{
    // extension_cpp::register_aabb(m);
    pt_mcc::register_muladd(m);
}
