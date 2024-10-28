#include <torch/extension.h>

namespace extension_cpp
{

    void register_aabb(torch::Library &m);
    void register_muladd(torch::Library &m);
    void impl_muladd(torch::Library &m);

}

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
}

// Defines the operators
TORCH_LIBRARY(extension_cpp, m)
{
    extension_cpp::register_aabb(m);
    extension_cpp::register_muladd(m);
}
