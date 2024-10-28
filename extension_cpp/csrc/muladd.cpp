#include <torch/extension.h>

#include <vector>

namespace extension_cpp
{

  at::Tensor mymuladd_cpu(const at::Tensor &a, const at::Tensor &b, double c)
  {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    const float *a_ptr = a_contig.data_ptr<float>();
    const float *b_ptr = b_contig.data_ptr<float>();
    float *result_ptr = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); i++)
    {
      result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
    }
    return result;
  }

  at::Tensor mymul_cpu(const at::Tensor &a, const at::Tensor &b)
  {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    const float *a_ptr = a_contig.data_ptr<float>();
    const float *b_ptr = b_contig.data_ptr<float>();
    float *result_ptr = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); i++)
    {
      result_ptr[i] = a_ptr[i] * b_ptr[i];
    }
    return result;
  }

  // An example of an operator that mutates one of its inputs.
  void myadd_out_cpu(const at::Tensor &a, const at::Tensor &b, at::Tensor &out)
  {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(b.sizes() == out.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_CHECK(out.dtype() == at::kFloat);
    TORCH_CHECK(out.is_contiguous());
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CPU);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    const float *a_ptr = a_contig.data_ptr<float>();
    const float *b_ptr = b_contig.data_ptr<float>();
    float *result_ptr = out.data_ptr<float>();
    for (int64_t i = 0; i < out.numel(); i++)
    {
      result_ptr[i] = a_ptr[i] + b_ptr[i];
    }
  }

  std::tuple<torch::Tensor, at::Tensor> compute_aabb(
      torch::Tensor points, torch::Tensor batchIds, int64_t batchSize, bool scaleInv)
  {
    // Check input tensor dimensions and types
    TORCH_CHECK(points.dim() == 2, "Points should have 2 dimensions (numPoints, pointComponents)");
    TORCH_CHECK(points.size(1) >= 3, "Points should have at least 3 components");
    TORCH_CHECK(batchIds.dim() == 1, "Batch IDs should have 1 dimension (numPoints)");

    int numPoints = points.size(0);
    auto pointSize = points.size(1);

    // Ensure batch_ids size matches num_points
    TORCH_CHECK(batchIds.size(0) == numPoints, "Batch IDs should have the same number of points");

    // Allocate output tensors
    torch::Tensor aabbMin = torch::empty({batchSize, 3}, points.options());
    torch::Tensor aabbMax = torch::empty({batchSize, 3}, points.options());

    // Call the CUDA kernel (passing raw pointers to the tensors)
    // computeAABB(
    //     scaleInv, numPoints, batchSize,
    //     points.data_ptr<float>(), batchIds.data_ptr<int>(),
    //     aabbMin.data_ptr<float>(), aabbMax.data_ptr<float>());

    return std::make_tuple(aabbMin, aabbMax);
  }

  // Defines the operators
  void register_muladd(torch::Library &m)
  {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
    m.def("mymul(Tensor a, Tensor b) -> Tensor");
    m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
    m.def("compute_aabb(Tensor points, Tensor batchIds, int batchSize, bool scaleInv) -> (Tensor, Tensor)");
  }

  // Registers CPU implementations for mymuladd, mymul, myadd_out
  TORCH_LIBRARY_IMPL(extension_cpp, CPU, m)
  {

    m.impl("mymuladd", &mymuladd_cpu);
    m.impl("mymul", &mymul_cpu);
    m.impl("myadd_out", &myadd_out_cpu);
    m.impl("compute_aabb", &compute_aabb); // define compute_aabb here
  }
}
