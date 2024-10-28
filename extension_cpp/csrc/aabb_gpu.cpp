/////////////////////////////////////////////////////////////////////////////
/// \file aabb_gpu.cc
///
/// \brief C++ operation definition to compute the axis aligned bounding box
///        of a batch of point clouds.
///
/// \copyright Copyright (c) 2018 Visual Computing group of Ulm University,
///            Germany. See the LICENSE file at the top-level directory of
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include <torch/extension.h>
#include <tuple>

namespace extension_cpp
{

    void computeAABB(
        const bool pScaleInv, const int pNumPoints, const int pBatchSize,
        const float *pPoints, const int *pBatchIds, float *pAABBMin, float *pAABBMax);

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

        return 10;
    }

    void register_aabb(torch::Library &m)
    {
        m.def("compute_aabb(Tensor points, Tensor batchIds, int batchSize, bool scaleInv) -> int");
    }

    TORCH_LIBRARY_IMPL(extension_cpp, CPU, m)
    {
        m.impl("compute_aabb", &compute_aabb); // define compute_aabb here
    }
}