#include <torch/extension.h>
#include <cuda_runtime.h>

namespace pt_mcc
{
    void computeDPFsCPU(
        const bool pScaleInv,
        const float pWindow,
        const int64_t numSamples,
        const int64_t pNumNeighbors,
        const float pRadius,
        const float *pInPts,
        const int64_t *pInBatchIds,
        const float *pAABBMin,
        const float *pAABBMax,
        const int64_t *pStartIndexs,
        const int64_t *pPackedIndexs,
        float *pPDFs);

    // Define the computePDF function
    torch::Tensor compute_pdf(
        const torch::Tensor &points,
        const torch::Tensor &batch_ids,
        const torch::Tensor &aabb_min,
        const torch::Tensor &aabb_max,
        const torch::Tensor &start_indexes,
        const torch::Tensor &neighbors,
        float window,
        float radius,
        int64_t batch_size,
        bool scale_inv)
    {

        // Check input tensor dimensions and types
        TORCH_CHECK(window > 0.0, "window must be positive")
        TORCH_CHECK(radius > 0.0, "radius must be positive")
        TORCH_CHECK(batch_size > 0, "batch size must be positive")

        TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "Points should have shape (N, 3)");
        int num_points = points.size(0);

        TORCH_CHECK(batch_ids.dim() == 2 && batch_ids.size(1) == 1, "Batch IDs should have shape (N, 1)");
        TORCH_CHECK(batch_ids.size(0) == numPoints, "batch_ids should match the first dimension of points");

        TORCH_CHECK(start_indexes.dim() == 2 && start_indexes.size(1) == 1, "Start indexes (Samples) should have shape (N_sample, 1)");
        int num_samples = start_indexes.size(0);

        TORCH_CHECK(neighbors.dim() == 2 && neighbors.size(1) == 2,
                    "neighbors should have shape (N_neighbor, 2))");
        int num_neighbors = neighbors.size(0);

        TORCH_CHECK(aabb_min.dim() == 2 && aabb_min.size(1) == 3 && aabb_min.size(0) == batch_size,
                    "aabb_min must have shape (batch_size, 3)");
        TORCH_CHECK(aabb_max.dim() == 2 && aabb_max.size(1) == 3 && aabb_max.size(0) == batch_size,
                    "aabb_max must have shape (batch_size, 3)");

        // Allocate output tensor for the PDFs
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(points.device());
        torch::Tensor pdfs = torch::empty({num_neighbors, 1}, options);

        // Call the CPU or CUDA kernel (implement computeDPFsCPU accordingly)
        computeDPFsCPU(
            scale_inv, window, num_samples, num_neighbors, radius,
            points.data_ptr<float>(),
            batch_ids.data_ptr<int64_t>(),
            aabb_min.data_ptr<float>(),
            aabb_max.data_ptr<float>(),
            start_indexes.data_ptr<int64_t>(),
            neighbors.data_ptr<int64_t>(),
            pdfs.data_ptr<float>());

        return pdfs;
    }

    void register_aabb(torch::Library &m)
    {
        m.def("compute_pdf(Tensor points, Tensor batch_ids, Tensor start_indexes, Tensor neighbors, Tensor aabb_min, Tensor aabb_max, float window, float radius, int batch_size, bool scale_inv) -> Tensor");
    }

    // Register CPU implementations
    TORCH_LIBRARY_IMPL(pt_mcc, CPU, m)
    {

        m.impl("compute_pdf", &compute_pdf);
    }

    // Register CUDA implementations
    TORCH_LIBRARY_IMPL(pt_mcc, CUDA, m)
    {
        m.impl("compute_pdf", &compute_pdf);
    }
}
