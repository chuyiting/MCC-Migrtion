#include <torch/extension.h>
#include <cuda_runtime.h>

namespace pt_mcc
{
    // CUDA functions assumed to be defined elsewhere
    unsigned int countNeighborsCPU(
        const bool pScaleInv,
        const int pNumPoints,
        const int pNumCells,
        const double pRadius,
        const float *pInPts,
        const int *pInBatchIds,
        const float *pInPts2,
        const int *pCellIndexs,
        const float *pAABBMin,
        const float *pAABBMax,
        int *pStartIndex);

    void packNeighborsCPU(
        const bool pScaleInv,
        const int pNumPoints,
        const int pNumNeighbors,
        const int pNumCells,
        const double pRadius,
        const float *pInPts,
        const int *pInBatchIds,
        const float *pInPts2,
        const int *pCellIndexs,
        const float *pAABBMin,
        const float *pAABBMax,
        int *pAuxBuffOffsets,
        int *pAuxBuffOffsets2,
        int *pStartIndexs,
        int *pPackedIndexs);

    void computeAuxiliarBuffersSize(
        const int pNumPoints,
        int *PBufferSize1,
        int *PBufferSize2);

    std::tuple<torch::Tensor, torch::Tensor> find_neighbors(
        const torch::Tensor &points,
        const torch::Tensor &batch_ids,
        const torch::Tensor &points2,
        const torch::Tensor &cell_indices,
        const torch::Tensor &aabb_min,
        const torch::Tensor &aabb_max,
        double radius,
        int64_t batch_size,
        bool scale_inv)
    {
        // Check inputs
        TORCH_CHECK(radius > 0.0, "radius must be positive")
        TORCH_CHECK(batch_size > 0, "batch size must be positive")

        TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points should have dimensions (N, 3)");
        TORCH_CHECK(batch_ids.dim() == 2 && batch_ids.size(0) == points.size(0) && batch_ids.size(1) == 1, "batch_ids should have dimensions (N, 1)");
        TORCH_CHECK(points2.dim() == 2 && points2.size(1) == 3, "points2 should have dimensions (M, 3)");
        TORCH_CHECK(cell_indices.dim() == 5 && cell_indices.size(0) == batch_size, "cell_indices should have dimensions (B, numCells, ...)");
        TORCH_CHECK(aabb_min.dim() == 2 && aabb_min.size(0) == batch_size && aabb_min.size(1) == 3, "aabb_min should have dimensions (B, 3)");
        TORCH_CHECK(aabb_max.dim() == 2 && aabb_max.size(0) == batch_size && aabb_max.size(1) == 3, "aabb_max should have dimensions (B, 3)");

        // Convert tensors to pointers
        const float *points_ptr = points.data_ptr<float>();
        const int *batch_ids_ptr = batch_ids.data_ptr<int>();
        const float *points2_ptr = points2.data_ptr<float>();
        const int *cell_indices_ptr = cell_indices.data_ptr<int>();
        const float *aabb_min_ptr = aabb_min.data_ptr<float>();
        const float *aabb_max_ptr = aabb_max.data_ptr<float>();

        // Determine the number of neighbors
        int num_points = points.size(0);
        int num_cells = cell_indices.size(1);

        // Allocate output tensor for start indexes
        torch::Tensor start_indexes = torch::zeros({num_points, 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        int *start_indexes_ptr = start_indexes.data_ptr<int>();

        unsigned int num_neighbors = countNeighborsCPU(scale_inv, num_points, num_cells, radius,
                                                       points_ptr, batch_ids_ptr, points2_ptr, cell_indices_ptr, aabb_min_ptr, aabb_max_ptr, start_indexes_ptr);

        // Allocate output tensor for neighbor indexes
        torch::Tensor neigh_indexes = torch::zeros({static_cast<int64_t>(num_neighbors), 2}, torch::dtype(torch::kInt32));
        int *neigh_indexes_ptr = neigh_indexes.data_ptr<int>();

        // Allocate temporary buffers
        int tmp_buff1_size, tmp_buff2_size;
        computeAuxiliarBuffersSize(num_points, &tmp_buff1_size, &tmp_buff2_size);
        torch::Tensor tmp_buff1 = torch::empty({tmp_buff1_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        torch::Tensor tmp_buff2 = torch::empty({tmp_buff2_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        int *tmp_buff1_ptr = tmp_buff1.data_ptr<int>();
        int *tmp_buff2_ptr = tmp_buff2.data_ptr<int>();

        // Pack neighbors
        packNeighborsCPU(scale_inv, num_points, num_neighbors, num_cells, radius,
                         points_ptr, batch_ids_ptr, points2_ptr, cell_indices_ptr, aabb_min_ptr, aabb_max_ptr,
                         tmp_buff1_ptr, tmp_buff2_ptr, start_indexes_ptr, neigh_indexes_ptr);

        return std::make_tuple(start_indexes, neigh_indexes);
    }

    void register_find_neighbors(torch::Library &m)
    {
        m.def("find_neighbors(Tensor points, Tensor batch_ids, Tensor points2, Tensor cell_indices, Tensor aabb_min, Tensor aabb_max, float radius, int batch_size, bool scale_inv) -> (Tensor, Tensor)");
    }

    // Register CPU implementations
    TORCH_LIBRARY_IMPL(pt_mcc, CPU, m)
    {

        m.impl("find_neighbors", &find_neighbors);
    }

    // Register CUDA implementations
    TORCH_LIBRARY_IMPL(pt_mcc, CUDA, m)
    {
        m.impl("find_neighbors", &find_neighbors);
    }
}
