#include <torch/extension.h>
#include <vector>

namespace pt_mcc
{
    int determineNumCells(
        const bool scaleInv,
        const int64_t batchSize,
        const double cellSize,
        const float *aabbMin,
        const float *aabbMax);

    void computeAuxiliarBuffersSize(
        const int64_t batchSize,
        const int numCells,
        int *bufferSize1,
        int *bufferSize2,
        int *bufferSize3);

    void sortPointsStep1GPUKernel(
        const int pNumPoints,
        const int64_t pBatchSize,
        const int pNumCells,
        const float *pAABBMin,
        const float *pAABBMax,
        const float *pPoints,
        const int *pBatchIds,
        int *pAuxBuffCounters,
        int *pAuxBuffOffsets,
        int *pAuxBuffOffsets2,
        int *pKeys,
        int *pNewIndexs);

    void sortPointsStep2GPUKernel(
        const int pNumPoints,
        const int64_t pBatchSize,
        const int pNumFeatures,
        const int pNumCells,
        const float *pPoints,
        const int *pBatchIds,
        const float *pFeatures,
        const int *pKeys,
        const int *pNewIndexs,
        int *pAuxBuffer,
        float *pOutPoints,
        int *pOutBatchIds,
        float *pOutFeatures,
        int *pOutCellIndexs);

    void sortPointsStep2GradGPUKernel(
        const int pNumPoints,
        const int pNumFeatures,
        const float *pOutGradients,
        const float *pOutFeatureGradients,
        const int *pNewIndexs,
        float *pInGradients,
        float *pInFeatureGradients);

    void sortFeaturesBack(
        const int pNumPoints,
        const int pNumFeatures,
        const float *pInFeatures,
        const int *pIndexs,
        float *pOutFeatures);

    void sortFeaturesBackGrad(
        const int pNumPoints,
        const int pNumFeatures,
        const float *pOutFeatureGrads,
        const int *pIndexs,
        float *pInFeatureGrads);

    void computeInverseIndexs(
        const int pNumPoints,
        const int *pIndexs,
        int *pOutIndexs);

    void transformIndexs(
        const int pNumIndexs,
        const int pNumPoints,
        const int *pInStartIndexs,
        const int *pInNewIndexs,
        int *pOutIndexs);

    /**
     * Did not fully grasp the implementation details, but the idea is to sort each poinnt into a 3D grid cell
     */
    std::tuple<torch::Tensor, torch::Tensor> sort_points_step1(
        const torch::Tensor &points,
        const torch::Tensor &batch_ids,
        const torch::Tensor &aabb_min,
        const torch::Tensor &aabb_max,
        const int64_t batch_size,
        const double cell_size,
        const bool scale_inv)
    {
        TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "Points tensor must have shape (N, 3)");
        TORCH_CHECK(batch_ids.dim() == 2 && batch_ids.size(0) == points.size(0) && batch_ids.size(1) == 1, "Batch IDs tensor must have shape (N)");
        TORCH_CHECK(aabb_min.dim() == 2 && aabb_min.size(0) == batch_size && aabb_min.size(1) == 3, "AABB min tensor must have shape (batch_size, 3)");
        TORCH_CHECK(aabb_max.dim() == 2 && aabb_max.size(0) == batch_size && aabb_max.size(1) == 3, "AABB max tensor must have shape (batch_size, 3)");

        int num_points = points.size(0);

        // Output tensors
        auto keys = torch::zeros({num_points}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        auto new_indices = torch::zeros({num_points}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        // Prepare CUDA pointers
        const float *points_ptr = points.data_ptr<float>();
        const int *batch_ids_ptr = batch_ids.data_ptr<int>();
        const float *aabb_min_ptr = aabb_min.data_ptr<float>();
        const float *aabb_max_ptr = aabb_max.data_ptr<float>();
        int *keys_ptr = keys.data_ptr<int>();
        int *new_indices_ptr = new_indices.data_ptr<int>();

        // Determine the number of cells based on inputs
        int num_cells = determineNumCells(scale_inv, batch_size, cell_size, aabb_min_ptr, aabb_max_ptr);

        // Auxiliary buffers
        int buffer_size1, buffer_size2, buffer_size3;
        computeAuxiliarBuffersSize(batch_size, num_cells, &buffer_size1, &buffer_size2, &buffer_size3);

        auto aux_buffer_counters = torch::zeros({buffer_size1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        auto aux_buffer_offsets = torch::zeros({buffer_size2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        auto aux_buffer_offsets2 = torch::zeros({buffer_size3}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        int *aux_buffer_counters_ptr = aux_buffer_counters.data_ptr<int>();
        int *aux_buffer_offsets_ptr = aux_buffer_offsets.data_ptr<int>();
        int *aux_buffer_offsets2_ptr = aux_buffer_offsets2.data_ptr<int>();

        // Call the CUDA kernel
        sortPointsStep1GPUKernel(
            num_points, batch_size, num_cells,
            aabb_min_ptr, aabb_max_ptr,
            points_ptr, batch_ids_ptr,
            aux_buffer_counters_ptr, aux_buffer_offsets_ptr, aux_buffer_offsets2_ptr,
            keys_ptr, new_indices_ptr);

        return std::make_tuple(keys, new_indices);
    }
    /**
     * Apply the sorted indices computed from sort_points_step1 to sort the points, batch_ids, features, and cell_indices
     */
    std::vector<torch::Tensor> sort_points_step2(
        torch::Tensor points, torch::Tensor batch_ids,
        torch::Tensor features, torch::Tensor keys,
        torch::Tensor new_indices, torch::Tensor aabb_min,
        torch::Tensor aabb_max, int64_t batch_size, double cell_size, bool scale_inv)
    {
        // Validate inputs
        TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
                    "Expected points to have shape (num_points, 3)");
        int num_points = points.size(0);
        TORCH_CHECK(batch_ids.dim() == 2 && batch_ids.size(0) == num_points && batch_ids.size(1) == 1,
                    "Expected batch_ids to have shape (num_points, 1)");
        TORCH_CHECK(features.dim() == 2 && features.size(1) > 0,
                    "Expected features to have shape (num_points, num_features)");
        int num_features = features.size(1);
        TORCH_CHECK(keys.dim() == 1 && keys.size(0) == num_points,
                    "Expected keys to have shape (num_points)");
        TORCH_CHECK(new_indices.dim() == 1 && new_indices.size(0) == num_points,
                    "Expected new_indices to have shape (num_points)");
        TORCH_CHECK(aabb_min.dim() == 2 && aabb_min.size(0) == batch_size && aabb_min.size(1) == 3,
                    "Expected aabb_min to have shape (batch_size, 3)");
        TORCH_CHECK(aabb_max.dim() == 2 && aabb_max.size(0) == batch_size && aabb_max.size(1) == 3,
                    "Expected aabb_max to have shape (batch_size, 3)");

        // Compute the number of cells
        const float *aabb_min_ptr = aabb_min.data_ptr<float>();
        const float *aabb_max_ptr = aabb_max.data_ptr<float>();
        int num_cells = determineNumCells(scale_inv, batch_size, cell_size, aabb_min_ptr, aabb_max_ptr);

        // Allocate output tensors
        auto out_points = torch::empty_like(points, torch::TensorOptions().device(torch::kCUDA));
        auto out_batch_ids = torch::empty_like(batch_ids, torch::TensorOptions().device(torch::kCUDA));
        auto out_features = torch::empty_like(features, torch::TensorOptions().device(torch::kCUDA));
        auto out_cell_indices = torch::zeros({batch_size, num_cells, num_cells, num_cells, 2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        // Allocate temporary buffer
        auto temp_buffer = torch::empty({num_points}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        // Pointers to input data
        const float *points_ptr = points.data_ptr<float>();
        const int *batch_ids_ptr = batch_ids.data_ptr<int>();
        const float *features_ptr = features.data_ptr<float>();
        const int *keys_ptr = keys.data_ptr<int>();
        const int *new_indices_ptr = new_indices.data_ptr<int>();

        // Pointers to output data
        float *out_points_ptr = out_points.data_ptr<float>();
        int *out_batch_ids_ptr = out_batch_ids.data_ptr<int>();
        float *out_features_ptr = out_features.data_ptr<float>();
        int *out_cell_indices_ptr = out_cell_indices.data_ptr<int>();
        int *temp_buffer_ptr = temp_buffer.data_ptr<int>();

        // Call the GPU kernel
        sortPointsStep2GPUKernel(num_points, batch_size, num_features, num_cells,
                                 points_ptr, batch_ids_ptr, features_ptr, keys_ptr,
                                 new_indices_ptr, temp_buffer_ptr,
                                 out_points_ptr, out_batch_ids_ptr, out_features_ptr, out_cell_indices_ptr);

        return {out_points, out_batch_ids, out_features, out_cell_indices};
    }

    std::tuple<torch::Tensor, torch::Tensor> sort_points_step2_grad(
        const torch::Tensor &new_indices,
        const torch::Tensor &output_grad,
        const torch::Tensor &output_feature_grad)
    {
        // Check the dimensions and types
        TORCH_CHECK(new_indices.dim() == 1, "Expected new_indices to be 1D (num_points)");
        TORCH_CHECK(output_grad.dim() == 2 && output_grad.size(1) >= 3, "Expected output_grad to be 2D with at least 3 components per point");
        TORCH_CHECK(output_feature_grad.dim() == 2 && output_feature_grad.size(1) > 0, "Expected output_feature_grad to be 2D with at least one component per point");

        int num_points = new_indices.size(0);
        int num_features = output_feature_grad.size(1);

        // Ensure tensors are on the same device and of compatible types
        TORCH_CHECK(new_indices.device().is_cuda(), "Input tensors must be on CUDA device");
        TORCH_CHECK(output_grad.device().is_cuda(), "Input tensors must be on CUDA device");
        TORCH_CHECK(output_feature_grad.device().is_cuda(), "Input tensors must be on CUDA device");

        // Allocate output tensors
        auto out_input_grads = torch::empty_like(output_grad);                 // will be on CUDA device
        auto out_input_feature_grads = torch::empty_like(output_feature_grad); // will be on CUDA device

        // Get pointers to data
        const int *new_indexs_ptr = new_indices.data_ptr<int>();
        const float *output_grad_ptr = output_grad.data_ptr<float>();
        const float *output_feature_grad_ptr = output_feature_grad.data_ptr<float>();
        float *out_input_grads_ptr = out_input_grads.data_ptr<float>();
        float *out_input_feature_grads_ptr = out_input_feature_grads.data_ptr<float>();

        // Call the GPU kernel
        sortPointsStep2GradGPUKernel(
            num_points,
            num_features,
            output_grad_ptr,
            output_feature_grad_ptr,
            new_indexs_ptr,
            out_input_grads_ptr,
            out_input_feature_grads_ptr);

        // Return both output tensors as a tuple
        return std::make_tuple(out_input_grads, out_input_feature_grads);
    }

    torch::Tensor sort_features_back(
        torch::Tensor features,
        torch::Tensor new_indices)
    {
        // Check input shapes
        TORCH_CHECK(new_indices.dim() == 1, "sort_features_back expects indices with shape (num_points)");
        int num_points = new_indices.size(0);

        TORCH_CHECK(features.dim() == 2, "sort_features_back expects features with shape (num_points, num_features)");
        int num_features = features.size(1);

        TORCH_CHECK(features.device().is_cuda(), "Input tensors must be on CUDA device");
        TORCH_CHECK(new_indices.device().is_cuda(), "Input tensors must be on CUDA device");

        // Allocate output tensor
        auto out_features = torch::empty_like(features); // features should be in CUDA device

        // Call the kernel function
        sortFeaturesBack(
            num_points,
            num_features,
            features.data_ptr<float>(),
            new_indices.data_ptr<int>(),
            out_features.data_ptr<float>());

        return out_features;
    }

    torch::Tensor sort_features_back_grad(
        torch::Tensor new_indices,
        torch::Tensor output_feature_grad)
    {

        // Check input shapes
        TORCH_CHECK(new_indices.dim() == 1, "sort_features_back_grad expects indices with shape (num_points)");
        int num_points = new_indices.size(0);

        TORCH_CHECK(output_feature_grad.dim() == 2, "sort_features_back_grad expects gradients of features with shape (num_points, num_features)");
        int num_features = output_feature_grad.size(1);

        TORCH_CHECK(output_feature_grad.device().is_cuda(), "Input tensors must be on CUDA device");
        TORCH_CHECK(new_indices.device().is_cuda(), "Input tensors must be on CUDA device");

        // Allocate output tensor
        auto out_input_feature_grads = torch::empty_like(output_feature_grad);

        // Call the kernel function
        sortFeaturesBackGrad(
            num_points,
            num_features,
            output_feature_grad.data_ptr<float>(),
            new_indices.data_ptr<int>(),
            out_input_feature_grads.data_ptr<float>());

        return out_input_feature_grads;
    }

    torch::Tensor transform_indices(
        torch::Tensor start_indices,
        torch::Tensor new_indices)
    {

        // Validate input tensor shapes
        TORCH_CHECK(start_indices.dim() == 1, "transform_indices expects start_indices with shape (num_indices)");
        int num_indices = start_indices.size(0);

        TORCH_CHECK(new_indices.dim() == 1, "transform_indices expects new_indices with shape (num_points)");
        int num_points = new_indices.size(0);

        TORCH_CHECK(start_indices.device().is_cuda(), "Input tensors must be on CUDA device");
        TORCH_CHECK(new_indices.device().is_cuda(), "Input tensors must be on CUDA device");

        // Allocate temporary tensor
        auto tmp_indices = torch::empty_like(new_indices);

        // Call the kernel for computing inverse indices
        computeInverseIndexs(
            num_points,
            new_indices.data_ptr<int>(),
            tmp_indices.data_ptr<int>());

        // Allocate output tensor
        auto out_indices = torch::empty_like(start_indices);

        // Call the kernel for transforming indices
        transformIndexs(
            num_indices,
            num_points,
            start_indices.data_ptr<int>(),
            tmp_indices.data_ptr<int>(),
            out_indices.data_ptr<int>());

        return out_indices;
    }

    void register_sort(torch::Library &m)
    {
        m.def("sort_points_step1(Tensor points, Tensor batch_ids, Tensor aabb_min, Tensor aabb_max, int batch_size, float cell_size, bool scale_inv) -> (Tensor, Tensor)");
        m.def("sort_points_step2(Tensor points, Tensor batch_ids, Tensor features, Tensor keys, Tensor new_indices, Tensor aabb_min, Tensor aabb_max, int batch_size, float cell_size, bool scale_inv) -> (Tensor, Tensor, Tensor, Tensor)");
        m.def("sort_points_step2_grad(Tensor new_indices, Tensor output_grad, Tensor output_feature_grad) -> (Tensor, Tensor)");
        m.def("sort_features_back(Tensor features, Tensor new_indices) -> Tensor");
        m.def("sort_features_back_grad(Tensor new_indices, Tensor output_feature_grad) -> Tensor");
        m.def("transform_indices(Tensor start_indices, Tensor new_indices) -> Tensor");
    }

    // Register CUDA implementations
    TORCH_LIBRARY_IMPL(pt_mcc, CUDA, m)
    {
        m.impl("sort_points_step1", &sort_points_step1);
        m.impl("sort_points_step2", &sort_points_step2);
        m.impl("sort_points_step2_grad", &sort_points_step2_grad);
        m.impl("sort_features_back", &sort_features_back);
        m.impl("sort_features_back_grad", &sort_features_back_grad);
        m.impl("transform_indices", &transform_indices);
    }

}
