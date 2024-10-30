#include <torch/extension.h>
#include <cuda_runtime.h>

namespace pt_mcc
{

    int samplePointCloud(
        bool scaleInv,
        double pRadius,
        int pNumPoints,
        int64_t pBatchSize,
        int pNumCells,
        const float *pAABBMin,
        const float *pAABBMax,
        const float *pPoints,
        const int *pBatchIds,
        const int *pCellIndexs,
        float *pSelectedPts,
        int *pSelectedBatchIds,
        int *pSelectedIndexs,
        bool *pAuxBoolBuffer);

    void copyPoints(
        float *pSelectedPts,
        int *pSelectedBatchIds,
        int *pSelectedIndexs,
        int pNumPts,
        float *pDestPts,
        int *pDestBatchIds,
        int *pDestIndexs);

    void getFeaturesSampledPoints(
        int pNumPoints,
        int pNumFeatures,
        int pNumSampledPoints,
        const int *pInPointsIndexs,
        const float *pInFeature,
        float *pOutSelFeatures);

    void getFeaturesSampledPointsGradients(
        int pNumPoints,
        int pNumFeatures,
        int pNumSampledPoints,
        const int *pInPointsIndexs,
        const float *pInOutFeatureGrad,
        float *pOutInFeaturesGradients);

    // Poisson sampling function
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> poisson_sampling(
        const torch::Tensor &points,
        const torch::Tensor &batch_ids,
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
        TORCH_CHECK(cell_indices.dim() == 5 && cell_indices.size(0) == batch_size, "cell_indices should have dimensions (B, numCells, ...)");
        TORCH_CHECK(aabb_min.dim() == 2 && aabb_min.size(0) == batch_size && aabb_min.size(1) == 3, "aabb_min should have dimensions (B, 3)");
        TORCH_CHECK(aabb_max.dim() == 2 && aabb_max.size(0) == batch_size && aabb_max.size(1) == 3, "aabb_max should have dimensions (B, 3)");

        int num_points = points.size(0);
        int num_cells = cell_indices.size(1);

        printf("number of points: %d\n", num_points);
        torch::Tensor tmp_pts = torch::empty({num_points, 3}, points.options());
        torch::Tensor tmp_batchs = torch::empty({num_points, 1}, batch_ids.options());
        torch::Tensor tmp_indexs = torch::empty({num_points, 1}, batch_ids.options());
        torch::Tensor tmp_used_bool = torch::empty({num_points, 1}, torch::kBool);
        // Calculate the size in bytes
        size_t total_size = tmp_used_bool.numel() * tmp_used_bool.element_size();
        // Print the size
        std::cout << "Size of tmp_used_bool tensor: " << total_size << " bytes" << std::endl;

        int num_sel_samples = samplePointCloud(
            scale_inv, radius, num_points, batch_size, num_cells,
            aabb_min.data_ptr<float>(), aabb_max.data_ptr<float>(),
            points.data_ptr<float>(), batch_ids.data_ptr<int>(),
            cell_indices.data_ptr<int>(), tmp_pts.data_ptr<float>(),
            tmp_batchs.data_ptr<int>(), tmp_indexs.data_ptr<int>(),
            tmp_used_bool.data_ptr<bool>());

        torch::Tensor out_pts = torch::empty({num_sel_samples, 3}, points.options());
        torch::Tensor out_batchs = torch::empty({num_sel_samples, 1}, batch_ids.options());
        torch::Tensor out_indices = torch::empty({num_sel_samples}, batch_ids.options());

        copyPoints(tmp_pts.data_ptr<float>(), tmp_batchs.data_ptr<int>(), tmp_indexs.data_ptr<int>(),
                   num_sel_samples, out_pts.data_ptr<float>(), out_batchs.data_ptr<int>(),
                   out_indices.data_ptr<int>());

        return std::make_tuple(out_pts, out_batchs, out_indices);
    }

    // Feature sampling function
    torch::Tensor get_sampled_features(const torch::Tensor &pts_indices, const torch::Tensor &features)
    {
        TORCH_CHECK(pts_indices.dim() == 1, "point indices should have dimensions (N)");
        TORCH_CHECK(features.dim() == 2, "features should have dimensions (N, m)");
        int num_sampled_points = pts_indices.size(0);
        int num_points = features.size(0);
        int num_features = features.size(1);

        torch::Tensor out_sel_features = torch::empty({num_sampled_points, num_features}, features.options());

        getFeaturesSampledPoints(num_points, num_features, num_sampled_points,
                                 pts_indices.data_ptr<int>(), features.data_ptr<float>(),
                                 out_sel_features.data_ptr<float>());

        return out_sel_features;
    }

    // Gradient function for feature sampling
    torch::Tensor get_sampled_features_grad(const torch::Tensor &pts_indices, const torch::Tensor &features, const torch::Tensor &sampled_features_grad)
    {
        TORCH_CHECK(pts_indices.dim() == 1, "point indices should have dimensions (N)");
        TORCH_CHECK(features.dim() == 2, "features should have dimensions (N, m)");
        int num_sampled_points = pts_indices.size(0);
        int num_points = features.size(0);
        int num_features = features.size(1);

        TORCH_CHECK(sampled_features_grad.dim() == 2 && sampled_features_grad.size(0) == num_sampled_points && sampled_features_grad.size(1) == num_features, "features should have dimensions (N, m)");
        torch::Tensor out_features_grad = torch::empty({num_points, num_features}, features.options());

        getFeaturesSampledPointsGradients(num_points, num_features, num_sampled_points,
                                          pts_indices.data_ptr<int>(), sampled_features_grad.data_ptr<float>(),
                                          out_features_grad.data_ptr<float>());

        return out_features_grad;
    }

    void register_poisson_sampling(torch::Library &m)
    {
        m.def("poisson_sampling(Tensor points, Tensor batch_ids, Tensor cell_indices, Tensor aabb_min, Tensor aabb_max, float radius, int batch_size, bool scale_inv) -> (Tensor, Tensor, Tensor)");
        m.def("get_sampled_features(Tensor pts_indices, Tensor features) ->  Tensor");
        m.def("get_sampled_features_grad(Tensor pts_indices, Tensor features, Tensor sampled_features_grad) ->  Tensor");
    }

    // Register CPU implementations
    TORCH_LIBRARY_IMPL(pt_mcc, CPU, m)
    {
        m.impl("poisson_sampling", &poisson_sampling);
        m.impl("get_sampled_features", &get_sampled_features);
        m.impl("get_sampled_features_grad", &get_sampled_features_grad);
    }

    // Register CUDA implementations
    TORCH_LIBRARY_IMPL(pt_mcc, CUDA, m)
    {
        m.impl("poisson_sampling", &poisson_sampling);
        m.impl("get_sampled_features", &get_sampled_features);
        m.impl("get_sampled_features_grad", &get_sampled_features_grad);
    }

}