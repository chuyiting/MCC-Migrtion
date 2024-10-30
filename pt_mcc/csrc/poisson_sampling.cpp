#include <torch/extension.h>
#include <cuda_runtime.h>

using namespace at;

namespace pt_mcc
{

    int samplePointCloud(
        bool scaleInv,
        float pRadius,
        int pNumPoints,
        int pBatchSize,
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
    Tensor poisson_sampling(
        const Tensor &points,
        const Tensor &batch_ids,
        const Tensor &cell_indexs,
        const Tensor &aabb_min,
        const Tensor &aabb_max,
        float radius,
        int batch_size,
        bool scale_inv)
    {

        int num_points = points.size(0);
        int num_cells = cell_indexs.size(1);

        Tensor tmp_pts = at::empty({num_points, 3}, points.options());
        Tensor tmp_batchs = at::empty({num_points, 1}, batch_ids.options());
        Tensor tmp_indexs = at::empty({num_points, 1}, batch_ids.options());
        Tensor tmp_used_bool = at::empty({num_points, 1}, at::kBool);

        int num_sel_samples = samplePointCloud(
            scale_inv, radius, num_points, batch_size, num_cells,
            aabb_min.data_ptr<float>(), aabb_max.data_ptr<float>(),
            points.data_ptr<float>(), batch_ids.data_ptr<int>(),
            cell_indexs.data_ptr<int>(), tmp_pts.data_ptr<float>(),
            tmp_batchs.data_ptr<int>(), tmp_indexs.data_ptr<int>(),
            tmp_used_bool.data_ptr<bool>);

        Tensor out_pts = at::empty({num_sel_samples, 3}, points.options());
        Tensor out_batchs = at::empty({num_sel_samples, 1}, batch_ids.options());
        Tensor out_indexs = at::empty({num_sel_samples}, batch_ids.options());

        copyPoints(tmp_pts.data_ptr<float>(), tmp_batchs.data_ptr<int>(), tmp_indexs.data_ptr<int>(),
                   num_sel_samples, out_pts.data_ptr<float>(), out_batchs.data_ptr<int>(),
                   out_indexs.data_ptr<int>());

        return std::make_tuple(out_pts, out_batchs, out_indexs);
    }

    // Feature sampling function
    Tensor get_sampled_features(const Tensor &pts_indexs, const Tensor &features)
    {
        int num_sampled_points = pts_indexs.size(0);
        int num_features = features.size(1);
        Tensor out_sel_features = at::empty({num_sampled_points, num_features}, features.options());

        getFeaturesSampledPoints(features.size(0), num_features, num_sampled_points,
                                 pts_indexs.data_ptr<int>(), features.data_ptr<float>(),
                                 out_sel_features.data_ptr<float>());

        return out_sel_features;
    }

    // Gradient function for feature sampling
    Tensor get_sampled_features_grad(const Tensor &pts_indexs, const Tensor &in_features, const Tensor &sampled_features_grad)
    {
        int num_sampled_points = pts_indexs.size(0);
        int num_features = in_features.size(1);
        Tensor out_in_features_grad = at::empty({in_features.size(0), num_features}, in_features.options());

        getFeaturesSampledPointsGradients(in_features.size(0), num_features, num_sampled_points,
                                          pts_indexs.data_ptr<int>(), sampled_features_grad.data_ptr<float>(),
                                          out_in_features_grad.data_ptr<float>());

        return out_in_features_grad;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
    {
        m.def("poisson_sampling", &poisson_sampling, "Poisson Disk Sampling (CUDA)");
        m.def("get_sampled_features", &get_sampled_features, "Get Sampled Features (CUDA)");
        m.def("get_sampled_features_grad", &get_sampled_features_grad, "Get Sampled Features Gradient (CUDA)");
    }

}