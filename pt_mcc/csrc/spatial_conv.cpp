#include <torch/extension.h>

#ifndef BLOCK_MLP_SIZE
#define BLOCK_MLP_SIZE 8
#endif

namespace pt_mcc
{
    void spatialConvCPU(
        bool pAvg,
        bool pScaleInv,
        int pNumNeighbors,
        int pNumInFeatures,
        int pNumOutFeatures,
        int pNumSamples,
        bool pCombin,
        double pRadius,
        const float *pInPoints,
        const int *pBatchIds,
        const float *pInFeatures,
        const float *pPDFs,
        const float *pSamples,
        const int *pStartIndexs,
        const int *pPackedNeighs,
        const float *pAABBMin,
        const float *pAABBMax,
        const float *pWeights1,
        const float *pBiases1,
        const float *pWeights2,
        const float *pBiases2,
        const float *pWeightsOut,
        const float *pBiasesOut,
        float *pOutFeatues);

    void spatialConvGradsCPU(
        bool pAvg,
        bool pScaleInv,
        int pNumNeighbors,
        int pNumInFeatures,
        int pNumOutFeatures,
        int pNumSamples,
        int pNumSamples2,
        bool pCombin,
        double pRadius,
        const float *pInPoints,
        const int *pBatchIds,
        const float *pInFeatures,
        const float *pPDFs,
        const float *pSamples,
        const int *pStartIndexs,
        const int *pPackedNeighs,
        const float *pAABBMin,
        const float *pAABBMax,
        const float *pWeights1,
        const float *pBiases1,
        const float *pWeights2,
        const float *pBiases2,
        const float *pWeightsOut,
        const float *pBiasesOut,
        const float *pInOutFeatueGrads,
        float *pOutFeatureGrads,
        float *pWeights1Grads,
        float *pWeight2Grads,
        float *pWeightOutGrads,
        float *pBiases1Grads,
        float *pBiases2Grads,
        float *pBiasesOutGrads);

    int64_t get_block_size()
    {
        return BLOCK_MLP_SIZE;
    }

    torch::Tensor spatial_conv(
        torch::Tensor in_points, torch::Tensor in_features, torch::Tensor batch_ids, torch::Tensor in_pdfs,
        torch::Tensor in_samples, torch::Tensor start_index, torch::Tensor packed_neigh, torch::Tensor in_aabb_min,
        torch::Tensor in_aabb_max, torch::Tensor in_weights_hidd1, torch::Tensor in_weights_hidd2, torch::Tensor in_weights_out,
        torch::Tensor in_bias_hidd1, torch::Tensor in_bias_hidd2, torch::Tensor in_bias_out,
        int64_t num_out_features, bool combin, int64_t batch_size, double radius,
        bool scale_inv, bool avg)
    {
        TORCH_CHECK(in_points.is_cuda() && in_features.is_cuda() && batch_ids.is_cuda() && in_pdfs.is_cuda() && in_samples.is_cuda(), "all inputs should be on CUDA - 1");
        TORCH_CHECK(start_index.is_cuda() && packed_neigh.is_cuda() && in_aabb_min.is_cuda() && in_aabb_max.is_cuda(), "all inputs should be on CUDA - 2");
        TORCH_CHECK(in_weights_hidd1.is_cuda() && in_weights_hidd2.is_cuda() && in_weights_out.is_cuda(), "all inputs should be on CUDA - 3");
        TORCH_CHECK(in_bias_hidd1.is_cuda() && in_bias_hidd2.is_cuda() && in_bias_out.is_cuda(), "all inputs should be on CUDA - 4");
        // Input validation
        TORCH_CHECK(num_out_features > 0, "SpatialConvOp expects a positive number of output features")
        TORCH_CHECK(radius > 0.0, "SpatialConvOp expects a positive radius")
        TORCH_CHECK(batch_size > 0, "SpatialConvOp expects a positive batch size")
        TORCH_CHECK(in_points.dim() == 2 && in_points.size(1) == 3, "in_points must be of shape (num_points, 3)");
        int num_points = in_points.size(0);

        TORCH_CHECK(in_features.dim() == 2 && in_features.size(0) == num_points, "in_features must be of shape (num_points, num_in_features)");
        int num_in_features = in_features.size(1);

        TORCH_CHECK(batch_ids.dim() == 2 && batch_ids.size(1) == 1 && batch_ids.size(0) == num_points, "batch_ids must be of shape (num_points, 1)");
        TORCH_CHECK(in_pdfs.dim() == 2 && in_pdfs.size(1) == 1, "in_pdfs must be of shape (num_neighs, 1)");
        int num_neighs = in_pdfs.size(0);

        TORCH_CHECK(in_samples.dim() == 2 && in_samples.size(1) == 3, "in_samples must be of shape (num_samples, 3)");
        int num_samples = in_samples.size(0);

        TORCH_CHECK(start_index.dim() == 2 && start_index.size(1) == 1 && start_index.size(0) == num_samples, "start_index must be of shape (num_samples, 1)");
        TORCH_CHECK(packed_neigh.dim() == 2 && packed_neigh.size(0) == num_neighs && packed_neigh.size(1) == 2, "packed_neigh must be of shape (num_neighs, 2)");
        TORCH_CHECK(in_aabb_min.dim() == 2 && in_aabb_min.size(0) == batch_size && in_aabb_min.size(1) == 3, "in_aabb_min must be of shape (batch_size, 3)");
        TORCH_CHECK(in_aabb_max.dim() == 2 && in_aabb_max.size(0) == batch_size && in_aabb_max.size(1) == 3, "in_aabb_max must be of shape (batch_size, 3)");
        TORCH_CHECK(in_weights_hidd1.dim() == 2 && in_weights_hidd1.size(0) == 3 && in_bias_hidd1.dim() == 1 &&
                        in_weights_hidd1.size(1) == in_bias_hidd1.size(0) && in_weights_hidd1.size(1) % BLOCK_MLP_SIZE == 0,
                    "Invalid shape for first hidden layer weights and bias");
        TORCH_CHECK(in_weights_hidd2.dim() == 2 && in_weights_hidd2.size(0) == BLOCK_MLP_SIZE &&
                        in_bias_hidd2.dim() == 1 && in_weights_hidd2.size(1) == in_bias_hidd2.size(0) &&
                        in_weights_hidd2.size(1) == in_weights_hidd1.size(1),
                    "Invalid shape for second hidden layer weights and bias");
        TORCH_CHECK(in_weights_out.dim() == 2 && in_weights_out.size(0) == BLOCK_MLP_SIZE && in_bias_out.dim() == 1 &&
                        in_weights_out.size(1) == in_bias_out.size(0),
                    "Invalid shape for output layer weights and bias");
        TORCH_CHECK(in_weights_out.size(1) % in_features.size(1) == 0, "Output neurons must be multiple of input features");
        if (!combin)
        {
            TORCH_CHECK(in_weights_out.size(1) == num_in_features, "Input and output features must match for combin=false");
        }

        // Initialize output tensor
        torch::Tensor out_conv_features = torch::zeros({num_samples, num_out_features}, torch::dtype(torch::kFloat32).device(in_points.device()));

        // Call the kernel function
        spatialConvCPU(
            avg, scale_inv, num_neighs, num_in_features, num_out_features, num_samples, combin, radius,
            in_points.data_ptr<float>(), batch_ids.data_ptr<int>(), in_features.data_ptr<float>(),
            in_pdfs.data_ptr<float>(), in_samples.data_ptr<float>(), start_index.data_ptr<int>(),
            packed_neigh.data_ptr<int>(), in_aabb_min.data_ptr<float>(), in_aabb_max.data_ptr<float>(),
            in_weights_hidd1.data_ptr<float>(), in_bias_hidd1.data_ptr<float>(), in_weights_hidd2.data_ptr<float>(),
            in_bias_hidd2.data_ptr<float>(), in_weights_out.data_ptr<float>(), in_bias_out.data_ptr<float>(),
            out_conv_features.data_ptr<float>());

        return out_conv_features;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> spatial_conv_grad(
        torch::Tensor in_points,
        torch::Tensor in_features,
        torch::Tensor batch_ids,
        torch::Tensor in_pdfs,
        torch::Tensor in_samples,
        torch::Tensor start_index,
        torch::Tensor packed_neigh,
        torch::Tensor in_aabb_min,
        torch::Tensor in_aabb_max,
        torch::Tensor in_weights_hidd1,
        torch::Tensor in_weights_hidd2,
        torch::Tensor in_weights_out,
        torch::Tensor in_bias_hidd1,
        torch::Tensor in_bias_hidd2,
        torch::Tensor in_bias_out,
        torch::Tensor in_out_feature_grads,
        int64_t num_out_features,
        bool combin,
        int64_t batch_size,
        double radius,
        bool scale_inv,
        bool avg)
    {
        // Input validation checks
        TORCH_CHECK(in_points.dim() == 2 && in_points.size(1) == 3, "in_points must have dimensions (batchSize, 3)");
        int num_points = in_points.size(0);

        TORCH_CHECK(in_features.dim() == 2 && in_features.size(0) == num_points, "in_features must have dimensions (numPoints, numInFeatures)");
        int num_in_features = in_features.size(1);

        TORCH_CHECK(batch_ids.dim() == 2 && batch_ids.size(1) == 1 && batch_ids.size(0) == num_points, "batch_ids must have dimensions (numPoints, 1)");
        TORCH_CHECK(in_pdfs.dim() == 2 && in_pdfs.size(1) == 1, "in_pdfs must have dimensions (num_neighs, 1)");
        int num_neighs = in_pdfs.size(0);

        TORCH_CHECK(in_samples.dim() == 2 && in_samples.size(1) == 3, "in_samples must have dimensions (numSamples, 3)");
        int num_samples = in_samples.size(0);

        TORCH_CHECK(start_index.dim() == 2 && start_index.size(1) == 1 && start_index.size(0) == num_samples, "start_index must have dimensions (numSamples, 1)");
        TORCH_CHECK(packed_neigh.dim() == 2 && packed_neigh.size(0) == num_neighs && packed_neigh.size(1) == 2, "packed_neigh must have dimensions (numNeighs, 2)");
        TORCH_CHECK(in_aabb_min.dim() == 2 && in_aabb_min.size(0) == batch_size && in_aabb_min.size(1) == 3, "in_aabb_min must have dimensions (batchSize, 3)");
        TORCH_CHECK(in_aabb_max.dim() == 2 && in_aabb_max.size(0) == batch_size && in_aabb_max.size(1) == 3, "in_aabb_max must have dimensions (batchSize, 3)");
        TORCH_CHECK(in_weights_hidd1.dim() == 2 && in_weights_hidd1.size(0) == 3 &&
                        in_bias_hidd1.dim() == 1 && in_weights_hidd1.size(1) == in_bias_hidd1.size(0) &&
                        in_weights_hidd1.size(1) % BLOCK_MLP_SIZE == 0,
                    "First hidden layer dimensions are incorrect");
        TORCH_CHECK(in_weights_hidd2.dim() == 2 && in_weights_hidd2.size(0) == BLOCK_MLP_SIZE &&
                        in_bias_hidd2.dim() == 1 && in_weights_hidd2.size(1) == in_bias_hidd2.size(0) &&
                        in_weights_hidd2.size(1) == in_weights_hidd1.size(1),
                    "Second hidden layer dimensions are incorrect");
        TORCH_CHECK(in_weights_out.dim() == 2 && in_weights_out.size(0) == BLOCK_MLP_SIZE &&
                        in_bias_out.dim() == 1 && in_weights_out.size(1) == in_bias_out.size(0),
                    "Output layer dimensions are incorrect");
        TORCH_CHECK(in_weights_out.size(1) % in_features.size(1) == 0,
                    "Output layer output neurons must be a multiple of input features");

        if (!combin)
        {
            TORCH_CHECK(in_weights_out.size(1) == in_features.size(1),
                        "If combin is false, input and output features must match");
        }
        TORCH_CHECK(in_out_feature_grads.dim() == 2 && in_out_feature_grads.size(0) == in_samples.size(0) &&
                        in_out_feature_grads.size(1) == num_out_features,
                    "in_out_feature_grads dimensions are incorrect");

        // Prepare output tensors
        auto feature_gradients = torch::zeros_like(in_features);
        auto weight1_grads = torch::zeros_like(in_weights_hidd1);
        auto bias1_grads = torch::zeros_like(in_bias_hidd1);
        auto weight2_grads = torch::zeros_like(in_weights_hidd2);
        auto bias2_grads = torch::zeros_like(in_bias_hidd2);
        auto weight_out_grads = torch::zeros_like(in_weights_out);
        auto bias_out_grads = torch::zeros_like(in_bias_out);

        // Extract pointers
        const float *in_points_ptr = in_points.data_ptr<float>();
        const float *in_features_ptr = in_features.data_ptr<float>();
        const int *batch_ids_ptr = batch_ids.data_ptr<int>();
        const float *in_pdfs_ptr = in_pdfs.data_ptr<float>();
        const float *in_samples_ptr = in_samples.data_ptr<float>();
        const int *start_index_ptr = start_index.data_ptr<int>();
        const int *packed_neigh_ptr = packed_neigh.data_ptr<int>();
        const float *in_aabb_min_ptr = in_aabb_min.data_ptr<float>();
        const float *in_aabb_max_ptr = in_aabb_max.data_ptr<float>();
        const float *in_weights_hidd1_ptr = in_weights_hidd1.data_ptr<float>();
        const float *in_bias_hidd1_ptr = in_bias_hidd1.data_ptr<float>();
        const float *in_weights_hidd2_ptr = in_weights_hidd2.data_ptr<float>();
        const float *in_bias_hidd2_ptr = in_bias_hidd2.data_ptr<float>();
        const float *in_weights_out_ptr = in_weights_out.data_ptr<float>();
        const float *in_bias_out_ptr = in_bias_out.data_ptr<float>();
        const float *in_out_feature_grads_ptr = in_out_feature_grads.data_ptr<float>();

        float *feature_gradients_ptr = feature_gradients.data_ptr<float>();
        float *weight1_grads_ptr = weight1_grads.data_ptr<float>();
        float *bias1_grads_ptr = bias1_grads.data_ptr<float>();
        float *weight2_grads_ptr = weight2_grads.data_ptr<float>();
        float *bias2_grads_ptr = bias2_grads.data_ptr<float>();
        float *weight_out_grads_ptr = weight_out_grads.data_ptr<float>();
        float *bias_out_grads_ptr = bias_out_grads.data_ptr<float>();

        // Call the CPU function
        spatialConvGradsCPU(
            avg, scale_inv, num_neighs, num_in_features, num_out_features, num_samples, num_points,
            combin, radius, in_points_ptr, batch_ids_ptr, in_features_ptr, in_pdfs_ptr, in_samples_ptr,
            start_index_ptr, packed_neigh_ptr, in_aabb_min_ptr, in_aabb_max_ptr, in_weights_hidd1_ptr,
            in_bias_hidd1_ptr, in_weights_hidd2_ptr, in_bias_hidd2_ptr, in_weights_out_ptr, in_bias_out_ptr,
            in_out_feature_grads_ptr, feature_gradients_ptr, weight1_grads_ptr, weight2_grads_ptr,
            weight_out_grads_ptr, bias1_grads_ptr, bias2_grads_ptr, bias_out_grads_ptr);

        // Return outputs as a tuple
        return {feature_gradients, weight1_grads, bias1_grads, weight2_grads,
                bias2_grads, weight_out_grads, bias_out_grads};
    }

    void register_spatial_connv(torch::Library &m)
    {
        m.def("spatial_conv(Tensor in_points, Tensor in_features, Tensor batch_ids, Tensor in_pdfs, Tensor in_samples, Tensor start_index, Tensor packed_neigh, Tensor in_aabb_min, Tensor in_aabb_max, Tensor in_weights_hidd1, Tensor in_weights_hidd2, Tensor in_weights_out, Tensor in_bias_hidd1, Tensor in_bias_hidd2, Tensor in_bias_out, int num_out_features, bool combin, int batch_size, float radius, bool scale_inv, bool avg) -> Tensor");
        m.def("spatial_conv_grad(Tensor in_points, Tensor in_features, Tensor batch_ids, Tensor in_pdfs, Tensor in_samples, Tensor start_index, Tensor packed_neigh, Tensor in_aabb_min, Tensor in_aabb_max, Tensor in_weights_hidd1, Tensor in_weights_hidd2, Tensor in_weights_out, Tensor in_bias_hidd1, Tensor in_bias_hidd2, Tensor in_bias_out, Tensor in_out_feature_grads, int num_out_features, bool combin, int batch_size, float radius, bool scale_inv, bool avg) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
        m.def("get_block_size", &get_block_size);
    }

    // Register CUDA implementations
    TORCH_LIBRARY_IMPL(pt_mcc, CUDA, m)
    {
        m.impl("spatial_conv", &spatial_conv);
        m.impl("spatial_conv_grad", &spatial_conv_grad);
    }
}
