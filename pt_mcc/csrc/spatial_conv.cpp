#include <torch/extension.h>
#include <vector>

#define BLOCK_MLP_SIZE 128 // Example size; adjust as needed

void spatialConvCPU(
    bool pAvg,
    bool pScaleInv,
    int pNumNeighbors,
    int pNumInFeatures,
    int pNumOutFeatures,
    int pNumSamples,
    bool pCombin,
    float pRadius,
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
    float pRadius,
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

torch::Tensor spatial_conv(
    torch::Tensor in_points, torch::Tensor in_features, torch::Tensor batch_ids, torch::Tensor in_pdfs,
    torch::Tensor in_samples, torch::Tensor start_index, torch::Tensor packed_neigh, torch::Tensor in_aabb_min,
    torch::Tensor in_aabb_max, torch::Tensor in_weights_hidd1, torch::Tensor in_bias_hidd1,
    torch::Tensor in_weights_hidd2, torch::Tensor in_bias_hidd2, torch::Tensor in_weights_out,
    torch::Tensor in_bias_out, int64_t num_out_features, bool combin, double radius, int64_t batch_size,
    bool scale_inv, bool avg)
{
    // Input validation
    TORCH_CHECK(num_out_features > 0, "SpatialConvOp expects a positive number of output features")
    TORCH_CHECK(radius > 0.0, "SpatialConvOp expects a positive radius")
    TORCH_CHECK(batch_size > 0, "SpatialConvOp expects a positive batch size")
    TORCH_CHECK(in_points.dim() == 2 && in_points.size(1) == 3, "in_points must be of shape (num_points, 3)");
    int num_points = in_points.size(0);

    TORCH_CHECK(in_features.dim() == 2 && in_features.size(0) == in_points.size(0), "in_features must be of shape (num_points, num_in_features)");
    int num_in_features = in_features.size(1);

    TORCH_CHECK(batch_ids.dim() == 2 && batch_ids.size(1) == 1 && batch_ids.size(0) == in_points.size(0), "batch_ids must be of shape (num_points, 1)");
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
        TORCH_CHECK(in_weights_out.size(1) == in_features.size(1), "Input and output features must match for combin=false");
    }

    // Initialize output tensor
    int num_samples = in_samples.size(0);
    torch::Tensor out_conv_features = torch::zeros({num_samples, num_out_features}, torch::dtype(torch::kFloat32).device(in_points.device()));

    // Call the kernel function
    spatial_conv_cpu(
        avg, scale_inv, num_neighs, num_in_features, num_out_features, num_samples, combin, radius,
        in_points.data_ptr<float>(), batch_ids.data_ptr<int>(), in_features.data_ptr<float>(),
        in_pdfs.data_ptr<float>(), in_samples.data_ptr<float>(), start_index.data_ptr<int>(),
        packed_neigh.data_ptr<int>(), in_aabb_min.data_ptr<float>(), in_aabb_max.data_ptr<float>(),
        in_weights_hidd1.data_ptr<float>(), in_bias_hidd1.data_ptr<float>(), in_weights_hidd2.data_ptr<float>(),
        in_bias_hidd2.data_ptr<float>(), in_weights_out.data_ptr<float>(), in_bias_out.data_ptr<float>(),
        out_conv_features.data_ptr<float>());

    return out_conv_features;
}
