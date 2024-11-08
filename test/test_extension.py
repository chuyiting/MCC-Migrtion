import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import pt_mcc
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F


def reference_muladd(a, b, c):
    return a * b + c

class TestMyMulAdd(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(3), make_tensor(3), 1],
            [make_tensor(20), make_tensor(20), 3.14],
            [make_tensor(20), make_nondiff_tensor(20), -123],
            [make_nondiff_tensor(2, 3), make_tensor(2, 3), -0.3],
        ]

    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            result = pt_mcc.ops.mymuladd(*args)
            expected = reference_muladd(*args)
            torch.testing.assert_close(result, expected)

    def test_correctness_cpu(self):
        self._test_correctness("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        self._test_correctness("cuda")

    def _test_gradients(self, device):
        samples = self.sample_inputs(device, requires_grad=True)
        for args in samples:
            diff_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
            out = pt_mcc.ops.mymuladd(*args)
            grad_out = torch.randn_like(out)
            result = torch.autograd.grad(out, diff_tensors, grad_out)

            out = reference_muladd(*args)
            expected = torch.autograd.grad(out, diff_tensors, grad_out)

            torch.testing.assert_close(result, expected)

    def test_gradients_cpu(self):
        self._test_gradients("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_gradients_cuda(self):
        self._test_gradients("cuda")

    def _opcheck(self, device):
        # Use opcheck to check for incorrect usage of operator registration APIs
        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            opcheck(torch.ops.pt_mcc.mymuladd.default, args)

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestMyAddOut(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(3), make_tensor(3), make_tensor(3)],
            [make_tensor(20), make_tensor(20), make_tensor(20)],
        ]

    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            result = args[-1]
            pt_mcc.ops.myadd_out(*args)
            expected = torch.add(*args[:2])
            torch.testing.assert_close(result, expected)

    def _opcheck(self, device):
        # Use opcheck to check for incorrect usage of operator registration APIs
        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            opcheck(torch.ops.pt_mcc.myadd_out.default, args)

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


if __name__ == "__main__":
    print('##################### Test compute_aabb #####################')
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).cuda()
    batch_ids = torch.tensor([[0], [0], [0], [0]]).int().cuda()
    res = pt_mcc.ops.compute_aabb(pts, batch_ids, 1, True)
    print(f'the result is: {res[0].shape} {res}\n')

    print('##################### Test compute_pdf #####################')
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).cuda()
    batch = torch.tensor([[1], [1], [1], [1]]).int().cuda()
    aabb_min = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    aabb_max = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).cuda()
    # Define the starting indexes, indicating where each point's neighbors start in the neighbors list
    start_indexes = torch.tensor([[0], [2], [4], [6]]).int().cuda()

    # Define the neighbors list, where each point has two neighbors for simplicity
    neighbors = torch.tensor([
        [1, 0], [2, 0],  # Neighbors for point 0
        [0, 1], [3, 1],  # Neighbors for point 1
        [0, 2], [3, 2],  # Neighbors for point 2
        [1, 3], [2, 3]   # Neighbors for point 3
    ]).int().cuda()

    # Set other parameters
    window = 1.0
    radius = 1.0
    batch_size = 4
    scale_inv = True
    res = pt_mcc.ops.compute_pdf(pts, batch, aabb_min, aabb_max, start_indexes, neighbors, window, radius, batch_size, scale_inv)
    print(f'the result is: {res}\n')

    print('##################### Test find_neighbors #####################')
    pts = torch.tensor([[0.1, 0.2, 0.3],[0.5, 0.5, 0.5], [0.7, 0.7, 0.7], [0.9, 0.9, 0.9]], dtype=torch.float32).cuda()
    batch_ids = torch.tensor([[0], [0], [0], [0]], dtype=torch.int32).cuda()  # All points belong to batch 0
    pts2 = torch.tensor([[0.15, 0.25, 0.35], [0.55, 0.55, 0.55], [0.7, 0.7, 0.7], [0.95, 0.95, 0.95]], dtype=torch.float32).cuda()  # Points in the second set to check as potential neighbors

    # Define cell_indices in 5D: [batch_size, pNumCells, pNumCells, pNumCells, 2]
    # For simplicity, assume each cell contains exactly one point (start and end indices are consecutive)
    cell_indices = torch.tensor([[
        [[[0, 2]]]
    ]], dtype=torch.int32).cuda()  # Batch 0, 1x1x1 cells

    aabb_min = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32).cuda()  # Minimum AABB for batch 0

    aabb_max = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).cuda()  # Maximum AABB for batch 0
    radius = 1  # Radius for neighbor search
    batch_size = 1  # Single batch
    scale_inv = False  # Do not scale radius by AABB size

    # Call the find_neighbors function
    start_idx, neigh_idx = pt_mcc.ops.find_neighbors(pts, batch_ids, pts2, cell_indices, aabb_min, aabb_max, radius, batch_size, scale_inv)
    print(start_idx.cpu())
    print(neigh_idx.cpu())
    # print('Skip the test for find_neighbors for now... Find issues later...')

    print('##################### Test poisson_sampling #####################')
    out_pts, out_batchs, out_indices = pt_mcc.ops.poisson_sampling(pts, batch_ids, cell_indices, aabb_min, aabb_max, radius, batch_size, scale_inv)
    out_pts1, out_batchs1, out_indice1= pt_mcc.ops.poisson_sampling(pts, batch_ids, cell_indices, aabb_min, aabb_max, radius, batch_size, scale_inv)
    print(f'stable output: {torch.allclose(out_pts, out_pts1)}  {torch.allclose(out_batchs1, out_batchs)} {torch.allclose(out_indices, out_indice1)}')
    
    print('##################### Test get_sampled_features(pts_indices, features) #####################')
    pts_indices = torch.tensor([0, 3], dtype=torch.int).cuda()
    features = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=torch.float32).cuda()
    features.requires_grad_()
    res = pt_mcc.ops.get_sampled_features(pts_indices, features)
    num_sampled_point = pts_indices.shape[0]
    num_feature = features.shape[1]
    grad = torch.ones(num_sampled_point, num_feature, dtype=torch.float).cuda()
    print(f'output features: {res}')
    res.backward(grad)
    print(f'grad: {features.grad}')

    print('##################### Test sort_points_step1 #####################')
    batch_size = 2
    num_points = 4
    cell_size = 0.5
    scale_inv = True

    # Example input tensors
    pts = torch.tensor([[0.1, 0.2, 0.3],
                        [0.9, 0.8, 0.7],
                        [0.4, 0.5, 0.6],
                        [1.0, 1.1, 1.2]], dtype=torch.float32, device="cuda")
    
    batch_ids = torch.tensor([[0], [0], [1], [1]], dtype=torch.int32, device="cuda")

    aabb_min = torch.tensor([[0.0, 0.0, 0.0],
                                [0.5, 0.5, 0.5]], dtype=torch.float32, device="cuda")
    
    aabb_max = torch.tensor([[1.0, 1.0, 1.0],
                                [1.5, 1.5, 1.5]], dtype=torch.float32, device="cuda")

    # Run the function
    keys, new_indices = pt_mcc.ops.sort_points_step1(pts, batch_ids, aabb_min, aabb_max, batch_size, cell_size, scale_inv)
    print(f'keys: {keys.cpu()}')
    print(f'new_indices: {new_indices.cpu()}')

    print('##################### Test sort_points_step2 #####################')
    features = torch.tensor([[0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0],
                        [3.0, 3.0, 3.0]], dtype=torch.float32, device="cuda")
    features.requires_grad_()
    out_points, out_batch_ids, out_features, out_cell_indices = pt_mcc.ops.sort_points_step2(pts, batch_ids, features, keys, new_indices, aabb_min, aabb_max, batch_size, cell_size, scale_inv)
    print(f'out points: {out_points}')
    print(f'out batch ids: {out_batch_ids}')
    print(f'out features: {out_features}')
    print(f'out cell indices: {out_cell_indices}')



    print('##################### Test sort_features_back #####################')
    features = torch.tensor([[0, 0, 0], # Features for point 0
                            [1, 1, 1], # Features for point 1
                            [2, 2, 2], # Features for point 2
                            [3, 3, 3], # Features for point 3
                            [4, 4, 4]], dtype=torch.float32).cuda()
    num_points, num_features = features.shape
    features.requires_grad_()
    # Define new_indices that will sort the features in reverse order
    new_indices = torch.tensor([4, 3, 2, 1, 0], dtype=torch.int32).cuda()

    out_features = pt_mcc.ops.sort_features_back(features, new_indices)
    print(out_features.cpu())

    grad = torch.ones(num_points, num_features, dtype=torch.float).cuda()
    out_features.backward(grad)
    print(f'grad: {out_features.grad}')

    print('##################### Test sort_features #####################')
    features = torch.tensor([[0, 0, 0],  # Features for point 0
                            [1, 1, 1],  # Features for point 1
                            [2, 2, 2],  # Features for point 2
                            [3, 3, 3],# Features for point 3
                            [4, 4, 4]], dtype=torch.float32).cuda()
    num_points, num_features = features.shape
    # Define new_indices that will sort the features in reverse order
    new_indices = torch.tensor([4, 3, 2, 1, 0], dtype=torch.int32).cuda()
    features.requires_grad_()

    out_features = pt_mcc.ops.sort_features(features, new_indices)
    print(out_features.cpu())

    grad = torch.tensor([[4, 4, 4],  # Features for point 0
                        [3, 3, 3],  # Features for point 1
                        [2, 2, 2],  # Features for point 2
                        [1, 1, 1],# Features for point 3
                        [0, 0, 0]], dtype=torch.float32).cuda()
    out_features.backward(grad)
    print(f'grad: {features.grad}')

    print('##################### Test get_block_size #####################')
    block_size = pt_mcc.ops.get_block_size()
    print(block_size)

    print('##################### Test conv #####################')

    # Mock data (smaller sizes for easier debugging)
    num_points = 4
    num_neighs = 3
    num_samples = 2
    num_out_features = 2
    num_in_features = 3
    batch_size = 2
    radius = 1.0
    block_mlp_size = 2  # Matches size of hidden layers

    # Input tensors
    in_points = torch.tensor([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6],
                            [0.7, 0.8, 0.9],
                            [1.0, 1.1, 1.2]], device='cuda', dtype=torch.float32)

    in_features = torch.tensor([[1.0, 1.0],
                                [0.9, 0.8],
                                [0.6, 0.5],
                                [0.3, 0.2]], device='cuda', dtype=torch.float32)

    batch_ids = torch.tensor([[0], [1], [0], [1]], device='cuda', dtype=torch.int32)

    in_pdfs = torch.tensor([[1.0], [0.8], [0.5]], device='cuda', dtype=torch.float32)

    in_samples = torch.tensor([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]], device='cuda', dtype=torch.float32)

    start_index = torch.tensor([[0], [1]], device='cuda', dtype=torch.int32)

    packed_neigh = torch.tensor([[0, 1],
                                [1, 2],
                                [2, 3]], device='cuda', dtype=torch.int32)

    in_aabb_min = torch.tensor([[0.0, 0.0, 0.0],
                                [0.5, 0.5, 0.5]], device='cuda', dtype=torch.float32)

    in_aabb_max = torch.tensor([[1.0, 1.0, 1.0],
                                [1.5, 1.5, 1.5]], device='cuda', dtype=torch.float32)

    # First hidden layer weights and bias
    # Shape (3, 8) for weights to satisfy in_weights_hidd1.size(1) % BLOCK_MLP_SIZE == 0 and in_weights_hidd1.size(1) == in_bias_hidd1.size(0)
    in_weights_hidd1 = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                    [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                                    [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]], 
                                    device='cuda', dtype=torch.float32)

    in_bias_hidd1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], device='cuda', dtype=torch.float32)

    # Second hidden layer weights and bias
    # Shape (8, 8) for weights to satisfy in_weights_hidd2.size(0) == BLOCK_MLP_SIZE and matching in_weights_hidd1.size(1)
    in_weights_hidd2 = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                    [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                                    [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
                                    [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2],
                                    [3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
                                    [4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8],
                                    [4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6],
                                    [5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4]], 
                                    device='cuda', dtype=torch.float32)

    in_bias_hidd2 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], device='cuda', dtype=torch.float32)

    # Output layer weights and bias
    # Shape (8, 8) for weights to match BLOCK_MLP_SIZE and also ensure compatibility with in_weights_hidd2 size
    in_weights_out = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                                [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
                                [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2],
                                [3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
                                [4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8],
                                [4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6],
                                [5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4]], 
                                device='cuda', dtype=torch.float32)

    in_bias_out = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], device='cuda', dtype=torch.float32)


    # Scalar tensors
    num_out_features_tensor = torch.tensor(num_out_features, device='cuda', dtype=torch.int64)
    combin_tensor = torch.tensor(True, device='cuda', dtype=torch.bool)  # Example with combin=True
    batch_size_tensor = torch.tensor(batch_size, device='cuda', dtype=torch.int64)
    radius_tensor = torch.tensor(radius, device='cuda', dtype=torch.float64)
    scale_inv_tensor = torch.tensor(True, device='cuda', dtype=torch.bool)
    avg_tensor = torch.tensor(True, device='cuda', dtype=torch.bool)

    # Call the function
    output = pt_mcc.ops.spatial_conv(
        in_points, in_features, batch_ids, in_pdfs,
        in_samples, start_index, packed_neigh, in_aabb_min,
        in_aabb_max, in_weights_hidd1, in_weights_hidd2, in_weights_out,
        in_bias_hidd1, in_bias_hidd2, in_bias_out,
        num_out_features_tensor, combin_tensor, batch_size_tensor, radius_tensor,
        scale_inv_tensor, avg_tensor
    )
    output1 = pt_mcc.ops.spatial_conv(
        in_points, in_features, batch_ids, in_pdfs,
        in_samples, start_index, packed_neigh, in_aabb_min,
        in_aabb_max, in_weights_hidd1, in_weights_hidd2, in_weights_out,
        in_bias_hidd1, in_bias_hidd2, in_bias_out,
        num_out_features_tensor, combin_tensor, batch_size_tensor, radius_tensor,
        scale_inv_tensor, avg_tensor
    )
    
    print("Output shape:", output.shape)  # Expected shape: (num_samples, num_out_features)
    print("Output:", output)
    print(f'stable output: {torch.allclose(output1, output)}')

    unittest.main()
