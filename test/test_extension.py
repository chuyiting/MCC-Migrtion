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
    batch_ids = torch.tensor([[0], [0], [0], [0]]).cuda()
    res = pt_mcc.ops.compute_aabb(pts, batch_ids, 1, True)
    print(f'the result is: {res[0].shape} {res}\n')

    print('##################### Test compute_pdf #####################')
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).cuda()
    batch = torch.tensor([[1], [1], [1], [1]]).cuda()
    aabb_min = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    aabb_max = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).cuda()
    # Define the starting indexes, indicating where each point's neighbors start in the neighbors list
    start_indexes = torch.tensor([[0], [2], [4], [6]]).cuda()

    # Define the neighbors list, where each point has two neighbors for simplicity
    neighbors = torch.tensor([
        [1, 0], [2, 0],  # Neighbors for point 0
        [0, 1], [3, 1],  # Neighbors for point 1
        [0, 2], [3, 2],  # Neighbors for point 2
        [1, 3], [2, 3]   # Neighbors for point 3
    ]).cuda()

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
    print (start_idx.device)
    print (neigh_idx.device)
    print(start_idx.cpu())
    print(neigh_idx.cpu())
    # print('Skip the test for find_neighbors for now... Find issues later...')

    print('##################### Test poisson_sampling #####################')
    res = pt_mcc.ops.poisson_sampling(pts, batch_ids, cell_indices, aabb_min, aabb_max, radius, batch_size, scale_inv)
    print(res)

    print('##################### Test get_sampled_features(pts_indices, features) #####################')
    pts_indices = torch.tensor([0, 3], dtype=torch.int).cuda()
    features = torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=torch.float, requires_grad=True).cuda()
    res = pt_mcc.ops.get_sampled_features(pts_indices, features)
    num_sampled_point = pts_indices.shape[0]
    num_feature = features.shape[1]
    grad = torch.ones(num_sampled_point, num_feature, dtype=torch.float).cuda()
    print(f'output features: {res}')
    res.backward(grad)
    print(f'grad: {res.grad}')


    unittest.main()
