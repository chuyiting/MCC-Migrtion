import torch
from torch import Tensor

__all__ = ["mymuladd", "myadd_out", "compute_aabb", "compute_pdf"]

def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.pt_mcc.mymuladd.default(a, b, c)

def compute_aabb(pts: Tensor, batch_ids: Tensor, batch_size: int, inv_inf: bool):
    return torch.ops.pt_mcc.compute_aabb.default(pts, batch_ids, batch_size, inv_inf)

def compute_pdf(pts, batch_ids, aabb_min, aabb_max, start_indexes, neighbors, window, radius, batch_size, scale_inv):
    return torch.ops.pt_mcc.compute_pdf(pts, batch_ids, aabb_min, aabb_max, start_indexes, neighbors, window, radius, batch_size, scale_inv)

def find_neighbors(pts, batch_ids, pts2, cell_indices, aabb_min, aabb_max, radius, batch_size, scale_inv):
    return torch.ops.pt_mcc.find_neighbors(pts, batch_ids, pts2, cell_indices, aabb_min, aabb_max, radius, batch_size, scale_inv)

# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("pt_mcc::mymuladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)

@torch.library.register_fake("pt_mcc::compute_aabb")
def _(pts, batch_ids, batch_size, scale_inv):
    torch._check(pts.device == batch_ids.device)
    torch._check(pts.dim() == 2)
    torch._check(batch_ids.dim() == 2)
    torch._check(pts.shape[0] == batch_size.shape[0])
    torch._check(batch_size.dtype == torch.int)
    torch._check(scale_inv.dtype == torch.bool)
    return (torch.empty_like(pts), torch.empty_like(pts))

@torch.library.register_fake("pt_mcc::compute_pdf")
def _(pts, batch_ids, aabb_min, aabb_max, start_indexes, neighbors, window, radius, batch_size, scale_inv):
    torch._check(pts.device == batch_ids.device)
    torch._check(pts.dim() == 2)
    torch._check(batch_ids.dim() == 2)
    torch._check(aabb_min.dim() == 2)
    torch._check(aabb_max.dim() == 2)
    torch._check(start_indexes.dim() == 2)
    torch._check(neighbors.dim() == 2)
    torch._check(pts.shape[0] == batch_size.shape[0])
    torch._check(neighbors.shape[1] == 2)
    torch._check(aabb_min.shape[0] == batch_size)
    torch._check(aabb_max.shape[0] == batch_size)
    torch._check(window.dtype == torch.float)
    torch._check(radius.dtype == torch.float)
    torch._check(batch_size.dtype == torch.int)
    torch._check(scale_inv.dtype == torch.bool)
    num_neighbors = neighbors.shape[0]
    return torch.empty((num_neighbors, 1), dtype=torch.float)

@torch.library.register_fake("pt_mcc::find_neighbors")
def _(pts, batch_ids, pts2, cell_indices, aabb_min, aabb_max, radius, batch_size, scale_inv):
    torch._check(pts.device == batch_ids.device)
    torch._check(pts.dim() == 2)
    torch._check(pts2.dim() == 2)
    torch._check(cell_indices.dim() == 5 and cell_indices.shape[0] == batch_size)

    torch._check(batch_ids.dim() == 2)
    torch._check(aabb_min.dim() == 2)
    torch._check(aabb_max.dim() == 2)
    torch._check(aabb_min.shape[0] == batch_size)
    torch._check(aabb_max.shape[0] == batch_size)
    torch._check(radius.dtype == torch.float)
    torch._check(batch_size.dtype == torch.int)
    torch._check(scale_inv.dtype == torch.bool)

    num_pts = pts.shape[0]
    num_neigh = 100 # just some placeholder, it will be different for different input
    return torch.empty((num_pts, 1), dtype=torch.int), torch.empty((num_neigh, 1), dtype=torch.int)

def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.pt_mcc.mymul.default(grad, b)
    if ctx.needs_input_grad[1]:
        grad_b = torch.ops.pt_mcc.mymul.default(grad, a)
    return grad_a, grad_b, None


def _setup_context(ctx, inputs, output):
    a, b, c = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "pt_mcc::mymuladd", _backward, setup_context=_setup_context)


@torch.library.register_fake("pt_mcc::mymul")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def myadd_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    """Writes a + b into out"""
    torch.ops.pt_mcc.myadd_out.default(a, b, out)
