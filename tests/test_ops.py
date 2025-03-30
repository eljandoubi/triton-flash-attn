import torch
import pytest
from ops.add import Add


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available")
def test_add():
    x = torch.randn((7, 3, 64, 64, 5), device="cuda", requires_grad=True)
    y = torch.randn((7, 3, 64, 64, 5), device="cuda", requires_grad=True)
    dz = torch.randn((7, 3, 64, 64, 5), device="cuda")
    z_ref = x + y
    z_ref.backward(dz, retain_graph=True)
    dx_ref, dy_ref = [_.grad.clone() for _ in [x, y]]
    x.grad, y.grad = None, None
    z_output = Add.apply(x, y)
    z_output.backward(dz)
    dx_tri, dy_tri = x.grad, y.grad
    assert torch.norm(z_output - z_ref, p=1) == 0
    assert torch.norm(dx_ref - dx_tri, p=1) == 0
    assert torch.norm(dy_ref - dy_tri, p=1) == 0
