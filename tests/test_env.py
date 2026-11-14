"""
Test the environment compatibility
"""

import importlib.util as ut
import pytest


def cuda_available() -> bool:
    if ut.find_spec("torch") is None:
        return False
    import torch  # pylint: disable=C0415

    return torch.cuda.is_available()


requires_cuda = pytest.mark.skipif(not cuda_available(), reason="Cuda is not available")


@pytest.mark.parametrize("name", ["torch", "triton"])
def test_module(name: str):
    """Check if 'name' lib exists"""
    assert ut.find_spec(name) is not None


@requires_cuda
def test_torch_cuda():
    """check cuda availability under torch"""
    from torch import cuda  # pylint: disable=C0415

    assert cuda.is_available()


@requires_cuda
def test_triton_cuda():
    """check cuda availability under triton"""
    import triton  # pylint: disable=C0415

    assert triton.runtime.driver.active.get_current_target().backend == "cuda"
