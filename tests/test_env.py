"""
Test the environment compatibility
"""

import importlib.util as ut
import pytest


@pytest.mark.parametrize("name", ["torch", "triton"])
def test_module(name: str):
    """Check if 'name' lib exists"""
    assert ut.find_spec(name) is not None


def test_torch_cuda():
    """check cuda availability under torch"""
    from torch import cuda   # pylint: disable=C0415
    assert cuda.is_available()


def test_triton_cuda():
    """check cuda availability under triton"""
    import triton   # pylint: disable=C0415
    assert triton.runtime.driver.active.get_current_target().backend == "cuda"
