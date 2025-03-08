"""
Test the environment compatibility
"""

import importlib.util as ut
import pytest


@pytest.mark.parametrize("name", ["torch", "triton"])
def test_module(name: str):
    """Check if 'name' lib exists"""
    assert ut.find_spec(name) is not None


def test_cuda():
    """check cuda availability"""
    from torch import cuda   # pylint: disable=C0415
    assert cuda.is_available()
