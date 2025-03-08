"""
Test the environment compatibility
"""

from torch import cuda

def test_cuda():
    """check cuda availability"""
    assert cuda.is_available()
