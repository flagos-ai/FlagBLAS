"""
flag_blas - BLAS operations implemented with Triton
"""


import torch
from packaging import version
from flag_blas import runtime
from flag_blas import testing
from flag_blas import ops
device = runtime.device.name
vendor_name = runtime.device.vendor_name
__version__ = "0.1.0"

__all__ = ["ops"]
