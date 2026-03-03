"""
flag_blas - BLAS operations implemented with Triton
"""

from . import ops

from . import runtime
from . import testing
device = runtime.device.name
vendor_name = runtime.device.vendor_name
__version__ = "0.1.0"

__all__ = ["ops"]
