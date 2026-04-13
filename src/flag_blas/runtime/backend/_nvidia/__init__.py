import torch
import triton

from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="nvidia", device_name="cuda", device_query_cmd="nvidia-smi"
)
ARCH_MAP = {"9": "hopper", "8": "ampere"}


def _alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc_fn)


__all__ = ["*"]
