# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def get_stream_id(device_fn, device):
    """Return a stable stream handle for CUDA-like torch backends."""
    stream = device_fn.current_stream(device)
    for attr in ("cuda_stream", "musa_stream", "npu_stream", "stream_id"):
        value = getattr(stream, attr, None)
        if value is not None:
            return value
    return id(stream)
